#include "acr_source_linearization.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <new>
#include <utility>

namespace imprint::acr {
namespace {

constexpr std::uint32_t kMainU16Entries = 0x10000u;
constexpr std::uint32_t kMainU8Entries = 0x100u;
constexpr double kRationalOverflow = 1966080000.0;
constexpr std::uint32_t kRationalMaxDenominator = 0x100u;

bool finite(double value) noexcept { return std::isfinite(value); }

// The recovered rounding sequence is `x + (x <= 0 ? -0.5 : 0.5)` followed by
// a truncating conversion and an explicit int32 range check
// ("Overflow in Round_int32", FUN_012e9c48(100000, ...)).
bool round_half_away(double value, std::int32_t* out) noexcept {
    if (!finite(value)) return false;
    double adjusted = value <= 0.0 ? value - 0.5 : value + 0.5;
    if (!(adjusted > -2147483649.0 && adjusted < 2147483648.0)) return false;
    *out = static_cast<std::int32_t>(adjusted);
    return true;
}

// FUN_0130312c saturating uint16 store: values above 0xffff become 0xffff,
// negative values become 0, in-range values are truncated (not rounded).
std::uint16_t saturate_u16(std::int32_t value) noexcept {
    std::uint32_t bits = static_cast<std::uint32_t>(value);
    if (bits <= 0xffffu) return static_cast<std::uint16_t>(bits);
    return value >= 0 ? std::uint16_t{0xffff} : std::uint16_t{0};
}

// Branch A clamp: FEMM fminnm(d, 1.0) then max(d, 0.0). fminnm returns the
// non-NaN operand, so a NaN here becomes 1.0 and then survives the lower bound.
double clamp01_fminnm(double value) noexcept {
    double upper = std::isnan(value) ? 1.0 : std::min(value, 1.0);
    return 0.0 <= upper ? upper : 0.0;
}


// Byte span touched by `height` rows of `width` samples at `row_stride_samples`
// with `element_size` bytes per sample. Returns false when any multiplication or
// addition would wrap size_t, so the caller never forms an out-of-range pointer
// from the declared geometry. The buffer capacity itself remains the caller's
// contract (a raw pointer carries no length).
bool checked_span_bytes(std::size_t width,
                        std::size_t height,
                        std::size_t row_stride_samples,
                        std::size_t element_size,
                        std::size_t* out_bytes) noexcept {
    constexpr std::size_t kMax = std::numeric_limits<std::size_t>::max();
    if (out_bytes == nullptr || element_size == 0 || width == 0 || height == 0) return false;
    if (row_stride_samples < width) return false;
    if (row_stride_samples > kMax / element_size) return false;
    if (width > kMax / element_size) return false;
    const std::size_t stride_bytes = row_stride_samples * element_size;
    const std::size_t row_bytes = width * element_size;
    if (height - 1 > (kMax - row_bytes) / stride_bytes) return false;
    *out_bytes = (height - 1) * stride_bytes + row_bytes;
    return true;
}

// FUN_01304c54 unsigned rational quantization (FUN_0126b200 + FUN_0126b1dc).
bool quantize_unsigned(double value, std::uint32_t denominator, double* out) noexcept {
    if (!finite(value)) return false;
    if (value <= 0.0) {
        *out = 0.0;
        return true;
    }
    double scaled = static_cast<double>(denominator) * value + 0.5;
    if (!finite(scaled) || scaled < 0.0 || scaled >= 4294967296.0) return false;
    double numerator = std::floor(scaled);
    *out = numerator / static_cast<double>(denominator);
    return true;
}

// FUN_01304c54 signed rational quantization (FUN_0126b0e0 + FUN_0126b0b8).
bool quantize_signed(double value, std::uint32_t denominator, double* out) noexcept {
    if (!finite(value)) return false;
    double scaled = static_cast<double>(denominator) * value;
    double adjusted = scaled <= 0.0 ? scaled - 0.5 : scaled + 0.5;
    if (!(adjusted > -2147483649.0 && adjusted < 2147483648.0)) return false;
    std::int32_t numerator = static_cast<std::int32_t>(adjusted);
    *out = static_cast<double>(numerator) / static_cast<double>(denominator);
    return true;
}

}  // namespace

double max_black_level(const LinearizationInfo& info, std::uint32_t plane) noexcept {
    const std::uint32_t rows = info.black_repeat_rows;
    const std::uint32_t cols = info.black_repeat_cols;
    // Public helper: reject out-of-range prepared fields before indexing the
    // fixed [8][8][4] arrays (FUN_01303a18 trusted its parsed inputs; the
    // wrapper must not). NaN means "prepared state is not usable".
    if (rows < 1 || rows > kLinearizationMaxBlackRepeat ||
        cols < 1 || cols > kLinearizationMaxBlackRepeat ||
        plane >= kLinearizationMaxPlanes) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    double column_max[kLinearizationMaxBlackRepeat] = {};
    double row_max[kLinearizationMaxBlackRepeat] = {};

    if (info.has_black_level_delta_h && !info.black_level_delta_h.empty()) {
        const std::size_t count = info.black_level_delta_h.size();
        for (std::size_t i = 0; i < count; ++i) {
            std::uint32_t column = static_cast<std::uint32_t>(i % cols);
            // FUN_01303a18 seeds each column with the first occurrence instead
            // of folding it against the initial 0.0.
            if (i < cols) {
                column_max[column] = info.black_level_delta_h[i];
            } else if (column_max[column] <= info.black_level_delta_h[i]) {
                column_max[column] = info.black_level_delta_h[i];
            }
        }
    }
    if (info.has_black_level_delta_v && !info.black_level_delta_v.empty()) {
        const std::size_t count = info.black_level_delta_v.size();
        for (std::size_t i = 0; i < count; ++i) {
            std::uint32_t row = static_cast<std::uint32_t>(i % rows);
            if (i < rows) {
                row_max[row] = info.black_level_delta_v[i];
            } else if (row_max[row] <= info.black_level_delta_v[i]) {
                row_max[row] = info.black_level_delta_v[i];
            }
        }
    }

    double maximum = 0.0;
    for (std::uint32_t row = 0; row < rows; ++row) {
        for (std::uint32_t col = 0; col < cols; ++col) {
            double value = info.black_level[row][col][plane] + column_max[col] + row_max[row];
            if (row == 0 && col == 0) {
                maximum = value;
            } else if (maximum <= value) {
                maximum = value;
            }
        }
    }
    return maximum;
}

LinearizationStatus quantize_linearization_info(LinearizationInfo* info) noexcept {
    if (info == nullptr) return LinearizationStatus::invalid_argument;
    if (info->black_repeat_rows < 1 || info->black_repeat_rows > kLinearizationMaxBlackRepeat ||
        info->black_repeat_cols < 1 || info->black_repeat_cols > kLinearizationMaxBlackRepeat) {
        return LinearizationStatus::invalid_black_repeat;
    }
    if (info->plane_count < 1 || info->plane_count > kLinearizationMaxPlanes) {
        return LinearizationStatus::invalid_plane;
    }
    if (info->has_linearization_table && info->linearization_table.empty()) {
        return LinearizationStatus::invalid_table;
    }

    double absolute_max = 0.0;
    for (std::uint32_t row = 0; row < info->black_repeat_rows; ++row) {
        for (std::uint32_t col = 0; col < info->black_repeat_cols; ++col) {
            for (int plane = 0; plane < kLinearizationMaxPlanes; ++plane) {
                double value = info->black_level[row][col][plane];
                if (!finite(value)) return LinearizationStatus::invalid_value;
                double magnitude = value <= 0.0 ? -value : value;
                if (absolute_max <= magnitude) absolute_max = magnitude;
            }
        }
    }
    if (info->has_black_level_delta_h) {
        for (double value : info->black_level_delta_h) {
            if (!finite(value)) return LinearizationStatus::invalid_value;
            double magnitude = value <= 0.0 ? -value : value;
            if (absolute_max <= magnitude) absolute_max = magnitude;
        }
    }
    if (info->has_black_level_delta_v) {
        for (double value : info->black_level_delta_v) {
            if (!finite(value)) return LinearizationStatus::invalid_value;
            double magnitude = value <= 0.0 ? -value : value;
            if (absolute_max <= magnitude) absolute_max = magnitude;
        }
    }

    std::uint32_t denominator = kRationalMaxDenominator;
    if (kRationalOverflow <= absolute_max * 256.0) {
        if (kRationalOverflow <= absolute_max * 128.0) {
            if (kRationalOverflow <= absolute_max * 64.0) {
                if (kRationalOverflow <= absolute_max * 32.0) {
                    if (kRationalOverflow <= absolute_max * 16.0) {
                        if (kRationalOverflow <= absolute_max * 8.0) {
                            if (kRationalOverflow <= absolute_max * 4.0) {
                                denominator = kRationalOverflow <= absolute_max + absolute_max ? 1u : 2u;
                            } else {
                                denominator = 4u;
                            }
                        } else {
                            denominator = 8u;
                        }
                    } else {
                        denominator = 0x10u;
                    }
                } else {
                    denominator = 0x20u;
                }
            } else {
                denominator = 0x40u;
            }
        } else {
            denominator = 0x80u;
        }
    }

    for (std::uint32_t row = 0; row < info->black_repeat_rows; ++row) {
        for (std::uint32_t col = 0; col < info->black_repeat_cols; ++col) {
            for (int plane = 0; plane < kLinearizationMaxPlanes; ++plane) {
                double quantized = 0.0;
                if (!quantize_unsigned(info->black_level[row][col][plane], denominator,
                                       &quantized)) {
                    return LinearizationStatus::arithmetic_overflow;
                }
                info->black_level[row][col][plane] = quantized;
            }
        }
    }
    if (info->has_black_level_delta_h) {
        for (double& value : info->black_level_delta_h) {
            double quantized = 0.0;
            if (!quantize_signed(value, denominator, &quantized)) {
                return LinearizationStatus::arithmetic_overflow;
            }
            value = quantized;
        }
    }
    if (info->has_black_level_delta_v) {
        for (double& value : info->black_level_delta_v) {
            double quantized = 0.0;
            if (!quantize_signed(value, denominator, &quantized)) {
                return LinearizationStatus::arithmetic_overflow;
            }
            value = quantized;
        }
    }
    return LinearizationStatus::ok;
}

DestinationFloor compute_destination_floor(
    const LinearizationInfo& info,
    std::uint32_t plane_count,
    LinearizationOutputType output_type,
    bool source_gate,
    bool skip_recompute,
    std::uint16_t cached_max16,
    bool source_has_mosaic) noexcept {
    DestinationFloor floor;
    // clampFlag = uVar3 ^ 1: the recovered XOR is preserved even though the
    // only known producer (00affadc) returns exactly 0 or 1.
    floor.clamp_flag = source_gate ^ true;
    // FUN_01305598 reads the same prepared linObj the build consumes. Validate
    // the repeat/plane fields here as well so this public helper cannot index
    // the fixed [8][8][4] arrays without the build-time checks.
    if (plane_count < 1 ||
        plane_count > static_cast<std::uint32_t>(kLinearizationMaxPlanes)) {
        floor.status = LinearizationStatus::invalid_plane;
        floor.max16 = cached_max16;
        return floor;
    }
    if (info.black_repeat_rows < 1 ||
        info.black_repeat_rows > kLinearizationMaxBlackRepeat ||
        info.black_repeat_cols < 1 ||
        info.black_repeat_cols > kLinearizationMaxBlackRepeat) {
        floor.status = LinearizationStatus::invalid_black_repeat;
        floor.max16 = cached_max16;
        return floor;
    }
    if (!source_gate || skip_recompute || output_type != LinearizationOutputType::u16) {
        floor.max16 = cached_max16;
        return floor;
    }

    double ratio = 0.0;
    for (std::uint32_t plane = 0; plane < plane_count && plane < kLinearizationMaxPlanes; ++plane) {
        double black = max_black_level(info, plane);
        double white = info.white_level[plane];
        if (!finite(black) || !finite(white)) continue;
        if (!(0.0 < black) || !(black < white)) continue;
        if (ratio <= black / white) ratio = black / white;
    }

    double value = ratio * 65535.0 + 0.5;
    if (0.2 <= ratio) value = 13107.5;
    if (!(0.0 <= value)) value = 0.0;
    if (4294967296.0 <= value) {
        // FUN_01305598 raises "Overflow in Floor_uint32". Saturating here keeps
        // the function noexcept; callers see the recovered 0xffff ceiling.
        value = 4294967295.0;
    }
    std::uint32_t truncated = static_cast<std::uint32_t>(value);
    std::uint32_t result = truncated;
    if (source_has_mosaic) {
        // source byte +0x600 != 0 selects the 0x404 destination floor variant.
        std::uint32_t floored = truncated & 0xffffu;
        if (floored < 0x405u) floored = kLinearizationDestinationFloor;
        result = floored;
    }
    floor.max16 = static_cast<std::uint16_t>(result);
    return floor;
}

void SourceLinearization::reset() noexcept {
    built_ = false;
    plane_ = 0;
    input_type_ = LinearizationInputType::u16;
    output_type_ = LinearizationOutputType::u16;
    rect_top_ = rect_left_ = rect_bottom_ = rect_right_ = 0;
    black_2d_rows_ = black_2d_cols_ = row_lut_count_ = 0;
    main_u16_.clear();
    main_i24_.clear();
    black_2d_.clear();
    row_lut_.clear();
}

LinearizationStatus SourceLinearization::build(
    const LinearizationInfo& info,
    std::uint32_t plane,
    LinearizationInputType input_type,
    LinearizationOutputType output_type,
    std::uint16_t max16,
    bool clamp_flag) noexcept {
    try {
        return build_impl(info, plane, input_type, output_type, max16, clamp_flag);
    } catch (const std::bad_alloc&) {
        return LinearizationStatus::allocation_failure;
    } catch (...) {
        return LinearizationStatus::allocation_failure;
    }
}

LinearizationStatus SourceLinearization::build_impl(
    const LinearizationInfo& info,
    std::uint32_t plane,
    LinearizationInputType input_type,
    LinearizationOutputType output_type,
    std::uint16_t max16,
    bool clamp_flag) noexcept(false) {
    if (input_type != LinearizationInputType::u8 &&
        input_type != LinearizationInputType::u16) {
        return LinearizationStatus::unsupported_type;
    }
    if (output_type != LinearizationOutputType::u16) {
        return LinearizationStatus::unsupported_type;
    }
    if (plane >= kLinearizationMaxPlanes || plane >= info.plane_count) {
        return LinearizationStatus::invalid_plane;
    }
    if (info.black_repeat_rows < 1 || info.black_repeat_rows > kLinearizationMaxBlackRepeat ||
        info.black_repeat_cols < 1 || info.black_repeat_cols > kLinearizationMaxBlackRepeat) {
        return LinearizationStatus::invalid_black_repeat;
    }
    // Widen every rect difference before narrowing. `right - left` in int32 is
    // signed-overflow UB for rect_left=INT32_MIN, rect_right=INT32_MAX; the
    // recovered code consumes 32-bit counts, so require each difference to be
    // representable as the uint32 count it becomes.
    const std::int64_t rect_width = static_cast<std::int64_t>(info.rect_right) -
        static_cast<std::int64_t>(info.rect_left);
    const std::int64_t rect_height = static_cast<std::int64_t>(info.rect_bottom) -
        static_cast<std::int64_t>(info.rect_top);
    if (rect_width <= 0 || rect_height <= 0) return LinearizationStatus::invalid_rect;
    if (rect_width > static_cast<std::int64_t>(std::numeric_limits<std::uint32_t>::max()) ||
        rect_height > static_cast<std::int64_t>(std::numeric_limits<std::uint32_t>::max())) {
        return LinearizationStatus::invalid_rect;
    }
    if (info.has_linearization_table && info.linearization_table.empty()) {
        return LinearizationStatus::invalid_table;
    }
    if (info.has_black_level_delta_h &&
        info.black_level_delta_h.size() < static_cast<std::size_t>(rect_width)) {
        return LinearizationStatus::invalid_table;
    }
    if (info.has_black_level_delta_v &&
        info.black_level_delta_v.size() < static_cast<std::size_t>(rect_height)) {
        return LinearizationStatus::invalid_table;
    }

    const double white = info.white_level[plane];
    if (!finite(white)) return LinearizationStatus::invalid_value;
    const double black_max = max_black_level(info, plane);
    if (!finite(black_max)) return LinearizationStatus::invalid_value;
    const double span = white - black_max;
    if (!(span > 0.0) || !finite(span)) return LinearizationStatus::invalid_value;
    const double scale = 1.0 / span;

    const std::uint32_t width = static_cast<std::uint32_t>(rect_width);
    const std::uint32_t height = static_cast<std::uint32_t>(rect_height);

    std::uint32_t rows_2d = 0;
    std::uint32_t cols_2d = 0;
    if (info.has_black_level_delta_h) {
        rows_2d = info.black_repeat_rows;
        cols_2d = width;
    } else if (info.black_repeat_cols >= 2) {
        rows_2d = info.black_repeat_rows;
        cols_2d = info.black_repeat_cols;
    }
    if (rows_2d == 0) cols_2d = 0;

    std::uint32_t row_count = 0;
    if (info.has_black_level_delta_v) {
        row_count = height;
    } else if (rows_2d == 0 && info.black_repeat_rows > 1) {
        row_count = info.black_repeat_rows;
    }

    if (rows_2d != 0 && cols_2d != 0) {
        // rows_2d <= 8 but cols_2d may be the full uint32 rect width, so the
        // product is formed in 64 bits before it is compared with the guard.
        const std::uint64_t entries =
            static_cast<std::uint64_t>(rows_2d) * static_cast<std::uint64_t>(cols_2d);
        if (entries > kLinearizationMaxBlackTableEntries) {
            return LinearizationStatus::resource_limit;
        }
    }
    if (row_count > kLinearizationMaxBlackTableEntries) {
        return LinearizationStatus::resource_limit;
    }

    // Build into candidates so a failure leaves the previous stage untouched.
    std::vector<std::uint16_t> main_u16;
    std::vector<std::int32_t> main_i24;
    std::vector<std::int32_t> black_2d;
    std::vector<std::int32_t> row_lut;

    const std::uint32_t weight8 = static_cast<std::uint32_t>(max16 ^ 0xffffu) << 8;
    const double weight16 = static_cast<double>(max16 ^ 0xffffu);
    const double base16 = static_cast<double>(max16);

    if (rows_2d == 0 && row_count == 0) {
        const std::uint32_t entries =
            input_type == LinearizationInputType::u8 ? kMainU8Entries : kMainU16Entries;
        main_u16.resize(entries);
        for (std::uint32_t index = 0; index < entries; ++index) {
            std::uint32_t value = index;
            if (info.has_linearization_table) {
                std::size_t table_index = index;
                if (table_index + 1 >= info.linearization_table.size()) {
                    table_index = info.linearization_table.size() - 1;
                }
                value = info.linearization_table[table_index];
            }
            double normalized = scale *
                (static_cast<double>(value) - info.black_level[0][0][plane]);
            double mapped = clamp_flag ? clamp01_fminnm(normalized) : normalized;
            // FUN_0130312c @01303844 is a double FMADD: d = mapped * weight16 +
            // base16, one rounding. The strict `-ffp-contract=off` build cannot
            // be relied on to fuse, so the fusion is explicit here.
            double affine = std::fma(weight16, mapped, base16);
            std::int32_t rounded = 0;
            if (!round_half_away(affine, &rounded)) {
                return LinearizationStatus::arithmetic_overflow;
            }
            main_u16[index] = saturate_u16(rounded);
        }
    } else {
        main_i24.resize(kMainU16Entries);
        for (std::uint32_t index = 0; index < kMainU16Entries; ++index) {
            std::uint32_t value = index;
            if (info.has_linearization_table) {
                std::size_t table_index = index;
                if (table_index + 1 >= info.linearization_table.size()) {
                    table_index = info.linearization_table.size() - 1;
                }
                value = info.linearization_table[table_index];
            }
            double normalized = scale * static_cast<double>(value);
            // FUN_0130312c @013036d4 is the same double FMADD, followed by the
            // exact *256 scaling and the shared round-half-away sequence.
            double fixed = std::fma(weight16, normalized, base16) * 256.0;
            std::int32_t rounded = 0;
            if (!round_half_away(fixed, &rounded)) {
                return LinearizationStatus::arithmetic_overflow;
            }
            main_i24[index] = rounded;
        }

        if (rows_2d != 0 && cols_2d != 0) {
            black_2d.resize(static_cast<std::size_t>(rows_2d) * cols_2d);
            for (std::uint32_t row = 0; row < rows_2d; ++row) {
                for (std::uint32_t col = 0; col < cols_2d; ++col) {
                    double value = info.black_level[row][col % info.black_repeat_cols][plane];
                    if (info.has_black_level_delta_h) value += info.black_level_delta_h[col];
                    double fixed =
                        scale * value * static_cast<double>(static_cast<std::int32_t>(weight8));
                    std::int32_t rounded = 0;
                    if (!round_half_away(fixed, &rounded)) {
                        return LinearizationStatus::arithmetic_overflow;
                    }
                    black_2d[static_cast<std::size_t>(row) * cols_2d + col] = rounded;
                }
            }
        }

        if (row_count != 0) {
            row_lut.resize(row_count);
            for (std::uint32_t row = 0; row < row_count; ++row) {
                double value = 0.0;
                if (rows_2d == 0) {
                    value = info.black_level[row % info.black_repeat_rows][0][plane];
                }
                if (info.has_black_level_delta_v) value += info.black_level_delta_v[row];
                double fixed =
                    scale * value * static_cast<double>(static_cast<std::int32_t>(weight8));
                std::int32_t rounded = 0;
                if (!round_half_away(fixed, &rounded)) {
                    return LinearizationStatus::arithmetic_overflow;
                }
                row_lut[row] = rounded;
            }
        }
    }

    main_u16_ = std::move(main_u16);
    main_i24_ = std::move(main_i24);
    black_2d_ = std::move(black_2d);
    row_lut_ = std::move(row_lut);
    plane_ = plane;
    input_type_ = input_type;
    output_type_ = output_type;
    rect_top_ = info.rect_top;
    rect_left_ = info.rect_left;
    rect_bottom_ = info.rect_bottom;
    rect_right_ = info.rect_right;
    black_2d_rows_ = rows_2d;
    black_2d_cols_ = cols_2d;
    row_lut_count_ = row_count;
    built_ = true;
    return LinearizationStatus::ok;
}

LinearizationStatus SourceLinearization::process(
    const void* input,
    std::size_t input_row_stride_samples,
    void* output,
    std::size_t output_row_stride_samples,
    std::int32_t region_top,
    std::int32_t region_left,
    std::int32_t region_bottom,
    std::int32_t region_right) const noexcept {
    if (input == nullptr || output == nullptr) return LinearizationStatus::invalid_argument;
    if (!built_) return LinearizationStatus::not_built;
    if (region_top < rect_top_ || region_left < rect_left_ ||
        region_bottom > rect_bottom_ || region_right > rect_right_) {
        return LinearizationStatus::invalid_region;
    }
    if (region_bottom <= region_top || region_right <= region_left) {
        return LinearizationStatus::invalid_region;
    }
    // Widened geometry: `right - left` in int32 is signed-overflow UB for
    // INT32_MIN/INT32_MAX. The recovered Process counts rows/cols in 32-bit
    // registers (01303c7c/01303c84), so each count must fit uint32.
    const std::int64_t width_wide = static_cast<std::int64_t>(region_right) -
        static_cast<std::int64_t>(region_left);
    const std::int64_t height_wide = static_cast<std::int64_t>(region_bottom) -
        static_cast<std::int64_t>(region_top);
    if (width_wide <= 0 || height_wide <= 0) return LinearizationStatus::invalid_region;
    if (width_wide > static_cast<std::int64_t>(std::numeric_limits<std::uint32_t>::max()) ||
        height_wide > static_cast<std::int64_t>(std::numeric_limits<std::uint32_t>::max())) {
        return LinearizationStatus::invalid_region;
    }
    const std::size_t width = static_cast<std::size_t>(width_wide);
    const std::size_t height = static_cast<std::size_t>(height_wide);

    // Every span multiplication (row * stride, samples -> bytes) is checked
    // before the first store; a failing call leaves `output` untouched.
    const std::size_t input_element_bytes =
        input_type_ == LinearizationInputType::u16 ? 2u : 1u;
    std::size_t input_bytes = 0;
    std::size_t output_bytes = 0;
    if (!checked_span_bytes(width, height, input_row_stride_samples, input_element_bytes,
                            &input_bytes) ||
        !checked_span_bytes(width, height, output_row_stride_samples, 2u, &output_bytes)) {
        return LinearizationStatus::invalid_region;
    }

    // Alias contract: only the recovered u16 -> u16 in-place call (identical
    // base pointer and identical row stride) is implemented. Any other overlap
    // would read input elements a previous store already rewrote (u8 -> u16 at
    // the same address overwrites the next source byte), so it is rejected
    // before the first write instead of producing silently wrong pixels.
    const std::uintptr_t input_begin = reinterpret_cast<std::uintptr_t>(input);
    const std::uintptr_t output_begin = reinterpret_cast<std::uintptr_t>(output);
    if (input_bytes > std::numeric_limits<std::uintptr_t>::max() - input_begin ||
        output_bytes > std::numeric_limits<std::uintptr_t>::max() - output_begin) {
        return LinearizationStatus::invalid_region;
    }
    const bool overlap = input_begin < output_begin + output_bytes &&
        output_begin < input_begin + input_bytes;
    if (overlap) {
        const bool in_place = input == output &&
            input_type_ == LinearizationInputType::u16 &&
            input_row_stride_samples == output_row_stride_samples;
        if (!in_place) return LinearizationStatus::unsupported_aliasing;
    }

    const std::int64_t row_origin = static_cast<std::int64_t>(region_top) - rect_top_;
    const std::int64_t col_origin = static_cast<std::int64_t>(region_left) - rect_left_;
    const bool branch_b = !main_i24_.empty();
    const bool has_2d = black_2d_cols_ != 0;
    const bool has_row = row_lut_count_ != 0;

    for (std::size_t row = 0; row < height; ++row) {
        const std::int64_t absolute_row = row_origin + static_cast<std::int64_t>(row);
        // 01303fe8 / 013043c4: `sub w22,w22,#0x80` on the loaded row-LUT entry
        // (zero when the row LUT is absent) is a 32-bit wrap, not a widened
        // subtraction.
        std::uint32_t row_offset = 0u - 0x80u;
        if (has_row) {
            row_offset = static_cast<std::uint32_t>(
                row_lut_[static_cast<std::size_t>(absolute_row) % row_lut_count_]) - 0x80u;
        }
        const std::size_t black_row_offset = has_2d
            ? static_cast<std::size_t>(absolute_row % black_2d_rows_) * black_2d_cols_
            : 0;
        std::uint32_t black_col = 0;
        if (has_2d) {
            black_col = static_cast<std::uint32_t>(
                static_cast<std::uint64_t>(col_origin) % black_2d_cols_);
        }

        const std::uint16_t* input_u16 = nullptr;
        const std::uint8_t* input_u8 = nullptr;
        if (input_type_ == LinearizationInputType::u16) {
            input_u16 = reinterpret_cast<const std::uint16_t*>(input) +
                row * input_row_stride_samples;
        } else {
            input_u8 = reinterpret_cast<const std::uint8_t*>(input) +
                row * input_row_stride_samples;
        }
        std::uint16_t* output_row =
            reinterpret_cast<std::uint16_t*>(output) + row * output_row_stride_samples;

        for (std::size_t col = 0; col < width; ++col) {
            const std::uint32_t pixel = input_u16 != nullptr
                ? static_cast<std::uint32_t>(input_u16[col])
                : static_cast<std::uint32_t>(input_u8[col]);
            if (!branch_b) {
                output_row[col] = main_u16_[pixel];
                continue;
            }
            // 01304050/0130405c (u8 input) and 01304440/0130444c (u16 input):
            // both `sub w` operations are 32-bit and wrap modulo 2^32. The
            // stored value is then selected by the bit patterns, not by an
            // int64 comparison: 0130401c..01304030 keeps `v >> 8` only when
            // `v >> 24 == 0`, and otherwise stores 0xffff when bit 31 is clear
            // (01304020..01304024) or 0 when bit 31 is set.
            std::uint32_t value = static_cast<std::uint32_t>(main_i24_[pixel]) - row_offset;
            if (has_2d) {
                value -= static_cast<std::uint32_t>(black_2d_[black_row_offset + black_col]);
                if (++black_col == black_2d_cols_) black_col = 0;
            }
            if ((value >> 24) == 0) {
                output_row[col] = static_cast<std::uint16_t>(value >> 8);
            } else {
                output_row[col] = (value >> 31) != 0 ? std::uint16_t{0} : std::uint16_t{0xffff};
            }
        }
    }
    return LinearizationStatus::ok;
}

}  // namespace imprint::acr
