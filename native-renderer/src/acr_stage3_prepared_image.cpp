#include "acr_stage3_prepared_image.hpp"

#include "acr_bayer_gain.hpp"
#include "acr_float_lut.hpp"
#include "acr_hotpixel.hpp"
#include "acr_stage3_core3_compose.hpp"
#include "acr_stage3_core3_format.hpp"
#include "acr_stage3_curve.hpp"
#include "acr_u16_lut.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>

namespace imprint::acr {
namespace {

constexpr std::size_t kMaximumRgbSamples = 128u * 1024u * 1024u;

struct AddressSpan {
    std::uintptr_t begin = 0;
    std::uintptr_t end = 0;
};

bool checked_add(std::size_t a, std::size_t b, std::size_t* out) noexcept {
    if (b > std::numeric_limits<std::size_t>::max() - a) return false;
    *out = a + b;
    return true;
}

bool checked_mul(std::size_t a, std::size_t b, std::size_t* out) noexcept {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
        return false;
    }
    *out = a * b;
    return true;
}

bool checked_sample_span(std::size_t width, std::size_t height,
                         std::size_t stride, std::size_t* out) noexcept {
    if (width == 0 || height == 0 || stride < width) return false;
    std::size_t preceding_rows = 0;
    return checked_mul(height - 1, stride, &preceding_rows) &&
           checked_add(preceding_rows, width, out);
}

bool checked_byte_count(std::size_t count, std::size_t item_size,
                        std::size_t* out) noexcept {
    if (item_size == 0 || count > std::numeric_limits<std::size_t>::max() /
                                      item_size ||
        static_cast<std::uintmax_t>(count) >
            static_cast<std::uintmax_t>(
                std::numeric_limits<std::ptrdiff_t>::max()) /
                item_size) {
        return false;
    }
    *out = count * item_size;
    return true;
}

bool address_span(const void* pointer, std::size_t count,
                  std::size_t item_size, std::size_t alignment,
                  AddressSpan* out) noexcept {
    if (pointer == nullptr || out == nullptr || alignment == 0) return false;
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (begin % alignment != 0) return false;
    std::size_t bytes = 0;
    if (!checked_byte_count(count, item_size, &bytes) ||
        bytes > std::numeric_limits<std::uintptr_t>::max() - begin) {
        return false;
    }
    out->begin = begin;
    out->end = begin + static_cast<std::uintptr_t>(bytes);
    return true;
}

bool overlaps(AddressSpan a, AddressSpan b) noexcept {
    return a.begin < b.end && b.begin < a.end;
}

bool aligned_16(const void* pointer) noexcept {
    return (reinterpret_cast<std::uintptr_t>(pointer) & 0xFu) == 0;
}

std::int64_t floor_mod2(std::int64_t value) noexcept {
    const std::int64_t remainder = value % 2;
    return remainder < 0 ? remainder + 2 : remainder;
}

bool round_up(std::size_t value, std::size_t alignment,
              std::size_t* out) noexcept {
    if (alignment == 0) return false;
    const std::size_t remainder = value % alignment;
    return remainder == 0 ? (*out = value, true)
                          : checked_add(value, alignment - remainder, out);
}

bool fits_i32(std::int64_t value) noexcept {
    return value >= std::numeric_limits<std::int32_t>::min() &&
           value <= std::numeric_limits<std::int32_t>::max();
}

// std::vector only promises alignof(T). Keep enough extra elements to advance
// to an actual 16-byte boundary without relying on allocator implementation.
template <typename T>
class AlignedArray {
  public:
    explicit AlignedArray(std::size_t count) {
        constexpr std::size_t kAlignment = 16;
        const std::size_t extra = (kAlignment - 1 + sizeof(T) - 1) / sizeof(T);
        std::size_t allocation_count = 0;
        if (!checked_add(count, extra, &allocation_count)) {
            throw std::length_error("aligned scratch count overflow");
        }
        std::size_t bytes = 0;
        if (!checked_byte_count(allocation_count, sizeof(T), &bytes)) {
            throw std::length_error("aligned scratch byte count overflow");
        }
        storage_.resize(allocation_count);
        const auto begin = reinterpret_cast<std::uintptr_t>(storage_.data());
        if (kAlignment - 1 >
            std::numeric_limits<std::uintptr_t>::max() - begin) {
            throw std::length_error("aligned scratch address overflow");
        }
        const auto aligned = (begin + (kAlignment - 1)) &
                             ~static_cast<std::uintptr_t>(kAlignment - 1);
        const auto offset_bytes = aligned - begin;
        if (offset_bytes % sizeof(T) != 0 ||
            offset_bytes / sizeof(T) > extra ||
            count > allocation_count - offset_bytes / sizeof(T)) {
            throw std::length_error("aligned scratch span overflow");
        }
        data_ = storage_.data() + offset_bytes / sizeof(T);
        (void)bytes;
    }

    T* data() noexcept { return data_; }
    const T* data() const noexcept { return data_; }

  private:
    std::vector<T> storage_;
    T* data_ = nullptr;
};

PreparedImageStatus map_core3_control(Core3ComposeStatus status) noexcept {
    return status == Core3ComposeStatus::ok
               ? PreparedImageStatus::ok
               : PreparedImageStatus::prepare_core3_controls_failure;
}

PreparedImageStatus validate_expanded_tiles(
    std::size_t width, std::size_t height, BayerRepeatRect rect,
    std::size_t tile_width, std::size_t tile_height,
    std::size_t inset) noexcept {
    const std::size_t first_padded_width = std::min(width, tile_width);
    const std::size_t last_x = ((width - 1) / tile_width) * tile_width;
    const std::size_t last_y = ((height - 1) / tile_height) * tile_height;
    const std::size_t last_tile_width = width - last_x;
    const std::size_t last_tile_height = height - last_y;
    std::size_t first_padded = 0;
    std::size_t last_padded_width = 0;
    if (!round_up(first_padded_width, 8, &first_padded) ||
        !round_up(last_tile_width, 8, &last_padded_width)) {
        return PreparedImageStatus::arithmetic_overflow;
    }

    // Every request is translated monotonically with the tile origin. Its
    // smallest top/left is the first tile's request origin; its largest
    // bottom/right is the last tile's exclusive request endpoint.
    const std::int64_t first_raw_top = static_cast<std::int64_t>(rect.top) -
                                       static_cast<std::int64_t>(inset) - 2;
    const std::int64_t first_raw_left = static_cast<std::int64_t>(rect.left) -
                                        static_cast<std::int64_t>(inset) - 2;
    std::size_t last_hot_width = 0;
    std::size_t last_hot_height = 0;
    std::size_t last_raw_width = 0;
    std::size_t last_raw_height = 0;
    if (!checked_add(last_padded_width, 2 * inset, &last_hot_width) ||
        !checked_add(last_tile_height, 2 * inset, &last_hot_height) ||
        !checked_add(last_hot_width, 4, &last_raw_width) ||
        !checked_add(last_hot_height, 4, &last_raw_height)) {
        return PreparedImageStatus::arithmetic_overflow;
    }
    const std::int64_t last_origin_y =
        static_cast<std::int64_t>(rect.top) +
        static_cast<std::int64_t>(last_y);
    const std::int64_t last_origin_x =
        static_cast<std::int64_t>(rect.left) +
        static_cast<std::int64_t>(last_x);
    const std::int64_t last_raw_bottom =
        last_origin_y - static_cast<std::int64_t>(inset) - 2 +
        static_cast<std::int64_t>(last_raw_height);
    const std::int64_t last_raw_right =
        last_origin_x - static_cast<std::int64_t>(inset) - 2 +
        static_cast<std::int64_t>(last_raw_width);
    (void)first_padded;
    if (!fits_i32(first_raw_top) || !fits_i32(first_raw_left) ||
        !fits_i32(last_raw_bottom) || !fits_i32(last_raw_right)) {
        return PreparedImageStatus::invalid_rect;
    }
    return PreparedImageStatus::ok;
}

PreparedImageStatus render_tile(
    const std::uint16_t* source, std::size_t source_sample_count,
    std::size_t source_stride, BayerRepeatRect source_rect,
    std::size_t image_width, std::size_t tile_x, std::size_t tile_y,
    std::size_t tile_width, std::size_t tile_height, std::size_t inset,
    const BayerGainPrepared& forward_gain,
    const BayerGainPrepared& inverse_gain,
    const U16LutPrepared& forward_lut, const U16LutPrepared& inverse_lut,
    double amount, bool suppress_refine, std::uint16_t hot_threshold,
    std::uint16_t cold_threshold, std::vector<std::uint16_t>* image) {
    std::size_t padded_width = 0;
    if (!round_up(tile_width, 8, &padded_width)) {
        return PreparedImageStatus::arithmetic_overflow;
    }
    std::size_t hot_width = 0;
    std::size_t hot_height = 0;
    std::size_t raw_width = 0;
    std::size_t raw_height = 0;
    if (!checked_add(padded_width, 2 * inset, &hot_width) ||
        !checked_add(tile_height, 2 * inset, &hot_height) ||
        !checked_add(hot_width, 4, &raw_width) ||
        !checked_add(hot_height, 4, &raw_height)) {
        return PreparedImageStatus::arithmetic_overflow;
    }
    std::size_t raw_stride = 0;
    std::size_t raw_float_stride = 0;
    std::size_t hot_stride = 0;
    std::size_t hot_float_stride = 0;
    if (!round_up(raw_width, 8, &raw_stride) ||
        !round_up(raw_width, 4, &raw_float_stride) ||
        !round_up(hot_width, 8, &hot_stride) ||
        !round_up(hot_width, 4, &hot_float_stride)) {
        return PreparedImageStatus::arithmetic_overflow;
    }

    std::size_t raw_count = 0;
    std::size_t raw_float_count = 0;
    std::size_t hot_count = 0;
    std::size_t hot_float_count = 0;
    if (!checked_mul(raw_stride, raw_height, &raw_count) ||
        !checked_mul(raw_float_stride, raw_height, &raw_float_count) ||
        !checked_mul(hot_stride, hot_height, &hot_count) ||
        !checked_mul(hot_float_stride, hot_height, &hot_float_count)) {
        return PreparedImageStatus::arithmetic_overflow;
    }

    const std::int64_t target_y = static_cast<std::int64_t>(source_rect.top) +
                                  static_cast<std::int64_t>(tile_y);
    const std::int64_t target_x = static_cast<std::int64_t>(source_rect.left) +
                                  static_cast<std::int64_t>(tile_x);
    const std::int64_t raw_top = target_y - static_cast<std::int64_t>(inset) - 2;
    const std::int64_t raw_left = target_x - static_cast<std::int64_t>(inset) - 2;
    const std::int64_t raw_bottom = raw_top + static_cast<std::int64_t>(raw_height);
    const std::int64_t raw_right = raw_left + static_cast<std::int64_t>(raw_width);
    if (!fits_i32(raw_top) || !fits_i32(raw_left) || !fits_i32(raw_bottom) ||
        !fits_i32(raw_right)) {
        return PreparedImageStatus::invalid_rect;
    }
    const BayerRepeatRect raw_rect{
        static_cast<std::int32_t>(raw_top), static_cast<std::int32_t>(raw_left),
        static_cast<std::int32_t>(raw_bottom), static_cast<std::int32_t>(raw_right)};

    AlignedArray<std::uint16_t> copied_raw(raw_count);
    AlignedArray<std::uint16_t> transformed_raw(raw_count);
    AlignedArray<float> raw_float(raw_float_count);
    AlignedArray<std::uint16_t> hot_u16(hot_count);
    AlignedArray<float> hot_float(hot_float_count);

    const BayerBorderStatus copy_status = copy_bayer_repeat2_u16(
        source, source_sample_count, source_stride, source_rect, raw_rect,
        copied_raw.data(), raw_count, raw_stride);
    if (copy_status != BayerBorderStatus::ok) {
        return PreparedImageStatus::copy_bayer_failure;
    }

    for (std::size_t row = 0; row < raw_height; ++row) {
        const std::uint16_t* input = copied_raw.data() + row * raw_stride;
        float* working = raw_float.data() + row * raw_float_stride;
        std::uint16_t* encoded = transformed_raw.data() + row * raw_stride;
        if (core3_u16_to_f32_prepared_row_008096e4(
                input, raw_stride, working, raw_float_stride, raw_width) !=
            Core3FormatStatus::ok) {
            return PreparedImageStatus::u16_to_f32_failure;
        }
        const std::int64_t global_y = raw_top + static_cast<std::int64_t>(row);
        if (apply_bayer_gain_cfa_row_007ef1f4(
                working, raw_float_stride, raw_width,
                static_cast<unsigned>(floor_mod2(global_y)),
                static_cast<unsigned>(floor_mod2(raw_left)), forward_gain) !=
            BayerGainStatus::ok) {
            return PreparedImageStatus::cfa_gain_failure;
        }
        if (core3_f32_to_u16_prepared_row_008103dc(
                working, raw_float_stride, encoded, raw_stride, raw_width) !=
            Core3FormatStatus::ok) {
            return PreparedImageStatus::f32_to_u16_failure;
        }
        if (apply_u16_lut_row_0132d484(
                forward_lut, encoded, raw_stride, encoded, raw_stride,
                raw_width) != FloatLutStatus::ok) {
            return PreparedImageStatus::forward_lut_failure;
        }
    }

    const std::int64_t hot_top = raw_top + 2;
    const std::int64_t hot_left = raw_left + 2;
    for (std::size_t row = 0; row < hot_height; ++row) {
        const std::uint16_t* rows[5];
        std::size_t row_counts[5];
        for (std::size_t k = 0; k < 5; ++k) {
            rows[k] = transformed_raw.data() + (row + k) * raw_stride;
            row_counts[k] = raw_stride;
        }
        std::uint16_t* hot_row = hot_u16.data() + row * hot_stride;
        const std::uint32_t phase = static_cast<std::uint32_t>(
            floor_mod2(hot_top + static_cast<std::int64_t>(row)) ^
            floor_mod2(hot_left));
        if (hotpixel_u16_row_0132d7c8(
                rows, row_counts, 2, hot_row, hot_stride, hot_width, phase,
                hot_threshold, cold_threshold) != HotPixelStatus::ok) {
            return PreparedImageStatus::hotpixel_failure;
        }
        if (core3_u16_to_f32_prepared_row_008096e4(
                hot_row, hot_stride, hot_float.data() + row * hot_float_stride,
                hot_float_stride, hot_width) != Core3FormatStatus::ok) {
            return PreparedImageStatus::u16_to_f32_failure;
        }
    }

    Core3ComposeControls core_controls{};
    const auto control_status = prepare_core3_controls_007f2124(
        amount, suppress_refine,
        static_cast<std::uint32_t>(floor_mod2(hot_top)),
        static_cast<std::uint32_t>(floor_mod2(hot_left)), &core_controls);
    if (map_core3_control(control_status) != PreparedImageStatus::ok) {
        return PreparedImageStatus::prepare_core3_controls_failure;
    }
    Core3ComposeOutput core_output{};
    if (run_core3_prepared_f32_tile(
            hot_float.data(), hot_float_count, hot_width, hot_height,
            hot_float_stride, core_controls, &core_output) !=
        Core3ComposeStatus::ok ||
        core_output.output_width != padded_width ||
        core_output.output_height != tile_height) {
        return PreparedImageStatus::core3_failure;
    }

    std::size_t plane_samples = 0;
    std::size_t expected_core_samples = 0;
    if (!checked_mul(padded_width, tile_height, &plane_samples) ||
        !checked_mul(plane_samples, 3, &expected_core_samples) ||
        core_output.rgb.size() != expected_core_samples) {
        return PreparedImageStatus::core3_failure;
    }
    AlignedArray<float> planar_float[3] = {
        AlignedArray<float>(plane_samples), AlignedArray<float>(plane_samples),
        AlignedArray<float>(plane_samples)};
    AlignedArray<std::uint16_t> planar_u16[3] = {
        AlignedArray<std::uint16_t>(plane_samples),
        AlignedArray<std::uint16_t>(plane_samples),
        AlignedArray<std::uint16_t>(plane_samples)};
    AlignedArray<std::uint16_t> final_u16[3] = {
        AlignedArray<std::uint16_t>(plane_samples),
        AlignedArray<std::uint16_t>(plane_samples),
        AlignedArray<std::uint16_t>(plane_samples)};

    for (std::size_t row = 0; row < tile_height; ++row) {
        const std::size_t row_offset = row * padded_width;
        const std::size_t rgb_offset = row * padded_width * 3;
        for (std::size_t column = 0; column < padded_width; ++column) {
            const std::size_t source_index = rgb_offset + column * 3;
            for (std::size_t channel = 0; channel < 3; ++channel) {
                planar_float[channel].data()[row_offset + column] =
                    core_output.rgb[source_index + channel];
            }
        }
    }

    for (std::size_t channel = 0; channel < 3; ++channel) {
        for (std::size_t row = 0; row < tile_height; ++row) {
            const std::size_t offset = row * padded_width;
            float* floats = planar_float[channel].data() + offset;
            std::uint16_t* integers = planar_u16[channel].data() + offset;
            if (core3_f32_to_u16_prepared_row_008103dc(
                    floats, padded_width, integers, padded_width,
                    padded_width) != Core3FormatStatus::ok) {
                return PreparedImageStatus::f32_to_u16_failure;
            }
            if (apply_u16_lut_row_0132d484(
                    inverse_lut, integers, padded_width, integers,
                    padded_width, padded_width) != FloatLutStatus::ok) {
                return PreparedImageStatus::inverse_lut_failure;
            }
            if (core3_u16_to_f32_prepared_row_008096e4(
                    integers, padded_width, floats, padded_width,
                    padded_width) != Core3FormatStatus::ok) {
                return PreparedImageStatus::u16_to_f32_failure;
            }
        }
    }

    for (std::size_t row = 0; row < tile_height; ++row) {
        const std::size_t offset = row * padded_width;
        if (apply_bayer_gain_rgb_row_007ef1f4(
                planar_float[0].data() + offset, padded_width,
                planar_float[1].data() + offset, padded_width,
                planar_float[2].data() + offset, padded_width, padded_width,
                inverse_gain) != BayerGainStatus::ok) {
            return PreparedImageStatus::rgb_gain_failure;
        }
    }

    for (std::size_t channel = 0; channel < 3; ++channel) {
        for (std::size_t row = 0; row < tile_height; ++row) {
            const std::size_t offset = row * padded_width;
            if (core3_f32_to_u16_prepared_row_008103dc(
                    planar_float[channel].data() + offset, padded_width,
                    final_u16[channel].data() + offset, padded_width,
                    padded_width) != Core3FormatStatus::ok) {
                return PreparedImageStatus::final_conversion_failure;
            }
        }
    }

    const std::size_t actual_width = std::min(tile_width, image_width - tile_x);
    for (std::size_t row = 0; row < tile_height; ++row) {
        const std::size_t image_row = tile_y + row;
        const std::size_t output_base = (image_row * image_width + tile_x) * 3;
        const std::size_t plane_base = row * padded_width;
        for (std::size_t column = 0; column < actual_width; ++column) {
            const std::size_t output_index = output_base + column * 3;
            for (std::size_t channel = 0; channel < 3; ++channel) {
                (*image)[output_index + channel] =
                    final_u16[channel].data()[plane_base + column];
            }
        }
    }
    return PreparedImageStatus::ok;
}

}  // namespace

PreparedImageStatus run_stage3_prepared_image_u16(
    const std::uint16_t* source, std::size_t source_sample_count,
    std::size_t source_row_stride, BayerRepeatRect source_rect,
    const PreparedImageControls& controls,
    PreparedImageOutput* output) noexcept {
    AddressSpan output_object_span{};
    AddressSpan controls_span{};
    if (!address_span(output, 1, sizeof(*output), alignof(PreparedImageOutput),
                      &output_object_span) ||
        !address_span(&controls, 1, sizeof(controls),
                      alignof(PreparedImageControls), &controls_span)) {
        return PreparedImageStatus::invalid_argument;
    }
    if (overlaps(output_object_span, controls_span)) {
        return PreparedImageStatus::overlap;
    }
    if (source == nullptr) return PreparedImageStatus::invalid_argument;

    const std::int64_t height64 = static_cast<std::int64_t>(source_rect.bottom) -
                                  static_cast<std::int64_t>(source_rect.top);
    const std::int64_t width64 = static_cast<std::int64_t>(source_rect.right) -
                                 static_cast<std::int64_t>(source_rect.left);
    if (width64 <= 0 || height64 <= 0) return PreparedImageStatus::invalid_rect;
    if (width64 < 2 || height64 < 2 ||
        width64 > std::numeric_limits<std::int32_t>::max() ||
        height64 > std::numeric_limits<std::int32_t>::max()) {
        return PreparedImageStatus::invalid_dimensions;
    }
    const std::size_t width = static_cast<std::size_t>(width64);
    const std::size_t height = static_cast<std::size_t>(height64);
    if (source_row_stride < width) return PreparedImageStatus::invalid_stride;

    std::size_t source_required = 0;
    AddressSpan source_capacity_span{};
    if (!checked_sample_span(width, height, source_row_stride, &source_required) ||
        !address_span(source, source_sample_count, sizeof(*source),
                      alignof(std::uint16_t), &source_capacity_span)) {
        return PreparedImageStatus::arithmetic_overflow;
    }
    if (source_sample_count < source_required) {
        return PreparedImageStatus::buffer_too_small;
    }
    if (overlaps(output_object_span, source_capacity_span)) {
        return PreparedImageStatus::overlap;
    }

    if (controls.tile_width == 0 || controls.tile_height == 0 ||
        controls.tile_width > 1024 || controls.tile_height > 1024 ||
        !std::isfinite(controls.amount) ||
        !std::isfinite(controls.gains[0]) ||
        !std::isfinite(controls.gains[1]) ||
        !std::isfinite(controls.gains[2])) {
        return PreparedImageStatus::invalid_controls;
    }
    std::size_t image_pixels = 0;
    std::size_t output_samples = 0;
    std::size_t output_bytes = 0;
    if (!checked_mul(width, height, &image_pixels) ||
        !checked_mul(image_pixels, 3, &output_samples) ||
        !checked_byte_count(output_samples, sizeof(std::uint16_t),
                            &output_bytes)) {
        return PreparedImageStatus::arithmetic_overflow;
    }
    (void)output_bytes;
    if (output_samples > kMaximumRgbSamples) {
        return PreparedImageStatus::resource_limit;
    }

    PreparedImageStatus expansion_status = PreparedImageStatus::ok;
    // The selected inset is derived from prepared green.m. Both amount==1
    // (m==0) and amount==6 (m!=0) are represented by the same preparation
    // helper; no metadata or runtime selection is inferred here.
    Core3ComposeControls prepared_phase{};
    if (prepare_core3_controls_007f2124(controls.amount,
                                        controls.suppress_refine, 0, 0,
                                        &prepared_phase) !=
        Core3ComposeStatus::ok) {
        return PreparedImageStatus::prepare_core3_controls_failure;
    }
    const std::size_t inset = prepared_phase.green.m == 0.0f ? 8u : 12u;
    expansion_status = validate_expanded_tiles(
        width, height, source_rect, controls.tile_width,
        controls.tile_height, inset);
    if (expansion_status != PreparedImageStatus::ok) return expansion_status;

    BayerGainPrepared forward_gain{};
    BayerGainPrepared inverse_gain{};
    if (prepare_bayer_gain_from_source_floor_007ef070(
            controls.gains, 3, controls.source_floor, false,
            &forward_gain) != BayerGainStatus::ok ||
        prepare_bayer_gain_from_source_floor_007ef070(
            controls.gains, 3, controls.source_floor, true,
            &inverse_gain) != BayerGainStatus::ok) {
        return PreparedImageStatus::prepare_gain_failure;
    }

    constexpr double kReciprocal65535 = 0x1.0001000100010p-16;
    const double pivot = static_cast<double>(controls.source_floor) *
                         kReciprocal65535;
    Stage3CurvePrepared curve{};
    if (prepare_stage3_curve(pivot, controls.cubic, &curve) !=
        Stage3CurveStatus::ok) {
        return PreparedImageStatus::prepare_curve_failure;
    }
    FloatLutPrepared forward_float_lut{};
    FloatLutPrepared inverse_float_lut{};
    if (build_stage3_curve_lut(curve, false, &forward_float_lut) !=
            Stage3CurveStatus::ok ||
        build_stage3_curve_lut(curve, true, &inverse_float_lut) !=
            Stage3CurveStatus::ok) {
        return PreparedImageStatus::build_curve_lut_failure;
    }
    U16LutPrepared forward_lut{};
    U16LutPrepared inverse_lut{};
    if (build_u16_lut_01300600(forward_float_lut, &forward_lut) !=
            FloatLutStatus::ok ||
        build_u16_lut_01300600(inverse_float_lut, &inverse_lut) !=
            FloatLutStatus::ok) {
        return PreparedImageStatus::build_u16_lut_failure;
    }

    try {
        std::vector<std::uint16_t> candidate;
        candidate.resize(output_samples);
        for (std::size_t tile_y = 0; tile_y < height;) {
            const std::size_t current_height =
                std::min(controls.tile_height, height - tile_y);
            for (std::size_t tile_x = 0; tile_x < width;) {
                const std::size_t current_width =
                    std::min(controls.tile_width, width - tile_x);
                const PreparedImageStatus tile_status = render_tile(
                    source, source_sample_count, source_row_stride, source_rect,
                    width, tile_x, tile_y, current_width, current_height, inset,
                    forward_gain, inverse_gain, forward_lut, inverse_lut,
                    controls.amount, controls.suppress_refine,
                    controls.hot_threshold, controls.cold_threshold, &candidate);
                if (tile_status != PreparedImageStatus::ok) return tile_status;
                tile_x += current_width;
            }
            tile_y += current_height;
        }

        output->rgb.swap(candidate);
        output->width = width;
        output->height = height;
        return PreparedImageStatus::ok;
    } catch (const std::bad_alloc&) {
        return PreparedImageStatus::allocation_failure;
    } catch (const std::length_error&) {
        return PreparedImageStatus::length_error;
    } catch (...) {
        return PreparedImageStatus::allocation_failure;
    }
}

PreparedImageStatus run_stage3_prepared_image_u16_to_buffer(
    const std::uint16_t* source, std::size_t source_sample_count,
    std::size_t source_row_stride, const std::int32_t rect4[4],
    const double gains3[3], std::uint16_t source_floor, int cubic,
    double amount, int suppress_refine, std::uint16_t hot_threshold,
    std::uint16_t cold_threshold, std::size_t tile_width,
    std::size_t tile_height, std::uint16_t* destination,
    std::size_t destination_sample_count) noexcept {
    if (source == nullptr || rect4 == nullptr || gains3 == nullptr ||
        destination == nullptr) {
        return PreparedImageStatus::invalid_argument;
    }
    if (cubic != 0 && cubic != 1) return PreparedImageStatus::invalid_controls;
    if (suppress_refine != 0 && suppress_refine != 1) {
        return PreparedImageStatus::invalid_controls;
    }

    AddressSpan source_capacity_span{};
    AddressSpan destination_capacity_span{};
    AddressSpan rect_span{};
    AddressSpan gains_span{};
    if (!address_span(source, source_sample_count, sizeof(*source),
                      alignof(std::uint16_t), &source_capacity_span) ||
        !address_span(destination, destination_sample_count,
                      sizeof(*destination), alignof(std::uint16_t),
                      &destination_capacity_span) ||
        !address_span(rect4, 4, sizeof(*rect4), alignof(std::int32_t),
                      &rect_span) ||
        !address_span(gains3, 3, sizeof(*gains3), alignof(double),
                      &gains_span)) {
        return PreparedImageStatus::arithmetic_overflow;
    }
    if (overlaps(destination_capacity_span, source_capacity_span) ||
        overlaps(destination_capacity_span, rect_span) ||
        overlaps(destination_capacity_span, gains_span)) {
        return PreparedImageStatus::overlap;
    }

    const std::int64_t width64 = static_cast<std::int64_t>(rect4[3]) -
                                 static_cast<std::int64_t>(rect4[1]);
    const std::int64_t height64 = static_cast<std::int64_t>(rect4[2]) -
                                  static_cast<std::int64_t>(rect4[0]);
    if (width64 <= 0 || height64 <= 0) return PreparedImageStatus::invalid_rect;
    if (width64 < 2 || height64 < 2 ||
        width64 > std::numeric_limits<std::int32_t>::max() ||
        height64 > std::numeric_limits<std::int32_t>::max()) {
        return PreparedImageStatus::invalid_dimensions;
    }
    const std::size_t width = static_cast<std::size_t>(width64);
    const std::size_t height = static_cast<std::size_t>(height64);
    std::size_t pixels = 0;
    std::size_t expected_samples = 0;
    std::size_t expected_bytes = 0;
    if (!checked_mul(width, height, &pixels) ||
        !checked_mul(pixels, 3, &expected_samples) ||
        !checked_byte_count(expected_samples, sizeof(*destination),
                            &expected_bytes)) {
        return PreparedImageStatus::arithmetic_overflow;
    }
    (void)expected_bytes;
    if (expected_samples > kMaximumRgbSamples) {
        return PreparedImageStatus::resource_limit;
    }
    if (destination_sample_count < expected_samples) {
        return PreparedImageStatus::buffer_too_small;
    }

    PreparedImageControls controls{};
    controls.gains[0] = gains3[0];
    controls.gains[1] = gains3[1];
    controls.gains[2] = gains3[2];
    controls.source_floor = source_floor;
    controls.cubic = cubic != 0;
    controls.amount = amount;
    controls.suppress_refine = suppress_refine != 0;
    controls.hot_threshold = hot_threshold;
    controls.cold_threshold = cold_threshold;
    controls.tile_width = tile_width;
    controls.tile_height = tile_height;
    const BayerRepeatRect source_rect{rect4[0], rect4[1], rect4[2], rect4[3]};

    PreparedImageOutput rendered{};
    const PreparedImageStatus status = run_stage3_prepared_image_u16(
        source, source_sample_count, source_row_stride, source_rect, controls,
        &rendered);
    if (status != PreparedImageStatus::ok) return status;
    if (rendered.width != width || rendered.height != height ||
        rendered.rgb.size() != expected_samples) {
        return PreparedImageStatus::arithmetic_overflow;
    }
    std::copy(rendered.rgb.begin(), rendered.rgb.end(), destination);
    return PreparedImageStatus::ok;
}

}  // namespace imprint::acr
