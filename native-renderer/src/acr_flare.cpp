#include "acr_flare.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

namespace acr_flare {
namespace {

constexpr std::size_t kMaximumPixels = std::size_t{1} << 26;
constexpr std::size_t kTableIntervals = 4096;
constexpr std::size_t kTableSamplesWithGuard = kTableIntervals + 2;
constexpr double kU16Maximum = 65535.0;
constexpr double kFlareScale = 1.0 / 1048576.0;
constexpr double kEpsilonMaximum = 1.0 / 4096.0;

struct AddressRange {
    std::uintptr_t begin;
    std::uintptr_t end;
};

bool make_range(const void* pointer, std::size_t bytes, AddressRange* range) noexcept {
    if (pointer == nullptr || bytes == 0 || range == nullptr) {
        return false;
    }
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    const auto maximum = std::numeric_limits<std::uintptr_t>::max();
    if (bytes > maximum - begin) {
        return false;
    }
    range->begin = begin;
    range->end = begin + bytes;
    return true;
}

bool overlaps(AddressRange left, AddressRange right) noexcept {
    return left.begin < right.end && right.begin < left.end;
}

float signed_sqrt_forward(float value, bool nonnegative) noexcept {
    constexpr float kQuarter = 0.00390625f;
    constexpr float kSixteenth = 0.0625f;
    constexpr float kScale = 0x1.107fe0p+0f;
    float root_input = kQuarter - value;
    float scale = -kScale;
    if (nonnegative) {
        root_input = value + kQuarter;
        scale = kScale;
    }
    const float root = std::sqrt(root_input);
    const float centered = root + -kSixteenth;
    return centered * scale;
}

float signed_sqrt_inverse(float value, bool negative) noexcept {
    constexpr float kSixteenth = 0.0625f;
    constexpr float kScale = 0x1.e0ffc0p-1f;
    const float signed_scale = negative ? -kScale : kScale;
    const float z = std::fma(value, signed_scale, kSixteenth);
    const float v = std::fma(z, z, -0.00390625f);
    return negative ? -v : v;
}

enum class TableFunction {
    forward_bracket,
    inverse_bracket,
    black_normalization,
};

double clamp_unit(double value) noexcept {
    if (value < 0.0) {
        return 0.0;
    }
    if (value > 1.0) {
        return 1.0;
    }
    return value;
}

float sample_table_function(double input,
                            TableFunction function,
                            double black,
                            double q,
                            bool zero_black_special) noexcept {
    if (function == TableFunction::black_normalization) {
        const double d = black - 1.0;
        const double scale = -1.0 / d;
        const double u = input - black;
        const double normalized = std::fma(scale, u, +0.0);
        return static_cast<float>(clamp_unit(normalized));
    }

    if (zero_black_special) {
        const float input_float = static_cast<float>(input);
        return function == TableFunction::forward_bracket
                   ? signed_sqrt_forward(input_float, true)
                   : signed_sqrt_inverse(input_float, false);
    }

    const double base = function == TableFunction::forward_bracket ? black : q;
    const double d = base - 1.0;
    const double scale = -1.0 / d;
    const double u = input - base;
    const double normalized = std::fma(scale, u, +0.0);
    const float normalized_float = static_cast<float>(normalized);
    const bool negative = normalized < 0.0;
    const float nonlinear = function == TableFunction::forward_bracket
                                ? signed_sqrt_forward(normalized_float, !negative)
                                : signed_sqrt_inverse(normalized_float, negative);
    const double nonlinear_double = static_cast<double>(nonlinear);
    const double mapped = function == TableFunction::forward_bracket
                              ? std::fma(-(q - 1.0), nonlinear_double, q)
                              : std::fma(-(black - 1.0), nonlinear_double, black);
    return static_cast<float>(clamp_unit(mapped));
}

void build_u16_lut(double black,
                   double q,
                   TableFunction function,
                   std::vector<std::uint16_t>* output) {
    std::vector<float> sampled(kTableSamplesWithGuard);
    const bool zero_black_special = black == 0.0 && q == 0.0;
    const double table_step = 1.0 / static_cast<double>(kTableIntervals);
    for (std::size_t index = 0; index <= kTableIntervals; ++index) {
        const double input = table_step * static_cast<double>(index);
        sampled[index] = sample_table_function(
            input, function, black, q, zero_black_special);
    }
    sampled[kTableIntervals + 1] = sampled[kTableIntervals];

    output->resize(65536);
    double fraction = 0.0;
    std::size_t right_index = 1;
    double right = static_cast<double>(sampled[1]);
    double intercept = std::fma(static_cast<double>(sampled[0]),
                                kU16Maximum,
                                0.5);
    double slope = (right - static_cast<double>(sampled[0])) * kU16Maximum;
    const double fraction_step = static_cast<double>(kTableIntervals) / kU16Maximum;

    for (std::size_t index = 0; index < output->size(); ++index) {
        const double expanded = std::fma(slope, fraction, intercept);
        const auto integer = static_cast<std::uint32_t>(expanded);
        (*output)[index] = static_cast<std::uint16_t>(integer);

        fraction = fraction_step + fraction;
        double next_right = right;
        if (1.0 < fraction) {
            ++right_index;
            fraction = fraction + -1.0;
            next_right = static_cast<double>(sampled[right_index]);
            intercept = std::fma(right, kU16Maximum, 0.5);
            slope = (next_right - right) * kU16Maximum;
        }
        right = next_right;
    }
}

std::size_t rounded_level_dimension(std::size_t original, std::size_t level) noexcept {
    const std::size_t divisor_offset = std::size_t{1} << level;
    const std::size_t shift = level + 1;
    const std::size_t rounded = (original + divisor_offset) >> shift;
    return std::max<std::size_t>(1, rounded);
}

std::size_t clamped_coordinate(std::size_t coordinate, std::size_t extent) noexcept {
    return coordinate < extent ? coordinate : extent - 1;
}

void downsample_level(const std::uint16_t* source,
                      std::size_t source_width,
                      std::size_t source_height,
                      std::size_t origin_x,
                      std::size_t origin_y,
                      std::size_t destination_width,
                      std::size_t destination_height,
                      const std::vector<std::uint16_t>* forward_lut,
                      const std::vector<std::uint16_t>* inverse_lut,
                      std::vector<std::uint16_t>* destination) {
    destination->resize(destination_width * destination_height * 3);
    const bool nonlinear = forward_lut != nullptr;
    std::size_t output_index = 0;
    const std::size_t source_stride = source_width * 3;
    for (std::size_t y = 0; y < destination_height; ++y) {
        const std::size_t y0 = clamped_coordinate(origin_y + y * 2, source_height);
        const std::size_t y1 = clamped_coordinate(origin_y + y * 2 + 1, source_height);
        for (std::size_t x = 0; x < destination_width; ++x) {
            const std::size_t x0 = clamped_coordinate(origin_x + x * 2, source_width);
            const std::size_t x1 = clamped_coordinate(origin_x + x * 2 + 1, source_width);
            const std::size_t a = y0 * source_stride + x0 * 3;
            const std::size_t b = y0 * source_stride + x1 * 3;
            const std::size_t c = y1 * source_stride + x0 * 3;
            const std::size_t d = y1 * source_stride + x1 * 3;
            for (std::size_t channel = 0; channel < 3; ++channel) {
                const std::uint32_t av = nonlinear ? (*forward_lut)[source[a + channel]]
                                                   : source[a + channel];
                const std::uint32_t bv = nonlinear ? (*forward_lut)[source[b + channel]]
                                                   : source[b + channel];
                const std::uint32_t cv = nonlinear ? (*forward_lut)[source[c + channel]]
                                                   : source[c + channel];
                const std::uint32_t dv = nonlinear ? (*forward_lut)[source[d + channel]]
                                                   : source[d + channel];
                const std::uint16_t average = static_cast<std::uint16_t>(
                    (av + bv + cv + dv + 2u) >> 2u);
                (*destination)[output_index++] =
                    nonlinear ? (*inverse_lut)[average] : average;
            }
        }
    }
}

}  // namespace

Status prepare_integer_lut(std::uint16_t black_offset,
                           IntegerTableKind kind,
                           std::uint16_t* output,
                           std::size_t count) noexcept {
    constexpr std::size_t kEntryCount = 65536;
    constexpr std::size_t kOutputBytes = kEntryCount * sizeof(std::uint16_t);
    if (output == nullptr || count != kEntryCount || black_offset > 65534 ||
        (kind != IntegerTableKind::black && kind != IntegerTableKind::forward &&
         kind != IntegerTableKind::inverse)) {
        return Status::invalid_argument;
    }

    const auto begin = reinterpret_cast<std::uintptr_t>(output);
    if (begin % alignof(std::uint16_t) != 0) {
        return Status::invalid_argument;
    }
    if (kOutputBytes > std::numeric_limits<std::uintptr_t>::max() - begin) {
        return Status::size_overflow;
    }

    TableFunction function = TableFunction::black_normalization;
    if (kind == IntegerTableKind::forward) {
        function = TableFunction::forward_bracket;
    } else if (kind == IntegerTableKind::inverse) {
        function = TableFunction::inverse_bracket;
    }

    try {
        const double black = static_cast<double>(black_offset) * (1.0 / kU16Maximum);
        const double q = black > 0.0 ? 0.25 : 0.0;
        std::vector<std::uint16_t> candidate;
        build_u16_lut(black, q, function, &candidate);
        std::memcpy(output, candidate.data(), kOutputBytes);
        return Status::ok;
    } catch (const std::bad_alloc&) {
        return Status::allocation_failure;
    } catch (...) {
        return Status::allocation_failure;
    }
}

Status calculate(const std::uint16_t* packed_rgb,
                 std::size_t image_width,
                 std::size_t image_height,
                 Crop crop,
                 std::uint16_t black_offset,
                 bool nonlinear_enabled,
                 bool scene_referred,
                 Result* output,
                 std::uint16_t* last_level_rgb,
                 std::size_t last_level_capacity_samples,
                 StatisticsLevel* statistics_level) noexcept {
    if (packed_rgb == nullptr || output == nullptr || image_width == 0 ||
        image_height == 0 || crop.width == 0 || crop.height == 0 ||
        black_offset > 65534 ||
        (last_level_rgb == nullptr && last_level_capacity_samples != 0)) {
        return Status::invalid_argument;
    }
    if (reinterpret_cast<std::uintptr_t>(packed_rgb) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(output) % alignof(Result) != 0 ||
        (last_level_rgb != nullptr &&
         reinterpret_cast<std::uintptr_t>(last_level_rgb) % alignof(std::uint16_t) != 0)) {
        return Status::invalid_argument;
    }
    if (image_width > kMaximumPixels / image_height) {
        return Status::too_many_pixels;
    }
    const std::size_t image_pixels = image_width * image_height;
    if (image_pixels > kMaximumPixels) {
        return Status::too_many_pixels;
    }
    if (crop.x > image_width || crop.y > image_height ||
        crop.width > image_width - crop.x || crop.height > image_height - crop.y) {
        return Status::invalid_argument;
    }

    const bool build_pyramid = crop.width > 64 || crop.height > 64;
    const std::size_t last_width = build_pyramid
                                       ? rounded_level_dimension(crop.width, 4)
                                       : image_width;
    const std::size_t last_height = build_pyramid
                                        ? rounded_level_dimension(crop.height, 4)
                                        : image_height;
    const std::size_t last_pixels = last_width * last_height;
    const std::size_t input_samples = image_pixels * 3;
    const std::size_t input_bytes = input_samples * sizeof(std::uint16_t);
    const std::size_t last_samples = last_pixels * 3;
    const std::size_t last_bytes = last_samples * sizeof(std::uint16_t);
    if (last_level_rgb != nullptr && last_level_capacity_samples < last_samples) {
        return Status::invalid_argument;
    }

    AddressRange input_range{};
    AddressRange result_range{};
    if (!make_range(packed_rgb, input_bytes, &input_range) ||
        !make_range(output, sizeof(Result), &result_range)) {
        return Status::size_overflow;
    }
    AddressRange last_range{};
    if (last_level_rgb != nullptr && !make_range(last_level_rgb, last_bytes, &last_range)) {
        return Status::size_overflow;
    }
    AddressRange statistics_range{};
    if (statistics_level != nullptr) {
        if (reinterpret_cast<std::uintptr_t>(statistics_level) % alignof(StatisticsLevel) != 0) {
            return Status::invalid_argument;
        }
        if (!make_range(statistics_level, sizeof(*statistics_level), &statistics_range)) {
            return Status::size_overflow;
        }
    }
    if (overlaps(input_range, result_range) ||
        (last_level_rgb != nullptr &&
         (overlaps(input_range, last_range) || overlaps(result_range, last_range))) ||
        (statistics_level != nullptr &&
         (overlaps(input_range, statistics_range) || overlaps(result_range, statistics_range) ||
          (last_level_rgb != nullptr && overlaps(last_range, statistics_range))))) {
        return Status::overlapping_buffers;
    }

    if (statistics_level != nullptr && statistics_level->pixels.capacity() != 0) {
        if (statistics_level->pixels.capacity() >
            std::numeric_limits<std::size_t>::max() / sizeof(std::uint16_t)) {
            return Status::size_overflow;
        }
        const std::size_t existing_statistics_bytes =
            statistics_level->pixels.capacity() * sizeof(std::uint16_t);
        AddressRange existing_statistics_range{};
        if (!make_range(statistics_level->pixels.data(),
                        existing_statistics_bytes,
                        &existing_statistics_range)) {
            return Status::size_overflow;
        }
        if (overlaps(input_range, existing_statistics_range) ||
            overlaps(result_range, existing_statistics_range) ||
            overlaps(statistics_range, existing_statistics_range) ||
            (last_level_rgb != nullptr && overlaps(last_range, existing_statistics_range))) {
            return Status::overlapping_buffers;
        }
    }

    try {
        std::vector<std::uint16_t> forward_lut;
        std::vector<std::uint16_t> inverse_lut;
        std::vector<std::uint16_t> black_lut;
        if (nonlinear_enabled) {
            const double black =
                static_cast<double>(black_offset) * (1.0 / kU16Maximum);
            const double q = black > 0.0 ? 0.25 : 0.0;
            build_u16_lut(black, q, TableFunction::forward_bracket, &forward_lut);
            build_u16_lut(black, q, TableFunction::inverse_bracket, &inverse_lut);
        }
        if (black_offset > 0) {
            const double black =
                static_cast<double>(black_offset) * (1.0 / kU16Maximum);
            build_u16_lut(black, 0.0, TableFunction::black_normalization, &black_lut);
        }

        const std::uint16_t* selected_pixels = packed_rgb;
        std::size_t selected_width = image_width;
        std::size_t selected_height = image_height;

        StatisticsLevel computed_statistics;
        std::uint32_t selected_statistics_level = 0;
        std::size_t statistics_width = image_width;
        std::size_t statistics_height = image_height;
        if (statistics_level != nullptr) {
            const bool statistics_has_pyramid = crop.width > 64 || crop.height > 64;
            if (statistics_has_pyramid) {
                for (std::uint32_t candidate = 1; candidate <= 5; ++candidate) {
                    const std::size_t candidate_width =
                        rounded_level_dimension(crop.width, candidate - 1);
                    const std::size_t candidate_height =
                        rounded_level_dimension(crop.height, candidate - 1);
                    if (std::max(candidate_width, candidate_height) < 128) {
                        break;
                    }
                    selected_statistics_level = candidate;
                    statistics_width = candidate_width;
                    statistics_height = candidate_height;
                }
            }
            if (selected_statistics_level == 0) {
                computed_statistics.pixels.assign(packed_rgb,
                                                   packed_rgb + input_samples);
            }
            computed_statistics.width = statistics_width;
            computed_statistics.height = statistics_height;
            computed_statistics.level = selected_statistics_level;
        }

        std::vector<std::uint16_t> previous;
        std::vector<std::uint16_t> next;
        if (build_pyramid) {
            for (std::size_t level = 0; level < 5; ++level) {
                const std::size_t destination_width =
                    rounded_level_dimension(crop.width, level);
                const std::size_t destination_height =
                    rounded_level_dimension(crop.height, level);
                const std::uint16_t* source =
                    level == 0 ? packed_rgb : previous.data();
                const std::size_t source_width =
                    level == 0 ? image_width : selected_width;
                const std::size_t source_height =
                    level == 0 ? image_height : selected_height;
                const std::size_t origin_x = level == 0 ? crop.x : 0;
                const std::size_t origin_y = level == 0 ? crop.y : 0;
                downsample_level(source,
                                 source_width,
                                 source_height,
                                 origin_x,
                                 origin_y,
                                 destination_width,
                                 destination_height,
                                 nonlinear_enabled ? &forward_lut : nullptr,
                                 nonlinear_enabled ? &inverse_lut : nullptr,
                                 &next);
                previous.swap(next);
                selected_pixels = previous.data();
                selected_width = destination_width;
                selected_height = destination_height;
                if (statistics_level != nullptr && level + 1 == selected_statistics_level) {
                    computed_statistics.pixels = previous;
                }
            }
        }

        double channel_sums[3] = {0.0, 0.0, 0.0};
        std::size_t sample_index = 0;
        for (std::size_t y = 0; y < selected_height; ++y) {
            for (std::size_t x = 0; x < selected_width; ++x) {
                for (std::size_t channel = 0; channel < 3; ++channel) {
                    std::uint16_t value = selected_pixels[sample_index];
                    if (black_offset > 0) {
                        value = black_lut[value];
                    }
                    if (last_level_rgb != nullptr) {
                        last_level_rgb[sample_index] = value;
                    }
                    channel_sums[channel] += static_cast<double>(value);
                    ++sample_index;
                }
            }
        }

        const double count = static_cast<double>(last_pixels);
        const double red_mean = channel_sums[0] / count;
        const double green_mean = channel_sums[1] / count;
        const double blue_mean = channel_sums[2] / count;
        const double normalized_mean =
            (((red_mean + green_mean) + blue_mean) / 3.0) * (1.0 / kU16Maximum);
        double flare = normalized_mean * kFlareScale;
        if (flare < 0.0) {
            flare = 0.0;
        }
        if (flare > kEpsilonMaximum) {
            flare = kEpsilonMaximum;
        }
        const double epsilon = scene_referred && flare > 0.0
                                   ? std::min(flare, kEpsilonMaximum)
                                   : kEpsilonMaximum;

        const Result computed{normalized_mean,
                              flare,
                              epsilon,
                              selected_width,
                              selected_height,
                              build_pyramid ? 5u : 0u};
        *output = computed;
        if (statistics_level != nullptr) {
            statistics_level->pixels.swap(computed_statistics.pixels);
            statistics_level->width = computed_statistics.width;
            statistics_level->height = computed_statistics.height;
            statistics_level->level = computed_statistics.level;
        }
        return Status::ok;
    } catch (const std::bad_alloc&) {
        return Status::allocation_failure;
    } catch (...) {
        return Status::allocation_failure;
    }
}

}  // namespace acr_flare
