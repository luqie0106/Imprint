#include "acr_display.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

namespace imprint::acr {
namespace {

constexpr std::size_t kMaximumPixels = std::size_t{1} << 26;
constexpr double kSrgbToXyz50[9] = {
    0.4361, 0.3851, 0.1431,
    0.2225, 0.7169, 0.0606,
    0.0139, 0.0971, 0.7141,
};

bool overlaps(const void *left, std::size_t left_size,
              const void *right, std::size_t right_size) noexcept {
    const auto left_start = reinterpret_cast<std::uintptr_t>(left);
    const auto right_start = reinterpret_cast<std::uintptr_t>(right);
    if (left_start <= right_start) return right_start - left_start < left_size;
    return left_start - right_start < right_size;
}

bool range_nonwrapping(const void *pointer, std::size_t byte_length) noexcept {
    const auto start = reinterpret_cast<std::uintptr_t>(pointer);
    return byte_length <= std::numeric_limits<std::uintptr_t>::max() - start;
}

bool pixel_count(std::uint32_t width, std::uint32_t height,
                 std::size_t &pixels) noexcept {
    if (width == 0 || height == 0) return false;
    const std::uint64_t count = static_cast<std::uint64_t>(width) * height;
    if (count > kMaximumPixels) return false;
    pixels = static_cast<std::size_t>(count);
    return true;
}

// Explicit cofactor inverse of the sRGB working matrix after the same D50 row
// normalization used by ACR's ProPhoto working matrix. The base values come
// from src/camera_profile.py's SRGB_TO_XYZ50 constant; keeping both working
// whites normalized makes the resulting ProPhoto-to-sRGB matrix map neutral
// white to neutral white.
bool inverse_srgb_to_xyz50(double inverse[9]) noexcept {
    constexpr double kD50White[3] = {
        0.3457 / 0.3585,
        1.0,
        ((1.0 - 0.3457) - 0.3585) / 0.3585,
    };
    double normalized[9]{};
    for (std::size_t row = 0; row < 3; ++row) {
        const double row_sum = (kSrgbToXyz50[row * 3] +
                                kSrgbToXyz50[row * 3 + 1]) +
                               kSrgbToXyz50[row * 3 + 2];
        if (!std::isfinite(row_sum) || row_sum == 0.0) return false;
        const double scale = kD50White[row] / row_sum;
        if (!std::isfinite(scale)) return false;
        for (std::size_t column = 0; column < 3; ++column) {
            normalized[row * 3 + column] =
                scale * kSrgbToXyz50[row * 3 + column];
            if (!std::isfinite(normalized[row * 3 + column])) return false;
        }
    }
    const double a = normalized[0];
    const double b = normalized[1];
    const double c = normalized[2];
    const double d = normalized[3];
    const double e = normalized[4];
    const double f = normalized[5];
    const double g = normalized[6];
    const double h = normalized[7];
    const double i = normalized[8];
    const double ei_fh = e * i - f * h;
    const double di_fg = d * i - f * g;
    const double dh_eg = d * h - e * g;
    const double determinant = a * ei_fh - b * di_fg + c * dh_eg;
    if (!std::isfinite(determinant) || determinant == 0.0) return false;
    const double reciprocal = 1.0 / determinant;
    const double candidate[9] = {
        ei_fh * reciprocal, (c * h - b * i) * reciprocal,
        (b * f - c * e) * reciprocal,
        (f * g - d * i) * reciprocal, (a * i - c * g) * reciprocal,
        (c * d - a * f) * reciprocal,
        dh_eg * reciprocal, (b * g - a * h) * reciprocal,
        (a * e - b * d) * reciprocal,
    };
    for (std::size_t index = 0; index < 9; ++index) {
        if (!std::isfinite(candidate[index])) return false;
        inverse[index] = candidate[index];
    }
    return true;
}

bool make_prophoto_to_srgb(const double q[9], float matrix[9]) noexcept {
    double inverse[9]{};
    if (!inverse_srgb_to_xyz50(inverse)) return false;
    constexpr double kFloatMaximum = std::numeric_limits<float>::max();
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t column = 0; column < 3; ++column) {
            double sum = 0.0;
            for (std::size_t inner = 0; inner < 3; ++inner) {
                const double product = inverse[row * 3 + inner] *
                                       q[inner * 3 + column];
                if (!std::isfinite(product)) return false;
                const double next = sum + product;
                if (!std::isfinite(next)) return false;
                sum = next;
            }
            if (std::abs(sum) > kFloatMaximum) return false;
            const float converted = static_cast<float>(sum);
            if (!std::isfinite(converted)) return false;
            matrix[row * 3 + column] = converted;
        }
    }
    return true;
}

bool finite_nonnegative(const float *source, std::size_t values) noexcept {
    for (std::size_t index = 0; index < values; ++index) {
        if (!std::isfinite(source[index]) || source[index] < 0.0f) return false;
    }
    return true;
}

bool finite_unit_interval(const float *source, std::size_t values) noexcept {
    for (std::size_t index = 0; index < values; ++index) {
        if (!std::isfinite(source[index]) || source[index] < 0.0f ||
            source[index] > 1.0f) {
            return false;
        }
    }
    return true;
}

std::uint8_t round_ties_even_u8(float value) noexcept {
    // value is already clipped to [0,255]. Using floor and an explicit tie
    // rule avoids depending on a caller-modified floating-point rounding mode.
    const float floor_value = std::floor(value);
    const auto lower = static_cast<unsigned int>(floor_value);
    const float fraction = value - floor_value;
    unsigned int rounded = lower;
    if (fraction > 0.5f || (fraction == 0.5f && (lower & 1u) != 0u)) {
        ++rounded;
    }
    return static_cast<std::uint8_t>(rounded);
}

bool project_pixel(const float matrix[9], const float *source,
                   std::uint8_t *output) noexcept {
    for (std::size_t row = 0; row < 3; ++row) {
        const float product0 = matrix[row * 3] * source[0];
        const float product1 = matrix[row * 3 + 1] * source[1];
        const float product2 = matrix[row * 3 + 2] * source[2];
        if (!std::isfinite(product0) || !std::isfinite(product1) ||
            !std::isfinite(product2)) {
            return false;
        }
        const float sum01 = product0 + product1;
        if (!std::isfinite(sum01)) return false;
        const float linear = sum01 + product2;
        if (!std::isfinite(linear)) return false;

        const float nonnegative = std::max(0.0f, linear);
        float encoded = 0.0f;
        if (nonnegative <= 0.0031308f) {
            encoded = 12.92f * nonnegative;
        } else {
            const float powered = std::pow(nonnegative, 1.0f / 2.4f);
            if (!std::isfinite(powered)) return false;
            encoded = 1.055f * powered - 0.055f;
        }
        if (!std::isfinite(encoded)) return false;
        const float scaled = encoded * 255.0f;
        if (!std::isfinite(scaled)) return false;
        const float clipped = std::min(255.0f, std::max(0.0f, scaled));
        output[row] = round_ties_even_u8(clipped);
    }
    return true;
}

}  // namespace

DisplayStatus project_display_srgb8(
    std::uint32_t width, std::uint32_t height,
    const float *source_rgb, std::size_t source_values,
    const double *normalized_prophoto_to_pcs_q, std::size_t q_values,
    std::uint8_t *output_rgb8, std::size_t output_values) noexcept {
    std::size_t pixels = 0;
    if (source_rgb == nullptr || normalized_prophoto_to_pcs_q == nullptr ||
        output_rgb8 == nullptr || q_values != 9 ||
        !pixel_count(width, height, pixels) ||
        pixels > std::numeric_limits<std::size_t>::max() / 3) {
        return DisplayStatus::invalid_input;
    }
    const std::size_t values = pixels * 3;
    const std::size_t source_bytes = values * sizeof(float);
    const std::size_t output_bytes = values * sizeof(std::uint8_t);
    const std::size_t q_bytes = 9 * sizeof(double);
    if (source_values != values || output_values != values ||
        !range_nonwrapping(source_rgb, source_bytes) ||
        !range_nonwrapping(normalized_prophoto_to_pcs_q, q_bytes) ||
        !range_nonwrapping(output_rgb8, output_bytes) ||
        overlaps(source_rgb, values * sizeof(float),
                 output_rgb8, output_bytes) ||
        overlaps(normalized_prophoto_to_pcs_q, q_bytes,
                 output_rgb8, output_bytes) ||
        overlaps(source_rgb, source_bytes,
                 normalized_prophoto_to_pcs_q, q_bytes)) {
        return DisplayStatus::invalid_input;
    }
    if (!finite_nonnegative(source_rgb, values)) return DisplayStatus::invalid_input;
    double q[9]{};
    for (std::size_t index = 0; index < 9; ++index) {
        q[index] = normalized_prophoto_to_pcs_q[index];
        if (!std::isfinite(q[index])) return DisplayStatus::invalid_input;
    }
    float matrix[9]{};
    if (!make_prophoto_to_srgb(q, matrix)) {
        return DisplayStatus::non_finite_intermediate;
    }

    try {
        std::vector<std::uint8_t> candidate(values);
        for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
            if (!project_pixel(matrix, source_rgb + pixel * 3,
                               candidate.data() + pixel * 3)) {
                return DisplayStatus::non_finite_intermediate;
            }
        }
        std::memcpy(output_rgb8, candidate.data(), values);
    } catch (const std::bad_alloc &) {
        return DisplayStatus::allocation_failure;
    } catch (...) {
        return DisplayStatus::allocation_failure;
    }
    return DisplayStatus::ok;
}

DisplayStatus scale_sdr_rgb(
    std::uint32_t width, std::uint32_t height,
    const float *source_rgb, std::size_t source_values, float gain,
    float *destination_rgb, std::size_t destination_values) noexcept {
    std::size_t pixels = 0;
    if (source_rgb == nullptr || destination_rgb == nullptr ||
        !pixel_count(width, height, pixels) ||
        pixels > std::numeric_limits<std::size_t>::max() / 3 ||
        !std::isfinite(gain) || gain < 0.0f) {
        return DisplayStatus::invalid_input;
    }
    const std::size_t values = pixels * 3;
    const std::size_t buffer_bytes = values * sizeof(float);
    if (source_values != values || destination_values != values ||
        !range_nonwrapping(source_rgb, buffer_bytes) ||
        !range_nonwrapping(destination_rgb, buffer_bytes) ||
        overlaps(source_rgb, buffer_bytes, destination_rgb, buffer_bytes)) {
        return DisplayStatus::invalid_input;
    }
    if (!finite_unit_interval(source_rgb, values)) return DisplayStatus::invalid_input;
    try {
        std::vector<float> candidate(values);
        for (std::size_t index = 0; index < values; ++index) {
            const float scaled = source_rgb[index] * gain;
            if (!std::isfinite(scaled)) return DisplayStatus::non_finite_intermediate;
            candidate[index] = std::min(scaled, 1.0f);
        }
        std::memcpy(destination_rgb, candidate.data(), values * sizeof(float));
    } catch (const std::bad_alloc &) {
        return DisplayStatus::allocation_failure;
    } catch (...) {
        return DisplayStatus::allocation_failure;
    }
    return DisplayStatus::ok;
}

}  // namespace imprint::acr
