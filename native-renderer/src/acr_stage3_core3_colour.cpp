#include "acr_stage3_core3_colour.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace imprint::acr {
namespace {

bool aligned_16(const void *pointer) noexcept {
    return (reinterpret_cast<std::uintptr_t>(pointer) & 0xFu) == 0;
}

bool valid_shape(std::size_t width, std::size_t stride) noexcept {
    if (width == 0 || (width & 3u) != 0 || stride < width ||
        (stride & 3u) != 0) {
        return false;
    }

    // Pointer arithmetic is performed in float samples, but the byte distance
    // must also fit ptrdiff_t. The largest center-relative access combines one
    // row stride with the right neighbor of the final output column; retain an
    // extra guard sample in the check so stride + width + guard is representable
    // before any derived row pointer is formed.
    const auto max_samples = static_cast<std::uintmax_t>(
        std::numeric_limits<std::ptrdiff_t>::max()) /
        sizeof(float);
    const auto width_samples = static_cast<std::uintmax_t>(width);
    const auto stride_samples = static_cast<std::uintmax_t>(stride);
    constexpr std::uintmax_t kHorizontalGuard = 1;
    if (width_samples > max_samples - kHorizontalGuard ||
        stride_samples > max_samples - width_samples - kHorizontalGuard) {
        return false;
    }
    return true;
}

float clamp01_propagating_nan(float value) noexcept {
    // Match the recovered FMIN(value, 1) followed by FMAX(0, value) while
    // retaining NaNs. In particular, the lower clamp selects positive zero
    // for either signed zero and for negative finite values.
    if (std::isnan(value)) return value;

    const float upper_clamped = value <= 1.0f ? value : 1.0f;
    if (upper_clamped > 0.0f) return upper_clamped;
    return 0.0f;
}

}  // namespace

Core3ColourStatus core3_colour_difference_row_008134b8(
    const float *source_center,
    const float *green_center,
    std::size_t source_stride,
    std::size_t green_stride,
    std::size_t width,
    std::uint32_t phase,
    float *out_a,
    float *out_b) noexcept {
    if (source_center == nullptr || green_center == nullptr || out_a == nullptr ||
        out_b == nullptr) {
        return Core3ColourStatus::invalid_argument;
    }
    if (phase > 1) return Core3ColourStatus::invalid_phase;
    if (!valid_shape(width, source_stride) ||
        !valid_shape(width, green_stride)) {
        return Core3ColourStatus::invalid_dimensions;
    }
    if (!aligned_16(source_center) || !aligned_16(green_center) ||
        !aligned_16(out_a) || !aligned_16(out_b)) {
        return Core3ColourStatus::unsupported_alignment;
    }

    const auto source_row_offset = static_cast<std::ptrdiff_t>(source_stride);
    const auto green_row_offset = static_cast<std::ptrdiff_t>(green_stride);
    const float *source_up = source_center - source_row_offset;
    const float *source_down = source_center + source_row_offset;
    const float *green_up = green_center - green_row_offset;
    const float *green_down = green_center + green_row_offset;

    for (std::size_t column = 0; column < width; ++column) {
        const auto j = static_cast<std::ptrdiff_t>(column);
        const float source = source_center[j];
        const float green = green_center[j];

        const float dx_left = source_center[j - 1] - green_center[j - 1];
        const float dx_right = source_center[j + 1] - green_center[j + 1];
        const float horizontal_delta = dx_left + dx_right;
        const float horizontal = clamp01_propagating_nan(
            std::fma(0.5f, horizontal_delta, green));

        const float dy_up = source_up[j] - green_up[j];
        const float dy_down = source_down[j] - green_down[j];
        const float vertical_delta = dy_up + dy_down;
        const float vertical = clamp01_propagating_nan(
            std::fma(0.5f, vertical_delta, green));

        const float a = source_up[j - 1] - green_up[j - 1];
        const float b = source_up[j + 1] - green_up[j + 1];
        const float c = source_down[j - 1] - green_down[j - 1];
        const float d = source_down[j + 1] - green_down[j + 1];
        const float c_plus_d = c + d;
        const float b_plus_cd = b + c_plus_d;
        const float diagonal_delta = a + b_plus_cd;
        const float diagonal = clamp01_propagating_nan(
            std::fma(0.25f, diagonal_delta, green));

        const bool use_source = ((column & 1u) == 0u) == (phase == 0);
        if (use_source) {
            out_a[column] = source;
            out_b[column] = diagonal;
        } else {
            out_a[column] = horizontal;
            out_b[column] = vertical;
        }
    }
    return Core3ColourStatus::ok;
}

}  // namespace imprint::acr
