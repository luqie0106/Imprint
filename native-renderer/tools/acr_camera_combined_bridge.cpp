// Validation seam only; pixel arithmetic remains in the native CPU kernel.
#include "../src/acr_camera_combined.hpp"

#include <cstddef>
#include <cstdint>

extern "C" int acr_test_camera_combined(
    const float *src, std::uint64_t src_count, std::uint64_t src_row_stride,
    std::uint64_t height, std::uint64_t width, const double *controls,
    std::uint64_t control_count, const std::uint32_t order3[3],
    std::uint32_t flags, float *dst, std::uint64_t dst_count,
    std::uint64_t dst_row_stride) noexcept {
    return static_cast<int>(
        imprint::acr::apply_camera_combined_prepared_f32_to_buffer(
            src, static_cast<std::size_t>(src_count),
            static_cast<std::size_t>(src_row_stride),
            static_cast<std::size_t>(height), static_cast<std::size_t>(width),
            controls, static_cast<std::size_t>(control_count), order3, flags,
            dst, static_cast<std::size_t>(dst_count),
            static_cast<std::size_t>(dst_row_stride)));
}
