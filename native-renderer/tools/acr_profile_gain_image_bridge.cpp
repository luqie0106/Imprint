#include "../src/acr_profile_gain_image.hpp"

extern "C" int acr_profile_gain_image_bridge(
    const float *src, std::size_t srcCount, std::size_t srcStride,
    float *dst, std::size_t dstCount, std::size_t dstStride,
    const std::uint8_t *payload, std::size_t bytes, std::uint32_t tag,
    std::uint32_t byte_order, const std::int32_t *rects,
    std::size_t rectCount, const double *matrix50,
    std::size_t matrixCount, double scalar, std::int32_t amount,
    std::uint32_t lookup_flag, std::uint32_t predicate,
    std::uint32_t has_task) noexcept {
    return static_cast<int>(imprint::acr::process_profile_gain_image_rgbf32(
        src, srcCount, srcStride, dst, dstCount, dstStride, payload, bytes,
        tag, byte_order, rects, rectCount, matrix50, matrixCount, scalar,
        amount, lookup_flag, predicate, has_task));
}
