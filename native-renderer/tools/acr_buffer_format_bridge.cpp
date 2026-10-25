#include "../src/acr_buffer_format.hpp"

extern "C" int acr_test_buffer_f32_to_u16(
    const float *src, std::size_t src_count, std::size_t src_row_stride,
    std::size_t height, std::size_t row_samples, std::uint16_t *dst,
    std::size_t dst_count, std::size_t dst_row_stride) {
    return static_cast<int>(imprint::acr::copy_buffer_f32_to_u16_012eaeb8(
        src, src_count, src_row_stride, height, row_samples, dst, dst_count,
        dst_row_stride));
}

extern "C" int acr_test_buffer_u16_to_f32(
    const std::uint16_t *src, std::size_t src_count,
    std::size_t src_row_stride, std::size_t height, std::size_t row_samples,
    float *dst, std::size_t dst_count, std::size_t dst_row_stride) {
    return static_cast<int>(imprint::acr::copy_buffer_u16_to_f32_012eac64(
        src, src_count, src_row_stride, height, row_samples, dst, dst_count,
        dst_row_stride));
}
