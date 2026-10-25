#include "../src/acr_exposure_sdr_mask.hpp"
#include "../include/imprint_renderer.h"

#include <cstddef>
#include <cstdint>
#include <limits>

extern "C" int acr_test_exposure_sdr_mask(
    const float *src, std::size_t src_count, std::size_t src_stride,
    std::size_t height, std::size_t width, const float *mask,
    std::size_t mask_count, std::size_t mask_stride, const float *controls,
    std::size_t controls_count, const float *table, std::size_t table_count,
    float *dst, std::size_t dst_count, std::size_t dst_stride) noexcept {
    if (controls == nullptr || controls_count != 6 ||
        reinterpret_cast<std::uintptr_t>(controls) % alignof(float) != 0) {
        return static_cast<int>(imprint::acr::ExposureSdrMaskStatus::invalid_argument);
    }

    constexpr std::size_t control_bytes = 6 * sizeof(float);
    const auto controls_begin = reinterpret_cast<std::uintptr_t>(controls);
    if (control_bytes > static_cast<std::size_t>(
                            std::numeric_limits<std::ptrdiff_t>::max()) ||
        control_bytes > std::numeric_limits<std::uintptr_t>::max() -
                            controls_begin) {
        return static_cast<int>(
            imprint::acr::ExposureSdrMaskStatus::arithmetic_overflow);
    }
    const auto controls_end = controls_begin + control_bytes;
    if (dst != nullptr) {
        const auto dst_begin = reinterpret_cast<std::uintptr_t>(dst);
        if (dst_begin % alignof(float) != 0) {
            return static_cast<int>(
                imprint::acr::ExposureSdrMaskStatus::invalid_argument);
        }
        if (dst_count > static_cast<std::size_t>(
                            std::numeric_limits<std::ptrdiff_t>::max()) /
                            sizeof(float)) {
            return static_cast<int>(
                imprint::acr::ExposureSdrMaskStatus::arithmetic_overflow);
        }
        const std::size_t dst_bytes = dst_count * sizeof(float);
        if (dst_bytes > std::numeric_limits<std::uintptr_t>::max() - dst_begin) {
            return static_cast<int>(
                imprint::acr::ExposureSdrMaskStatus::arithmetic_overflow);
        }
        const auto dst_end = dst_begin + dst_bytes;
        if (controls_begin < dst_end && dst_begin < controls_end) {
            return static_cast<int>(
                imprint::acr::ExposureSdrMaskStatus::unsupported_aliasing);
        }
    }

    imprint::acr::PreparedExposureSdrMask prepared;
    prepared.gain = controls[0];
    prepared.main_lut_scale = controls[1];
    prepared.rational_scale = controls[2];
    prepared.sqrt_amount = controls[3];
    prepared.rational_denominator = controls[4];
    prepared.epsilon = controls[5];
    prepared.main_lut = table;
    prepared.main_lut_count = table_count;

    return static_cast<int>(imprint::acr::apply_prepared_exposure_sdr_mask(
        src, src_count, src_stride, height, width, mask, mask_count,
        mask_stride, &prepared, dst, dst_count, dst_stride));
}

extern "C" int acr_test_exposure_sdr_mask_context(
    const float *src, std::size_t src_count, std::size_t src_stride,
    std::size_t height, std::size_t width, const float *mask,
    std::size_t mask_count, std::size_t mask_stride, const float *controls,
    std::size_t controls_count, const float *table, std::size_t table_count,
    const im_acr_exposure_lut_f32 *tables, std::size_t lut_count,
    float *dst, std::size_t dst_count, std::size_t dst_stride) noexcept {
    using imprint::acr::ExposureSdrMaskStatus;
    if (controls == nullptr || controls_count != 6 ||
        reinterpret_cast<std::uintptr_t>(controls) % alignof(float) != 0 ||
        tables == nullptr || lut_count != 5 ||
        reinterpret_cast<std::uintptr_t>(tables) %
                alignof(im_acr_exposure_lut_f32) != 0) {
        return static_cast<int>(ExposureSdrMaskStatus::invalid_argument);
    }

    constexpr std::size_t control_bytes = 6 * sizeof(float);
    constexpr std::size_t table_bytes =
        5 * sizeof(im_acr_exposure_lut_f32);
    const auto controls_begin = reinterpret_cast<std::uintptr_t>(controls);
    const auto tables_begin = reinterpret_cast<std::uintptr_t>(tables);
    if (control_bytes > static_cast<std::size_t>(
                            std::numeric_limits<std::ptrdiff_t>::max()) ||
        table_bytes > static_cast<std::size_t>(
                          std::numeric_limits<std::ptrdiff_t>::max()) ||
        control_bytes > std::numeric_limits<std::uintptr_t>::max() -
                            controls_begin ||
        table_bytes > std::numeric_limits<std::uintptr_t>::max() -
                          tables_begin) {
        return static_cast<int>(ExposureSdrMaskStatus::arithmetic_overflow);
    }
    if (dst != nullptr) {
        const auto dst_begin = reinterpret_cast<std::uintptr_t>(dst);
        if (dst_begin % alignof(float) != 0) {
            return static_cast<int>(ExposureSdrMaskStatus::invalid_argument);
        }
        if (dst_count > static_cast<std::size_t>(
                            std::numeric_limits<std::ptrdiff_t>::max()) /
                            sizeof(float)) {
            return static_cast<int>(ExposureSdrMaskStatus::arithmetic_overflow);
        }
        const std::size_t dst_bytes = dst_count * sizeof(float);
        if (dst_bytes > std::numeric_limits<std::uintptr_t>::max() - dst_begin) {
            return static_cast<int>(ExposureSdrMaskStatus::arithmetic_overflow);
        }
        const auto dst_end = dst_begin + dst_bytes;
        const auto controls_end = controls_begin + control_bytes;
        const auto tables_end = tables_begin + table_bytes;
        if ((controls_begin < dst_end && dst_begin < controls_end) ||
            (tables_begin < dst_end && dst_begin < tables_end)) {
            return static_cast<int>(ExposureSdrMaskStatus::unsupported_aliasing);
        }
    }

    imprint::acr::PreparedExposureSdrMask prepared;
    prepared.gain = controls[0];
    prepared.main_lut_scale = controls[1];
    prepared.rational_scale = controls[2];
    prepared.sqrt_amount = controls[3];
    prepared.rational_denominator = controls[4];
    prepared.epsilon = controls[5];
    prepared.main_lut = table;
    prepared.main_lut_count = table_count;
    for (std::size_t index = 0; index < 5; ++index) {
        prepared.context[index].values = tables[index].values;
        prepared.context[index].count = tables[index].count;
        prepared.context[index].max_index = tables[index].max_index;
        prepared.context[index].scale = tables[index].scale;
    }

    return static_cast<int>(imprint::acr::apply_prepared_exposure_sdr_mask(
        src, src_count, src_stride, height, width, mask, mask_count,
        mask_stride, &prepared, dst, dst_count, dst_stride));
}

extern "C" int acr_test_exposure_sdr_nomask_context(
    const float *src, std::size_t src_count, std::size_t src_stride,
    std::size_t height, std::size_t width, const float *controls,
    std::size_t controls_count, const float *table, std::size_t table_count,
    const im_acr_exposure_lut_f32 *tables, std::size_t lut_count,
    float *dst, std::size_t dst_count, std::size_t dst_stride) noexcept {
    using imprint::acr::ExposureSdrMaskStatus;
    if (controls == nullptr || controls_count != 6 ||
        reinterpret_cast<std::uintptr_t>(controls) % alignof(float) != 0 ||
        tables == nullptr || lut_count != 5 ||
        reinterpret_cast<std::uintptr_t>(tables) %
                alignof(im_acr_exposure_lut_f32) != 0) {
        return static_cast<int>(ExposureSdrMaskStatus::invalid_argument);
    }

    constexpr std::size_t control_bytes = 6 * sizeof(float);
    constexpr std::size_t table_bytes = 5 * sizeof(im_acr_exposure_lut_f32);
    const auto controls_begin = reinterpret_cast<std::uintptr_t>(controls);
    const auto tables_begin = reinterpret_cast<std::uintptr_t>(tables);
    if (control_bytes > static_cast<std::size_t>(
                            std::numeric_limits<std::ptrdiff_t>::max()) ||
        table_bytes > static_cast<std::size_t>(
                          std::numeric_limits<std::ptrdiff_t>::max()) ||
        control_bytes > std::numeric_limits<std::uintptr_t>::max() -
                            controls_begin ||
        table_bytes > std::numeric_limits<std::uintptr_t>::max() -
                          tables_begin) {
        return static_cast<int>(ExposureSdrMaskStatus::arithmetic_overflow);
    }
    if (dst != nullptr) {
        const auto dst_begin = reinterpret_cast<std::uintptr_t>(dst);
        if (dst_begin % alignof(float) != 0) {
            return static_cast<int>(ExposureSdrMaskStatus::invalid_argument);
        }
        if (dst_count > static_cast<std::size_t>(
                            std::numeric_limits<std::ptrdiff_t>::max()) /
                            sizeof(float)) {
            return static_cast<int>(ExposureSdrMaskStatus::arithmetic_overflow);
        }
        const std::size_t dst_bytes = dst_count * sizeof(float);
        if (dst_bytes > std::numeric_limits<std::uintptr_t>::max() - dst_begin) {
            return static_cast<int>(ExposureSdrMaskStatus::arithmetic_overflow);
        }
        const auto dst_end = dst_begin + dst_bytes;
        const auto controls_end = controls_begin + control_bytes;
        const auto tables_end = tables_begin + table_bytes;
        if ((controls_begin < dst_end && dst_begin < controls_end) ||
            (tables_begin < dst_end && dst_begin < tables_end)) {
            return static_cast<int>(ExposureSdrMaskStatus::unsupported_aliasing);
        }
    }

    imprint::acr::PreparedExposureSdrMask prepared;
    prepared.gain = controls[0];
    prepared.main_lut_scale = controls[1];
    prepared.rational_scale = controls[2];
    prepared.sqrt_amount = controls[3];
    prepared.rational_denominator = controls[4];
    prepared.epsilon = controls[5];
    prepared.main_lut = table;
    prepared.main_lut_count = table_count;
    for (std::size_t index = 0; index < 5; ++index) {
        prepared.context[index].values = tables[index].values;
        prepared.context[index].count = tables[index].count;
        prepared.context[index].max_index = tables[index].max_index;
        prepared.context[index].scale = tables[index].scale;
    }

    return static_cast<int>(imprint::acr::apply_prepared_exposure_sdr_nomask(
        src, src_count, src_stride, height, width, &prepared, dst, dst_count,
        dst_stride));
}
