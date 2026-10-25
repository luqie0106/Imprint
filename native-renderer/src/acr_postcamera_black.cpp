#include "acr_postcamera_black.hpp"
#include "acr_u16_lut.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>

namespace imprint::acr {
namespace {

constexpr std::size_t kActiveSampleLimit = 128u * 1024u * 1024u;

static_assert(sizeof(float) == sizeof(std::uint32_t) &&
              std::numeric_limits<float>::is_iec559 &&
              std::numeric_limits<float>::digits == 24);

struct AddressSpan {
    std::uintptr_t begin;
    std::uintptr_t end;
};

struct BlackFunction {
    double black;
    double scale;
};

bool checked_multiply(std::size_t left, std::size_t right,
                      std::size_t* result) noexcept {
    if (left != 0u && right > std::numeric_limits<std::size_t>::max() / left)
        return false;
    *result = left * right;
    return true;
}

bool checked_add(std::size_t left, std::size_t right,
                 std::size_t* result) noexcept {
    if (right > std::numeric_limits<std::size_t>::max() - left) return false;
    *result = left + right;
    return true;
}

bool make_address_span(const void* pointer, std::size_t byte_count,
                       AddressSpan* span) noexcept {
    if (pointer == nullptr || span == nullptr ||
        byte_count > static_cast<std::size_t>(
                         std::numeric_limits<std::ptrdiff_t>::max())) {
        return false;
    }
    const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (byte_count > std::numeric_limits<std::uintptr_t>::max() - begin)
        return false;
    span->begin = begin;
    span->end = begin + static_cast<std::uintptr_t>(byte_count);
    return true;
}

bool make_declared_span(const void* pointer, std::size_t count,
                        AddressSpan* span) noexcept {
    std::size_t byte_count = 0u;
    return checked_multiply(count, sizeof(float), &byte_count) &&
           make_address_span(pointer, byte_count, span);
}

bool spans_overlap(const AddressSpan& left,
                   const AddressSpan& right) noexcept {
    return left.begin < right.end && right.begin < left.end;
}

bool valid_black(double b) noexcept {
    return std::isfinite(b) && b >= 0.0 && b < 1.0;
}

bool evaluate_black(double x, const void* context,
                    double* out) noexcept {
    if (context == nullptr || out == nullptr) return false;
    const auto* function = static_cast<const BlackFunction*>(context);
    const double u = x - function->black;
    const double v = std::fma(function->scale, u, +0.0);
    if (!std::isfinite(u) || !std::isfinite(v)) return false;
    double pinned = std::fmin(v, 1.0);
    if (!std::isfinite(pinned)) return false;
    if (pinned < 0.0) pinned = +0.0;
    *out = pinned;
    return true;
}

PostcameraBlackStatus map_lut_status(FloatLutStatus status) noexcept {
    switch (status) {
        case FloatLutStatus::ok:
            return PostcameraBlackStatus::ok;
        case FloatLutStatus::invalid_argument:
            return PostcameraBlackStatus::invalid_argument;
        case FloatLutStatus::invalid_parameters:
            return PostcameraBlackStatus::invalid_argument;
        case FloatLutStatus::overlap:
            return PostcameraBlackStatus::overlap;
        case FloatLutStatus::evaluation_failure:
            return PostcameraBlackStatus::non_finite_intermediate;
    }
    return PostcameraBlackStatus::non_finite_intermediate;
}

}  // namespace

PostcameraBlackStatus build_postcamera_black_lut(
    double b, FloatLutPrepared* out) noexcept {
    using Status = PostcameraBlackStatus;
    if (out == nullptr) return Status::invalid_argument;
    const std::uintptr_t out_address = reinterpret_cast<std::uintptr_t>(out);
    if (out_address % alignof(FloatLutPrepared) != 0u)
        return Status::unsupported_alignment;

    AddressSpan out_span{};
    if (!make_address_span(out, sizeof(FloatLutPrepared), &out_span))
        return Status::arithmetic_overflow;
    (void)out_span;
    if (!valid_black(b)) return Status::invalid_black;

    const double d = b - 1.0;
    const double scale = -1.0 / d;
    if (!std::isfinite(d) || !std::isfinite(scale))
        return Status::non_finite_intermediate;

    const BlackFunction function{b, scale};
    FloatLutPrepared candidate{};
    const FloatLutStatus lut_status = build_float_lut_01300490(
        evaluate_black, &function, &candidate);
    if (lut_status != FloatLutStatus::ok) return map_lut_status(lut_status);
    *out = candidate;
    return Status::ok;
}

PostcameraBlackStatus apply_postcamera_black_prepared_f32(
    const float* src, std::size_t srcCount, std::size_t srcRowStride,
    std::size_t height, std::size_t rowSamples, double b, float* dst,
    std::size_t dstCount, std::size_t dstRowStride) noexcept {
    using Status = PostcameraBlackStatus;

    if (src == nullptr || dst == nullptr) return Status::invalid_argument;
    if (height == 0u || rowSamples == 0u) return Status::invalid_dimensions;
    if (srcRowStride < rowSamples || dstRowStride < rowSamples)
        return Status::invalid_stride;

    std::size_t active_sample_count = 0u;
    std::size_t src_last_row = 0u;
    std::size_t dst_last_row = 0u;
    std::size_t src_required = 0u;
    std::size_t dst_required = 0u;
    std::size_t row_bytes = 0u;
    if (!checked_multiply(height, rowSamples, &active_sample_count) ||
        !checked_multiply(height - 1u, srcRowStride, &src_last_row) ||
        !checked_multiply(height - 1u, dstRowStride, &dst_last_row) ||
        !checked_add(src_last_row, rowSamples, &src_required) ||
        !checked_add(dst_last_row, rowSamples, &dst_required) ||
        !checked_multiply(rowSamples, sizeof(float), &row_bytes)) {
        return Status::arithmetic_overflow;
    }
    const std::size_t ptrdiff_sample_limit =
        static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) /
        sizeof(float);
    if (src_required > ptrdiff_sample_limit ||
        dst_required > ptrdiff_sample_limit) {
        return Status::arithmetic_overflow;
    }

    AddressSpan src_span{};
    AddressSpan dst_span{};
    if (!make_declared_span(src, srcCount, &src_span) ||
        !make_declared_span(dst, dstCount, &dst_span)) {
        return Status::arithmetic_overflow;
    }
    if (active_sample_count > kActiveSampleLimit)
        return Status::resource_limit;
    if (srcCount < src_required || dstCount < dst_required)
        return Status::buffer_too_small;

    if (reinterpret_cast<std::uintptr_t>(src) % alignof(float) != 0u ||
        reinterpret_cast<std::uintptr_t>(dst) % alignof(float) != 0u) {
        return Status::unsupported_alignment;
    }

    const bool exact_in_place = src == dst && srcCount == dstCount &&
                                srcRowStride == dstRowStride;
    if (spans_overlap(src_span, dst_span) && !exact_in_place)
        return Status::overlap;

    if (!valid_black(b)) return Status::invalid_black;

    for (std::size_t y = 0u; y < height; ++y) {
        const std::size_t row_base = y * srcRowStride;
        for (std::size_t sample = 0u; sample < rowSamples; ++sample) {
            const float value = src[row_base + sample];
            if (!std::isfinite(value)) return Status::non_finite_input;
            if (value < 0.0f || value > 1.0f) return Status::out_of_domain;
        }
    }

    FloatLutPrepared lut{};
    if (b != 0.0) {
        const Status build_status = build_postcamera_black_lut(b, &lut);
        if (build_status != Status::ok) return build_status;
    }

    std::vector<float> candidate;
    try {
        candidate.resize(active_sample_count);
    } catch (const std::bad_alloc&) {
        return Status::allocation_failure;
    } catch (const std::length_error&) {
        return Status::length_error;
    } catch (...) {
        return Status::allocation_failure;
    }

    for (std::size_t y = 0u; y < height; ++y) {
        const std::size_t src_row = y * srcRowStride;
        const std::size_t candidate_row = y * rowSamples;
        if (b == 0.0) {
            std::memcpy(candidate.data() + candidate_row, src + src_row,
                        row_bytes);
            continue;
        }
        const FloatLutStatus apply_status = apply_float_lut_row_00875edc(
            lut, src + src_row, srcCount - src_row,
            candidate.data() + candidate_row,
            active_sample_count - candidate_row, rowSamples, false);
        if (apply_status != FloatLutStatus::ok)
            return map_lut_status(apply_status);
    }

    for (std::size_t y = 0u; y < height; ++y) {
        const std::size_t dst_row = y * dstRowStride;
        const std::size_t candidate_row = y * rowSamples;
        std::memcpy(dst + dst_row, candidate.data() + candidate_row, row_bytes);
    }
    return Status::ok;
}

namespace {

bool make_declared_u16_span(const void* pointer, std::size_t count,
                            AddressSpan* span) noexcept {
    std::size_t byte_count = 0u;
    return checked_multiply(count, sizeof(std::uint16_t), &byte_count) &&
           make_address_span(pointer, byte_count, span);
}

}  // namespace

PostcameraBlackStatus build_postcamera_black_u16_lut(
    double b, U16LutPrepared* out) noexcept {
    using Status = PostcameraBlackStatus;
    if (out == nullptr) return Status::invalid_argument;
    const std::uintptr_t out_address = reinterpret_cast<std::uintptr_t>(out);
    if (out_address % alignof(U16LutPrepared) != 0u)
        return Status::unsupported_alignment;

    AddressSpan out_span{};
    if (!make_address_span(out, sizeof(U16LutPrepared), &out_span))
        return Status::arithmetic_overflow;
    (void)out_span;

    FloatLutPrepared float_lut{};
    const Status float_status = build_postcamera_black_lut(b, &float_lut);
    if (float_status != Status::ok) return float_status;

    U16LutPrepared candidate{};
    const FloatLutStatus lut_status =
        build_u16_lut_01300600(float_lut, &candidate);
    if (lut_status != FloatLutStatus::ok) return map_lut_status(lut_status);
    *out = candidate;
    return Status::ok;
}

PostcameraBlackStatus apply_postcamera_black_prepared_u16(
    const std::uint16_t* src, std::size_t srcCount,
    std::size_t srcRowStride, std::size_t height, std::size_t rowSamples,
    double b, std::uint16_t* dst, std::size_t dstCount,
    std::size_t dstRowStride) noexcept {
    using Status = PostcameraBlackStatus;

    if (src == nullptr || dst == nullptr) return Status::invalid_argument;
    if (height == 0u || rowSamples == 0u) return Status::invalid_dimensions;
    if (srcRowStride < rowSamples || dstRowStride < rowSamples)
        return Status::invalid_stride;

    std::size_t active_sample_count = 0u;
    std::size_t src_last_row = 0u;
    std::size_t dst_last_row = 0u;
    std::size_t src_required = 0u;
    std::size_t dst_required = 0u;
    std::size_t row_bytes = 0u;
    if (!checked_multiply(height, rowSamples, &active_sample_count) ||
        !checked_multiply(height - 1u, srcRowStride, &src_last_row) ||
        !checked_multiply(height - 1u, dstRowStride, &dst_last_row) ||
        !checked_add(src_last_row, rowSamples, &src_required) ||
        !checked_add(dst_last_row, rowSamples, &dst_required) ||
        !checked_multiply(rowSamples, sizeof(std::uint16_t), &row_bytes)) {
        return Status::arithmetic_overflow;
    }
    const std::size_t ptrdiff_sample_limit =
        static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) /
        sizeof(std::uint16_t);
    if (src_required > ptrdiff_sample_limit ||
        dst_required > ptrdiff_sample_limit) {
        return Status::arithmetic_overflow;
    }

    AddressSpan src_span{};
    AddressSpan dst_span{};
    if (!make_declared_u16_span(src, srcCount, &src_span) ||
        !make_declared_u16_span(dst, dstCount, &dst_span)) {
        return Status::arithmetic_overflow;
    }
    if (active_sample_count > kActiveSampleLimit)
        return Status::resource_limit;
    if (srcCount < src_required || dstCount < dst_required)
        return Status::buffer_too_small;

    if (reinterpret_cast<std::uintptr_t>(src) % alignof(std::uint16_t) != 0u ||
        reinterpret_cast<std::uintptr_t>(dst) % alignof(std::uint16_t) != 0u) {
        return Status::unsupported_alignment;
    }

    const bool exact_in_place = src == dst && srcCount == dstCount &&
                                srcRowStride == dstRowStride;
    if (spans_overlap(src_span, dst_span) && !exact_in_place)
        return Status::overlap;
    if (!valid_black(b)) return Status::invalid_black;

    U16LutPrepared lut{};
    if (b != 0.0) {
        const Status build_status = build_postcamera_black_u16_lut(b, &lut);
        if (build_status != Status::ok) return build_status;
    }

    std::vector<std::uint16_t> candidate;
    try {
        candidate.resize(active_sample_count);
    } catch (const std::bad_alloc&) {
        return Status::allocation_failure;
    } catch (const std::length_error&) {
        return Status::length_error;
    } catch (...) {
        return Status::allocation_failure;
    }

    for (std::size_t y = 0u; y < height; ++y) {
        const std::size_t src_row = y * srcRowStride;
        const std::size_t candidate_row = y * rowSamples;
        if (b == 0.0) {
            std::memcpy(candidate.data() + candidate_row, src + src_row,
                        row_bytes);
            continue;
        }
        const FloatLutStatus apply_status = apply_u16_lut_row_0132d484(
            lut, src + src_row, srcCount - src_row,
            candidate.data() + candidate_row,
            active_sample_count - candidate_row, rowSamples);
        if (apply_status != FloatLutStatus::ok)
            return map_lut_status(apply_status);
    }

    for (std::size_t y = 0u; y < height; ++y) {
        const std::size_t dst_row = y * dstRowStride;
        const std::size_t candidate_row = y * rowSamples;
        std::memcpy(dst + dst_row, candidate.data() + candidate_row, row_bytes);
    }
    return Status::ok;
}

}  // namespace imprint::acr
