#include "../src/acr_postcamera_black.hpp"
#include "../src/acr_u16_lut.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace {

using imprint::acr::FloatLutPrepared;
using imprint::acr::PostcameraBlackStatus;
using imprint::acr::U16LutPrepared;

bool valid_float_output_span(float* out, std::size_t count) noexcept {
    if (out == nullptr) return false;
    const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(out);
    if (begin % alignof(float) != 0u) return false;
    if (count > static_cast<std::size_t>(
                    std::numeric_limits<std::ptrdiff_t>::max()) /
                    sizeof(float)) {
        return false;
    }
    const std::size_t byte_count = count * sizeof(float);
    return byte_count <= std::numeric_limits<std::uintptr_t>::max() - begin;
}

bool valid_u16_output_span(std::uint16_t* out, std::size_t count) noexcept {
    if (out == nullptr) return false;
    const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(out);
    if (begin % alignof(std::uint16_t) != 0u) return false;
    if (count > static_cast<std::size_t>(
                    std::numeric_limits<std::ptrdiff_t>::max()) /
                    sizeof(std::uint16_t)) {
        return false;
    }
    const std::size_t byte_count = count * sizeof(std::uint16_t);
    return byte_count <= std::numeric_limits<std::uintptr_t>::max() - begin;
}

}  // namespace

extern "C" {

int acr_test_postcamera_black(
    const float* src, std::size_t srcCount, std::size_t srcRowStride,
    std::size_t height, std::size_t rowSamples, double b, float* dst,
    std::size_t dstCount, std::size_t dstRowStride) noexcept {
    return static_cast<int>(
        imprint::acr::apply_postcamera_black_prepared_f32(
            src, srcCount, srcRowStride, height, rowSamples, b, dst, dstCount,
            dstRowStride));
}

int acr_test_postcamera_black_lut(double b, float* out,
                                  std::size_t count) noexcept {
    if (out == nullptr) return static_cast<int>(PostcameraBlackStatus::invalid_argument);
    if (count != imprint::acr::kFloatLutEntryCount) {
        return static_cast<int>(PostcameraBlackStatus::invalid_argument);
    }
    if (reinterpret_cast<std::uintptr_t>(out) % alignof(float) != 0u) {
        return static_cast<int>(PostcameraBlackStatus::unsupported_alignment);
    }
    if (!valid_float_output_span(out, count)) {
        return static_cast<int>(PostcameraBlackStatus::arithmetic_overflow);
    }

    FloatLutPrepared candidate{};
    const PostcameraBlackStatus status =
        imprint::acr::build_postcamera_black_lut(b, &candidate);
    if (status != PostcameraBlackStatus::ok) return static_cast<int>(status);
    std::memcpy(out, candidate.values.data(),
                imprint::acr::kFloatLutEntryCount * sizeof(float));
    return static_cast<int>(PostcameraBlackStatus::ok);
}

int acr_test_postcamera_black_u16(
    const std::uint16_t* src, std::size_t srcCount,
    std::size_t srcRowStride, std::size_t height, std::size_t rowSamples,
    double b, std::uint16_t* dst, std::size_t dstCount,
    std::size_t dstRowStride) noexcept {
    return static_cast<int>(
        imprint::acr::apply_postcamera_black_prepared_u16(
            src, srcCount, srcRowStride, height, rowSamples, b, dst, dstCount,
            dstRowStride));
}

int acr_test_postcamera_black_u16_lut(double b, std::uint16_t* out,
                                      std::size_t count) noexcept {
    if (out == nullptr)
        return static_cast<int>(PostcameraBlackStatus::invalid_argument);
    if (count != imprint::acr::kU16LutEntryCount) {
        return static_cast<int>(PostcameraBlackStatus::invalid_argument);
    }
    if (reinterpret_cast<std::uintptr_t>(out) % alignof(std::uint16_t) != 0u) {
        return static_cast<int>(PostcameraBlackStatus::unsupported_alignment);
    }
    if (!valid_u16_output_span(out, count)) {
        return static_cast<int>(PostcameraBlackStatus::arithmetic_overflow);
    }

    U16LutPrepared candidate{};
    const PostcameraBlackStatus status =
        imprint::acr::build_postcamera_black_u16_lut(b, &candidate);
    if (status != PostcameraBlackStatus::ok) return static_cast<int>(status);
    std::memcpy(out, candidate.values.data(),
                imprint::acr::kU16LutEntryCount * sizeof(std::uint16_t));
    return static_cast<int>(PostcameraBlackStatus::ok);
}

}  // extern "C"
