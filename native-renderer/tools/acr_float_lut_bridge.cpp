#include "acr_float_lut.hpp"
#include "acr_noise_transform.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace {

using imprint::acr::FloatLutPrepared;
using imprint::acr::FloatLutStatus;
using imprint::acr::NoiseTransformPrepared;

struct AddressRange {
    std::uintptr_t begin;
    std::uintptr_t end;
};

bool address_range(const void* pointer, std::size_t size,
                   std::size_t alignment, AddressRange* out) noexcept {
    if (pointer == nullptr || out == nullptr || alignment == 0) return false;
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (begin % alignment != 0) return false;
    constexpr auto kMax = std::numeric_limits<std::uintptr_t>::max();
    if (size > static_cast<std::size_t>(kMax - begin)) return false;
    out->begin = begin;
    out->end = begin + static_cast<std::uintptr_t>(size);
    return true;
}

bool overlaps(const AddressRange& left, const AddressRange& right) noexcept {
    return left.begin < right.end && right.begin < left.end;
}

NoiseTransformPrepared load_noise_prepared(const double* values) noexcept {
    return NoiseTransformPrepared{
        values[0], values[1], values[2], values[3], values[4],
        values[5], values[6], values[7], values[8], values[9]};
}

bool identity(double x, const void*, double* out) noexcept {
    *out = x;
    return true;
}

bool polynomial(double x, const void*, double* out) noexcept {
    *out = std::fma(0.37, x * x, 0.2);
    return true;
}

bool noise_forward(double x, const void* context, double* out) noexcept {
    const auto* prepared = static_cast<const NoiseTransformPrepared*>(context);
    return imprint::acr::apply_noise_transform_forward_00b4bb88(
               x, *prepared, out) == imprint::acr::NoiseTransformStatus::ok;
}

bool noise_inverse(double x, const void* context, double* out) noexcept {
    const auto* prepared = static_cast<const NoiseTransformPrepared*>(context);
    return imprint::acr::apply_noise_transform_inverse_00b4bc38(
               x, *prepared, out) == imprint::acr::NoiseTransformStatus::ok;
}

}  // namespace

extern "C" {

int acr_test_lut_build(int kind, double* params, std::size_t params_count,
                       float* out, std::size_t capacity) {
    if (kind < 0 || kind > 3 || capacity < imprint::acr::kFloatLutEntryCount) {
        return static_cast<int>(FloatLutStatus::invalid_argument);
    }

    AddressRange output_range{};
    if (!address_range(out,
                       imprint::acr::kFloatLutEntryCount * sizeof(float),
                       alignof(float), &output_range)) {
        return static_cast<int>(FloatLutStatus::invalid_argument);
    }

    NoiseTransformPrepared noise{};
    AddressRange parameter_range{};
    if (kind == 2 || kind == 3) {
        if (params_count != 10 ||
            !address_range(params, 10 * sizeof(double), alignof(double),
                           &parameter_range)) {
            return static_cast<int>(FloatLutStatus::invalid_argument);
        }
        if (overlaps(parameter_range, output_range)) {
            return static_cast<int>(FloatLutStatus::overlap);
        }
        noise = load_noise_prepared(params);
    } else if (params_count != 0) {
        return static_cast<int>(FloatLutStatus::invalid_argument);
    }

    imprint::acr::FloatLutEvaluator evaluator = identity;
    const void* context = nullptr;
    if (kind == 1) {
        evaluator = polynomial;
    } else if (kind == 2) {
        evaluator = noise_forward;
        context = &noise;
    } else if (kind == 3) {
        evaluator = noise_inverse;
        context = &noise;
    }

    FloatLutPrepared prepared{};
    const FloatLutStatus status = imprint::acr::build_float_lut_01300490(
        evaluator, context, &prepared);
    if (status != FloatLutStatus::ok) return static_cast<int>(status);

    std::memcpy(out, prepared.values.data(),
                imprint::acr::kFloatLutEntryCount * sizeof(float));
    return static_cast<int>(FloatLutStatus::ok);
}

int acr_test_lut_apply(float* table, std::size_t table_count,
                       float* source, std::size_t source_count,
                       float* destination, std::size_t destination_count,
                       std::size_t width, int signed_extension) {
    if (signed_extension != 0 && signed_extension != 1) {
        return static_cast<int>(FloatLutStatus::invalid_argument);
    }
    if (table_count < imprint::acr::kFloatLutEntryCount || width == 0 ||
        source_count < width || destination_count < width ||
        width > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
        width > static_cast<std::size_t>(
                    std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(float)) {
        return static_cast<int>(FloatLutStatus::invalid_argument);
    }

    const std::size_t table_bytes =
        imprint::acr::kFloatLutEntryCount * sizeof(float);
    const std::size_t processed_bytes = width * sizeof(float);
    AddressRange table_range{};
    AddressRange source_range{};
    AddressRange destination_range{};
    if (!address_range(table, table_bytes, alignof(float), &table_range) ||
        !address_range(source, processed_bytes, alignof(float), &source_range) ||
        !address_range(destination, processed_bytes, alignof(float),
                       &destination_range)) {
        return static_cast<int>(FloatLutStatus::invalid_argument);
    }
    if (overlaps(table_range, destination_range)) {
        return static_cast<int>(FloatLutStatus::overlap);
    }

    FloatLutPrepared prepared{};
    std::memcpy(prepared.values.data(), table, table_bytes);
    return static_cast<int>(imprint::acr::apply_float_lut_row_00875edc(
        prepared, source, source_count, destination, destination_count, width,
        signed_extension != 0));
}

}  // extern "C"
