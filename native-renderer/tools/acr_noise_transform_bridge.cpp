#include "acr_noise_transform.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>

namespace {

struct AddressRange {
    std::uintptr_t begin;
    std::uintptr_t end;
};

bool address_range(const void* pointer, std::size_t size, std::size_t alignment,
                   AddressRange* out) noexcept {
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

void store_prepared(const imprint::acr::NoiseTransformPrepared& prepared,
                    double* out10) noexcept {
    const double values[10] = {
        prepared.slope,
        prepared.offset,
        prepared.baseline,
        prepared.scale,
        prepared.input_black,
        prepared.output_black,
        prepared.input_reciprocal,
        prepared.input_range,
        prepared.output_reciprocal,
        prepared.output_range,
    };
    for (std::size_t i = 0; i < 10; ++i) out10[i] = values[i];
}

imprint::acr::NoiseTransformPrepared load_prepared(const double* values) noexcept {
    return imprint::acr::NoiseTransformPrepared{
        values[0], values[1], values[2], values[3], values[4],
        values[5], values[6], values[7], values[8], values[9]};
}

}  // namespace

extern "C" {

int acr_test_noise_prepare(double slope, double offset, double input_black,
                           double output_black, double* out10) {
    AddressRange output_range{};
    if (!address_range(out10, 10 * sizeof(double), alignof(double),
                       &output_range)) {
        return static_cast<int>(imprint::acr::NoiseTransformStatus::invalid_argument);
    }

    imprint::acr::NoiseTransformPrepared prepared{};
    const auto status = imprint::acr::prepare_noise_transform_00b31b40(
        slope, offset, input_black, output_black, &prepared);
    if (status != imprint::acr::NoiseTransformStatus::ok) {
        return static_cast<int>(status);
    }
    store_prepared(prepared, out10);
    return static_cast<int>(imprint::acr::NoiseTransformStatus::ok);
}

int acr_test_noise_apply(double* prepared10, double sample, int inverse,
                         double* out) {
    AddressRange prepared_range{};
    AddressRange output_range{};
    if (!address_range(prepared10, 10 * sizeof(double), alignof(double),
                       &prepared_range) ||
        !address_range(out, sizeof(double), alignof(double), &output_range)) {
        return static_cast<int>(imprint::acr::NoiseTransformStatus::invalid_argument);
    }
    if (overlaps(prepared_range, output_range)) {
        return static_cast<int>(imprint::acr::NoiseTransformStatus::overlap);
    }

    const auto prepared = load_prepared(prepared10);
    const auto status = inverse != 0
        ? imprint::acr::apply_noise_transform_inverse_00b4bc38(sample, prepared, out)
        : imprint::acr::apply_noise_transform_forward_00b4bb88(sample, prepared, out);
    return static_cast<int>(status);
}

}  // extern "C"
