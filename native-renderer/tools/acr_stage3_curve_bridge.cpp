#include "../src/acr_stage3_curve.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace {

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

bool same_value(double left, double right) noexcept {
    if (left != right) return false;
    return left != 0.0 || std::signbit(left) == std::signbit(right);
}

int status(imprint::acr::Stage3CurveStatus value) noexcept {
    return static_cast<int>(value);
}

imprint::acr::Stage3CurveStatus load_prepared(
    double* values, imprint::acr::Stage3CurvePrepared* prepared,
    AddressRange* values_range) noexcept {
    if (prepared == nullptr || values_range == nullptr ||
        !address_range(values, 3 * sizeof(double), alignof(double), values_range)) {
        return imprint::acr::Stage3CurveStatus::invalid_argument;
    }
    if (!std::isfinite(values[0]) || !std::isfinite(values[1]) ||
        !std::isfinite(values[2]) ||
        !(same_value(values[2], +0.0) || same_value(values[2], 1.0))) {
        return imprint::acr::Stage3CurveStatus::invalid_parameters;
    }
    const bool cubic = values[2] == 1.0;
    const auto result = imprint::acr::prepare_stage3_curve(
        values[0], cubic, prepared);
    if (result != imprint::acr::Stage3CurveStatus::ok) return result;
    if (!same_value(prepared->log_black, values[1])) {
        return imprint::acr::Stage3CurveStatus::invalid_parameters;
    }
    return imprint::acr::Stage3CurveStatus::ok;
}

}  // namespace

extern "C" {

int acr_test_stage3_curve_prepare(double pivot, int cubic, double* out3) {
    if (cubic != 0 && cubic != 1) {
        return status(imprint::acr::Stage3CurveStatus::invalid_argument);
    }
    AddressRange output_range{};
    if (!address_range(out3, 3 * sizeof(double), alignof(double),
                       &output_range)) {
        return status(imprint::acr::Stage3CurveStatus::invalid_argument);
    }
    imprint::acr::Stage3CurvePrepared prepared{};
    const auto result = imprint::acr::prepare_stage3_curve(
        pivot, cubic != 0, &prepared);
    if (result != imprint::acr::Stage3CurveStatus::ok) return status(result);
    const double values[3] = {
        prepared.pivot,
        prepared.log_black,
        prepared.cubic ? 1.0 : +0.0,
    };
    for (std::size_t i = 0; i < 3; ++i) out3[i] = values[i];
    return status(imprint::acr::Stage3CurveStatus::ok);
}

int acr_test_stage3_curve_eval(double* prepared3, double sample, int inverse,
                               double* out) {
    if (inverse != 0 && inverse != 1) {
        return status(imprint::acr::Stage3CurveStatus::invalid_argument);
    }
    AddressRange prepared_range{};
    AddressRange output_range{};
    if (!address_range(prepared3, 3 * sizeof(double), alignof(double),
                       &prepared_range) ||
        !address_range(out, sizeof(double), alignof(double), &output_range)) {
        return status(imprint::acr::Stage3CurveStatus::invalid_argument);
    }
    if (overlaps(prepared_range, output_range)) {
        return status(imprint::acr::Stage3CurveStatus::overlap);
    }
    imprint::acr::Stage3CurvePrepared prepared{};
    AddressRange ignored{};
    const auto loaded = load_prepared(prepared3, &prepared, &ignored);
    if (loaded != imprint::acr::Stage3CurveStatus::ok) return status(loaded);
    return status(imprint::acr::evaluate_stage3_curve(
        sample, prepared, inverse != 0, out));
}

int acr_test_stage3_curve_lut(double* prepared3, int inverse, float* out,
                              std::size_t capacity) {
    if (inverse != 0 && inverse != 1) {
        return status(imprint::acr::Stage3CurveStatus::invalid_argument);
    }
    if (capacity < imprint::acr::kFloatLutEntryCount) {
        return status(imprint::acr::Stage3CurveStatus::invalid_argument);
    }
    constexpr std::size_t kOutputBytes =
        imprint::acr::kFloatLutEntryCount * sizeof(float);
    AddressRange prepared_range{};
    AddressRange output_range{};
    if (!address_range(prepared3, 3 * sizeof(double), alignof(double),
                       &prepared_range) ||
        !address_range(out, kOutputBytes, alignof(float), &output_range)) {
        return status(imprint::acr::Stage3CurveStatus::invalid_argument);
    }
    if (overlaps(prepared_range, output_range)) {
        return status(imprint::acr::Stage3CurveStatus::overlap);
    }
    imprint::acr::Stage3CurvePrepared prepared{};
    AddressRange ignored{};
    const auto loaded = load_prepared(prepared3, &prepared, &ignored);
    if (loaded != imprint::acr::Stage3CurveStatus::ok) return status(loaded);

    imprint::acr::FloatLutPrepared candidate{};
    const auto result = imprint::acr::build_stage3_curve_lut(
        prepared, inverse != 0, &candidate);
    if (result != imprint::acr::Stage3CurveStatus::ok) return status(result);
    for (std::size_t i = 0; i < imprint::acr::kFloatLutEntryCount; ++i) {
        out[i] = candidate.values[i];
    }
    return status(imprint::acr::Stage3CurveStatus::ok);
}

}  // extern "C"
