#include "../src/acr_stage3_curve.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>

extern "C" int acr_test_stage3_curve_prepare(double pivot, int cubic,
                                               double* out3);
extern "C" int acr_test_stage3_curve_eval(double* prepared3, double sample,
                                            int inverse, double* out);
extern "C" int acr_test_stage3_curve_lut(double* prepared3, int inverse,
                                           float* out, std::size_t capacity);

namespace {

using imprint::acr::FloatLutPrepared;
using imprint::acr::Stage3CurvePrepared;
using imprint::acr::Stage3CurveStatus;

int failures = 0;

void check(bool condition, const char* label) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", label);
    ++failures;
}

std::uint32_t bits(float value) {
    std::uint32_t result = 0;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

bool same_bits(float left, float right) { return bits(left) == bits(right); }

bool same_value(double left, double right) {
    if (left != right) return false;
    return left != 0.0 || std::signbit(left) == std::signbit(right);
}

void validate_curve(double pivot, bool cubic) {
    double wire[3] = {-91.0, -92.0, -93.0};
    int status = acr_test_stage3_curve_prepare(pivot, cubic ? 1 : 0, wire);
    check(status == static_cast<int>(Stage3CurveStatus::ok),
          "bridge prepares supported pivot and mode");
    check(same_value(wire[0], pivot), "bridge preserves pivot");
    check(same_value(wire[1], pivot > 0.0 ? 0.25 : +0.0),
          "bridge derives log black from pivot");
    check(same_value(wire[2], cubic ? 1.0 : +0.0),
          "bridge preserves explicit cubic mode");

    Stage3CurvePrepared prepared{};
    auto core_status = imprint::acr::prepare_stage3_curve(
        pivot, cubic, &prepared);
    check(core_status == Stage3CurveStatus::ok,
          "core prepares supported pivot and mode");

    for (bool inverse : {false, true}) {
        for (double endpoint : {0.0, 1.0}) {
            double scalar = -101.0;
            status = acr_test_stage3_curve_eval(
                wire, endpoint, inverse ? 1 : 0, &scalar);
            check(status == static_cast<int>(Stage3CurveStatus::ok),
                  "bridge evaluates both scalar endpoints");
            check(std::isfinite(scalar) && scalar >= 0.0 && scalar <= 1.0,
                  "scalar endpoint output is finite and clamped");

            double direct = -102.0;
            core_status = imprint::acr::evaluate_stage3_curve(
                endpoint, prepared, inverse, &direct);
            check(core_status == Stage3CurveStatus::ok &&
                      same_value(scalar, direct),
                  "bridge and native scalar endpoint agree");
        }

        std::array<float, imprint::acr::kFloatLutEntryCount> bridge_lut{};
        status = acr_test_stage3_curve_lut(
            wire, inverse ? 1 : 0, bridge_lut.data(), bridge_lut.size());
        check(status == static_cast<int>(Stage3CurveStatus::ok),
              "bridge builds 4096 segment LUT");
        check(same_bits(bridge_lut[imprint::acr::kFloatLutSegments],
                        bridge_lut[imprint::acr::kFloatLutSegments + 1]),
              "LUT duplicates the terminal guard entry");

        FloatLutPrepared native_lut{};
        core_status = imprint::acr::build_stage3_curve_lut(
            prepared, inverse, &native_lut);
        check(core_status == Stage3CurveStatus::ok,
              "native curve builder uses shared float LUT builder");
        check(same_bits(bridge_lut[0], native_lut.values[0]) &&
                  same_bits(bridge_lut[imprint::acr::kFloatLutSegments],
                            native_lut.values[imprint::acr::kFloatLutSegments]),
              "bridge LUT endpoint values match native output");

        for (std::size_t i : {std::size_t{0}, std::size_t{1024},
                              std::size_t{2048}, std::size_t{3072},
                              imprint::acr::kFloatLutSegments}) {
            const double sample = static_cast<double>(i) * (1.0 / 4096.0);
            double scalar = 0.0;
            core_status = imprint::acr::evaluate_stage3_curve(
                sample, prepared, inverse, &scalar);
            check(core_status == Stage3CurveStatus::ok,
                  "native scalar succeeds at LUT control points");
            if (core_status == Stage3CurveStatus::ok) {
                check(same_bits(static_cast<float>(scalar),
                                bridge_lut[i]),
                      "LUT samples equal binary32 scalar results");
            }
        }
    }
}

void test_invalid_and_atomic() {
    constexpr double kSentinel = 8842.125;
    double prepared3[3] = {0.0, 0.0, 0.0};
    int status = acr_test_stage3_curve_prepare(0.0, 0, prepared3);
    check(status == static_cast<int>(Stage3CurveStatus::ok),
          "prepare baseline state for invalid tests");

    const double invalid_pivots[] = {
        std::numeric_limits<double>::quiet_NaN(),
        -0.01,
        1.0,
        std::numeric_limits<double>::infinity(),
    };
    for (double pivot : invalid_pivots) {
        double out[3] = {kSentinel, kSentinel, kSentinel};
        status = acr_test_stage3_curve_prepare(pivot, 1, out);
        check(status == static_cast<int>(Stage3CurveStatus::invalid_parameters),
              "prepare rejects invalid pivot");
        check(out[0] == kSentinel && out[1] == kSentinel && out[2] == kSentinel,
              "invalid pivot leaves bridge output untouched");
    }

    double out3[3] = {kSentinel, kSentinel, kSentinel};
    status = acr_test_stage3_curve_prepare(0.5, 2, out3);
    check(status == static_cast<int>(Stage3CurveStatus::invalid_argument),
          "prepare rejects unsupported cubic flag");
    check(out3[0] == kSentinel && out3[1] == kSentinel && out3[2] == kSentinel,
          "invalid cubic flag leaves output untouched");

    double result = kSentinel;
    status = acr_test_stage3_curve_eval(prepared3,
        std::numeric_limits<double>::quiet_NaN(), 0, &result);
    check(status == static_cast<int>(Stage3CurveStatus::invalid_parameters) &&
              result == kSentinel,
          "non-finite scalar sample fails atomically");
    status = acr_test_stage3_curve_eval(prepared3, 0.5, -1, &result);
    check(status == static_cast<int>(Stage3CurveStatus::invalid_argument) &&
              result == kSentinel,
          "eval rejects unsupported inverse flag atomically");

    double malformed[3] = {0.5, 0.0, 1.0};
    status = acr_test_stage3_curve_eval(malformed, 0.5, 0, &result);
    check(status == static_cast<int>(Stage3CurveStatus::invalid_parameters) &&
              result == kSentinel,
          "eval rejects constructor-mismatched prepared state atomically");

    status = acr_test_stage3_curve_eval(prepared3, 0.5, 0, &prepared3[0]);
    check(status == static_cast<int>(Stage3CurveStatus::overlap) &&
              prepared3[0] == 0.0 && prepared3[1] == 0.0 &&
              prepared3[2] == 0.0,
          "bridge rejects output overlap with prepared state atomically");

    Stage3CurvePrepared native{};
    auto core_status = imprint::acr::prepare_stage3_curve(0.0, false, &native);
    check(core_status == Stage3CurveStatus::ok, "prepare core alias test state");
    const double original_pivot = native.pivot;
    core_status = imprint::acr::evaluate_stage3_curve(
        0.5, native, false, &native.pivot);
    check(core_status == Stage3CurveStatus::overlap &&
              same_value(native.pivot, original_pivot),
          "core rejects scalar output overlap atomically");

    std::array<float, imprint::acr::kFloatLutEntryCount> lut{};
    lut.fill(-71.25f);
    status = acr_test_stage3_curve_lut(prepared3, 0, lut.data(), lut.size() - 1);
    check(status == static_cast<int>(Stage3CurveStatus::invalid_argument),
          "LUT bridge rejects insufficient capacity");
    bool lut_unchanged = true;
    for (float value : lut) lut_unchanged &= value == -71.25f;
    check(lut_unchanged, "short LUT capacity leaves output untouched");

    status = acr_test_stage3_curve_lut(prepared3, 0,
                                        reinterpret_cast<float*>(prepared3),
                                        lut.size());
    check(status == static_cast<int>(Stage3CurveStatus::overlap),
          "LUT bridge rejects output overlap with prepared state");

    FloatLutPrepared native_lut{};
    for (float& value : native_lut.values) value = -37.5f;
    Stage3CurvePrepared invalid_native{0.5, -0.0, true};
    core_status = imprint::acr::build_stage3_curve_lut(
        invalid_native, false, &native_lut);
    check(core_status == Stage3CurveStatus::invalid_parameters,
          "native LUT builder rejects prepared state not made by constructor");
    bool native_lut_unchanged = true;
    for (float value : native_lut.values) native_lut_unchanged &= value == -37.5f;
    check(native_lut_unchanged,
          "invalid native prepared state leaves LUT output untouched");
}

void test_alignment_and_wrapped_spans() {
    constexpr unsigned char kMarker = 0xA7;
    alignas(double) unsigned char double_storage[3 * sizeof(double) + 8];
    for (unsigned char& value : double_storage) value = kMarker;
    int status = acr_test_stage3_curve_prepare(
        0.25, 0, reinterpret_cast<double*>(double_storage + 1));
    check(status == static_cast<int>(Stage3CurveStatus::invalid_argument),
          "prepare rejects misaligned output span");
    bool unchanged = true;
    for (unsigned char value : double_storage) unchanged &= value == kMarker;
    check(unchanged, "misaligned prepare output remains untouched");

    const auto max_address = std::numeric_limits<std::uintptr_t>::max();
    const auto wrapped_double =
        (max_address - static_cast<std::uintptr_t>(7)) &
        ~static_cast<std::uintptr_t>(7);
    status = acr_test_stage3_curve_prepare(
        0.25, 0, reinterpret_cast<double*>(wrapped_double));
    check(status == static_cast<int>(Stage3CurveStatus::invalid_argument),
          "prepare rejects wrapped output span");

    double prepared3[3] = {0.25, 0.25, 0.0};
    double result = 667.25;
    status = acr_test_stage3_curve_eval(
        reinterpret_cast<double*>(double_storage + 1), 0.5, 0, &result);
    check(status == static_cast<int>(Stage3CurveStatus::invalid_argument) &&
              result == 667.25,
          "eval rejects misaligned prepared span atomically");
    status = acr_test_stage3_curve_eval(
        prepared3, 0.5, 0, reinterpret_cast<double*>(double_storage + 1));
    check(status == static_cast<int>(Stage3CurveStatus::invalid_argument),
          "eval rejects misaligned output span");
    unchanged = true;
    for (unsigned char value : double_storage) unchanged &= value == kMarker;
    check(unchanged, "misaligned eval output remains untouched");

    status = acr_test_stage3_curve_eval(
        reinterpret_cast<double*>(wrapped_double), 0.5, 0, &result);
    check(status == static_cast<int>(Stage3CurveStatus::invalid_argument) &&
              result == 667.25,
          "eval rejects wrapped prepared span atomically");
    status = acr_test_stage3_curve_eval(
        prepared3, 0.5, 0, reinterpret_cast<double*>(wrapped_double));
    check(status == static_cast<int>(Stage3CurveStatus::invalid_argument),
          "eval rejects wrapped output span");

    alignas(float) unsigned char float_storage[
        imprint::acr::kFloatLutEntryCount * sizeof(float) + 8];
    for (unsigned char& value : float_storage) value = kMarker;
    status = acr_test_stage3_curve_lut(
        prepared3, 0, reinterpret_cast<float*>(float_storage + 1),
        imprint::acr::kFloatLutEntryCount);
    check(status == static_cast<int>(Stage3CurveStatus::invalid_argument),
          "LUT bridge rejects misaligned output span");
    unchanged = true;
    for (unsigned char value : float_storage) unchanged &= value == kMarker;
    check(unchanged, "misaligned LUT output remains untouched");

    const auto wrapped_float =
        (max_address - static_cast<std::uintptr_t>(3)) &
        ~static_cast<std::uintptr_t>(3);
    status = acr_test_stage3_curve_lut(
        prepared3, 0, reinterpret_cast<float*>(wrapped_float),
        imprint::acr::kFloatLutEntryCount);
    check(status == static_cast<int>(Stage3CurveStatus::invalid_argument),
          "LUT bridge rejects wrapped output span");
}

}  // namespace

int main() {
    validate_curve(0.0, false);
    validate_curve(0.0, true);
    validate_curve(1.0 / 4100.0, false);
    validate_curve(1.0 / 4100.0, true);
    validate_curve(std::nextafter(1.0, 0.0), false);
    validate_curve(std::nextafter(1.0, 0.0), true);
    test_invalid_and_atomic();
    test_alignment_and_wrapped_spans();
    if (failures == 0) {
        std::puts("Stage3 curve validation passed");
        return 0;
    }
    std::fprintf(stderr, "%d Stage3 curve validation checks failed\n", failures);
    return 1;
}
