#include "acr_noise_transform.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>

extern "C" {
int acr_test_noise_prepare(double slope, double offset, double input_black,
                           double output_black, double* out10);
int acr_test_noise_apply(double* prepared10, double sample, int inverse,
                         double* out);
}

namespace {

using imprint::acr::NoiseTransformPrepared;
using imprint::acr::NoiseTransformStatus;

int failures = 0;

void check(bool condition, const char* label) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", label);
    ++failures;
}

bool close(double left, double right, double tolerance = 1e-12) {
    return std::isfinite(left) && std::isfinite(right) &&
           std::abs(left - right) <= tolerance;
}

}  // namespace

int main() {
    constexpr double kSlope = 5.88172456e-05;
    constexpr double kOffset = 3.241287029e-08;
    constexpr double kBlack = 0.06256;
    constexpr double kSentinel = 12345.5;

    const double black_cases[2][2] = {{0.0, kBlack}, {kBlack, 0.0}};
    for (const auto& black_case : black_cases) {
        NoiseTransformPrepared prepared{};
        auto status = imprint::acr::prepare_noise_transform_00b31b40(
            kSlope, kOffset, black_case[0], black_case[1], &prepared);
        check(status == NoiseTransformStatus::ok, "prepare fixed parameters");
        check(std::isfinite(prepared.baseline) && prepared.scale > 0.0 &&
                  std::isfinite(prepared.scale),
              "prepared finite baseline and positive scale");

        const double forward_samples[] = {
            -0.25, -0.0, 0.0, black_case[0], 0.5, 1.0, 1.25};
        for (double sample : forward_samples) {
            double result = kSentinel;
            status = imprint::acr::apply_noise_transform_forward_00b4bb88(
                sample, prepared, &result);
            check(status == NoiseTransformStatus::ok,
                  "forward accepts finite sample");
            check(std::isfinite(result) && result >= 0.0 && result <= 1.0,
                  "forward result is clamped to unit range");
        }
        double black_mapped = kSentinel;
        status = imprint::acr::apply_noise_transform_forward_00b4bb88(
            black_case[0], prepared, &black_mapped);
        check(status == NoiseTransformStatus::ok &&
                  close(black_mapped, black_case[1]),
              "forward maps input black to output black");

        const double inverse_samples[] = {
            -0.25, -0.0, 0.0, black_case[1], 0.5, 1.0, 1.25};
        for (double sample : inverse_samples) {
            double result = kSentinel;
            status = imprint::acr::apply_noise_transform_inverse_00b4bc38(
                sample, prepared, &result);
            check(status == NoiseTransformStatus::ok,
                  "inverse accepts finite sample");
            check(std::isfinite(result) && result >= 0.0 && result <= 1.0,
                  "inverse result is clamped to unit range");
        }
        double black_unmapped = kSentinel;
        status = imprint::acr::apply_noise_transform_inverse_00b4bc38(
            black_case[1], prepared, &black_unmapped);
        check(status == NoiseTransformStatus::ok &&
                  close(black_unmapped, black_case[0]),
              "inverse maps output black to input black");

        const double original_slope = prepared.slope;
        status = imprint::acr::apply_noise_transform_forward_00b4bb88(
            0.5, prepared, &prepared.slope);
        check(status == NoiseTransformStatus::overlap &&
                  prepared.slope == original_slope,
              "scalar apply rejects output overlap without writing");

        alignas(double) unsigned char misaligned_storage[sizeof(double) + 8] = {};
        for (unsigned char& byte : misaligned_storage) byte = 0xA5;
        status = imprint::acr::apply_noise_transform_forward_00b4bb88(
            0.5, prepared,
            reinterpret_cast<double*>(misaligned_storage + 1));
        check(status == NoiseTransformStatus::invalid_argument,
              "scalar apply rejects misaligned output");
        bool misaligned_unchanged = true;
        for (unsigned char byte : misaligned_storage) {
            misaligned_unchanged &= byte == 0xA5;
        }
        check(misaligned_unchanged, "misaligned output remains unchanged");
    }

    double prepared_out[10];
    for (double& value : prepared_out) value = kSentinel;
    int bridge_status = acr_test_noise_prepare(
        kSlope, kOffset, 0.0, kBlack, prepared_out);
    check(bridge_status == static_cast<int>(NoiseTransformStatus::ok),
          "C prepare bridge accepts fixed parameters");
    double bridge_result = kSentinel;
    bridge_status = acr_test_noise_apply(prepared_out, 0.5, 0, &bridge_result);
    check(bridge_status == static_cast<int>(NoiseTransformStatus::ok) &&
              std::isfinite(bridge_result),
          "C forward bridge applies prepared transform");
    bridge_status = acr_test_noise_apply(prepared_out, bridge_result, 1,
                                         &bridge_result);
    check(bridge_status == static_cast<int>(NoiseTransformStatus::ok) &&
              std::isfinite(bridge_result),
          "C inverse bridge applies prepared transform");

    double failure_out[10];
    for (double& value : failure_out) value = kSentinel;
    bridge_status = acr_test_noise_prepare(
        0.0, kOffset, 0.0, kBlack, failure_out);
    check(bridge_status == static_cast<int>(NoiseTransformStatus::invalid_parameters),
          "C prepare bridge rejects zero slope");
    for (double value : failure_out) {
        check(value == kSentinel, "prepare failure is atomic");
    }

    const double invalid_parameter_sets[][4] = {
        {-1.0, kOffset, 0.0, kBlack},
        {kSlope, -kOffset, 0.0, kBlack},
        {kSlope, kOffset, -0.01, kBlack},
        {kSlope, kOffset, 0.0, 1.0},
        {std::numeric_limits<double>::infinity(), kOffset, 0.0, kBlack},
    };
    for (const auto& parameters : invalid_parameter_sets) {
        for (double& value : failure_out) value = kSentinel;
        bridge_status = acr_test_noise_prepare(
            parameters[0], parameters[1], parameters[2], parameters[3],
            failure_out);
        check(bridge_status == static_cast<int>(NoiseTransformStatus::invalid_parameters),
              "C prepare bridge rejects invalid parameter set");
        bool unchanged = true;
        for (double value : failure_out) unchanged &= value == kSentinel;
        check(unchanged, "invalid parameters leave bridge output unchanged");
    }

    bridge_status = acr_test_noise_apply(prepared_out, 0.5, 0, &prepared_out[9]);
    check(bridge_status == static_cast<int>(NoiseTransformStatus::overlap),
          "C apply bridge rejects input/output overlap");

    alignas(double) unsigned char misaligned_storage[sizeof(double) * 10 + 8] = {};
    for (unsigned char& byte : misaligned_storage) byte = 0x5A;
    bridge_status = acr_test_noise_prepare(
        kSlope, kOffset, 0.0, kBlack,
        reinterpret_cast<double*>(misaligned_storage + 1));
    check(bridge_status == static_cast<int>(NoiseTransformStatus::invalid_argument),
          "C prepare bridge rejects misaligned output");
    bool storage_unchanged = true;
    for (unsigned char byte : misaligned_storage) storage_unchanged &= byte == 0x5A;
    check(storage_unchanged, "misaligned C prepare output remains unchanged");
    bridge_result = kSentinel;
    bridge_status = acr_test_noise_apply(
        reinterpret_cast<double*>(misaligned_storage + 1), 0.5, 0,
        &bridge_result);
    check(bridge_status == static_cast<int>(NoiseTransformStatus::invalid_argument),
          "C apply bridge rejects misaligned prepared array");
    check(bridge_result == kSentinel,
          "misaligned C apply leaves output unchanged");

    const auto max_address = std::numeric_limits<std::uintptr_t>::max();
    const auto near_wrap = (max_address - static_cast<std::uintptr_t>(7)) &
                           ~static_cast<std::uintptr_t>(7);
    bridge_status = acr_test_noise_prepare(
        kSlope, kOffset, 0.0, kBlack,
        reinterpret_cast<double*>(near_wrap));
    check(bridge_status == static_cast<int>(NoiseTransformStatus::invalid_argument),
          "C prepare bridge rejects wrapped output span");

    double invalid_samples[] = {
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::infinity()};
    for (double sample : invalid_samples) {
        double result = kSentinel;
        const NoiseTransformPrepared prepared{
            kSlope, kOffset, 0.0, 1.0, 0.0, kBlack,
            1.0, 1.0, 1.0 / (1.0 - kBlack), 1.0 - kBlack};
        const auto status = imprint::acr::apply_noise_transform_forward_00b4bb88(
            sample, prepared, &result);
        check(status == NoiseTransformStatus::invalid_parameters &&
                  result == kSentinel,
              "non-finite sample rejected atomically");
    }

    NoiseTransformPrepared invalid_prepared{};
    check(imprint::acr::prepare_noise_transform_00b31b40(
              kSlope, kOffset, 0.0, kBlack, &invalid_prepared) ==
              NoiseTransformStatus::ok,
          "prepare state for invalid-state check");
    invalid_prepared.scale = std::numeric_limits<double>::quiet_NaN();
    double invalid_result = kSentinel;
    const auto invalid_status =
        imprint::acr::apply_noise_transform_inverse_00b4bc38(
            0.5, invalid_prepared, &invalid_result);
    check(invalid_status == NoiseTransformStatus::invalid_parameters &&
              invalid_result == kSentinel,
          "invalid prepared state rejected atomically");

    if (failures != 0) {
        std::fprintf(stderr, "%d noise transform validation failure(s)\n", failures);
        return 1;
    }
    std::puts("ACR noise scalar transform validation passed");
    return 0;
}
