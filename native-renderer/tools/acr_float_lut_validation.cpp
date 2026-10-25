#include "acr_float_lut.hpp"
#include "acr_noise_transform.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>

extern "C" {
int acr_test_lut_build(int kind, double* params, std::size_t params_count,
                       float* out, std::size_t capacity);
int acr_test_lut_apply(float* table, std::size_t table_count,
                       float* source, std::size_t source_count,
                       float* destination, std::size_t destination_count,
                       std::size_t width, int signed_extension);
}

namespace {

using imprint::acr::FloatLutPrepared;
using imprint::acr::FloatLutStatus;
using imprint::acr::NoiseTransformPrepared;

int failures = 0;

void check(bool condition, const char* label) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", label);
    ++failures;
}

bool same_bits(float left, float right) {
    std::uint32_t left_bits = 0;
    std::uint32_t right_bits = 0;
    std::memcpy(&left_bits, &left, sizeof(left_bits));
    std::memcpy(&right_bits, &right, sizeof(right_bits));
    return left_bits == right_bits;
}

void set_sentinel(float* values, std::size_t count, float sentinel) {
    for (std::size_t i = 0; i < count; ++i) values[i] = sentinel;
}

bool all_sentinel(const float* values, std::size_t count, float sentinel) {
    for (std::size_t i = 0; i < count; ++i) {
        if (!same_bits(values[i], sentinel)) return false;
    }
    return true;
}

bool callback_false(double, const void*, double*) noexcept {
    return false;
}

bool callback_overflow_float(double, const void*, double* out) noexcept {
    *out = std::numeric_limits<double>::max();
    return true;
}

bool callback_nonfinite(double, const void*, double* out) noexcept {
    *out = std::numeric_limits<double>::infinity();
    return true;
}

bool identity_callback(double x, const void*, double* out) noexcept {
    *out = x;
    return true;
}

void store_noise_prepared(const NoiseTransformPrepared& prepared,
                          double* values) {
    values[0] = prepared.slope;
    values[1] = prepared.offset;
    values[2] = prepared.baseline;
    values[3] = prepared.scale;
    values[4] = prepared.input_black;
    values[5] = prepared.output_black;
    values[6] = prepared.input_reciprocal;
    values[7] = prepared.input_range;
    values[8] = prepared.output_reciprocal;
    values[9] = prepared.output_range;
}

}  // namespace

int main() {
    constexpr float kSentinel = -1234.5f;
    alignas(FloatLutPrepared) FloatLutPrepared identity_lut{};
    int status = acr_test_lut_build(0, nullptr, 0, identity_lut.values.data(),
                                    identity_lut.values.size());
    check(status == static_cast<int>(FloatLutStatus::ok),
          "identity LUT build succeeds");
    for (std::size_t i = 0; i <= imprint::acr::kFloatLutSegments; ++i) {
        const float expected = static_cast<float>(
            static_cast<double>(i) * (1.0 / 4096.0));
        check(same_bits(identity_lut.values[i], expected),
              "identity LUT sample matches fixed coordinate");
    }
    check(same_bits(identity_lut.values[4096], identity_lut.values[4097]),
          "identity LUT guard duplicates endpoint bits");

    FloatLutPrepared polynomial_lut{};
    status = acr_test_lut_build(1, nullptr, 0, polynomial_lut.values.data(),
                                polynomial_lut.values.size());
    check(status == static_cast<int>(FloatLutStatus::ok),
          "polynomial LUT build succeeds");
    for (std::size_t i = 0; i <= imprint::acr::kFloatLutSegments; ++i) {
        const double x = static_cast<double>(i) * (1.0 / 4096.0);
        const float expected = static_cast<float>(std::fma(0.37, x * x, 0.2));
        check(same_bits(polynomial_lut.values[i], expected),
              "polynomial LUT sample matches double FMA evaluator");
    }

    NoiseTransformPrepared noise{};
    auto noise_status = imprint::acr::prepare_noise_transform_00b31b40(
        5.88172456e-05, 3.241287029e-08, 0.0, 0.06256, &noise);
    check(noise_status == imprint::acr::NoiseTransformStatus::ok,
          "prepare fixed noise parameters");
    double noise_params[10]{};
    store_noise_prepared(noise, noise_params);
    FloatLutPrepared noise_forward_lut{};
    FloatLutPrepared noise_inverse_lut{};
    status = acr_test_lut_build(2, noise_params, 10,
                                noise_forward_lut.values.data(),
                                noise_forward_lut.values.size());
    check(status == static_cast<int>(FloatLutStatus::ok),
          "noise-forward LUT build succeeds");
    status = acr_test_lut_build(3, noise_params, 10,
                                noise_inverse_lut.values.data(),
                                noise_inverse_lut.values.size());
    check(status == static_cast<int>(FloatLutStatus::ok),
          "noise-inverse LUT build succeeds");
    for (std::size_t i = 0; i <= imprint::acr::kFloatLutSegments; ++i) {
        const double x = static_cast<double>(i) * (1.0 / 4096.0);
        double forward = 0.0;
        double inverse = 0.0;
        const auto forward_status =
            imprint::acr::apply_noise_transform_forward_00b4bb88(
                x, noise, &forward);
        const auto inverse_status =
            imprint::acr::apply_noise_transform_inverse_00b4bc38(
                x, noise, &inverse);
        check(forward_status == imprint::acr::NoiseTransformStatus::ok &&
                  same_bits(noise_forward_lut.values[i],
                            static_cast<float>(forward)),
              "noise-forward LUT sample matches scalar transform");
        check(inverse_status == imprint::acr::NoiseTransformStatus::ok &&
                  same_bits(noise_inverse_lut.values[i],
                            static_cast<float>(inverse)),
              "noise-inverse LUT sample matches scalar transform");
    }
    check(same_bits(noise_forward_lut.values[4096],
                    noise_forward_lut.values[4097]) &&
              same_bits(noise_inverse_lut.values[4096],
                        noise_inverse_lut.values[4097]),
          "noise LUT guards duplicate endpoint bits");

    FloatLutPrepared failed_build{};
    set_sentinel(failed_build.values.data(), failed_build.values.size(),
                 kSentinel);
    status = static_cast<int>(imprint::acr::build_float_lut_01300490(
        nullptr, nullptr, &failed_build));
    check(status == static_cast<int>(FloatLutStatus::invalid_argument) &&
              all_sentinel(failed_build.values.data(),
                           failed_build.values.size(), kSentinel),
          "null evaluator is rejected atomically");
    using FailureEvaluator = bool (*)(double, const void*, double*) noexcept;
    const FailureEvaluator fail_callbacks[] = {
        callback_false, callback_nonfinite, callback_overflow_float};
    const int callback_failures[] = {
        static_cast<int>(FloatLutStatus::evaluation_failure),
        static_cast<int>(FloatLutStatus::evaluation_failure),
        static_cast<int>(FloatLutStatus::evaluation_failure)};
    for (std::size_t i = 0; i < 3; ++i) {
        set_sentinel(failed_build.values.data(), failed_build.values.size(),
                     kSentinel);
        status = static_cast<int>(imprint::acr::build_float_lut_01300490(
            fail_callbacks[i], nullptr, &failed_build));
        const char* label = i == 2
            ? "DBL_MAX evaluator result is rejected before float narrowing"
            : "failed/non-finite LUT evaluation is rejected";
        check(status == callback_failures[i] &&
                  all_sentinel(failed_build.values.data(),
                               failed_build.values.size(), kSentinel),
              label);
    }

    alignas(FloatLutPrepared) FloatLutPrepared built_direct{};
    status = static_cast<int>(imprint::acr::build_float_lut_01300490(
        identity_callback, nullptr, &built_direct));
    check(status == static_cast<int>(FloatLutStatus::ok),
          "direct LUT builder succeeds");
    float samples[] = {-0.0f, 0.0f, 0.25f, 0.5f, 1.0f};
    float results[5]{};
    status = static_cast<int>(imprint::acr::apply_float_lut_row_00875edc(
        built_direct, samples, 5, results, 5, 5, false));
    check(status == static_cast<int>(FloatLutStatus::ok),
          "unsigned identity row succeeds including negative zero");
    for (std::size_t i = 0; i < 5; ++i) {
        check(results[i] == samples[i], "unsigned identity result matches input");
    }

    const float lower_index_boundary = -1.0f / 4096.0f;
    const float upper_index_boundary = 4097.0f / 4096.0f;
    const float just_inside_lower =
        std::nextafter(lower_index_boundary,
                       std::numeric_limits<float>::infinity());
    const float just_outside_lower =
        std::nextafter(lower_index_boundary,
                       -std::numeric_limits<float>::infinity());
    const float just_inside_upper =
        std::nextafter(upper_index_boundary,
                       -std::numeric_limits<float>::infinity());
    const float just_outside_upper =
        std::nextafter(upper_index_boundary,
                       std::numeric_limits<float>::infinity());
    float edge_result = kSentinel;
    status = static_cast<int>(imprint::acr::apply_float_lut_row_00875edc(
        built_direct, &just_inside_lower, 1, &edge_result, 1, 1, false));
    check(status == static_cast<int>(FloatLutStatus::ok) &&
              std::isfinite(edge_result),
          "unsigned sample immediately inside lower index threshold is accepted");
    edge_result = kSentinel;
    status = static_cast<int>(imprint::acr::apply_float_lut_row_00875edc(
        built_direct, &just_outside_lower, 1, &edge_result, 1, 1, false));
    check(status == static_cast<int>(FloatLutStatus::invalid_parameters) &&
              same_bits(edge_result, kSentinel),
          "unsigned sample immediately outside lower index threshold is rejected");
    edge_result = kSentinel;
    status = static_cast<int>(imprint::acr::apply_float_lut_row_00875edc(
        built_direct, &just_inside_upper, 1, &edge_result, 1, 1, false));
    check(status == static_cast<int>(FloatLutStatus::ok) && edge_result == 1.0f,
          "unsigned sample immediately inside upper index threshold is accepted");
    edge_result = kSentinel;
    status = static_cast<int>(imprint::acr::apply_float_lut_row_00875edc(
        built_direct, &just_outside_upper, 1, &edge_result, 1, 1, false));
    check(status == static_cast<int>(FloatLutStatus::invalid_parameters) &&
              same_bits(edge_result, kSentinel),
          "unsigned sample immediately outside upper index threshold is rejected");

    float signed_samples[] = {-0.5f, -1.25f, 0.25f, 1.25f};
    float signed_results[4]{};
    status = acr_test_lut_apply(built_direct.values.data(),
                                built_direct.values.size(),
                                signed_samples, 4, signed_results, 4, 4, 1);
    check(status == static_cast<int>(FloatLutStatus::ok) &&
              signed_results[0] == -0.5f && signed_results[1] == -1.25f &&
              signed_results[2] == 0.25f && signed_results[3] == 1.25f,
          "signed extension preserves negative and over-range values");

    float in_place[] = {0.125f, 0.5f, 0.875f, 77.0f, 88.0f};
    status = static_cast<int>(imprint::acr::apply_float_lut_row_00875edc(
        built_direct, in_place, 5, in_place, 5, 3, false));
    check(status == static_cast<int>(FloatLutStatus::ok) &&
              in_place[0] == 0.125f && in_place[1] == 0.5f &&
              in_place[2] == 0.875f && in_place[3] == 77.0f &&
              in_place[4] == 88.0f,
          "in-place row touches only processed width");

    float partial_alias[] = {0.1f, 0.2f, 0.3f, 0.4f, 91.0f};
    float partial_before[5];
    std::memcpy(partial_before, partial_alias, sizeof(partial_alias));
    status = static_cast<int>(imprint::acr::apply_float_lut_row_00875edc(
        built_direct, partial_alias, 4, partial_alias + 1, 4, 3, false));
    check(status == static_cast<int>(FloatLutStatus::overlap) &&
              std::memcmp(partial_alias, partial_before,
                          sizeof(partial_alias)) == 0,
          "partial source/destination alias is rejected atomically");

    float invalid_output[4];
    set_sentinel(invalid_output, 4, kSentinel);
    float invalid_input[] = {0.1f, std::numeric_limits<float>::quiet_NaN(), 0.3f};
    status = static_cast<int>(imprint::acr::apply_float_lut_row_00875edc(
        built_direct, invalid_input, 3, invalid_output, 4, 3, false));
    check(status == static_cast<int>(FloatLutStatus::invalid_parameters) &&
              all_sentinel(invalid_output, 4, kSentinel),
          "non-finite input rejects row without partial writes");
    invalid_input[1] = -0.1f;
    status = static_cast<int>(imprint::acr::apply_float_lut_row_00875edc(
        built_direct, invalid_input, 3, invalid_output, 4, 3, false));
    check(status == static_cast<int>(FloatLutStatus::invalid_parameters) &&
              all_sentinel(invalid_output, 4, kSentinel),
          "unsigned negative sample rejects row atomically");
    invalid_input[1] = std::numeric_limits<float>::infinity();
    status = static_cast<int>(imprint::acr::apply_float_lut_row_00875edc(
        built_direct, invalid_input, 3, invalid_output, 4, 3, true));
    check(status == static_cast<int>(FloatLutStatus::invalid_parameters) &&
              all_sentinel(invalid_output, 4, kSentinel),
          "signed non-finite input rejects row atomically");

    FloatLutPrepared bad_table = built_direct;
    bad_table.values[22] = std::numeric_limits<float>::quiet_NaN();
    const float one_sample[] = {0.5f};
    set_sentinel(invalid_output, 4, kSentinel);
    status = static_cast<int>(imprint::acr::apply_float_lut_row_00875edc(
        bad_table, one_sample, 1, invalid_output, 4, 1, false));
    check(status == static_cast<int>(FloatLutStatus::invalid_parameters) &&
              all_sentinel(invalid_output, 4, kSentinel),
          "NaN anywhere in LUT rejects row atomically");
    bad_table = built_direct;
    bad_table.values[4097] = std::nextafter(bad_table.values[4097], 0.0f);
    status = static_cast<int>(imprint::acr::apply_float_lut_row_00875edc(
        bad_table, one_sample, 1, invalid_output, 4, 1, false));
    check(status == static_cast<int>(FloatLutStatus::invalid_parameters) &&
              all_sentinel(invalid_output, 4, kSentinel),
          "different guard bits reject row atomically");

    FloatLutPrepared overflowing_table{};
    overflowing_table.values.fill(std::numeric_limits<float>::max());
    const float huge_signed_sample[] = {std::numeric_limits<float>::max()};
    status = static_cast<int>(imprint::acr::apply_float_lut_row_00875edc(
        overflowing_table, huge_signed_sample, 1, invalid_output, 4, 1, true));
    check(status == static_cast<int>(FloatLutStatus::invalid_parameters) &&
              all_sentinel(invalid_output, 4, kSentinel),
          "non-finite signed interpolation rejects row atomically");

    // The destination range overlaps the original bridge LUT even though the
    // bridge copies that LUT to a local object before invoking the core API.
    float bridge_overlap[imprint::acr::kFloatLutEntryCount];
    std::memcpy(bridge_overlap, built_direct.values.data(), sizeof(bridge_overlap));
    float bridge_overlap_before[imprint::acr::kFloatLutEntryCount];
    std::memcpy(bridge_overlap_before, bridge_overlap, sizeof(bridge_overlap));
    status = acr_test_lut_apply(bridge_overlap, imprint::acr::kFloatLutEntryCount,
                                bridge_overlap, imprint::acr::kFloatLutEntryCount,
                                bridge_overlap + 1,
                                imprint::acr::kFloatLutEntryCount - 1, 2, 0);
    check(status == static_cast<int>(FloatLutStatus::overlap) &&
              std::memcmp(bridge_overlap, bridge_overlap_before,
                          sizeof(bridge_overlap)) == 0,
          "bridge rejects destination overlap with original table");

    float bridge_invalid[4];
    set_sentinel(bridge_invalid, 4, kSentinel);
    status = acr_test_lut_apply(built_direct.values.data(),
                                built_direct.values.size(), samples, 5,
                                bridge_invalid, 4, 5, 0);
    check(status == static_cast<int>(FloatLutStatus::invalid_argument) &&
              all_sentinel(bridge_invalid, 4, kSentinel),
          "bridge rejects insufficient destination capacity atomically");
    status = acr_test_lut_apply(built_direct.values.data(),
                                built_direct.values.size(), samples, 5,
                                bridge_invalid, 4, 1, 2);
    check(status == static_cast<int>(FloatLutStatus::invalid_argument) &&
              all_sentinel(bridge_invalid, 4, kSentinel),
          "bridge rejects flags outside zero or one");
    status = acr_test_lut_build(0, nullptr, 0, identity_lut.values.data(), 12);
    check(status == static_cast<int>(FloatLutStatus::invalid_argument),
          "bridge rejects insufficient build capacity");
    status = acr_test_lut_build(2, noise_params, 9,
                                identity_lut.values.data(),
                                identity_lut.values.size());
    check(status == static_cast<int>(FloatLutStatus::invalid_argument),
          "bridge requires exactly ten noise parameters");
    status = acr_test_lut_build(4, nullptr, 0, identity_lut.values.data(),
                                identity_lut.values.size());
    check(status == static_cast<int>(FloatLutStatus::invalid_argument),
          "bridge rejects unknown LUT kind");

    const auto max_address = std::numeric_limits<std::uintptr_t>::max();
    const auto near_wrap = (max_address - static_cast<std::uintptr_t>(3)) &
                           ~static_cast<std::uintptr_t>(3);
    status = static_cast<int>(imprint::acr::apply_float_lut_row_00875edc(
        built_direct, reinterpret_cast<const float*>(near_wrap), 1,
        invalid_output, 1, 1, false));
    check(status == static_cast<int>(FloatLutStatus::invalid_argument) &&
              all_sentinel(invalid_output, 4, kSentinel),
          "row API rejects wrapping input span");
    const std::size_t overflowing_width =
        std::numeric_limits<std::size_t>::max() / sizeof(float) + 1;
    status = static_cast<int>(imprint::acr::apply_float_lut_row_00875edc(
        built_direct, one_sample, overflowing_width, invalid_output,
        overflowing_width, overflowing_width, false));
    check(status == static_cast<int>(FloatLutStatus::invalid_argument) &&
              all_sentinel(invalid_output, 4, kSentinel),
          "row API rejects byte-size overflow before access");

    if (failures != 0) {
        std::fprintf(stderr, "%d float LUT validation failure(s)\n", failures);
        return 1;
    }
    std::puts("ACR fixed float LUT validation passed");
    return 0;
}
