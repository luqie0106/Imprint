#include "../src/acr_resample_coefficients.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <new>

extern "C" int acr_test_build_resample_coefficients(
    double q, double *metadata, std::size_t metadata_count,
    std::size_t *bounds, std::size_t bounds_count, float *coefficients,
    std::size_t coefficient_count) noexcept {
    using imprint::acr::ResampleCoefficientStatus;
    if (metadata == nullptr || metadata_count != 4 || bounds == nullptr ||
        bounds_count != 3 || coefficients == nullptr) {
        return static_cast<int>(ResampleCoefficientStatus::invalid_input);
    }

    try {
        imprint::acr::ResampleCoefficientTable candidate;
        candidate.q = -11.25;
        candidate.scale = 12.5;
        candidate.parameter_a = -13.75;
        candidate.parameter_b = 14.125;
        candidate.tap_count = 103;
        candidate.stride = 104;
        candidate.coefficients.assign(8, 123.25f);
        const auto status = imprint::acr::build_resample_coefficients(q, &candidate);
        if (status != ResampleCoefficientStatus::ok) {
            const bool unchanged = candidate.q == -11.25 &&
                candidate.scale == 12.5 && candidate.parameter_a == -13.75 &&
                candidate.parameter_b == 14.125 && candidate.tap_count == 103 &&
                candidate.stride == 104 && candidate.coefficients.size() == 8 &&
                std::all_of(candidate.coefficients.begin(),
                            candidate.coefficients.end(),
                            [](float value) { return value == 123.25f; });
            if (!unchanged) {
                return static_cast<int>(ResampleCoefficientStatus::invalid_coefficients);
            }
            return static_cast<int>(status);
        }
        if (candidate.coefficients.size() != coefficient_count) {
            return static_cast<int>(ResampleCoefficientStatus::invalid_input);
        }

        const double candidate_metadata[4] = {
            candidate.q, candidate.scale, candidate.parameter_a,
            candidate.parameter_b,
        };
        const std::size_t candidate_bounds[3] = {
            candidate.tap_count / 2, candidate.tap_count, candidate.stride,
        };
        std::copy(candidate_metadata, candidate_metadata + 4, metadata);
        std::copy(candidate_bounds, candidate_bounds + 3, bounds);
        std::copy(candidate.coefficients.begin(), candidate.coefficients.end(),
                  coefficients);
        return static_cast<int>(ResampleCoefficientStatus::ok);
    } catch (const std::bad_alloc &) {
        return static_cast<int>(ResampleCoefficientStatus::allocation_failure);
    } catch (...) {
        return static_cast<int>(ResampleCoefficientStatus::invalid_coefficients);
    }
}

extern "C" int acr_test_evaluate_resample_weight(
    double parameter_a, double parameter_b, double x, double *output) noexcept {
    if (output == nullptr || !std::isfinite(parameter_a) ||
        !std::isfinite(parameter_b) || !std::isfinite(x)) {
        return static_cast<int>(imprint::acr::ResampleCoefficientStatus::invalid_input);
    }
    const double candidate = imprint::acr::resample_detail::evaluate_resample_weight(
        parameter_a, parameter_b, x);
    if (!std::isfinite(candidate)) {
        return static_cast<int>(
            imprint::acr::ResampleCoefficientStatus::invalid_coefficients);
    }
    *output = candidate;
    return static_cast<int>(imprint::acr::ResampleCoefficientStatus::ok);
}
