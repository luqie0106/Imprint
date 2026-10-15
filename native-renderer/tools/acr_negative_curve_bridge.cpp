#include "../src/acr_negative_curve.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>

namespace {

constexpr std::size_t kMaxSamples = std::size_t{1} << 26;

bool valid_range(const void *pointer, std::size_t bytes) noexcept {
    if (bytes == 0) return true;
    if (pointer == nullptr) return false;
    const auto start = reinterpret_cast<std::uintptr_t>(pointer);
    return bytes <= std::numeric_limits<std::uintptr_t>::max() - start;
}

bool overlaps(const void *left, std::size_t left_bytes,
              const void *right, std::size_t right_bytes) noexcept {
    if (left_bytes == 0 || right_bytes == 0) return false;
    const auto left_start = reinterpret_cast<std::uintptr_t>(left);
    const auto right_start = reinterpret_cast<std::uintptr_t>(right);
    if (left_start <= right_start) return right_start - left_start < left_bytes;
    return left_start - right_start < right_bytes;
}

template <typename Evaluator>
int apply_samples(const double *input, std::size_t input_count,
                 double curve_ev18, double *output,
                 std::size_t output_count, Evaluator evaluator) noexcept {
    if (input_count != output_count || input_count > kMaxSamples ||
        input_count > std::numeric_limits<std::size_t>::max() / sizeof(double)) {
        return 1;
    }
    const std::size_t bytes = input_count * sizeof(double);
    if (!valid_range(input, bytes) || !valid_range(output, bytes) ||
        (input_count != 0 && (input == nullptr || output == nullptr)) ||
        overlaps(input, bytes, output, bytes)) {
        return 1;
    }
    if (input_count == 0) return 0;

    // Validate and run a no-write pass first so any bad sample leaves the
    // caller's complete destination array untouched.
    for (std::size_t i = 0; i < input_count; ++i) {
        double ignored = 0.0;
        const auto status = evaluator(input[i], curve_ev18, ignored);
        if (status != imprint::acr::NegativeCurveStatus::ok) {
            return 10 + static_cast<int>(status);
        }
    }
    for (std::size_t i = 0; i < input_count; ++i) {
        double value = 0.0;
        const auto status = evaluator(input[i], curve_ev18, value);
        if (status != imprint::acr::NegativeCurveStatus::ok) {
            // Deterministic, allocation-free evaluation means this can only
            // occur if the floating-point environment changes between passes.
            return 10 + static_cast<int>(status);
        }
        output[i] = value;
    }
    return 0;
}

}  // namespace

extern "C" int acr_test_apply_negative_curve(
    const double *input, std::size_t input_count, double curve_ev18,
    double *output, std::size_t output_count) noexcept {
    return apply_samples(input, input_count, curve_ev18, output, output_count,
                         imprint::acr::evaluate_negative_curve_scalar);
}

extern "C" int acr_test_negative_curve_four_segment_inverse(
    const double *input, std::size_t input_count, double curve_ev18,
    double *output, std::size_t output_count) noexcept {
    return apply_samples(input, input_count, curve_ev18, output, output_count,
                         imprint::acr::evaluate_negative_curve_four_segment_inverse);
}
