#include "acr_source_log.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

namespace imprint::acr {
namespace {

constexpr std::size_t kMaxSampleCount = std::size_t{1} << 28;
constexpr float kMaxMagnitude = 0x1p120f;
constexpr double kMaxEpsilon = 0x1p-12;

bool valid_float_range(const void *pointer, std::size_t byte_count) noexcept {
    if (pointer == nullptr) return false;
    const auto start = reinterpret_cast<std::uintptr_t>(pointer);
    return start % alignof(float) == 0 &&
           byte_count <= std::numeric_limits<std::uintptr_t>::max() - start;
}

bool overlaps(const void *left, std::size_t left_size, const void *right,
              std::size_t right_size) noexcept {
    const auto left_start = reinterpret_cast<std::uintptr_t>(left);
    const auto right_start = reinterpret_cast<std::uintptr_t>(right);
    if (left_start <= right_start) return right_start - left_start < left_size;
    return left_start - right_start < right_size;
}

}  // namespace

SourceLogStatus apply_acr_source_log(const float *input, float *output,
                                    std::size_t count, double epsilon,
                                    bool pin_overrange) noexcept {
    if (count == 0 || count > kMaxSampleCount || !std::isfinite(epsilon) ||
        epsilon <= 0.0 || epsilon > kMaxEpsilon) {
        return SourceLogStatus::invalid_input;
    }

    const std::size_t byte_count = count * sizeof(float);
    if (!valid_float_range(input, byte_count) ||
        !valid_float_range(output, byte_count) ||
        overlaps(input, byte_count, output, byte_count)) {
        return SourceLogStatus::invalid_input;
    }

    const float epsilon_float = static_cast<float>(epsilon);
    if (!std::isfinite(epsilon_float) || epsilon_float <= 0.0f) {
        return SourceLogStatus::invalid_input;
    }

    for (std::size_t index = 0; index < count; ++index) {
        if (!std::isfinite(input[index])) return SourceLogStatus::invalid_input;
    }

    std::vector<float> staged;
    try {
        staged.resize(count);
    } catch (const std::bad_alloc &) {
        return SourceLogStatus::allocation_failure;
    } catch (...) {
        return SourceLogStatus::allocation_failure;
    }

    const float log_epsilon = ::log2f(epsilon_float + 0.0f);
    const float twice_log_epsilon = log_epsilon + log_epsilon;
    for (std::size_t index = 0; index < count; ++index) {
        float value = input[index];
        if (pin_overrange) {
            if (value < 0.0f) {
                value = 0.0f;
            } else if (value > 1.0f) {
                value = 1.0f;
            }
        }

        float magnitude = ::fabsf(value);
        if (magnitude > kMaxMagnitude) magnitude = kMaxMagnitude;
        const float log_value = ::log2f(epsilon_float + magnitude);
        const float result = value >= 0.0f
                                 ? log_value
                                 : twice_log_epsilon - log_value;
        if (!std::isfinite(result)) return SourceLogStatus::numeric_failure;
        staged[index] = result;
    }

    std::memcpy(output, staged.data(), byte_count);
    return SourceLogStatus::ok;
}

}  // namespace imprint::acr
