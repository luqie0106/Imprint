#include "acr_resample_row.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <new>

namespace imprint::acr {
namespace {
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
              "the recovered sampler requires IEEE-754 binary32");
bool valid_span(const float *data, std::size_t count) noexcept {
    if (!data || count > std::numeric_limits<std::size_t>::max() / sizeof(float))
        return false;
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    return address % alignof(float) == 0 &&
           count * sizeof(float) <=
               std::numeric_limits<std::uintptr_t>::max() - address;
}

std::int32_t integer_part(std::int64_t coordinate) noexcept {
    const auto high = static_cast<std::uint64_t>(coordinate) >> 32;
    const std::int64_t signed_high = high >= 0x80000000ULL
        ? static_cast<std::int64_t>(high) - 0x100000000LL
        : static_cast<std::int64_t>(high);
    return static_cast<std::int32_t>(signed_high);
}

unsigned phase_part(std::int64_t coordinate) noexcept {
    return static_cast<unsigned>((static_cast<std::uint64_t>(coordinate) >> 25) & 127);
}

bool advance(std::int64_t value, std::int64_t step,
             std::int64_t &next) noexcept {
    if ((step > 0 && value > std::numeric_limits<std::int64_t>::max() - step) ||
        (step < 0 && value < std::numeric_limits<std::int64_t>::min() - step))
        return false;
    next = value + step;
    return true;
}

float generic_dot(const float *source, const float *weights,
                  std::size_t taps) noexcept {
    float sum = weights[0] * source[0];
    const std::size_t block_end = (taps - 1) & ~std::size_t{15};
    for (std::size_t k = 1; k <= block_end; ++k) {
        const float product = weights[k] * source[k];
        sum = sum + product;
    }
    for (std::size_t k = block_end + 1; k < taps; ++k)
        sum = std::fma(weights[k], source[k], sum);
    return sum;
}

float scalar_dot(const float *source, const float *weights,
                 std::size_t taps) noexcept {
    if (taps >= 4 && taps <= 24 && taps % 2 == 0) {
        const float second = weights[1] * source[1];
        float sum = std::fma(weights[0], source[0], second);
        for (std::size_t k = 2; k < taps; ++k)
            sum = std::fma(weights[k], source[k], sum);
        return sum;
    }
    return generic_dot(source, weights, taps);
}

float matrix_horizontal_baseline_dot(const float *source, const float *weights,
                                    std::size_t taps) noexcept {
    if (taps == 4 || taps == 6 || taps == 8) {
        const float second = weights[1] * source[1];
        float sum = std::fma(weights[0], source[0], second);
        for (std::size_t k = 2; k < taps; ++k)
            sum = std::fma(weights[k], source[k], sum);
        return sum;
    }
    return generic_dot(source, weights, taps);
}

float optimized_dot(const float *source, const float *weights,
                    std::size_t taps) noexcept {
    float lanes[4];
    for (std::size_t k = 0; k < 4; ++k) {
        const float sample = k < taps ? source[k] : 0.0f;
        lanes[k] = weights[k] * sample;
        if (taps > 4) {
            const float upper = k + 4 < taps ? source[k + 4] : 0.0f;
            lanes[k] = std::fma(weights[k + 4], upper, lanes[k]);
        }
    }
    const float low_pair = lanes[0] + lanes[1];
    const float high_pair = lanes[2] + lanes[3];
    return low_pair + high_pair;
}
}  // namespace

ResampleRowStatus process_resample_row(
    const float *source, std::size_t source_count,
    const PreparedResampleRows &rows, const RowRequest &request,
    std::vector<float> &result) noexcept {
    if (!source_count || source_count > kResampleRowMaximumElements ||
        !valid_span(source, source_count))
        return ResampleRowStatus::invalid_input;
    if (!rows.tap_count || rows.tap_count > kResampleRowMaximumTaps ||
        rows.stride != ((rows.tap_count + 7) / 8) * 8 ||
        rows.count != kResampleRowPhaseCount * rows.stride ||
        !valid_span(rows.data, rows.count))
        return ResampleRowStatus::invalid_rows;
    if (!request.output_count || request.output_count > kResampleRowMaximumElements ||
        (request.mode != ResampleRowMode::scalar &&
         request.mode != ResampleRowMode::optimized &&
         request.mode != ResampleRowMode::matrix_horizontal_baseline))
        return ResampleRowStatus::invalid_request;
    const auto lower = integer_part(request.lower_bound);
    const auto upper = integer_part(request.upper_bound);
    const unsigned upper_phase = phase_part(request.upper_bound);
    if (lower > upper) return ResampleRowStatus::invalid_request;
    for (std::size_t k = 0; k < source_count; ++k)
        if (!std::isfinite(source[k])) return ResampleRowStatus::invalid_input;
    for (std::size_t k = 0; k < rows.count; ++k)
        if (!std::isfinite(rows.data[k])) return ResampleRowStatus::invalid_rows;

    try {
        std::vector<float> candidate(request.output_count);
        auto coordinate = request.coordinate_origin;
        for (std::size_t x = 0; x < request.output_count; ++x) {
            auto index = integer_part(coordinate);
            auto phase = phase_part(coordinate);
            if (index > upper || (index == upper && phase > upper_phase)) {
                index = upper;
                phase = upper_phase;
            }
            if (index < lower) {
                index = lower;
                phase = 0;
            }
            const std::int64_t relative = static_cast<std::int64_t>(index) -
                                          request.source_first_index;
            if (relative < 0 || static_cast<std::uint64_t>(relative) > source_count ||
                rows.tap_count > source_count - static_cast<std::size_t>(relative))
                return ResampleRowStatus::source_window_out_of_range;
            const float *window = source + static_cast<std::size_t>(relative);
            const float *weights = rows.data + phase * rows.stride;
            if (request.mode == ResampleRowMode::optimized && rows.tap_count <= 8) {
                candidate[x] = optimized_dot(window, weights, rows.tap_count);
            } else if (request.mode == ResampleRowMode::matrix_horizontal_baseline) {
                candidate[x] = matrix_horizontal_baseline_dot(
                    window, weights, rows.tap_count);
            } else {
                candidate[x] = scalar_dot(window, weights, rows.tap_count);
            }
            if (!std::isfinite(candidate[x])) return ResampleRowStatus::non_finite_output;
            if (x + 1 < request.output_count &&
                !advance(coordinate, request.coordinate_step, coordinate))
                return ResampleRowStatus::coordinate_overflow;
        }
        result.swap(candidate);
        return ResampleRowStatus::ok;
    } catch (const std::bad_alloc &) {
        return ResampleRowStatus::allocation_failure;
    } catch (...) {
        return ResampleRowStatus::allocation_failure;
    }
}
}  // namespace imprint::acr
