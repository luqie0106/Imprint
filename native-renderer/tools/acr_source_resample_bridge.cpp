#include "acr_source_resample.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace {

struct SourceMode2Args {
    const float *source;
    std::size_t source_count;
    std::int32_t source_top;
    std::int32_t source_left;
    std::size_t source_height;
    std::size_t source_width;
    std::size_t channels;
    std::int32_t destination_top;
    std::int32_t destination_left;
    std::size_t destination_height;
    std::size_t destination_width;
    const double *matrix;
    const std::int32_t *plan_bounds;
    float *output;
    std::size_t output_count;
};

bool valid_output(const SourceMode2Args &args) noexcept {
    if (args.destination_width != 0 &&
        args.destination_height >
            std::numeric_limits<std::size_t>::max() / args.destination_width)
        return false;
    const std::size_t pixels = args.destination_height * args.destination_width;
    if (args.channels != 0 &&
        pixels > std::numeric_limits<std::size_t>::max() / args.channels)
        return false;
    const std::size_t expected = pixels * args.channels;
    if (expected != args.output_count ||
        expected > imprint::acr::kResampleSeparableMaximumElements ||
        expected > std::numeric_limits<std::size_t>::max() / sizeof(float))
        return false;
    if (expected == 0) return true;
    if (!args.output) return false;
    const auto address = reinterpret_cast<std::uintptr_t>(args.output);
    const std::size_t bytes = expected * sizeof(float);
    return address % alignof(float) == 0 &&
           bytes <= std::numeric_limits<std::uintptr_t>::max() - address;
}

std::vector<unsigned char> snapshot(const std::vector<float> &values) {
    std::vector<unsigned char> bytes(values.size() * sizeof(float));
    if (!bytes.empty()) std::memcpy(bytes.data(), values.data(), bytes.size());
    return bytes;
}

bool unchanged(const std::vector<float> &values,
               const std::vector<unsigned char> &before) noexcept {
    const std::size_t bytes = values.size() * sizeof(float);
    return bytes == before.size() &&
           (bytes == 0 || std::memcmp(values.data(), before.data(), bytes) == 0);
}

int run_source_mode2(const SourceMode2Args *args, bool alias_result) noexcept {
    using namespace imprint::acr;
    if (!args || !valid_output(*args) || !args->matrix || !args->plan_bounds)
        return static_cast<int>(ResampleSeparableStatus::invalid_input);
    if (alias_result &&
        (args->source_count > kResampleSeparableMaximumElements ||
         args->source_count > std::numeric_limits<std::size_t>::max() /
                                  sizeof(float) ||
         (args->source_count != 0 && !args->source)))
        return static_cast<int>(ResampleSeparableStatus::invalid_input);

    try {
        const ResampleMatrix3 matrix{{
            args->matrix[0], args->matrix[1], args->matrix[2],
            args->matrix[3], args->matrix[4], args->matrix[5],
            args->matrix[6], args->matrix[7], args->matrix[8]}};
        const std::array<std::int32_t, 4> bounds{{
            args->plan_bounds[0], args->plan_bounds[1],
            args->plan_bounds[2], args->plan_bounds[3]}};
        OrdinaryResamplePlan plan;
        const auto plan_status = prepare_ordinary_resample_plan(
            matrix, bounds, plan);
        if (plan_status != ResamplePlanStatus::ok)
            return 100 + static_cast<int>(plan_status);

        std::vector<float> result;
        if (alias_result) {
            result.assign(args->source, args->source + args->source_count);
        } else if (args->output_count != 0) {
            result.assign(args->output, args->output + args->output_count);
        }
        const auto before = snapshot(result);
        const auto before_size = result.size();
        const auto before_capacity = result.capacity();
        const auto *const before_data = result.data();
        const SourceView source{
            alias_result ? result.data() : args->source,
            args->source_count, args->source_top, args->source_left,
            args->source_height, args->source_width, args->channels};
        const DestinationRect destination{
            args->destination_top, args->destination_left,
            args->destination_height, args->destination_width};

        const auto status = process_ordinary_source_mode2_f32(
            source, destination, plan, result);
        if (static_cast<int>(status) != 0) {
            if (result.size() != before_size ||
                result.capacity() != before_capacity ||
                result.data() != before_data || !unchanged(result, before))
                return 99;
            return static_cast<int>(status);
        }
        if (result.size() != args->output_count) return 99;
        if (!result.empty()) std::copy(result.begin(), result.end(), args->output);
        return 0;
    } catch (...) {
        return static_cast<int>(ResampleSeparableStatus::allocation_failure);
    }
}

int run_physical_window_baseline(const SourceMode2Args *args) noexcept {
    using namespace imprint::acr;
    if (!args || !valid_output(*args) || !args->matrix ||
        !args->plan_bounds)
        return static_cast<int>(ResampleSeparableStatus::invalid_input);
    try {
        const ResampleMatrix3 matrix{{
            args->matrix[0], args->matrix[1], args->matrix[2],
            args->matrix[3], args->matrix[4], args->matrix[5],
            args->matrix[6], args->matrix[7], args->matrix[8]}};
        const std::array<std::int32_t, 4> bounds{{
            args->plan_bounds[0], args->plan_bounds[1],
            args->plan_bounds[2], args->plan_bounds[3]}};
        OrdinaryResamplePlan plan;
        const auto plan_status = prepare_ordinary_resample_plan(
            matrix, bounds, plan);
        if (plan_status != ResamplePlanStatus::ok)
            return 100 + static_cast<int>(plan_status);

        std::vector<float> result;
        if (args->output_count != 0)
            result.assign(args->output, args->output + args->output_count);
        const auto before = snapshot(result);
        const auto before_size = result.size();
        const auto before_capacity = result.capacity();
        const auto *const before_data = result.data();
        const SourceView source{
            args->source, args->source_count, args->source_top,
            args->source_left, args->source_height, args->source_width,
            args->channels};
        const DestinationRect destination{
            args->destination_top, args->destination_left,
            args->destination_height, args->destination_width};
        const auto status = process_ordinary_resample(
            source, destination, plan, result);
        if (status != ResampleSeparableStatus::ok) {
            if (result.size() != before_size ||
                result.capacity() != before_capacity ||
                result.data() != before_data || !unchanged(result, before))
                return 99;
            return static_cast<int>(status);
        }
        if (result.size() != args->output_count) return 99;
        if (!result.empty()) std::copy(result.begin(), result.end(), args->output);
        return 0;
    } catch (...) {
        return static_cast<int>(ResampleSeparableStatus::allocation_failure);
    }
}

}  // namespace

extern "C" int acr_test_ordinary_source_mode2(
    const SourceMode2Args *args) noexcept {
    return run_source_mode2(args, false);
}

extern "C" int acr_test_ordinary_source_mode2_alias(
    const SourceMode2Args *args) noexcept {
    return run_source_mode2(args, true);
}

extern "C" int acr_test_ordinary_source_physical_window_baseline(
    const SourceMode2Args *args) noexcept {
    return run_physical_window_baseline(args);
}
