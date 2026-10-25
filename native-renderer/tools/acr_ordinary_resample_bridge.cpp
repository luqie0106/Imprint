#include "acr_resample_separable.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace {

struct OrdinaryAxisArgs {
    int kind;
    std::int32_t integer_offset;
    double factory_translation;
    std::int64_t origin;
    std::int64_t step;
    std::int64_t lower;
    std::int64_t upper;
    const float *coefficients;
    std::size_t coefficient_count;
    std::size_t tap_count;
    std::size_t stride;
};

struct OrdinaryResampleArgs {
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
    OrdinaryAxisArgs vertical;
    OrdinaryAxisArgs horizontal;
    float *output;
    std::size_t output_count;
};

bool checked_destination_end(std::int32_t start, std::size_t extent,
                             std::int32_t &end) noexcept {
    if (extent > static_cast<std::size_t>(
                     std::numeric_limits<std::int32_t>::max()))
        return false;
    const auto value = static_cast<std::int64_t>(start) +
                       static_cast<std::int64_t>(extent);
    if (value > std::numeric_limits<std::int32_t>::max() ||
        value < std::numeric_limits<std::int32_t>::min())
        return false;
    end = static_cast<std::int32_t>(value);
    return true;
}

bool valid_output_span(const float *output, std::size_t count) noexcept {
    if (count > imprint::acr::kResampleSeparableMaximumElements ||
        count > std::numeric_limits<std::size_t>::max() / sizeof(float))
        return false;
    if (count == 0) return true;
    if (!output) return false;
    const auto address = reinterpret_cast<std::uintptr_t>(output);
    const auto bytes = count * sizeof(float);
    return address % alignof(float) == 0 &&
           bytes <= std::numeric_limits<std::uintptr_t>::max() - address;
}

bool expected_output_size(const OrdinaryResampleArgs &args) noexcept {
    if (args.destination_width != 0 &&
        args.destination_height >
            std::numeric_limits<std::size_t>::max() / args.destination_width)
        return false;
    const auto pixels = args.destination_height * args.destination_width;
    if (args.channels != 0 &&
        pixels > std::numeric_limits<std::size_t>::max() / args.channels)
        return false;
    return pixels * args.channels == args.output_count;
}

std::vector<unsigned char> byte_snapshot(
    const std::vector<float> &values) {
    std::vector<unsigned char> snapshot(values.size() * sizeof(float));
    if (!snapshot.empty())
        std::memcpy(snapshot.data(), values.data(), snapshot.size());
    return snapshot;
}

bool matches_snapshot(const std::vector<float> &values,
                      const std::vector<unsigned char> &snapshot) noexcept {
    const auto bytes = values.size() * sizeof(float);
    return bytes == snapshot.size() &&
           (bytes == 0 || std::memcmp(values.data(), snapshot.data(), bytes) == 0);
}

imprint::acr::ResampleSeparableStatus prepare_axis(
    const OrdinaryAxisArgs &args, bool use_factory, std::int32_t bound_start,
    std::int32_t bound_end,
    imprint::acr::PreparedOrdinaryAxis &output) noexcept {
    using namespace imprint::acr;
    ResampleAxisPlan geometry{
        static_cast<ResampleAxisKind>(args.kind), args.origin, args.step,
        args.lower, args.upper, args.integer_offset};
    if (use_factory && geometry.kind == ResampleAxisKind::integer_translation) {
        const auto status = prepare_ordinary_resample_axis(
            args.factory_translation, 1.0, 0.0, bound_start, bound_end, 0,
            geometry);
        if (status != ResampleAxisStatus::ok)
            return ResampleSeparableStatus::invalid_request;
    }
    output = {geometry, {args.coefficients, args.coefficient_count,
                         args.tap_count, args.stride}};
    return ResampleSeparableStatus::ok;
}

int run_ordinary_resample(const OrdinaryResampleArgs *args,
                          bool use_factory) noexcept {
    using namespace imprint::acr;
    if (!args) return static_cast<int>(ResampleSeparableStatus::invalid_input);
    if (!valid_output_span(args->output, args->output_count))
        return static_cast<int>(ResampleSeparableStatus::invalid_input);
    if (!expected_output_size(*args))
        return static_cast<int>(ResampleSeparableStatus::invalid_request);

    std::int32_t destination_bottom = 0;
    std::int32_t destination_right = 0;
    if (use_factory &&
        (!checked_destination_end(args->destination_top,
                                  args->destination_height,
                                  destination_bottom) ||
         !checked_destination_end(args->destination_left,
                                  args->destination_width,
                                  destination_right)))
        return static_cast<int>(ResampleSeparableStatus::invalid_request);

    PreparedOrdinaryAxis vertical{};
    PreparedOrdinaryAxis horizontal{};
    auto status = prepare_axis(args->vertical, use_factory,
                               args->destination_top, destination_bottom,
                               vertical);
    if (status != ResampleSeparableStatus::ok) return static_cast<int>(status);
    status = prepare_axis(args->horizontal, use_factory,
                          args->destination_left, destination_right,
                          horizontal);
    if (status != ResampleSeparableStatus::ok) return static_cast<int>(status);

    try {
        std::vector<float> result;
        if (args->output_count)
            result.assign(args->output, args->output + args->output_count);
        // A non-success status is valid only if the C++ vector retains every
        // original bit. Status 99 exposes any mutation through this bridge;
        // byte comparison deliberately distinguishes signed zero payloads.
        const auto initial_bytes = byte_snapshot(result);
        const SourceView source{
            args->source, args->source_count, args->source_top,
            args->source_left, args->source_height, args->source_width,
            args->channels};
        const DestinationRect destination{
            args->destination_top, args->destination_left,
            args->destination_height, args->destination_width};
        status = process_ordinary_resample(source, destination, vertical,
                                           horizontal, result);
        if (status != ResampleSeparableStatus::ok &&
            !matches_snapshot(result, initial_bytes))
            return 99;
        if (status == ResampleSeparableStatus::ok) {
            if (result.size() != args->output_count)
                return 99;
            std::copy(result.begin(), result.end(), args->output);
        }
        return static_cast<int>(status);
    } catch (...) {
        return static_cast<int>(ResampleSeparableStatus::allocation_failure);
    }
}

}  // namespace

extern "C" int acr_test_ordinary_resample(
    const OrdinaryResampleArgs *args) noexcept {
    return run_ordinary_resample(args, false);
}

extern "C" int acr_test_ordinary_resample_factory_translation(
    const OrdinaryResampleArgs *args) noexcept {
    return run_ordinary_resample(args, true);
}

extern "C" int acr_test_legacy_resample_separable(
    const OrdinaryResampleArgs *args) noexcept {
    using namespace imprint::acr;
    if (!args) return static_cast<int>(ResampleSeparableStatus::invalid_input);
    if (!valid_output_span(args->output, args->output_count))
        return static_cast<int>(ResampleSeparableStatus::invalid_input);
    if (!expected_output_size(*args))
        return static_cast<int>(ResampleSeparableStatus::invalid_request);
    try {
        std::vector<float> result;
        if (args->output_count)
            result.assign(args->output, args->output + args->output_count);
        const auto initial_bytes = byte_snapshot(result);
        const SourceView source{
            args->source, args->source_count, args->source_top,
            args->source_left, args->source_height, args->source_width,
            args->channels};
        const DestinationRect destination{
            args->destination_top, args->destination_left,
            args->destination_height, args->destination_width};
        const PreparedSeparableAxis vertical{
            args->vertical.origin, args->vertical.step, args->vertical.lower,
            args->vertical.upper,
            {args->vertical.coefficients, args->vertical.coefficient_count,
             args->vertical.tap_count, args->vertical.stride}};
        const PreparedSeparableAxis horizontal{
            args->horizontal.origin, args->horizontal.step,
            args->horizontal.lower, args->horizontal.upper,
            {args->horizontal.coefficients,
             args->horizontal.coefficient_count, args->horizontal.tap_count,
             args->horizontal.stride}};
        const auto status = process_resample_separable(
            source, destination, vertical, horizontal, result);
        if (status != ResampleSeparableStatus::ok &&
            !matches_snapshot(result, initial_bytes))
            return 99;
        if (status == ResampleSeparableStatus::ok) {
            if (result.size() != args->output_count)
                return 99;
            std::copy(result.begin(), result.end(), args->output);
        }
        return static_cast<int>(status);
    } catch (...) {
        return static_cast<int>(ResampleSeparableStatus::allocation_failure);
    }
}

extern "C" int acr_test_ordinary_resample_result_alias(
    float *output, std::size_t output_count) noexcept {
    using namespace imprint::acr;
    if (!output || output_count != 2)
        return static_cast<int>(ResampleSeparableStatus::invalid_input);
    std::vector<float> pixels{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    const SourceView source{pixels.data(), pixels.size(), -2, 3, 2, 3, 1};
    const DestinationRect destination{-1, 4, 1, 2};
    const PreparedOrdinaryAxis identity{
        {ResampleAxisKind::identity, INT64_MIN, INT64_MAX, 123, -456, 0},
        {nullptr, 0, 0, 0}};
    const auto status = process_ordinary_resample(
        source, destination, identity, identity, pixels);
    if (status == ResampleSeparableStatus::ok)
        std::copy(pixels.begin(), pixels.end(), output);
    return static_cast<int>(status);
}

extern "C" int acr_test_ordinary_resample_result_alias_failure(
    float *output, std::size_t output_count) noexcept {
    using namespace imprint::acr;
    if (!output || output_count != 6)
        return static_cast<int>(ResampleSeparableStatus::invalid_input);
    try {
        std::vector<float> pixels{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        const auto before = byte_snapshot(pixels);
        const SourceView source{pixels.data(), pixels.size(), -2, 3, 2, 3, 1};
        const DestinationRect destination{-3, 3, 1, 2};
        const PreparedOrdinaryAxis identity{
            {ResampleAxisKind::identity, 0, 0, 0, 0, 0}, {nullptr, 0, 0, 0}};
        const auto status = process_ordinary_resample(
            source, destination, identity, identity, pixels);
        std::copy(pixels.begin(), pixels.end(), output);
        if (status == ResampleSeparableStatus::ok ||
            !matches_snapshot(pixels, before))
            return 99;
        return static_cast<int>(status);
    } catch (...) {
        return static_cast<int>(ResampleSeparableStatus::allocation_failure);
    }
}

extern "C" int acr_test_ordinary_failure_mutation_guard() noexcept {
    // Self-test only: mutate a private signed-zero bit to prove the bridge
    // guard detects failed-operation writes without changing executor code.
    try {
        std::vector<float> original{-0.0f, 1.0f};
        const auto snapshot = byte_snapshot(original);
        std::vector<float> mutated = original;
        mutated[0] = 0.0f;
        return matches_snapshot(mutated, snapshot) ? 0 : 99;
    } catch (...) {
        return static_cast<int>(imprint::acr::ResampleSeparableStatus::allocation_failure);
    }
}
