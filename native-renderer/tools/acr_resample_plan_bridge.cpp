#include "acr_resample_plan.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

namespace {

using imprint::acr::OrdinaryResamplePlan;
using imprint::acr::ResampleCoefficientTable;
using imprint::acr::ResamplePlanStatus;
using imprint::acr::ResampleSeparableStatus;

constexpr std::size_t kMetadataWords = 26;
constexpr int kMutationDetected = 99;
constexpr int kBridgeInvalidRequest = 98;
constexpr int kExecutionStatusBase = 100;

struct OrdinaryPlanArgs {
    const double *matrix;
    const std::int32_t *bounds;
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
    float *output;
    std::size_t output_count;
    std::uint64_t *metadata;
    std::size_t metadata_count;
    std::uint32_t *vertical_coefficient_bits;
    std::size_t vertical_coefficient_capacity;
    std::uint32_t *horizontal_coefficient_bits;
    std::size_t horizontal_coefficient_capacity;
};

std::uint64_t bits_of_double(double value) noexcept {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

std::uint64_t bits_of_i64(std::int64_t value) noexcept {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

std::uint32_t bits_of_float(float value) noexcept {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

void append_axis(std::vector<std::uint64_t> &words,
                 const imprint::acr::ResampleAxisPlan &axis) {
    words.push_back(static_cast<std::uint64_t>(axis.kind));
    words.push_back(bits_of_i64(axis.integer_offset));
    words.push_back(bits_of_i64(axis.origin));
    words.push_back(bits_of_i64(axis.step));
    words.push_back(bits_of_i64(axis.lower));
    words.push_back(bits_of_i64(axis.upper));
}

void append_table_metadata(std::vector<std::uint64_t> &words,
                           const ResampleCoefficientTable &table) {
    words.push_back(bits_of_double(table.q));
    words.push_back(bits_of_double(table.scale));
    words.push_back(bits_of_double(table.parameter_a));
    words.push_back(bits_of_double(table.parameter_b));
    words.push_back(static_cast<std::uint64_t>(table.tap_count));
    words.push_back(static_cast<std::uint64_t>(table.stride));
    words.push_back(static_cast<std::uint64_t>(table.coefficients.size()));
}

std::vector<std::uint64_t> plan_signature(const OrdinaryResamplePlan &plan) {
    std::vector<std::uint64_t> words;
    words.reserve(kMetadataWords + plan.vertical_coefficients.coefficients.size()
                  + plan.horizontal_coefficients.coefficients.size());
    append_axis(words, plan.vertical);
    append_axis(words, plan.horizontal);
    append_table_metadata(words, plan.vertical_coefficients);
    append_table_metadata(words, plan.horizontal_coefficients);
    for (const float value : plan.vertical_coefficients.coefficients)
        words.push_back(bits_of_float(value));
    for (const float value : plan.horizontal_coefficients.coefficients)
        words.push_back(bits_of_float(value));
    return words;
}

OrdinaryResamplePlan sentinel_plan() {
    OrdinaryResamplePlan plan{};
    plan.vertical = {imprint::acr::ResampleAxisKind::integer_translation,
                     0x123456789LL, -0x23456789LL, 0x3456789LL,
                     -0x456789LL, -17};
    plan.horizontal = {imprint::acr::ResampleAxisKind::resample,
                       -0x56789LL, 0x6789LL, -0x789LL, 0x89LL, 23};
    plan.vertical_coefficients.q = -3.25;
    plan.vertical_coefficients.scale = 4.5;
    plan.vertical_coefficients.parameter_a = -5.75;
    plan.vertical_coefficients.parameter_b = 6.125;
    plan.vertical_coefficients.tap_count = 2;
    plan.vertical_coefficients.stride = 8;
    plan.vertical_coefficients.coefficients = {1.25f, -0.0f, -2.5f};
    plan.horizontal_coefficients.q = 7.25;
    plan.horizontal_coefficients.scale = -8.5;
    plan.horizontal_coefficients.parameter_a = 9.75;
    plan.horizontal_coefficients.parameter_b = -10.125;
    plan.horizontal_coefficients.tap_count = 4;
    plan.horizontal_coefficients.stride = 8;
    plan.horizontal_coefficients.coefficients = {-11.5f, 12.25f};
    return plan;
}

bool checked_byte_count(std::size_t count, std::size_t element_size,
                        std::size_t &bytes) noexcept {
    if (element_size != 0 &&
        count > std::numeric_limits<std::size_t>::max() / element_size)
        return false;
    bytes = count * element_size;
    return true;
}

std::vector<unsigned char> byte_snapshot(const void *data,
                                         std::size_t bytes) {
    std::vector<unsigned char> snapshot(bytes);
    if (bytes != 0) std::memcpy(snapshot.data(), data, bytes);
    return snapshot;
}

bool matches_bytes(const void *data, std::size_t bytes,
                   const std::vector<unsigned char> &snapshot) noexcept {
    return bytes == snapshot.size() &&
           (bytes == 0 || std::memcmp(data, snapshot.data(), bytes) == 0);
}

bool same_float_vector(const std::vector<float> &left,
                       const std::vector<float> &right) noexcept {
    if (left.size() != right.size()) return false;
    std::size_t bytes = 0;
    if (!checked_byte_count(left.size(), sizeof(float), bytes)) return false;
    return bytes == 0 || std::memcmp(left.data(), right.data(), bytes) == 0;
}

bool valid_args(const OrdinaryPlanArgs &args,
                std::size_t &source_bytes,
                std::size_t &output_bytes,
                std::size_t &metadata_bytes,
                std::size_t &vertical_bits_bytes,
                std::size_t &horizontal_bits_bytes) noexcept {
    if (!args.matrix || !args.bounds || args.metadata_count != kMetadataWords)
        return false;
    if ((args.source_count != 0 && !args.source) ||
        (args.output_count != 0 && !args.output) ||
        (args.metadata_count != 0 && !args.metadata) ||
        (args.vertical_coefficient_capacity != 0 &&
         !args.vertical_coefficient_bits) ||
        (args.horizontal_coefficient_capacity != 0 &&
         !args.horizontal_coefficient_bits))
        return false;
    if (!checked_byte_count(args.source_count, sizeof(float), source_bytes) ||
        !checked_byte_count(args.output_count, sizeof(float), output_bytes) ||
        !checked_byte_count(args.metadata_count, sizeof(std::uint64_t), metadata_bytes) ||
        !checked_byte_count(args.vertical_coefficient_capacity,
                            sizeof(std::uint32_t), vertical_bits_bytes) ||
        !checked_byte_count(args.horizontal_coefficient_capacity,
                            sizeof(std::uint32_t), horizontal_bits_bytes))
        return false;
    return true;
}

void write_report(const OrdinaryResamplePlan &plan,
                  const std::vector<std::uint64_t> &signature,
                  const OrdinaryPlanArgs &args) noexcept {
    std::copy_n(signature.begin(), kMetadataWords, args.metadata);
    std::size_t bit_index = kMetadataWords;
    for (std::size_t index = 0;
         index < plan.vertical_coefficients.coefficients.size(); ++index)
        args.vertical_coefficient_bits[index] =
            static_cast<std::uint32_t>(signature[bit_index++]);
    for (std::size_t index = 0;
         index < plan.horizontal_coefficients.coefficients.size(); ++index)
        args.horizontal_coefficient_bits[index] =
            static_cast<std::uint32_t>(signature[bit_index++]);
}

bool inputs_unchanged(const OrdinaryPlanArgs &args,
                      const std::vector<unsigned char> &matrix_before,
                      const std::vector<unsigned char> &bounds_before,
                      const std::vector<unsigned char> &source_before,
                      std::size_t source_bytes) noexcept {
    return matches_bytes(args.matrix, 9 * sizeof(double), matrix_before) &&
           matches_bytes(args.bounds, 4 * sizeof(std::int32_t), bounds_before) &&
           matches_bytes(args.source, source_bytes, source_before);
}

int run_plan(const OrdinaryPlanArgs *args) noexcept {
    if (!args) return kBridgeInvalidRequest;
    std::size_t source_bytes = 0;
    std::size_t output_bytes = 0;
    std::size_t metadata_bytes = 0;
    std::size_t vertical_bits_bytes = 0;
    std::size_t horizontal_bits_bytes = 0;
    if (!valid_args(*args, source_bytes, output_bytes, metadata_bytes,
                    vertical_bits_bytes, horizontal_bits_bytes))
        return kBridgeInvalidRequest;

    try {
        const auto matrix_before = byte_snapshot(args->matrix, 9 * sizeof(double));
        const auto bounds_before = byte_snapshot(args->bounds,
                                                 4 * sizeof(std::int32_t));
        const auto source_before = byte_snapshot(args->source, source_bytes);
        const auto output_before = byte_snapshot(args->output, output_bytes);
        const auto metadata_before = byte_snapshot(args->metadata, metadata_bytes);
        const auto vertical_bits_before = byte_snapshot(
            args->vertical_coefficient_bits, vertical_bits_bytes);
        const auto horizontal_bits_before = byte_snapshot(
            args->horizontal_coefficient_bits, horizontal_bits_bytes);

        OrdinaryResamplePlan plan = sentinel_plan();
        const auto original_plan = plan_signature(plan);
        const auto initial_result = args->output_count == 0
            ? std::vector<float>{}
            : std::vector<float>(args->output,
                                 args->output + args->output_count);
        auto result = initial_result;

        const imprint::acr::ResampleMatrix3 matrix = {
            args->matrix[0], args->matrix[1], args->matrix[2],
            args->matrix[3], args->matrix[4], args->matrix[5],
            args->matrix[6], args->matrix[7], args->matrix[8]};
        const std::array<std::int32_t, 4> bounds = {
            args->bounds[0], args->bounds[1], args->bounds[2], args->bounds[3]};
        const auto plan_status =
            imprint::acr::prepare_ordinary_resample_plan(matrix, bounds, plan);

        auto external_unchanged = [&]() noexcept {
            return inputs_unchanged(*args, matrix_before, bounds_before,
                                    source_before, source_bytes) &&
                   matches_bytes(args->output, output_bytes, output_before) &&
                   matches_bytes(args->metadata, metadata_bytes, metadata_before) &&
                   matches_bytes(args->vertical_coefficient_bits,
                                 vertical_bits_bytes, vertical_bits_before) &&
                   matches_bytes(args->horizontal_coefficient_bits,
                                 horizontal_bits_bytes, horizontal_bits_before);
        };

        if (plan_status != ResamplePlanStatus::ok) {
            if (plan_signature(plan) != original_plan ||
                !same_float_vector(result, initial_result) ||
                !external_unchanged())
                return kMutationDetected;
            return static_cast<int>(plan_status);
        }

        const auto prepared_plan = plan_signature(plan);
        const imprint::acr::SourceView source{
            args->source, args->source_count, args->source_top,
            args->source_left, args->source_height, args->source_width,
            args->channels};
        const imprint::acr::DestinationRect destination{
            args->destination_top, args->destination_left,
            args->destination_height, args->destination_width};
        const auto execute_status = imprint::acr::process_ordinary_resample(
            source, destination, plan, result);
        if (execute_status != ResampleSeparableStatus::ok) {
            if (plan_signature(plan) != prepared_plan ||
                !same_float_vector(result, initial_result) ||
                !external_unchanged())
                return kMutationDetected;
            return kExecutionStatusBase + static_cast<int>(execute_status);
        }
        if (result.size() != args->output_count ||
            !inputs_unchanged(*args, matrix_before, bounds_before,
                              source_before, source_bytes) ||
            plan_signature(plan) != prepared_plan)
            return kMutationDetected;

        // Execute independent copied and moved plans as a lifetime/self-pointer
        // check. Every result must match the original plan bit for bit.
        const auto expected_result = result;
        OrdinaryResamplePlan copied = plan;
        auto copied_result = initial_result;
        const auto copied_before = plan_signature(copied);
        const auto copied_status = imprint::acr::process_ordinary_resample(
            source, destination, copied, copied_result);
        if (copied_status != ResampleSeparableStatus::ok ||
            !same_float_vector(copied_result, expected_result) ||
            plan_signature(plan) != prepared_plan ||
            plan_signature(copied) != copied_before)
            return kMutationDetected;

        OrdinaryResamplePlan moved = std::move(copied);
        auto moved_result = initial_result;
        const auto moved_before = plan_signature(moved);
        const auto moved_status = imprint::acr::process_ordinary_resample(
            source, destination, moved, moved_result);
        if (moved_status != ResampleSeparableStatus::ok ||
            !same_float_vector(moved_result, expected_result) ||
            plan_signature(moved) != moved_before ||
            plan_signature(plan) != prepared_plan ||
            !inputs_unchanged(*args, matrix_before, bounds_before,
                              source_before, source_bytes) ||
            !external_unchanged())
            return kMutationDetected;

        if (plan.vertical_coefficients.coefficients.size() >
                args->vertical_coefficient_capacity ||
            plan.horizontal_coefficients.coefficients.size() >
                args->horizontal_coefficient_capacity)
            return kBridgeInvalidRequest;

        const auto completed_signature = plan_signature(plan);
        if (completed_signature != prepared_plan)
            return kMutationDetected;
        if (!result.empty())
            std::copy(result.begin(), result.end(), args->output);
        write_report(plan, completed_signature, *args);
        return static_cast<int>(ResamplePlanStatus::ok);
    } catch (...) {
        return kBridgeInvalidRequest;
    }
}

}  // namespace

extern "C" int acr_test_ordinary_resample_plan(
    const void *opaque_args) noexcept {
    return run_plan(static_cast<const OrdinaryPlanArgs *>(opaque_args));
}

extern "C" int acr_test_ordinary_resample_plan_alias(
    float *output, std::size_t output_count) noexcept {
    using namespace imprint::acr;
    if (!output || output_count != 4) return kBridgeInvalidRequest;
    try {
        std::vector<float> pixels = {
            0.0f, 1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f, 7.0f,
            8.0f, 9.0f, 10.0f, 11.0f};
        const auto original = pixels;
        OrdinaryResamplePlan plan{};
        const ResampleMatrix3 identity = {1.0, 0.0, 0.0,
                                          0.0, 1.0, 0.0,
                                          0.0, 0.0, 1.0};
        const std::array<std::int32_t, 4> bounds = {-1, 4, 1, 6};
        if (prepare_ordinary_resample_plan(identity, bounds, plan) !=
            ResamplePlanStatus::ok)
            return kMutationDetected;
        const SourceView source{pixels.data(), pixels.size(), -2, 3, 3, 4, 1};
        const DestinationRect destination{-1, 4, 2, 2};
        const auto status = process_ordinary_resample(
            source, destination, plan, pixels);
        if (status != ResampleSeparableStatus::ok || pixels.size() != 4)
            return kMutationDetected;
        const float expected[] = {5.0f, 6.0f, 9.0f, 10.0f};
        if (std::memcmp(pixels.data(), expected, sizeof(expected)) != 0)
            return kMutationDetected;
        std::copy(pixels.begin(), pixels.end(), output);
        (void)original;
        return static_cast<int>(status);
    } catch (...) {
        return kBridgeInvalidRequest;
    }
}

extern "C" int acr_test_ordinary_resample_plan_alias_failure(
    float *output, std::size_t output_count) noexcept {
    using namespace imprint::acr;
    if (!output || output_count != 12) return kBridgeInvalidRequest;
    try {
        std::vector<float> pixels = {
            0.0f, 1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f, 7.0f,
            8.0f, 9.0f, 10.0f, 11.0f};
        const auto original = pixels;
        OrdinaryResamplePlan plan{};
        const ResampleMatrix3 identity = {1.0, 0.0, 0.0,
                                          0.0, 1.0, 0.0,
                                          0.0, 0.0, 1.0};
        const std::array<std::int32_t, 4> bounds = {-2, 3, -1, 4};
        if (prepare_ordinary_resample_plan(identity, bounds, plan) !=
            ResamplePlanStatus::ok)
            return kMutationDetected;
        const SourceView source{pixels.data(), pixels.size(), -2, 3, 3, 4, 1};
        const DestinationRect destination{-3, 3, 1, 2};
        const auto status = process_ordinary_resample(
            source, destination, plan, pixels);
        if (status == ResampleSeparableStatus::ok ||
            !same_float_vector(pixels, original))
            return kMutationDetected;
        std::copy(pixels.begin(), pixels.end(), output);
        return static_cast<int>(status);
    } catch (...) {
        return kBridgeInvalidRequest;
    }
}
