#include "../src/acr_exposure_rgb_prepare.hpp"
#include "../src/acr_exposure_rgb.hpp"
#include "../src/acr_negative_curve.hpp"
#include "../src/acr_whites.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace {

constexpr std::size_t kInputCount = 8;
constexpr std::size_t kContextFlagCount = 5;
constexpr std::size_t kLaneCount = 11;
constexpr std::size_t kRecordFlagCount = 2;
constexpr std::size_t kLutSize = imprint::acr::kPositiveExposureLutSize;
constexpr int kApplyStatusOffset = 100;

struct AddressRange {
    std::uintptr_t begin = 0;
    std::uintptr_t end = 0;
};

bool make_range(const void *pointer, std::size_t bytes, std::size_t alignment,
                AddressRange *range) noexcept {
    if (range == nullptr || alignment == 0) return false;
    if (bytes == 0) {
        if (pointer != nullptr &&
            reinterpret_cast<std::uintptr_t>(pointer) % alignment != 0) {
            return false;
        }
        *range = {};
        return true;
    }
    if (pointer == nullptr) return false;
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (begin % alignment != 0 ||
        bytes > std::numeric_limits<std::uintptr_t>::max() - begin) {
        return false;
    }
    *range = {begin, begin + bytes};
    return true;
}

bool overlaps(const AddressRange &left, const AddressRange &right) noexcept {
    return left.begin < right.end && right.begin < left.end;
}

bool disjoint(const AddressRange &candidate, const AddressRange *ranges,
              std::size_t count) noexcept {
    for (std::size_t index = 0; index < count; ++index) {
        if (overlaps(candidate, ranges[index])) return false;
    }
    return true;
}

}  // namespace

// Inputs: [a98, bA0, amount58, grayDiffB8, anchor690, WhitesEV80,
//          postToneStat670, postToneStat678]. Context flags are
// [sdr, whites_zero, mask_enabled, color_transform_enabled, stats_prepared].
// Lanes: [split_a, split_b, LUT_EV, remainder_EV, gain, rational_gain,
//         LUT_scale, epsilon, positive_gain, white_clip_squared, whites_ev].
// Record flags: [positive exposure LUT enabled, negative Whites LUT enabled].
// Every output is staged until preparation and all pixels have succeeded.
extern "C" int acr_test_prepare_apply_exposure_rgb_whites(
    const double *inputs, std::size_t input_count, float pixel_epsilon,
    const std::uint8_t *context_flags, std::size_t context_flag_count,
    const float *pixels, std::size_t pixel_value_count, double *lanes,
    std::size_t lane_count, std::uint8_t *record_flags,
    std::size_t record_flag_count, float *positive_lut_output,
    std::size_t positive_lut_capacity,
    std::size_t *positive_lut_size_output, float *negative_whites_lut_output,
    std::size_t negative_whites_lut_capacity,
    std::size_t *negative_whites_lut_size_output, float *output,
    std::size_t output_value_count) noexcept {
    if (input_count != kInputCount ||
        context_flag_count != kContextFlagCount || lane_count != kLaneCount ||
        record_flag_count != kRecordFlagCount ||
        positive_lut_capacity != kLutSize ||
        negative_whites_lut_capacity != imprint::acr::kNegativeWhitesLutSize ||
        pixel_value_count % 3 != 0 || output_value_count != pixel_value_count ||
        input_count > std::numeric_limits<std::size_t>::max() / sizeof(double) ||
        pixel_value_count > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
        lane_count > std::numeric_limits<std::size_t>::max() / sizeof(double) ||
        positive_lut_capacity > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
        negative_whites_lut_capacity > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        return static_cast<int>(imprint::acr::ExposureRgbPrepareStatus::invalid_argument);
    }

    AddressRange input_range;
    AddressRange context_range;
    AddressRange pixel_range;
    AddressRange lane_range;
    AddressRange record_flag_range;
    AddressRange positive_lut_range;
    AddressRange positive_lut_size_range;
    AddressRange negative_lut_range;
    AddressRange negative_lut_size_range;
    AddressRange output_range;
    if (!make_range(inputs, input_count * sizeof(double), alignof(double),
                    &input_range) ||
        !make_range(context_flags, context_flag_count * sizeof(std::uint8_t),
                    alignof(std::uint8_t), &context_range) ||
        !make_range(pixels, pixel_value_count * sizeof(float), alignof(float),
                    &pixel_range) ||
        !make_range(lanes, lane_count * sizeof(double), alignof(double),
                    &lane_range) ||
        !make_range(record_flags, record_flag_count * sizeof(std::uint8_t),
                    alignof(std::uint8_t), &record_flag_range) ||
        !make_range(positive_lut_output, positive_lut_capacity * sizeof(float),
                    alignof(float), &positive_lut_range) ||
        !make_range(positive_lut_size_output, sizeof(std::size_t),
                    alignof(std::size_t), &positive_lut_size_range) ||
        !make_range(negative_whites_lut_output,
                    negative_whites_lut_capacity * sizeof(float), alignof(float),
                    &negative_lut_range) ||
        !make_range(negative_whites_lut_size_output, sizeof(std::size_t),
                    alignof(std::size_t), &negative_lut_size_range) ||
        !make_range(output, output_value_count * sizeof(float), alignof(float),
                    &output_range)) {
        return static_cast<int>(imprint::acr::ExposureRgbPrepareStatus::invalid_argument);
    }

    const AddressRange write_ranges[] = {
        lane_range, record_flag_range, positive_lut_range,
        positive_lut_size_range, negative_lut_range,
        negative_lut_size_range, output_range};
    if (!disjoint(lane_range, &input_range, 1) ||
        !disjoint(lane_range, &context_range, 1) ||
        !disjoint(lane_range, &pixel_range, 1) ||
        !disjoint(record_flag_range, &input_range, 1) ||
        !disjoint(record_flag_range, &context_range, 1) ||
        !disjoint(record_flag_range, &pixel_range, 1) ||
        !disjoint(positive_lut_range, &input_range, 1) ||
        !disjoint(positive_lut_range, &context_range, 1) ||
        !disjoint(positive_lut_range, &pixel_range, 1) ||
        !disjoint(positive_lut_size_range, &input_range, 1) ||
        !disjoint(positive_lut_size_range, &context_range, 1) ||
        !disjoint(positive_lut_size_range, &pixel_range, 1) ||
        !disjoint(negative_lut_range, &input_range, 1) ||
        !disjoint(negative_lut_range, &context_range, 1) ||
        !disjoint(negative_lut_range, &pixel_range, 1) ||
        !disjoint(negative_lut_size_range, &input_range, 1) ||
        !disjoint(negative_lut_size_range, &context_range, 1) ||
        !disjoint(negative_lut_size_range, &pixel_range, 1) ||
        !disjoint(output_range, &input_range, 1) ||
        !disjoint(output_range, &context_range, 1) ||
        (overlaps(output_range, pixel_range) && output != pixels)) {
        return static_cast<int>(imprint::acr::ExposureRgbPrepareStatus::invalid_argument);
    }
    for (std::size_t left = 0; left < sizeof(write_ranges) / sizeof(write_ranges[0]);
         ++left) {
        for (std::size_t right = left + 1;
             right < sizeof(write_ranges) / sizeof(write_ranges[0]); ++right) {
            if (overlaps(write_ranges[left], write_ranges[right])) {
                return static_cast<int>(
                    imprint::acr::ExposureRgbPrepareStatus::invalid_argument);
            }
        }
    }

    for (std::size_t index = 0; index < context_flag_count; ++index) {
        if (context_flags[index] > 1) {
            return static_cast<int>(
                imprint::acr::ExposureRgbPrepareStatus::invalid_argument);
        }
    }

    imprint::acr::ExposureRgbPrepareInputs prepare_inputs;
    prepare_inputs.a_ev = inputs[0];
    prepare_inputs.b_ev = inputs[1];
    prepare_inputs.amount58 = inputs[2];
    prepare_inputs.gray_diff_b8 = inputs[3];
    prepare_inputs.anchor690 = inputs[4];
    prepare_inputs.whites_ev = inputs[5];
    prepare_inputs.post_tone_stat670 = inputs[6];
    prepare_inputs.post_tone_stat678 = inputs[7];
    prepare_inputs.pixel_epsilon = pixel_epsilon;
    prepare_inputs.sdr_mode = context_flags[0] != 0;
    prepare_inputs.whites_zero = context_flags[1] != 0;
    prepare_inputs.mask_enabled = context_flags[2] != 0;
    prepare_inputs.color_transform_enabled = context_flags[3] != 0;
    prepare_inputs.post_tone_stats_prepared = context_flags[4] != 0;

    imprint::acr::PreparedExposureRgbPlan plan;
    const auto prepare_status =
        imprint::acr::prepare_exposure_rgb(prepare_inputs, &plan);
    if (prepare_status != imprint::acr::ExposureRgbPrepareStatus::ok) {
        return static_cast<int>(prepare_status);
    }

    std::vector<float> staged_output;
    try {
        staged_output.resize(pixel_value_count);
    } catch (...) {
        return kApplyStatusOffset +
               static_cast<int>(imprint::acr::ExposureRgbStatus::allocation_failure);
    }
    const auto apply_status = plan.apply(pixels, staged_output.data(),
                                         pixel_value_count / 3);
    if (apply_status != imprint::acr::ExposureRgbStatus::ok) {
        return kApplyStatusOffset + static_cast<int>(apply_status);
    }

    const std::array<double, kLaneCount> staged_lanes = {
        plan.split_a_ev,
        plan.split_b_ev,
        plan.positive_lut_ev,
        plan.rational_remainder_ev,
        static_cast<double>(plan.gain),
        static_cast<double>(plan.rational_gain),
        static_cast<double>(plan.lut_scale),
        static_cast<double>(plan.pixel_epsilon),
        static_cast<double>(plan.positive_gain),
        static_cast<double>(plan.white_clip_squared),
        plan.whites_ev,
    };
    std::copy(staged_lanes.begin(), staged_lanes.end(), lanes);
    record_flags[0] = plan.lut_enabled ? 1 : 0;
    record_flags[1] = plan.negative_whites_lut.empty() ? 0 : 1;
    if (plan.lut_enabled) {
        std::memcpy(positive_lut_output, plan.positive_lut.data(),
                    plan.positive_lut.size() * sizeof(float));
    }
    if (!plan.negative_whites_lut.empty()) {
        std::memcpy(negative_whites_lut_output,
                    plan.negative_whites_lut.data(),
                    plan.negative_whites_lut.size() * sizeof(float));
    }
    *positive_lut_size_output = plan.lut_enabled ? plan.positive_lut.size() : 0;
    *negative_whites_lut_size_output = plan.negative_whites_lut.size();
    if (pixel_value_count != 0) {
        std::memcpy(output, staged_output.data(), pixel_value_count * sizeof(float));
    }
    return static_cast<int>(imprint::acr::ExposureRgbPrepareStatus::ok);
}

// Preserve the original explicit zero-Whites bridge ABI for direct bitwise
// parity checks. It delegates to the same prepared plan with Whites fixed at
// zero and keeps its smaller output record unchanged.
extern "C" int acr_test_prepare_apply_exposure_rgb(
    const double *inputs, std::size_t input_count, float pixel_epsilon,
    const std::uint8_t *context_flags, std::size_t context_flag_count,
    const float *pixels, std::size_t pixel_value_count, double *lanes,
    std::size_t lane_count, std::uint8_t *record_flags,
    std::size_t record_flag_count, float *lut_output,
    std::size_t lut_capacity, std::size_t *lut_size_output, float *output,
    std::size_t output_value_count) noexcept {
    constexpr std::size_t kLegacyInputCount = 5;
    constexpr std::size_t kLegacyFlagCount = 4;
    constexpr std::size_t kLegacyLaneCount = 8;
    constexpr std::size_t kLegacyRecordFlagCount = 1;
    if (input_count != kLegacyInputCount ||
        context_flag_count != kLegacyFlagCount || lane_count != kLegacyLaneCount ||
        record_flag_count != kLegacyRecordFlagCount || lut_capacity != kLutSize ||
        pixel_value_count % 3 != 0 || output_value_count != pixel_value_count ||
        input_count > std::numeric_limits<std::size_t>::max() / sizeof(double) ||
        pixel_value_count > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
        lane_count > std::numeric_limits<std::size_t>::max() / sizeof(double) ||
        lut_capacity > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        return static_cast<int>(imprint::acr::ExposureRgbPrepareStatus::invalid_argument);
    }

    AddressRange input_range;
    AddressRange context_range;
    AddressRange pixel_range;
    AddressRange lane_range;
    AddressRange record_flag_range;
    AddressRange lut_range;
    AddressRange lut_size_range;
    AddressRange output_range;
    if (!make_range(inputs, input_count * sizeof(double), alignof(double),
                    &input_range) ||
        !make_range(context_flags, context_flag_count * sizeof(std::uint8_t),
                    alignof(std::uint8_t), &context_range) ||
        !make_range(pixels, pixel_value_count * sizeof(float), alignof(float),
                    &pixel_range) ||
        !make_range(lanes, lane_count * sizeof(double), alignof(double),
                    &lane_range) ||
        !make_range(record_flags, record_flag_count * sizeof(std::uint8_t),
                    alignof(std::uint8_t), &record_flag_range) ||
        !make_range(lut_output, lut_capacity * sizeof(float), alignof(float),
                    &lut_range) ||
        !make_range(lut_size_output, sizeof(std::size_t), alignof(std::size_t),
                    &lut_size_range) ||
        !make_range(output, output_value_count * sizeof(float), alignof(float),
                    &output_range)) {
        return static_cast<int>(imprint::acr::ExposureRgbPrepareStatus::invalid_argument);
    }
    const AddressRange write_ranges[] = {
        lane_range, record_flag_range, lut_range, lut_size_range, output_range};
    for (std::size_t left = 0; left < sizeof(write_ranges) / sizeof(write_ranges[0]);
         ++left) {
        for (std::size_t right = left + 1;
             right < sizeof(write_ranges) / sizeof(write_ranges[0]); ++right) {
            if (overlaps(write_ranges[left], write_ranges[right])) {
                return static_cast<int>(
                    imprint::acr::ExposureRgbPrepareStatus::invalid_argument);
            }
        }
    }
    if (!disjoint(lane_range, &input_range, 1) ||
        !disjoint(lane_range, &context_range, 1) ||
        !disjoint(lane_range, &pixel_range, 1) ||
        !disjoint(record_flag_range, &input_range, 1) ||
        !disjoint(record_flag_range, &context_range, 1) ||
        !disjoint(record_flag_range, &pixel_range, 1) ||
        !disjoint(lut_range, &input_range, 1) ||
        !disjoint(lut_range, &context_range, 1) ||
        !disjoint(lut_range, &pixel_range, 1) ||
        !disjoint(lut_size_range, &input_range, 1) ||
        !disjoint(lut_size_range, &context_range, 1) ||
        !disjoint(lut_size_range, &pixel_range, 1) ||
        !disjoint(output_range, &input_range, 1) ||
        !disjoint(output_range, &context_range, 1) ||
        (overlaps(output_range, pixel_range) && output != pixels)) {
        return static_cast<int>(imprint::acr::ExposureRgbPrepareStatus::invalid_argument);
    }
    for (std::size_t index = 0; index < context_flag_count; ++index) {
        if (context_flags[index] > 1) {
            return static_cast<int>(
                imprint::acr::ExposureRgbPrepareStatus::invalid_argument);
        }
    }

    std::array<double, kInputCount> expanded_inputs = {
        inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], 0.0, 0.0, 0.0};
    const std::array<std::uint8_t, kContextFlagCount> expanded_flags = {
        context_flags[0], context_flags[1], context_flags[2], context_flags[3], 0};
    std::array<double, kLaneCount> staged_lanes{};
    std::array<std::uint8_t, kRecordFlagCount> staged_flags{};
    std::array<float, kLutSize> staged_lut{};
    std::array<float, imprint::acr::kNegativeWhitesLutSize> staged_whites_lut{};
    std::size_t staged_lut_size = 0;
    std::size_t staged_whites_lut_size = 0;
    std::vector<float> staged_output;
    try {
        staged_output.resize(pixel_value_count);
    } catch (...) {
        return kApplyStatusOffset +
               static_cast<int>(imprint::acr::ExposureRgbStatus::allocation_failure);
    }

    const int status = acr_test_prepare_apply_exposure_rgb_whites(
        expanded_inputs.data(), expanded_inputs.size(), pixel_epsilon,
        expanded_flags.data(), expanded_flags.size(), pixels, pixel_value_count,
        staged_lanes.data(), staged_lanes.size(), staged_flags.data(),
        staged_flags.size(), staged_lut.data(), staged_lut.size(),
        &staged_lut_size, staged_whites_lut.data(), staged_whites_lut.size(),
        &staged_whites_lut_size, staged_output.data(), staged_output.size());
    if (status != static_cast<int>(imprint::acr::ExposureRgbPrepareStatus::ok)) {
        return status;
    }

    std::copy_n(staged_lanes.begin(), kLegacyLaneCount, lanes);
    record_flags[0] = staged_flags[0];
    if (staged_lut_size != 0) {
        std::memcpy(lut_output, staged_lut.data(), staged_lut_size * sizeof(float));
    }
    *lut_size_output = staged_lut_size;
    if (pixel_value_count != 0) {
        std::memcpy(output, staged_output.data(), pixel_value_count * sizeof(float));
    }
    return status;
}
