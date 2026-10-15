#include "../src/acr_exposure_rgb_prepare.hpp"
#include "../src/acr_negative_curve.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace {

constexpr std::size_t kInputCount = 5;
constexpr std::size_t kFlagCount = 4;
constexpr std::size_t kLaneCount = 8;
constexpr std::size_t kRecordFlagCount = 1;
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

// Inputs: [a, b, amount58, grayDiffB8, anchor690]. Context flags are
// [sdr_mode, whites_zero, mask_enabled, color_transform_enabled].
// Recorded lanes: [split_a, split_b, LUT_EV, remainder_EV, gain,
// rational_gain, LUT_scale, epsilon]. Pixel and record outputs are all
// committed only after preparation and the complete pixel pass succeed.
extern "C" int acr_test_prepare_apply_exposure_rgb(
    const double *inputs, std::size_t input_count, float pixel_epsilon,
    const std::uint8_t *context_flags, std::size_t context_flag_count,
    const float *pixels, std::size_t pixel_value_count, double *lanes,
    std::size_t lane_count, std::uint8_t *record_flags,
    std::size_t record_flag_count, float *lut_output,
    std::size_t lut_capacity, std::size_t *lut_size_output, float *output,
    std::size_t output_value_count) noexcept {
    if (input_count != kInputCount || context_flag_count != kFlagCount ||
        lane_count != kLaneCount || record_flag_count != kRecordFlagCount ||
        lut_capacity != imprint::acr::kPositiveExposureLutSize ||
        pixel_value_count % 3 != 0 ||
        output_value_count != pixel_value_count ||
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

    imprint::acr::ExposureRgbPrepareInputs prepare_inputs;
    prepare_inputs.a_ev = inputs[0];
    prepare_inputs.b_ev = inputs[1];
    prepare_inputs.amount58 = inputs[2];
    prepare_inputs.gray_diff_b8 = inputs[3];
    prepare_inputs.anchor690 = inputs[4];
    prepare_inputs.pixel_epsilon = pixel_epsilon;
    prepare_inputs.sdr_mode = context_flags[0] != 0;
    prepare_inputs.whites_zero = context_flags[1] != 0;
    prepare_inputs.mask_enabled = context_flags[2] != 0;
    prepare_inputs.color_transform_enabled = context_flags[3] != 0;

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
    };
    std::copy(staged_lanes.begin(), staged_lanes.end(), lanes);
    record_flags[0] = plan.lut_enabled ? 1 : 0;
    if (plan.lut_enabled) {
        std::memcpy(lut_output, plan.positive_lut.data(),
                    plan.positive_lut.size() * sizeof(float));
    }
    *lut_size_output = plan.lut_enabled ? plan.positive_lut.size() : 0;
    if (pixel_value_count != 0) {
        std::memcpy(output, staged_output.data(),
                    pixel_value_count * sizeof(float));
    }
    return static_cast<int>(imprint::acr::ExposureRgbPrepareStatus::ok);
}
