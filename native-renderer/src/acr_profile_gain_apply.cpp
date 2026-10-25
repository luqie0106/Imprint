#include "acr_profile_gain_apply.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>

namespace imprint::acr {
namespace {

constexpr std::size_t kMaximumActiveValues = std::size_t{1} << 26;

struct AddressRange {
    std::uintptr_t begin = 0;
    std::uintptr_t end = 0;
};

bool checked_product(std::size_t left, std::size_t right,
                     std::size_t *product) noexcept {
    if (product == nullptr ||
        (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)) {
        return false;
    }
    *product = left * right;
    return true;
}

ProfileGainApplyStatus make_range(const void *pointer, std::size_t count,
                                  std::size_t element_size,
                                  std::size_t alignment,
                                  AddressRange *range) noexcept {
    if (range == nullptr || alignment == 0 || pointer == nullptr) {
        return ProfileGainStatus::invalid_argument;
    }
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (begin % alignment != 0) {
        return ProfileGainStatus::unsupported_alignment;
    }

    std::size_t byte_count = 0;
    if (!checked_product(count, element_size, &byte_count) ||
        byte_count > static_cast<std::size_t>(
                         std::numeric_limits<std::ptrdiff_t>::max()) ||
        byte_count > std::numeric_limits<std::uintptr_t>::max() - begin) {
        return ProfileGainStatus::arithmetic_overflow;
    }
    *range = {begin, begin + byte_count};
    return ProfileGainStatus::ok;
}

ProfileGainApplyStatus make_float_range(const float *pointer,
                                        std::size_t count,
                                        AddressRange *range) noexcept {
    return make_range(pointer, count, sizeof(float), alignof(float), range);
}

ProfileGainApplyStatus make_float_range(float *pointer, std::size_t count,
                                        AddressRange *range) noexcept {
    return make_range(pointer, count, sizeof(float), alignof(float), range);
}

bool overlaps(const AddressRange &left, const AddressRange &right) noexcept {
    return left.begin < right.end && right.begin < left.end;
}

bool required_samples(std::size_t outer_count, std::size_t row_stride,
                      std::size_t inner_count,
                      std::size_t *required) noexcept {
    if (required == nullptr || outer_count == 0) return false;
    std::size_t row_offset = 0;
    if (!checked_product(outer_count - 1, row_stride, &row_offset) ||
        row_offset > std::numeric_limits<std::size_t>::max() - inner_count) {
        return false;
    }
    *required = row_offset + inner_count;
    return true;
}

float clip_value(float value, ProfileGainApplyMode mode) noexcept {
    if (mode == ProfileGainApplyMode::scalar) {
        return std::fmin(value, 1.0f);
    }
    // ARM FMIN propagates NaN; std::fmin has FMINNM semantics instead.
    if (std::isnan(value)) return value;
    return value > 1.0f ? 1.0f : value;
}

}  // namespace

ProfileGainApplyStatus process_profile_gain_apply_f32(
    const std::array<ProfileGainReadView, 4> &inputs,
    const std::array<ProfileGainWriteView, 3> &outputs,
    std::size_t outer_count, std::size_t inner_count,
    std::size_t value_row_stride, std::size_t gain_row_stride, double pivot,
    ProfileGainApplyMode mode, bool clip) noexcept {
    if (outer_count == 0 || inner_count == 0) {
        return ProfileGainStatus::invalid_dimensions;
    }
    constexpr std::size_t kMaximumStride =
        static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max());
    if (value_row_stride < inner_count || gain_row_stride < inner_count ||
        value_row_stride > kMaximumStride || gain_row_stride > kMaximumStride) {
        return ProfileGainStatus::invalid_stride;
    }
    if (mode != ProfileGainApplyMode::scalar &&
        mode != ProfileGainApplyMode::cpu_vector) {
        return ProfileGainStatus::invalid_argument;
    }

    if (!std::isfinite(pivot) ||
        std::fabs(pivot) > static_cast<double>(
                               std::numeric_limits<float>::max())) {
        return ProfileGainStatus::invalid_prepared;
    }
    const float a = static_cast<float>(pivot);
    if (!std::isfinite(a)) return ProfileGainStatus::invalid_prepared;

    std::size_t active_per_plane = 0;
    std::size_t active_total = 0;
    if (!checked_product(outer_count, inner_count, &active_per_plane) ||
        !checked_product(active_per_plane, outputs.size(), &active_total)) {
        return ProfileGainStatus::arithmetic_overflow;
    }
    if (active_total > kMaximumActiveValues) {
        return ProfileGainStatus::resource_limit;
    }

    std::size_t value_required = 0;
    std::size_t gain_required = 0;
    if (!required_samples(outer_count, value_row_stride, inner_count,
                         &value_required) ||
        !required_samples(outer_count, gain_row_stride, inner_count,
                         &gain_required)) {
        return ProfileGainStatus::arithmetic_overflow;
    }

    std::array<AddressRange, 4> input_ranges{};
    std::array<AddressRange, 3> output_ranges{};
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        if (inputs[i].data == nullptr) return ProfileGainStatus::invalid_argument;
        if (inputs[i].count < (i == 3 ? gain_required : value_required)) {
            return ProfileGainStatus::buffer_too_small;
        }
        const auto status = make_float_range(inputs[i].data, inputs[i].count,
                                             &input_ranges[i]);
        if (status != ProfileGainStatus::ok) return status;
    }
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        if (outputs[i].data == nullptr) return ProfileGainStatus::invalid_argument;
        if (outputs[i].count < value_required) {
            return ProfileGainStatus::buffer_too_small;
        }
        const auto status = make_float_range(outputs[i].data, outputs[i].count,
                                             &output_ranges[i]);
        if (status != ProfileGainStatus::ok) return status;
    }

    AddressRange input_views_range;
    auto status = make_range(&inputs, 1, sizeof(inputs), alignof(decltype(inputs)),
                             &input_views_range);
    if (status != ProfileGainStatus::ok) return status;
    AddressRange output_views_range;
    status = make_range(&outputs, 1, sizeof(outputs), alignof(decltype(outputs)),
                        &output_views_range);
    if (status != ProfileGainStatus::ok) return status;

    for (std::size_t i = 0; i < outputs.size(); ++i) {
        if (overlaps(output_ranges[i], input_ranges[3]) ||
            overlaps(output_ranges[i], input_views_range) ||
            overlaps(output_ranges[i], output_views_range)) {
            return ProfileGainStatus::overlap;
        }
        for (std::size_t j = 0; j < outputs.size(); ++j) {
            if (i != j && overlaps(output_ranges[i], output_ranges[j])) {
                return ProfileGainStatus::overlap;
            }
        }
        for (std::size_t j = 0; j < 3; ++j) {
            if (!overlaps(output_ranges[i], input_ranges[j])) continue;
            if (i != j || outputs[i].data != inputs[j].data ||
                outputs[i].count != inputs[j].count) {
                return ProfileGainStatus::overlap;
            }
        }
    }

    try {
        std::vector<float> staged(active_total);

        const float d = static_cast<float>(a + -1.0f);
        // Keep the division observable even when a == 1 selects +0 below.
        volatile float reciprocal_raw = -1.0f / d;
        const float reciprocal = reciprocal_raw;
        const float r = a == 1.0f ? 0.0f : reciprocal;
        const float factor = -d;
        const float bias = std::fma(-r, a, 0.0f);
        const float offset = std::fma(d, 0.0f, a);

        for (std::size_t channel = 0; channel < outputs.size(); ++channel) {
            const float *plane = inputs[channel].data;
            for (std::size_t outer = 0; outer < outer_count; ++outer) {
                const std::size_t row_offset = outer * value_row_stride;
                const std::size_t gain_offset = outer * gain_row_stride;
                const std::size_t stage_offset =
                    channel * active_per_plane + outer * inner_count;
                for (std::size_t inner = 0; inner < inner_count; ++inner) {
                    const float inner_value =
                        std::fma(plane[row_offset + inner], r, bias);
                    const float mult = static_cast<float>(
                        inner_value * inputs[3].data[gain_offset + inner]);
                    float value = std::fma(mult, factor, offset);
                    if (clip) value = clip_value(value, mode);
                    staged[stage_offset + inner] = value;
                }
            }
        }

        for (std::size_t channel = 0; channel < outputs.size(); ++channel) {
            for (std::size_t outer = 0; outer < outer_count; ++outer) {
                const std::size_t row_offset = outer * value_row_stride;
                const std::size_t stage_offset =
                    channel * active_per_plane + outer * inner_count;
                std::copy_n(staged.data() + stage_offset, inner_count,
                            outputs[channel].data + row_offset);
            }
        }
        return ProfileGainStatus::ok;
    } catch (const std::bad_alloc &) {
        return ProfileGainStatus::allocation_failure;
    } catch (const std::length_error &) {
        return ProfileGainStatus::allocation_failure;
    } catch (...) {
        return ProfileGainStatus::allocation_failure;
    }
}

}  // namespace imprint::acr
