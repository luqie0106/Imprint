#include "../src/acr_profile_gain_apply.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace {

using imprint::acr::ProfileGainApplyStatus;
using imprint::acr::ProfileGainStatus;

struct AddressRange {
    std::uintptr_t begin = 0;
    std::uintptr_t end = 0;
};

bool overlaps(const AddressRange &left, const AddressRange &right) noexcept {
    return left.begin < right.end && right.begin < left.end;
}

template <typename T>
ProfileGainApplyStatus validate_bridge_span(const T *pointer,
                                            std::size_t count,
                                            AddressRange *range) noexcept {
    if (range == nullptr || pointer == nullptr) {
        return ProfileGainStatus::invalid_argument;
    }
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (begin % alignof(T) != 0) {
        return ProfileGainStatus::unsupported_alignment;
    }
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
        return ProfileGainStatus::arithmetic_overflow;
    }
    const std::size_t bytes = count * sizeof(T);
    if (bytes > static_cast<std::size_t>(
                    std::numeric_limits<std::ptrdiff_t>::max()) ||
        bytes > std::numeric_limits<std::uintptr_t>::max() - begin) {
        return ProfileGainStatus::arithmetic_overflow;
    }
    *range = {begin, begin + bytes};
    return ProfileGainStatus::ok;
}

}  // namespace

extern "C" int acr_profile_gain_apply_bridge(
    const float *const *inputs, const std::size_t *input_counts,
    float *const *outputs, const std::size_t *output_counts,
    std::size_t outer, std::size_t inner, std::size_t value_stride,
    std::size_t gain_stride, double pivot, std::uint32_t mode,
    std::uint32_t clip) noexcept {
    if (inputs == nullptr || input_counts == nullptr || outputs == nullptr ||
        output_counts == nullptr) {
        return static_cast<int>(ProfileGainStatus::invalid_argument);
    }
    if (mode > static_cast<std::uint32_t>(
                   imprint::acr::ProfileGainApplyMode::cpu_vector) ||
        clip > 1) {
        return static_cast<int>(ProfileGainStatus::invalid_argument);
    }

    std::array<AddressRange, 4> controls{};
    auto status = validate_bridge_span(inputs, 4, &controls[0]);
    if (status != ProfileGainStatus::ok) return static_cast<int>(status);
    status = validate_bridge_span(input_counts, 4, &controls[1]);
    if (status != ProfileGainStatus::ok) return static_cast<int>(status);
    status = validate_bridge_span(outputs, 3, &controls[2]);
    if (status != ProfileGainStatus::ok) return static_cast<int>(status);
    status = validate_bridge_span(output_counts, 3, &controls[3]);
    if (status != ProfileGainStatus::ok) return static_cast<int>(status);

    std::array<imprint::acr::ProfileGainReadView, 4> input_views{};
    for (std::size_t i = 0; i < input_views.size(); ++i) {
        input_views[i] = {inputs[i], input_counts[i]};
    }
    std::array<imprint::acr::ProfileGainWriteView, 3> output_views{};
    std::array<AddressRange, 3> output_spans{};
    for (std::size_t i = 0; i < output_views.size(); ++i) {
        output_views[i] = {outputs[i], output_counts[i]};
        status = validate_bridge_span(outputs[i], output_counts[i],
                                      &output_spans[i]);
        if (status != ProfileGainStatus::ok) return static_cast<int>(status);
        for (const AddressRange &control : controls) {
            if (overlaps(output_spans[i], control)) {
                return static_cast<int>(ProfileGainStatus::overlap);
            }
        }
    }

    const auto result = imprint::acr::process_profile_gain_apply_f32(
        input_views, output_views, outer, inner, value_stride, gain_stride,
        pivot, static_cast<imprint::acr::ProfileGainApplyMode>(mode), clip != 0);
    return static_cast<int>(result);
}
