#include "../src/acr_resample_separable.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

constexpr float kSentinelA = 31.25f;
constexpr float kSentinelB = -12.5f;
constexpr int kBridgeInvalidOutput = 20;
constexpr int kBridgeUnexpectedResultMutation = 99;

bool valid_output_span(const float *output, std::size_t count) noexcept {
    if (!output || count == 0 ||
        count > std::numeric_limits<std::size_t>::max() / sizeof(float))
        return false;
    const auto address = reinterpret_cast<std::uintptr_t>(output);
    const auto bytes = count * sizeof(float);
    return address % alignof(float) == 0 &&
           bytes <= std::numeric_limits<std::uintptr_t>::max() - address;
}

}  // namespace

extern "C" int acr_test_resample_separable(
    const float *source, std::size_t source_count,
    std::int32_t top, std::int32_t left,
    std::size_t height, std::size_t width, std::size_t channels,
    std::int32_t dst_top, std::int32_t dst_left,
    std::size_t dst_height, std::size_t dst_width,
    const float *vcoeff, std::size_t vcount, std::size_t vtaps,
    std::size_t vstride, std::int64_t vorig, std::int64_t vstep,
    std::int64_t vlo, std::int64_t vhi,
    const float *hcoeff, std::size_t hcount, std::size_t htaps,
    std::size_t hstride, std::int64_t horig, std::int64_t hstep,
    std::int64_t hlo, std::int64_t hhi,
    float *output, std::size_t output_count) noexcept {
    if (!valid_output_span(output, output_count)) return kBridgeInvalidOutput;

    try {
        const imprint::acr::SourceView source_view{
            source, source_count, top, left, height, width, channels};
        const imprint::acr::DestinationRect destination{
            dst_top, dst_left, dst_height, dst_width};
        const imprint::acr::PreparedSeparableAxis vertical{
            vorig, vstep, vlo, vhi, {vcoeff, vcount, vtaps, vstride}};
        const imprint::acr::PreparedSeparableAxis horizontal{
            horig, hstep, hlo, hhi, {hcoeff, hcount, htaps, hstride}};

        std::vector<float> candidate{kSentinelA, kSentinelB};
        const auto status = imprint::acr::process_resample_separable(
            source_view, destination, vertical, horizontal, candidate);
        if (status != imprint::acr::ResampleSeparableStatus::ok) {
            if (candidate.size() != 2 || candidate[0] != kSentinelA ||
                candidate[1] != kSentinelB)
                return kBridgeUnexpectedResultMutation;
            return static_cast<int>(status);
        }
        if (candidate.size() != output_count) return kBridgeInvalidOutput;
        std::copy(candidate.begin(), candidate.end(), output);
        return static_cast<int>(imprint::acr::ResampleSeparableStatus::ok);
    } catch (...) {
        return static_cast<int>(
            imprint::acr::ResampleSeparableStatus::allocation_failure);
    }
}

extern "C" int acr_test_resample_separable_result_alias() noexcept {
    try {
        std::vector<float> vertical_coefficients(128 * 8, 0.0f);
        std::vector<float> horizontal_coefficients(128 * 8, 0.0f);
        for (std::size_t phase = 0; phase < 128; ++phase) {
            vertical_coefficients[phase * 8] = 1.0f;
            horizontal_coefficients[phase * 8] = 1.0f;
        }

        const std::int64_t one = std::int64_t{1} << 32;
        const std::int32_t top = -3;
        const std::int32_t left = -5;
        imprint::acr::SourceView source{
            nullptr, 4, top, left, 2, 2, 1};
        imprint::acr::DestinationRect destination{top, left, 2, 2};
        imprint::acr::PreparedSeparableAxis vertical{
            0, one,
            static_cast<std::int64_t>(top) * one,
            static_cast<std::int64_t>(top + 1) * one + 0xfe000000,
            {vertical_coefficients.data(), vertical_coefficients.size(), 1, 8}};
        imprint::acr::PreparedSeparableAxis horizontal{
            0, one,
            static_cast<std::int64_t>(left) * one,
            static_cast<std::int64_t>(left + 1) * one + 0xfe000000,
            {horizontal_coefficients.data(), horizontal_coefficients.size(), 1, 8}};
        std::vector<float> values{0.25f, 0.5f, 0.75f, 1.0f};
        source.data = values.data();
        const auto status = imprint::acr::process_resample_separable(
            source, destination, vertical, horizontal, values);
        if (status != imprint::acr::ResampleSeparableStatus::ok ||
            values != std::vector<float>{0.25f, 0.5f, 0.75f, 1.0f})
            return 1;

        source.data = values.data();
        const auto before = values;
        destination.left = 1;
        horizontal.origin = std::numeric_limits<std::int64_t>::max();
        horizontal.step = 1;
        const auto failure_status = imprint::acr::process_resample_separable(
            source, destination, vertical, horizontal, values);
        return failure_status ==
                       imprint::acr::ResampleSeparableStatus::coordinate_overflow &&
                       values == before
                   ? 0
                   : 2;
    } catch (...) {
        return 3;
    }
}
