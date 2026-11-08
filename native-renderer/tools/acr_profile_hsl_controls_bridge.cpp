// Test-only adapter; this is not the product ABI or profile metadata loader.
#include "../src/acr_profile_hsl_controls.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

extern "C" int acr_test_profile_hsl_controls(
    const std::int32_t *current, const std::int32_t *profile_defaults,
    double scalar, std::uint32_t monochrome, std::uint32_t convert_to_gray,
    std::int32_t *values, double *blended, std::uint32_t *wrote_mask) noexcept {
    // Bridge-only statuses: 2 invalid argument, 3 failure mutated result,
    // 4 input mutation. Core statuses remain ok=0/non_finite_scalar=1.
    if (!current || !profile_defaults || !values || !blended || !wrote_mask ||
        monochrome > 1 || convert_to_gray > 255) return 2;

    using namespace imprint::acr;
    ProfileHslControlsInput input{};
    std::copy_n(current, input.current.size(), input.current.begin());
    std::copy_n(profile_defaults, input.profile_defaults.size(), input.profile_defaults.begin());
    input.profile_scalar = scalar;
    input.monochrome_profile = monochrome != 0;
    input.convert_to_gray = static_cast<std::uint8_t>(convert_to_gray);

    ProfileHslControlsResult result{};
    static_assert(std::is_trivially_copyable_v<ProfileHslControlsResult>);
    static_assert(std::is_trivially_copyable_v<ProfileHslControlsInput>);
    // Seed padding too, so non-finite failure is checked over the full object,
    // with deliberately nonzero fields rather than a default-constructed result.
    std::fill_n(reinterpret_cast<unsigned char *>(&result), sizeof(result),
                static_cast<unsigned char>(0xa5));
    for (std::size_t lane = 0; lane < result.values.size(); ++lane) {
        result.values[lane] = -1234567 + static_cast<std::int32_t>(lane) * 7919;
        result.blended[lane] = lane % 2 ? -9123.25 - lane : 8123.5 + lane;
    }
    result.blended[0] = -0.0;
    result.wrote_mask = 0xa5c39e71u;
    std::array<unsigned char, sizeof(result)> result_before{};
    std::array<unsigned char, sizeof(input)> input_before{};
    std::memcpy(result_before.data(), &result, sizeof(result));
    std::memcpy(input_before.data(), &input, sizeof(input));

    const auto status = process_profile_hsl_controls(input, result);
    if (std::memcmp(input_before.data(), &input, sizeof(input)) != 0) return 4;
    if (status != ProfileHslControlsStatus::ok) {
        if (std::memcmp(result_before.data(), &result, sizeof(result)) != 0) return 3;
        return static_cast<int>(status);
    }

    std::copy(result.values.begin(), result.values.end(), values);
    std::copy(result.blended.begin(), result.blended.end(), blended);
    *wrote_mask = result.wrote_mask;
    return 0;
}
