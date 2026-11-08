#include "imprint_renderer.h"
#include "acr_color_kernels.hpp"
#include "acr_profile_hsl_controls.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr uint32_t kWidth = 17;
constexpr uint32_t kHeight = 13;
constexpr std::size_t kSamples = kWidth * kHeight * 3;
using Current = std::array<double, 25>;
using Defaults = std::array<int32_t, 24>;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

bool same_bits(const void* left, const void* right, std::size_t bytes) {
    return std::memcmp(left, right, bytes) == 0;
}

std::vector<float> source_pixels() {
    std::vector<float> source(kSamples);
    std::uint32_t state = 0x8fd7513bu;
    for (float& value : source) {
        state = state * 1664525u + 1013904223u;
        value = static_cast<float>(state >> 8) / 16777215.0f;
    }
    constexpr std::array<float, 42> edges{
        0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 0.0f, 0.0f, 1.0f, 0.5f, 0.0f,
        1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f,
        0.5f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f,
        0.5f, 0.5f, 0.5f, 0.001f, 0.001f, 0.001f,
        -0.0f, 0.0f, -0.0f, 0.9f, 0.899f, 0.898f};
    std::copy(edges.begin(), edges.end(), source.begin());
    return source;
}

imprint::acr::HslParams independent_params(const Current& current,
                                          const Defaults& defaults, double scalar) {
    imprint::acr::ProfileHslControlsInput input{};
    for (std::size_t lane = 0; lane < 24; ++lane) {
        input.current[lane] = static_cast<int32_t>(current[lane]);
        input.profile_defaults[lane] = defaults[lane];
    }
    input.profile_scalar = scalar;
    input.monochrome_profile = false;
    input.convert_to_gray = 0;
    imprint::acr::ProfileHslControlsResult controls{};
    require(imprint::acr::process_profile_hsl_controls(input, controls) ==
                imprint::acr::ProfileHslControlsStatus::ok,
            "independent profile control preparation failed");
    imprint::acr::HslParams params{};
    for (std::size_t lane = 0; lane < 8; ++lane) {
        params.hue[lane] = controls.values[lane];
        params.saturation[lane] = controls.values[8 + lane];
        params.luminance[lane] = controls.values[16 + lane];
    }
    params.vibrance = current[24];
    return params;
}

struct Invocation {
    uint32_t width = kWidth;
    uint32_t height = kHeight;
    const float* source;
    std::size_t source_samples = kSamples;
    const double* current;
    std::size_t current_count = 25;
    const int32_t* defaults;
    std::size_t default_count = 24;
    double scalar = 1.0;
    float* destination;
    std::size_t destination_samples = kSamples;

    im_status run() const {
        return im_native_acr_profile_hsl_sdr_rgbf32(
            width, height, source, source_samples, current, current_count,
            defaults, default_count, scalar, destination, destination_samples);
    }
};

void rejects_without_write(const Invocation& call, const void* watched,
                           std::size_t watched_bytes, const std::string& label) {
    std::vector<unsigned char> saved(watched_bytes);
    std::memcpy(saved.data(), watched, watched_bytes);
    require(call.run() == IM_STATUS_INVALID_ARGUMENT, label + ": wrong status");
    require(same_bits(watched, saved.data(), watched_bytes), label + ": modified destination");
}

std::size_t validate_parity(const std::vector<float>& source) {
    const auto saved_source = source;
    Defaults sentinel{};
    sentinel.fill(-999999);
    auto sparse = sentinel;
    sparse[9] = -14;  // Orange Saturation.
    sparse[17] = -7;  // Orange Luminance.
    Defaults mixed{};
    for (std::size_t lane = 0; lane < mixed.size(); ++lane) {
        mixed[lane] = static_cast<int32_t>((lane * 43) % 201) - 100;
    }
    mixed[0] = 100;
    mixed[1] = -100;
    mixed[2] = 0;
    mixed[3] = -101;
    mixed[4] = 101;
    mixed[5] = std::numeric_limits<int32_t>::min();
    mixed[6] = std::numeric_limits<int32_t>::max();
    const std::array<Defaults, 4> default_cases{sentinel, sparse, mixed, Defaults{}};
    std::array<Current, 6> current_cases{};
    for (std::size_t lane = 0; lane < 24; ++lane) {
        current_cases[1][lane] = lane % 2 == 0 ? 37.0 : -42.0;
        current_cases[2][lane] = lane % 2 == 0 ? 100.0 : -100.0;
        current_cases[3][lane] = static_cast<double>(static_cast<int>((lane * 61) % 201) - 100);
        current_cases[4][lane] = 100.0;
        current_cases[5][lane] = -100.0;
    }
    current_cases[1][24] = 35.75;
    current_cases[2][24] = -27.125;
    current_cases[3][24] = -0.0;
    current_cases[4][24] = 100.0;
    current_cases[5][24] = -100.0;
    constexpr std::array<double, 5> scalars{0.0, 0.25, 1.0, 1.75, 2.0};
    std::size_t cases = 0;
    for (const auto& defaults : default_cases) {
        const auto saved_defaults = defaults;
        for (const auto& current : current_cases) {
            const auto saved_current = current;
            for (double scalar : scalars) {
                const auto params = independent_params(current, defaults, scalar);
                std::vector<float> expected(kSamples);
                imprint::acr::apply_hsl(source.data(), expected.data(), kSamples / 3, params);
                std::vector<float> output(kSamples + 2, -123.0f);
                Invocation call{};
                call.source = source.data();
                call.current = current.data();
                call.defaults = defaults.data();
                call.scalar = scalar;
                call.destination = output.data() + 1;
                require(call.run() == IM_STATUS_OK, "valid prepared color SDR call rejected");
                require(same_bits(output.data() + 1, expected.data(), kSamples * sizeof(float)),
                        "composed pixels differ from independent process_profile_hsl_controls + apply_hsl");
                require(output.front() == -123.0f && output.back() == -123.0f,
                        "success wrote outside destination span");
                require(same_bits(current.data(), saved_current.data(), sizeof(Current)) &&
                            same_bits(defaults.data(), saved_defaults.data(), sizeof(Defaults)),
                        "success modified prepared controls");
                ++cases;
            }
        }
    }
    require(same_bits(source.data(), saved_source.data(), kSamples * sizeof(float)),
            "success modified source");

    Current zero{};
    std::vector<float> output(kSamples);
    Invocation call{};
    call.source = source.data();
    call.current = zero.data();
    call.defaults = sentinel.data();
    call.destination = output.data();
    require(call.run() == IM_STATUS_OK &&
                same_bits(output.data(), source.data(), kSamples * sizeof(float)),
            "all-sentinel neutral call is not bitwise identity");
    call.defaults = sparse.data();
    require(call.run() == IM_STATUS_OK &&
                !same_bits(output.data(), source.data(), kSamples * sizeof(float)),
            "sparse Orange defaults did not affect pixels");
    const auto prepared = independent_params(zero, sparse, 1.0);
    require(prepared.saturation[1] == -14.0 && prepared.luminance[1] == -7.0,
            "sparse defaults did not retain their recovered integer controls");
    call.scalar = 0.0;
    require(call.run() == IM_STATUS_OK &&
                same_bits(output.data(), source.data(), kSamples * sizeof(float)),
            "scalar zero did not disable profile adjustments");

    // Vibrance remains the current double, including fractional values and
    // both UI endpoints; profile shaping must never scale it.
    call.scalar = 1.75;
    call.defaults = sentinel.data();
    for (double vibrance : std::array<double, 4>{-100.0, -37.125, 35.75, 100.0}) {
        zero[24] = vibrance;
        imprint::acr::HslParams params{};
        params.vibrance = vibrance;
        std::vector<float> expected(kSamples);
        imprint::acr::apply_hsl(source.data(), expected.data(), kSamples / 3, params);
        require(call.run() == IM_STATUS_OK &&
                    same_bits(output.data(), expected.data(), kSamples * sizeof(float)) &&
                    !same_bits(output.data(), source.data(), kSamples * sizeof(float)),
                "current Vibrance was dropped or modified");
    }

    // Touching spans are disjoint, not overlapping.
    zero.fill(0.0);
    std::vector<float> adjacent(2 * kSamples, -123.0f);
    std::copy(source.begin(), source.end(), adjacent.begin());
    call.source = adjacent.data();
    call.destination = adjacent.data() + kSamples;
    require(call.run() == IM_STATUS_OK &&
                same_bits(adjacent.data(), adjacent.data() + kSamples, kSamples * sizeof(float)),
            "adjacent source/destination spans were rejected");
    return cases;
}

void validate_rejections(const std::vector<float>& source) {
    Current current{};
    current[0] = 14.0;
    current[9] = -11.0;
    current[24] = 12.25;
    Defaults defaults{};
    defaults.fill(-999999);
    defaults[9] = -14;
    defaults[17] = -7;
    std::vector<float> output(kSamples + 8, -123.0f);
    Invocation base{};
    base.source = source.data();
    base.current = current.data();
    base.defaults = defaults.data();
    base.destination = output.data() + 4;
    auto reject = [&](const Invocation& call, const std::string& label) {
        rejects_without_write(call, output.data(), output.size() * sizeof(float), label);
    };
    for (int pointer = 0; pointer < 4; ++pointer) {
        auto call = base;
        if (pointer == 0) call.source = nullptr;
        if (pointer == 1) call.current = nullptr;
        if (pointer == 2) call.defaults = nullptr;
        if (pointer == 3) call.destination = nullptr;
        reject(call, "null pointer");
    }
    for (int field = 0; field < 4; ++field) {
        const std::size_t expected = field < 2 ? kSamples : field == 2 ? 25 : 24;
        for (std::size_t count : {std::size_t{0}, expected - 1, expected + 1,
                                 std::numeric_limits<std::size_t>::max()}) {
            auto call = base;
            if (field == 0) call.source_samples = count;
            if (field == 1) call.destination_samples = count;
            if (field == 2) call.current_count = count;
            if (field == 3) call.default_count = count;
            reject(call, "incorrect declared count");
        }
    }
    for (const auto& dimensions : std::array<std::array<uint32_t, 2>, 5>{{
             {{0, kHeight}}, {{kWidth, 0}}, {{UINT32_MAX, UINT32_MAX}},
             {{UINT32_MAX, 1}}, {{1u << 29, 1}}}}) {
        auto call = base;
        call.width = dimensions[0];
        call.height = dimensions[1];
        reject(call, "zero or excessive image dimensions");
    }
    for (double scalar : std::array<double, 5>{
             -0.01, 2.01, std::numeric_limits<double>::quiet_NaN(),
             std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()}) {
        auto call = base;
        call.scalar = scalar;
        reject(call, "invalid profile scalar");
    }
    for (std::size_t lane = 0; lane < current.size(); ++lane) {
        const double saved = current[lane];
        for (double value : std::array<double, 5>{
                 -101.0, 101.0, std::numeric_limits<double>::quiet_NaN(),
                 std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()}) {
            current[lane] = value;
            reject(base, "nonfinite or out-of-range current lane " + std::to_string(lane));
        }
        if (lane < 24) {
            for (double value : {-0.25, 0.25}) {
                current[lane] = value;
                reject(base, "fractional HSL current lane " + std::to_string(lane));
            }
        }
        current[lane] = saved;
    }
    auto bad_source = source;
    for (float value : std::array<float, 5>{
             -0.01f, 1.01f, std::numeric_limits<float>::quiet_NaN(),
             std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()}) {
        bad_source.back() = value;
        auto call = base;
        call.source = bad_source.data();
        reject(call, "invalid last source sample");
        Current neutral{};
        call.current = neutral.data();
        Defaults skipped{};
        skipped.fill(-999999);
        call.defaults = skipped.data();
        reject(call, "invalid last source sample on neutral path");
    }

    alignas(double) std::array<unsigned char, kSamples * sizeof(float) + 32> unaligned{};
    for (int pointer = 0; pointer < 4; ++pointer) {
        auto call = base;
        if (pointer == 0) call.source = reinterpret_cast<const float*>(unaligned.data() + 1);
        if (pointer == 1) call.current = reinterpret_cast<const double*>(unaligned.data() + 1);
        if (pointer == 2) call.defaults = reinterpret_cast<const int32_t*>(unaligned.data() + 1);
        if (pointer == 3) call.destination = reinterpret_cast<float*>(unaligned.data() + 1);
        reject(call, "misaligned pointer");
        rejects_without_write(call, unaligned.data(), unaligned.size(), "misaligned storage");
    }
    const auto address_max = std::numeric_limits<std::uintptr_t>::max();
    for (int pointer = 0; pointer < 4; ++pointer) {
        auto call = base;
        if (pointer == 0) call.source = reinterpret_cast<const float*>(address_max - address_max % alignof(float));
        if (pointer == 1) call.current = reinterpret_cast<const double*>(address_max - address_max % alignof(double));
        if (pointer == 2) call.defaults = reinterpret_cast<const int32_t*>(address_max - address_max % alignof(int32_t));
        if (pointer == 3) call.destination = reinterpret_cast<float*>(address_max - address_max % alignof(float));
        reject(call, "wrapped pointer span");
    }

    std::vector<float> pixel_alias(kSamples + 8, 0.25f);
    for (int direction = 0; direction < 3; ++direction) {
        auto call = base;
        call.source = pixel_alias.data() + (direction == 2 ? 1 : 0);
        call.destination = pixel_alias.data() + (direction == 1 ? 1 : 0);
        rejects_without_write(call, pixel_alias.data(), pixel_alias.size() * sizeof(float),
                              "exact or partial source/destination alias");
    }
    // Keep every declared span backed by storage, including the case where
    // destination begins before a control array and only its tail overlaps.
    std::vector<double> current_alias((kSamples + 1) / 2 + 25, 0.0);
    for (bool tail : {false, true}) {
        auto call = base;
        auto* controls = current_alias.data() + (tail ? kSamples / 2 - 1 : 0);
        std::copy(current.begin(), current.end(), controls);
        call.current = controls;
        call.destination = reinterpret_cast<float*>(current_alias.data());
        rejects_without_write(call, current_alias.data(), current_alias.size() * sizeof(double),
                              "destination overlaps current controls");
    }
    std::vector<int32_t> default_alias(kSamples + 24, 0);
    for (bool tail : {false, true}) {
        auto call = base;
        auto* controls = default_alias.data() + (tail ? kSamples - 1 : 0);
        std::copy(defaults.begin(), defaults.end(), controls);
        call.defaults = controls;
        call.destination = reinterpret_cast<float*>(default_alias.data());
        rejects_without_write(call, default_alias.data(), default_alias.size() * sizeof(int32_t),
                              "destination overlaps profile defaults");
    }
}
}  // namespace

int main() {
    try {
        const auto source = source_pixels();
        const auto cases = validate_parity(source);
        validate_rejections(source);
        std::cout << "Prepared profile HSL CPU ABI passed: " << cases
                  << " bitwise composition cases, identity/Vibrance and failure-atomicity checks\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Prepared profile HSL CPU ABI validation failed: " << error.what() << '\n';
        return 1;
    }
}
