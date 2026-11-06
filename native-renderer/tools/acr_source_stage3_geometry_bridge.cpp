#include "../src/acr_source_stage3_geometry.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace {

constexpr std::size_t kRationalWordCount = 12U;
constexpr std::size_t kRationalBytes = kRationalWordCount * sizeof(std::uint32_t);
constexpr std::size_t kScaleCount = 2U;
constexpr std::size_t kScaleBytes = kScaleCount * sizeof(double);

bool is_aligned(const void *pointer, std::size_t alignment) noexcept {
    return reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0U;
}

bool make_end(std::uintptr_t start, std::size_t size,
              std::uintptr_t &end) noexcept {
    const auto max_address = std::numeric_limits<std::uintptr_t>::max();
    if (start > max_address - size) return false;
    end = start + size;
    return true;
}

bool ranges_overlap(std::uintptr_t first_start, std::uintptr_t first_end,
                    std::uintptr_t second_start,
                    std::uintptr_t second_end) noexcept {
    return first_start < second_end && second_start < first_end;
}

}  // namespace

extern "C" int acr_source_stage3_geometry_bridge(
    const std::uint32_t *rationals12, const double *scales2,
    std::uint32_t *out_rationals12, double *out_scales2) {
    if (rationals12 == nullptr || scales2 == nullptr ||
        out_rationals12 == nullptr || out_scales2 == nullptr) {
        return 3;
    }
    if (!is_aligned(rationals12, alignof(std::uint32_t)) ||
        !is_aligned(scales2, alignof(double)) ||
        !is_aligned(out_rationals12, alignof(std::uint32_t)) ||
        !is_aligned(out_scales2, alignof(double))) {
        return 3;
    }

    const auto rational_output_start =
        reinterpret_cast<std::uintptr_t>(out_rationals12);
    const auto scale_output_start = reinterpret_cast<std::uintptr_t>(out_scales2);
    std::uintptr_t rational_output_end = 0U;
    std::uintptr_t scale_output_end = 0U;
    if (!make_end(rational_output_start, kRationalBytes,
                  rational_output_end) ||
        !make_end(scale_output_start, kScaleBytes, scale_output_end) ||
        ranges_overlap(rational_output_start, rational_output_end,
                       scale_output_start, scale_output_end)) {
        return 3;
    }

    // Copy both complete input spans before any output write; input/output
    // aliasing is permitted by this test-only bridge.
    std::uint32_t rational_words[kRationalWordCount]{};
    double scale_values[kScaleCount]{};
    std::memcpy(rational_words, rationals12, kRationalBytes);
    std::memcpy(scale_values, scales2, kScaleBytes);

    imprint::acr::PreparedSourceStage3Geometry input{};
    input.rational_30 = {rational_words[0], rational_words[1]};
    input.rational_38 = {rational_words[2], rational_words[3]};
    input.rational_40 = {rational_words[4], rational_words[5]};
    input.rational_48 = {rational_words[6], rational_words[7]};
    input.rational_90 = {rational_words[8], rational_words[9]};
    input.rational_98 = {rational_words[10], rational_words[11]};
    input.scale_x = scale_values[0];
    input.scale_y = scale_values[1];

    imprint::acr::PreparedSourceStage3Geometry output{};
    const auto status =
        imprint::acr::normalize_source_stage3_geometry(input, output);
    if (status != imprint::acr::SourceStage3GeometryStatus::ok) {
        return static_cast<int>(status);
    }

    const std::uint32_t output_rational_words[kRationalWordCount] = {
        output.rational_30.numerator, output.rational_30.denominator,
        output.rational_38.numerator, output.rational_38.denominator,
        output.rational_40.numerator, output.rational_40.denominator,
        output.rational_48.numerator, output.rational_48.denominator,
        output.rational_90.numerator, output.rational_90.denominator,
        output.rational_98.numerator, output.rational_98.denominator,
    };
    const double output_scale_values[kScaleCount] = {
        output.scale_x, output.scale_y,
    };
    std::memcpy(out_rationals12, output_rational_words, kRationalBytes);
    std::memcpy(out_scales2, output_scale_values, kScaleBytes);
    return static_cast<int>(status);
}
