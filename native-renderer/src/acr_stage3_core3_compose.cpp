#include "acr_stage3_core3_compose.hpp"

#include "acr_stage3_core3_colour.hpp"
#include "acr_stage3_core3_format.hpp"
#include "acr_stage3_core3_refine.hpp"

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace imprint::acr {
namespace {

constexpr std::size_t kPlaneCount = 9;
constexpr std::size_t kWorkspaceFloatLimit = 64u * 1024u * 1024u;

static_assert(sizeof(float) == sizeof(std::uint32_t) &&
                  std::numeric_limits<float>::is_iec559 &&
                  std::numeric_limits<float>::digits == 24,
              "core3 controls require IEEE-754 binary32 float");

float float_from_bits(std::uint32_t bits) noexcept {
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

float sub_f32(float left, float right) noexcept {
    volatile float rounded = left - right;
    return rounded;
}

float div_f32(float numerator, float denominator) noexcept {
    volatile float rounded = numerator / denominator;
    return rounded;
}

float clamp01_fminnm(float value) noexcept {
    // The recovered sequence is FMINNM(value, 1), then FCSEL MI, +0, value.
    // In particular, the strict comparison preserves a resulting -0.
    const float upper_clamped = std::fmin(value, 1.0f);
    return upper_clamped < 0.0f ? 0.0f : upper_clamped;
}

bool checked_add(std::size_t a, std::size_t b,
                 std::size_t *out) noexcept {
    if (b > std::numeric_limits<std::size_t>::max() - a) return false;
    *out = a + b;
    return true;
}

bool checked_mul(std::size_t a, std::size_t b,
                 std::size_t *out) noexcept {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
        return false;
    }
    *out = a * b;
    return true;
}

bool checked_allocation_count(std::size_t count, std::size_t item_size,
                              std::size_t *bytes) noexcept {
    const std::size_t size_limit = std::numeric_limits<std::size_t>::max();
    const auto ptrdiff_limit = static_cast<std::uintmax_t>(
        std::numeric_limits<std::ptrdiff_t>::max());
    if (item_size == 0 || count > size_limit / item_size ||
        static_cast<std::uintmax_t>(count) > ptrdiff_limit / item_size) {
        return false;
    }
    *bytes = count * item_size;
    return true;
}

bool checked_address_span(const void *pointer, std::size_t count,
                         std::size_t item_size) noexcept {
    std::size_t bytes = 0;
    if (pointer == nullptr || !checked_allocation_count(count, item_size,
                                                        &bytes)) {
        return false;
    }
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    return bytes <= std::numeric_limits<std::uintptr_t>::max() - begin;
}

bool aligned_16(const void *pointer) noexcept {
    return (reinterpret_cast<std::uintptr_t>(pointer) & 0xFu) == 0;
}

float *plane_row(Core3PrefixWorkspace *workspace, std::size_t y,
                 std::size_t plane) noexcept {
    return workspace->samples.data() + y * workspace->row_stride +
           plane * workspace->plane_stride;
}

std::uint32_t row_parity(const Core3ComposeControls &controls,
                         std::size_t y) noexcept {
    return (controls.row_origin_parity +
            static_cast<std::uint32_t>(y & 1u)) & 1u;
}

std::uint32_t phase_at(const Core3ComposeControls &controls,
                       std::size_t y, std::size_t x) noexcept {
    return (controls.row_origin_parity + controls.column_origin_parity +
            static_cast<std::uint32_t>(y & 1u) +
            static_cast<std::uint32_t>(x & 1u)) & 1u;
}

Core3ComposeStatus map_prefix_status(Core3PrefixStatus status) noexcept {
    switch (status) {
        case Core3PrefixStatus::ok:
            return Core3ComposeStatus::ok;
        case Core3PrefixStatus::invalid_argument:
            return Core3ComposeStatus::invalid_argument;
        case Core3PrefixStatus::invalid_dimensions:
            return Core3ComposeStatus::invalid_dimensions;
        case Core3PrefixStatus::invalid_stride:
            return Core3ComposeStatus::invalid_stride;
        case Core3PrefixStatus::buffer_too_small:
            return Core3ComposeStatus::buffer_too_small;
        case Core3PrefixStatus::unsupported_alignment:
            return Core3ComposeStatus::unsupported_alignment;
        case Core3PrefixStatus::resource_limit:
            return Core3ComposeStatus::resource_limit;
        case Core3PrefixStatus::allocation_failure:
            return Core3ComposeStatus::allocation_failure;
        case Core3PrefixStatus::arithmetic_overflow:
            return Core3ComposeStatus::arithmetic_overflow;
        case Core3PrefixStatus::kernel_failure:
            return Core3ComposeStatus::kernel_failure;
    }
    return Core3ComposeStatus::kernel_failure;
}

}  // namespace

Core3ComposeStatus prepare_core3_controls_007f2124(
    double amount, bool suppress_refine,
    std::uint32_t row_origin_parity,
    std::uint32_t column_origin_parity,
    Core3ComposeControls *out) noexcept {
    const auto address = reinterpret_cast<std::uintptr_t>(out);
    if (out == nullptr || address % alignof(Core3ComposeControls) != 0u ||
        sizeof(*out) > std::numeric_limits<std::uintptr_t>::max() - address) {
        return Core3ComposeStatus::invalid_argument;
    }
    if (!std::isfinite(amount) ||
        std::fabs(amount) > static_cast<double>(
                                std::numeric_limits<float>::max())) {
        return Core3ComposeStatus::invalid_parameters;
    }
    if (row_origin_parity > 1u || column_origin_parity > 1u) {
        return Core3ComposeStatus::invalid_phase;
    }

    const float n = static_cast<float>(amount);
    const float a = sub_f32(n, 1.0f);
    const float b = sub_f32(n, 12.0f);
    const float x = clamp01_fminnm(div_f32(a, 5.0f));
    const float y = clamp01_fminnm(div_f32(b, -6.0f));
    const float m = suppress_refine ? 0.0f : (x < y ? x : y);
    const float q_ratio = clamp01_fminnm(
        div_f32(a, float_from_bits(0x40accccdU)));
    const float q = std::fma(q_ratio, float_from_bits(0xbecccccdU), 0.5f);

    const Core3ComposeControls candidate{
        {float_from_bits(0x41b40000u), float_from_bits(0x3cb60b61u), m, q},
        float_from_bits(0x3c3b80bcu),
        float_from_bits(0x3cbb80bcu),
        row_origin_parity,
        column_origin_parity,
    };
    *out = candidate;
    return Core3ComposeStatus::ok;
}

namespace {

template <typename SourceSample>
Core3ComposeStatus run_core3_prepared_tile(
    const SourceSample *src, std::size_t src_sample_count,
    std::size_t width, std::size_t height, std::size_t input_row_stride,
    const Core3ComposeControls &controls,
    Core3ComposeOutput *output) noexcept {
    if (src == nullptr || output == nullptr) {
        return Core3ComposeStatus::invalid_argument;
    }
    if (controls.row_origin_parity > 1u ||
        controls.column_origin_parity > 1u) {
        return Core3ComposeStatus::invalid_phase;
    }
    if (!std::isfinite(controls.green.t) ||
        !std::isfinite(controls.green.k) ||
        !std::isfinite(controls.green.m) ||
        !std::isfinite(controls.green.q) || controls.green.q < 0.0f ||
        !std::isfinite(controls.lower) ||
        !std::isfinite(controls.upper) || controls.lower < 0.0f ||
        controls.upper < 0.0f) {
        return Core3ComposeStatus::invalid_parameters;
    }

    const bool refine_enabled = controls.green.m != 0.0f;
    const std::size_t margin = refine_enabled ? 12u : 8u;
    std::size_t minimum_width = 0;
    std::size_t minimum_height = 0;
    if (!checked_mul(margin, 2u, &minimum_width) ||
        !checked_add(minimum_width, 4u, &minimum_width) ||
        !checked_mul(margin, 2u, &minimum_height) ||
        !checked_add(minimum_height, 1u, &minimum_height)) {
        return Core3ComposeStatus::arithmetic_overflow;
    }
    if ((width & 3u) != 0 || width < minimum_width ||
        height < minimum_height) {
        return Core3ComposeStatus::invalid_dimensions;
    }
    const std::size_t stride_alignment =
        std::is_same<SourceSample, std::uint16_t>::value ? 8u : 4u;
    if (input_row_stride < width ||
        input_row_stride % stride_alignment != 0) {
        return Core3ComposeStatus::invalid_stride;
    }

    std::size_t preceding_source_rows = 0;
    std::size_t required_source_samples = 0;
    if (!checked_mul(height - 1u, input_row_stride,
                     &preceding_source_rows) ||
        !checked_add(preceding_source_rows, width,
                     &required_source_samples)) {
        return Core3ComposeStatus::arithmetic_overflow;
    }
    if (src_sample_count < required_source_samples) {
        return Core3ComposeStatus::buffer_too_small;
    }
    if (!checked_address_span(src, src_sample_count, sizeof(*src))) {
        return Core3ComposeStatus::arithmetic_overflow;
    }
    if (!aligned_16(src)) return Core3ComposeStatus::unsupported_alignment;

    std::size_t prefix_pitch = 0;
    std::size_t prefix_row_stride = 0;
    std::size_t workspace_samples = 0;
    std::size_t workspace_bytes = 0;
    if (!checked_add(width, 4u, &prefix_pitch) ||
        !checked_mul(kPlaneCount, prefix_pitch, &prefix_row_stride) ||
        !checked_mul(height, prefix_row_stride, &workspace_samples) ||
        !checked_allocation_count(workspace_samples, sizeof(float),
                                  &workspace_bytes)) {
        return Core3ComposeStatus::arithmetic_overflow;
    }
    (void)workspace_bytes;
    if (workspace_samples > kWorkspaceFloatLimit) {
        return Core3ComposeStatus::resource_limit;
    }

    const std::size_t output_width = width - 2u * margin;
    const std::size_t output_height = height - 2u * margin;
    std::size_t output_pixels = 0;
    std::size_t output_components = 0;
    std::size_t output_float_bytes = 0;
    std::size_t output_u16_bytes = 0;
    std::size_t padded_input_stride = 0;
    std::size_t padded_input_samples = 0;
    std::size_t padded_input_bytes = 0;
    if (!checked_mul(output_width, output_height, &output_pixels) ||
        !checked_mul(output_pixels, 3u, &output_components) ||
        !checked_allocation_count(output_components, sizeof(float),
                                  &output_float_bytes) ||
        !checked_allocation_count(output_components, sizeof(std::uint16_t),
                                  &output_u16_bytes) ||
        !checked_add(width, 8u, &padded_input_stride) ||
        !checked_mul(height, padded_input_stride, &padded_input_samples) ||
        !checked_allocation_count(padded_input_samples, sizeof(float),
                                  &padded_input_bytes)) {
        return Core3ComposeStatus::arithmetic_overflow;
    }
    (void)output_float_bytes;
    (void)output_u16_bytes;
    (void)padded_input_bytes;

    if (output_components > std::vector<float>().max_size() ||
        output_components > std::vector<std::uint16_t>().max_size() ||
        padded_input_samples > std::vector<float>().max_size()) {
        return Core3ComposeStatus::resource_limit;
    }

    std::size_t padded_output_u16_stride = 0;
    if (!checked_add(output_width, 7u, &padded_output_u16_stride)) {
        return Core3ComposeStatus::arithmetic_overflow;
    }
    padded_output_u16_stride &= ~std::size_t(7u);
    std::size_t padded_output_u16_bytes = 0;
    if (!checked_allocation_count(padded_output_u16_stride,
                                  sizeof(std::uint16_t),
                                  &padded_output_u16_bytes)) {
        return Core3ComposeStatus::arithmetic_overflow;
    }
    (void)padded_output_u16_bytes;

    try {
        std::vector<float> padded_input(
            padded_input_samples, std::numeric_limits<float>::quiet_NaN());
        if (!aligned_16(padded_input.data())) {
            return Core3ComposeStatus::unsupported_alignment;
        }
        for (std::size_t y = 0; y < height; ++y) {
            const std::size_t source_offset = y * input_row_stride;
            float *destination_row =
                padded_input.data() + y * padded_input_stride + 4u;
            if constexpr (std::is_same<SourceSample,
                                       std::uint16_t>::value) {
                const Core3FormatStatus format_status =
                    core3_u16_to_f32_prepared_row_008096e4(
                        src + source_offset,
                        src_sample_count - source_offset, destination_row,
                        padded_input_stride - 4u, width);
                if (format_status != Core3FormatStatus::ok) {
                    return format_status ==
                                   Core3FormatStatus::unsupported_alignment
                               ? Core3ComposeStatus::unsupported_alignment
                               : format_status ==
                                         Core3FormatStatus::arithmetic_overflow
                                     ? Core3ComposeStatus::arithmetic_overflow
                                     : Core3ComposeStatus::kernel_failure;
                }
            } else {
                const SourceSample *source_row = src + source_offset;
                for (std::size_t x = 0; x < width; ++x) {
                    const float sample = source_row[x];
                    if (!std::isfinite(sample)) {
                        return Core3ComposeStatus::unsupported_domain;
                    }
                    destination_row[x] = sample;
                }
            }
        }

        Core3PrefixWorkspace workspace;
        const Core3ComposeStatus prefix_status = map_prefix_status(
            run_core3_gradient_energy_prefix(
                padded_input.data(), padded_input.size(), width, height,
                padded_input_stride, &workspace));
        if (prefix_status != Core3ComposeStatus::ok) return prefix_status;

        // Stage 1: aligned green rows at logical x=4. P4/P6 retain NaN outside
        // the rows and columns claimed by the recovered caller schedule.
        for (std::size_t y = 7u; y < height - 7u; ++y) {
            const Core3GreenStatus status =
                core3_green_prepared_float_row_0080f2b8(
                    padded_input.data() + y * padded_input_stride + 8u,
                    plane_row(&workspace, y, 0) + 4u,
                    plane_row(&workspace, y, 1) + 4u,
                    plane_row(&workspace, y, 2) + 4u,
                    plane_row(&workspace, y, 5) + 4u,
                    plane_row(&workspace, y, 7) + 4u,
                    plane_row(&workspace, y, 8) + 4u,
                    padded_input_stride, workspace.row_stride, width - 8u,
                    phase_at(controls, y, 4u), controls.green,
                    plane_row(&workspace, y, 4) + 4u,
                    refine_enabled ? plane_row(&workspace, y, 6) + 4u
                                   : nullptr);
            if (status != Core3GreenStatus::ok) {
                return Core3ComposeStatus::kernel_failure;
            }
        }

        // Stage 2: write the row-dependent A/B plane mapping selected by the
        // canonical row origin parity. P4 is the green plane from Stage 1.
        for (std::size_t y = 8u; y < height - 8u; ++y) {
            const bool even_row = row_parity(controls, y) == 0u;
            float *out_a = plane_row(&workspace, y, even_row ? 3u : 5u) + 8u;
            float *out_b = plane_row(&workspace, y, even_row ? 5u : 3u) + 8u;
            const Core3ColourStatus status =
                core3_colour_difference_row_008134b8(
                    padded_input.data() + y * padded_input_stride + 12u,
                    plane_row(&workspace, y, 4) + 8u,
                    padded_input_stride, workspace.row_stride, width - 16u,
                    phase_at(controls, y, 8u), out_a, out_b);
            if (status != Core3ColourStatus::ok) {
                return Core3ComposeStatus::kernel_failure;
            }
        }

        // Stage 3: the first refine writes only its own, separately mapped
        // destination planes. Its source rows remain live until every call is
        // complete.
        if (refine_enabled) {
            for (std::size_t y = 10u; y < height - 10u; ++y) {
                const bool even_row = row_parity(controls, y) == 0u;
                RefineInputs inputs;
                inputs.guide = plane_row(&workspace, y, even_row ? 3u : 5u) + 8u;
                inputs.channel_a = plane_row(&workspace, y, 4) + 8u;
                inputs.channel_b = plane_row(&workspace, y, even_row ? 5u : 3u) + 8u;
                inputs.mask = plane_row(&workspace, y, 6) + 8u;
                inputs.stride = workspace.row_stride;
                inputs.width = width - 16u;
                inputs.phase = phase_at(controls, y, 8u);
                inputs.lower = controls.lower;
                inputs.upper = controls.upper;
                RefineOutputs outputs;
                outputs.guide = plane_row(&workspace, y, even_row ? 0u : 2u) + 8u;
                outputs.a = plane_row(&workspace, y, 1) + 8u;
                outputs.b = plane_row(&workspace, y, even_row ? 2u : 0u) + 8u;
                if (core3_refine_prepared_float_row_00814230(inputs, outputs) !=
                    Core3RefineStatus::ok) {
                    return Core3ComposeStatus::kernel_failure;
                }
            }
        }

        Core3ComposeOutput candidate;
        candidate.width = width;
        candidate.height = height;
        candidate.valid_region = {margin, margin, height - margin,
                                  width - margin};
        candidate.output_width = output_width;
        candidate.output_height = output_height;
        candidate.rgb.resize(output_components);
        candidate.rgb_u16.resize(output_components);

        // Row-local outputs are separate from the workspace planes. In
        // particular, the second refine must not overwrite P0/P1/P2 while
        // neighboring spatial rows still read them.
        std::vector<float> refined_guide(output_width);
        std::vector<float> refined_a(output_width);
        std::vector<float> refined_b(output_width);
        std::vector<std::uint16_t> encoded_r(padded_output_u16_stride);
        std::vector<std::uint16_t> encoded_g(padded_output_u16_stride);
        std::vector<std::uint16_t> encoded_b(padded_output_u16_stride);
        if (!aligned_16(refined_guide.data()) || !aligned_16(refined_a.data()) ||
            !aligned_16(refined_b.data()) || !aligned_16(encoded_r.data()) ||
            !aligned_16(encoded_g.data()) || !aligned_16(encoded_b.data())) {
            return Core3ComposeStatus::unsupported_alignment;
        }

        for (std::size_t output_y = 0; output_y < output_height; ++output_y) {
            const std::size_t y = output_y + margin;
            const float *red_row = nullptr;
            const float *green_row = nullptr;
            const float *blue_row = nullptr;
            if (refine_enabled) {
                RefineInputs inputs;
                inputs.guide = plane_row(&workspace, y, 1) + 12u;
                inputs.channel_a = plane_row(&workspace, y, 0) + 12u;
                inputs.channel_b = plane_row(&workspace, y, 2) + 12u;
                inputs.mask = plane_row(&workspace, y, 6) + 12u;
                inputs.stride = workspace.row_stride;
                inputs.width = output_width;
                inputs.phase = phase_at(controls, y, 12u);
                inputs.lower = controls.lower;
                inputs.upper = controls.upper;
                const RefineOutputs outputs{refined_guide.data(),
                                            refined_a.data(),
                                            refined_b.data()};
                if (core3_refine_prepared_float_row_008145b4(inputs, outputs) !=
                    Core3RefineStatus::ok) {
                    return Core3ComposeStatus::kernel_failure;
                }
                red_row = refined_a.data();
                green_row = refined_guide.data();
                blue_row = refined_b.data();
            } else {
                red_row = plane_row(&workspace, y, 3) + margin;
                green_row = plane_row(&workspace, y, 4) + margin;
                blue_row = plane_row(&workspace, y, 5) + margin;
            }

            const Core3FormatStatus red_status =
                core3_f32_to_u16_prepared_row_008103dc(
                    red_row, output_width, encoded_r.data(),
                    encoded_r.size(), output_width);
            const Core3FormatStatus green_status =
                core3_f32_to_u16_prepared_row_008103dc(
                    green_row, output_width, encoded_g.data(),
                    encoded_g.size(), output_width);
            const Core3FormatStatus blue_status =
                core3_f32_to_u16_prepared_row_008103dc(
                    blue_row, output_width, encoded_b.data(),
                    encoded_b.size(), output_width);
            if (red_status == Core3FormatStatus::unsupported_domain ||
                green_status == Core3FormatStatus::unsupported_domain ||
                blue_status == Core3FormatStatus::unsupported_domain) {
                return Core3ComposeStatus::unsupported_domain;
            }
            if (red_status != Core3FormatStatus::ok ||
                green_status != Core3FormatStatus::ok ||
                blue_status != Core3FormatStatus::ok) {
                return Core3ComposeStatus::kernel_failure;
            }

            const std::size_t row_offset = output_y * output_width * 3u;
            for (std::size_t x = 0; x < output_width; ++x) {
                const std::size_t index = row_offset + x * 3u;
                candidate.rgb[index] = red_row[x];
                candidate.rgb[index + 1u] = green_row[x];
                candidate.rgb[index + 2u] = blue_row[x];
                candidate.rgb_u16[index] = encoded_r[x];
                candidate.rgb_u16[index + 1u] = encoded_g[x];
                candidate.rgb_u16[index + 2u] = encoded_b[x];
            }
        }

        static_assert(std::is_nothrow_move_assignable<Core3ComposeOutput>::value,
                      "output commit must not throw");
        *output = std::move(candidate);
        return Core3ComposeStatus::ok;
    } catch (const std::length_error &) {
        return Core3ComposeStatus::resource_limit;
    } catch (const std::bad_alloc &) {
        return Core3ComposeStatus::allocation_failure;
    }
}

}  // namespace

Core3ComposeStatus run_core3_prepared_u16_tile(
    const std::uint16_t *src, std::size_t src_sample_count,
    std::size_t width, std::size_t height, std::size_t input_row_stride,
    const Core3ComposeControls &controls,
    Core3ComposeOutput *output) noexcept {
    return run_core3_prepared_tile(src, src_sample_count, width, height,
                                   input_row_stride, controls, output);
}

Core3ComposeStatus run_core3_prepared_f32_tile(
    const float *src, std::size_t src_sample_count,
    std::size_t width, std::size_t height, std::size_t input_row_stride,
    const Core3ComposeControls &controls,
    Core3ComposeOutput *output) noexcept {
    return run_core3_prepared_tile(src, src_sample_count, width, height,
                                   input_row_stride, controls, output);
}

}  // namespace imprint::acr
