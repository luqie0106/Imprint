#include "acr_stage3_core3_prefix.hpp"

#include "acr_stage3_core3.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace imprint::acr {
namespace {

constexpr std::size_t kGuardSamples = 4;
constexpr std::size_t kPlaneCount = 9;
constexpr std::size_t kWorkspaceSampleLimit = 64u * 1024u * 1024u;

bool checked_add(std::size_t left, std::size_t right,
                 std::size_t *result) noexcept {
    if (right > std::numeric_limits<std::size_t>::max() - left) return false;
    *result = left + right;
    return true;
}

bool checked_mul(std::size_t left, std::size_t right,
                 std::size_t *result) noexcept {
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
        return false;
    }
    *result = left * right;
    return true;
}

bool aligned_16(const void *pointer) noexcept {
    return (reinterpret_cast<std::uintptr_t>(pointer) & 0xFu) == 0;
}

float *plane_row(Core3PrefixWorkspace *workspace, std::size_t y,
                 std::size_t plane) noexcept {
    return workspace->samples.data() + y * workspace->row_stride +
           plane * workspace->plane_stride;
}

}  // namespace

Core3PrefixStatus run_core3_gradient_energy_prefix(
    const float *padded_input,
    std::size_t input_sample_count,
    std::size_t width,
    std::size_t height,
    std::size_t input_row_stride,
    Core3PrefixWorkspace *output) noexcept {
    if (padded_input == nullptr || output == nullptr) {
        return Core3PrefixStatus::invalid_argument;
    }
    if (width < 12 || (width & 3u) != 0 || height < 9) {
        return Core3PrefixStatus::invalid_dimensions;
    }

    std::size_t input_row_minimum = 0;
    std::size_t pitch = 0;
    if (!checked_add(width, 2 * kGuardSamples, &input_row_minimum) ||
        !checked_add(width, kGuardSamples, &pitch)) {
        return Core3PrefixStatus::arithmetic_overflow;
    }
    if (input_row_stride < input_row_minimum ||
        (input_row_stride & 3u) != 0) {
        return Core3PrefixStatus::invalid_stride;
    }

    std::size_t preceding_rows = 0;
    std::size_t required_input_samples = 0;
    if (!checked_mul(height - 1, input_row_stride, &preceding_rows) ||
        !checked_add(preceding_rows, input_row_minimum,
                     &required_input_samples)) {
        return Core3PrefixStatus::arithmetic_overflow;
    }
    if (input_sample_count < required_input_samples) {
        return Core3PrefixStatus::buffer_too_small;
    }

    std::size_t input_span_bytes = 0;
    if (!checked_mul(input_sample_count, sizeof(float), &input_span_bytes)) {
        return Core3PrefixStatus::arithmetic_overflow;
    }
    const std::uintptr_t input_address =
        reinterpret_cast<std::uintptr_t>(padded_input);
    const std::uintptr_t uintptr_max =
        std::numeric_limits<std::uintptr_t>::max();
    if (input_span_bytes > uintptr_max ||
        static_cast<std::uintptr_t>(input_span_bytes) >
            uintptr_max - input_address) {
        return Core3PrefixStatus::arithmetic_overflow;
    }
    const std::size_t ptrdiff_max = static_cast<std::size_t>(
        std::numeric_limits<std::ptrdiff_t>::max());
    if (input_sample_count > ptrdiff_max ||
        required_input_samples > ptrdiff_max ||
        input_row_stride > ptrdiff_max / 2) {
        return Core3PrefixStatus::arithmetic_overflow;
    }
    if (!aligned_16(padded_input)) {
        return Core3PrefixStatus::unsupported_alignment;
    }

    std::size_t row_stride = 0;
    std::size_t workspace_samples = 0;
    std::size_t workspace_bytes = 0;
    if (!checked_mul(kPlaneCount, pitch, &row_stride) ||
        !checked_mul(height, row_stride, &workspace_samples) ||
        !checked_mul(workspace_samples, sizeof(float), &workspace_bytes)) {
        return Core3PrefixStatus::arithmetic_overflow;
    }
    if (row_stride > ptrdiff_max / 2 || workspace_samples > ptrdiff_max) {
        return Core3PrefixStatus::arithmetic_overflow;
    }
    if (workspace_samples > kWorkspaceSampleLimit) {
        return Core3PrefixStatus::resource_limit;
    }
    if (workspace_samples > std::vector<float>().max_size()) {
        return Core3PrefixStatus::resource_limit;
    }
    (void)workspace_bytes;  // The checked byte count documents allocation size.

    Core3PrefixWorkspace candidate;
    candidate.width = width;
    candidate.height = height;
    candidate.pitch = pitch;
    candidate.plane_stride = pitch;
    candidate.row_stride = row_stride;
    candidate.gradient_valid = {2, 0, height - 2, width};
    candidate.energy_valid = {4, 4, height - 4, width - 4};

    try {
        candidate.samples.assign(workspace_samples,
                                 std::numeric_limits<float>::quiet_NaN());
    } catch (const std::length_error &) {
        return Core3PrefixStatus::resource_limit;
    } catch (const std::bad_alloc &) {
        return Core3PrefixStatus::allocation_failure;
    }
    if (!aligned_16(candidate.samples.data())) {
        return Core3PrefixStatus::unsupported_alignment;
    }

    for (std::size_t y = 2; y < height - 2; ++y) {
        float *gradient_outputs[6] = {
            plane_row(&candidate, y, 0), plane_row(&candidate, y, 1),
            plane_row(&candidate, y, 2), plane_row(&candidate, y, 5),
            plane_row(&candidate, y, 3), plane_row(&candidate, y, 4),
        };
        const Core3Status status = core3_gradient_rows_0081376c(
            padded_input + y * input_row_stride + kGuardSamples,
            input_row_stride, width, gradient_outputs, row_stride);
        if (status != Core3Status::ok) {
            return Core3PrefixStatus::kernel_failure;
        }
    }

    for (std::size_t y = 4; y < height - 4; ++y) {
        const Core3Status status = core3_local_energy_rows_00813368(
            plane_row(&candidate, y, 3) + 4,
            plane_row(&candidate, y, 4) + 4, row_stride, width - 8,
            plane_row(&candidate, y, 7) + 4,
            plane_row(&candidate, y, 8) + 4);
        if (status != Core3Status::ok) {
            return Core3PrefixStatus::kernel_failure;
        }
    }

    static_assert(std::is_nothrow_move_assignable<Core3PrefixWorkspace>::value,
                  "workspace commit must not throw");
    *output = std::move(candidate);
    return Core3PrefixStatus::ok;
}

}  // namespace imprint::acr
