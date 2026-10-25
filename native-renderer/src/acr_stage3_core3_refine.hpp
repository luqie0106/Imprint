#pragma once

#include <cstddef>
#include <cstdint>

namespace imprint::acr {

enum class Core3RefineStatus : int {
    ok = 0,
    invalid_argument = 1,
    invalid_dimensions = 2,
    unsupported_alignment = 3,
    invalid_phase = 4,
    invalid_parameters = 5,
    arithmetic_overflow = 6,
};

// Caller-owned prepared-float planes. Each pointer addresses column zero of
// the center row. Width is positive and divisible by four; stride is in float
// samples, is at least width, and is divisible by four. Every pointer must be
// 16-byte aligned. The caller guarantees capacities and non-overlap among all
// input and output ranges, plus the guards required by the selected row kernel.
// `lower` and `upper` must be finite and nonnegative. Rejected calls do not
// write outputs.
//
// This API covers only the aligned prepared-float subset and writes exactly
// `width` output samples. Original pointer-alignment backup and tail-store
// behavior is not implemented, so this is not a full ABI replica.
struct RefineInputs {
    const float *guide = nullptr;
    const float *channel_a = nullptr;
    const float *channel_b = nullptr;
    const float *mask = nullptr;
    std::size_t stride = 0;  // float samples
    std::size_t width = 0;
    std::uint32_t phase = 0;
    float lower = 0.0f;
    float upper = 0.0f;
};

struct RefineOutputs {
    float *guide = nullptr;
    float *a = nullptr;
    float *b = nullptr;
};

// FUN_00814230: copies the guide row bit-for-bit, and conditionally refines
// channels A and B using the center mask. Phase 0 refines even columns and
// copies odd columns; phase 1 refines odd columns and copies even columns.
// Reads guide +/-2 rows/columns and channel +/-1 rows/columns.
Core3RefineStatus core3_refine_prepared_float_row_00814230(
    const RefineInputs &inputs, const RefineOutputs &outputs) noexcept;

// FUN_008145b4: conditionally refines channels A and B using the four-neighbor
// mask mean. Phase 0 refines odd columns and copies even columns; phase 1
// refines even columns and copies odd columns. Reads guide +/-2 rows/columns,
// channel +/-1 rows/columns, and mask +/-1 rows/columns.
Core3RefineStatus core3_refine_prepared_float_row_008145b4(
    const RefineInputs &inputs, const RefineOutputs &outputs) noexcept;

}  // namespace imprint::acr
