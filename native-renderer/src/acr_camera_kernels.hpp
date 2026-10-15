#pragma once

#include <array>
#include <cstddef>

namespace imprint::acr {

struct CameraFloatParams {
    // Per-channel gains, sorted by the caller in descending order. The input
    // channels must already use that same order; this kernel does not sort.
    float gain_max;     // A
    float gain_middle;  // B
    float upper;        // U
    float scale;        // S

    // Row-major 3x3 matrix. Columns correspond to the caller-sorted input
    // channels, and rows are the fixed output ProPhoto planes. In the recovered
    // scalar ABI, row 2 and row 3 coefficients were packed into stack values;
    // callers must unpack them into this order before invoking the kernel.
    std::array<float, 9> matrix;
};

// Applies the recovered scalar camera component.
//
// Input: `pixels` interleaved triples in caller-sorted channel order. Values
// must be finite and nonnegative; values above 1 are allowed.
// Output: interleaved RGB triples in fixed ProPhoto channel semantics. Matrix
// rows 0, 1, and 2 produce R, G, and B respectively; this kernel does not
// reorder those output channels.
//
// Supported parameter domain: finite A >= B >= 1, U in
// [32768/65535, 1], finite S > 0, and finite matrix coefficients. The recovered
// function itself does not validate these parameter relationships; they are a
// conservative API precondition. Every float intermediate is checked for
// overflow. On invalid input or overflow, this throws and leaves `output`
// unchanged. Compile this translation unit with FP contraction disabled for
// the intended non-FMA operation order. This is a scalar reconstruction, not
// a claim of bitwise equivalence to Adobe's runtime implementation.
void apply_camera_float(const float *sorted_input_rgb, float *output,
                        std::size_t pixels, const CameraFloatParams &params);

}  // namespace imprint::acr
