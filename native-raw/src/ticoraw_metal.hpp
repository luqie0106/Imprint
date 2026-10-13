/*
 * Nikon HE Bayer reconstruction Metal entry point.
 *
 * The integer reconstruction equations are a clean Metal reimplementation
 * of nikon_he_bayer.cpp by Dmitri Sotnikov (2026), from LibRaw. The reference
 * implementation is distributed under the LGPL-2.1 or CDDL-1.0; this
 * attribution is retained here for the derived reconstruction behavior.
 *
 * Build note: the Objective-C++ implementation embeds ticoraw.metal at build
 * time and expects generated/ticoraw_metal_shader.hpp to define the global
 * string variable `imprint_ticoraw_metal_source`. Runtime source-tree lookup is
 * intentionally not used so packaged builds contain the shader.
 */

#ifndef IMPRINT_TICORAW_METAL_HPP
#define IMPRINT_TICORAW_METAL_HPP

#include <cstdint>

// Nikon HE tone-curve constants used by step2_bayer_rows.
constexpr int kTicoRawLutShift = 2;
constexpr int kTicoRawLutBitWidth = 14;
constexpr int kTicoRawLutSize = 81792;
constexpr int kTicoRawClipMax = (1 << kTicoRawLutBitWidth) - 1;

// Reconstruct a complete Bayer image from four full-frame wavelet planes.
//
// Plane order follows the decoder: ll/lh/hh/hl. Each plane has `rows` rows,
// `cols` values per row, and the shared element stride `stride`. The function
// gathers strided planes into compact Metal shared buffers, runs the two
// integer reconstruction passes on Metal, then writes a contiguous
// (rows * 2) by (cols * 2) uint16 image to `out`.
//
// `lut` must contain at least kTicoRawLutSize signed 32-bit entries. Returns
// false for invalid inputs, unavailable Metal, shader/pipeline errors, or a
// failed command buffer. There is no CPU fallback.
extern "C" bool imprint_tico_reconstruct_metal(
    const int32_t* ll,
    const int32_t* lh,
    const int32_t* hh,
    const int32_t* hl,
    int rows,
    int cols,
    int stride,
    const int32_t* lut,
    int lut_size,
    uint16_t* out);

#endif  // IMPRINT_TICORAW_METAL_HPP
