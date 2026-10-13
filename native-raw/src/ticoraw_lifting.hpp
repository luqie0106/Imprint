/*
 * Imprint's C++ TicoRAW inverse wavelet and Bayer reconstruction operators.
 * The lifting/state equations are adapted from redlamp-libraw's Nikon HE
 * implementation, distributed under LGPL-2.1 or CDDL-1.0.
 */
#ifndef IMPRINT_TICORAW_LIFTING_HPP
#define IMPRINT_TICORAW_LIFTING_HPP

#include <cstdint>

namespace nikon_he {
struct SubbandConfig;
struct LayoutInfo;
}

namespace imprint_tico {

enum class VerticalState : uint8_t {
    Init0 = 0,
    Init1 = 1,
    Init4 = 4,
    Init2 = 2,
    State5 = 5,
    State7 = 7,
    State8 = 8,
    State9 = 9,
    State11 = 11,
};

struct VerticalLiftState {
    VerticalState state = VerticalState::Init0;
    int32_t* x2_carry = nullptr;
    int32_t* x3_carry = nullptr;
    bool failed = false;
};

// Entropy scatter -> one of the two horizontal inverse-wavelet passes.
bool horizontal_pass(const int32_t* input,
                     const nikon_he::SubbandConfig config[26],
                     bool pass_a,
                     int32_t* output,
                     int32_t* work) noexcept;

// One vertical lifting state transition. A null input is only valid for the
// final two flush states (State7 and State9).
void vertical_lift_step(const int32_t* input,
                        int32_t* output,
                        int count,
                        VerticalLiftState& state) noexcept;

// CPU fallback matching the full-frame Metal reconstruction ABI.
bool reconstruct_cpu(const int32_t* ll,
                     const int32_t* lh,
                     const int32_t* hh,
                     const int32_t* hl,
                     int rows,
                     int cols,
                     int stride,
                     const int32_t* lut,
                     int lut_size,
                     uint16_t* output) noexcept;

}  // namespace imprint_tico

#endif  // IMPRINT_TICORAW_LIFTING_HPP
