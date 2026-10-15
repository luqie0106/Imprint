#pragma once

#include <array>

namespace imprint::acr {

enum class ColorSpecStatus : int {
    ok = 0,
    invalid_input = 1,
    singular_matrix = 2,
    non_finite_intermediate = 3,
    invalid_denominator = 4,
};

// Three-plane RGB inputs already resolved by the caller. Matrices are
// row-major. AB is represented by its three diagonal values. Empty Forward
// matrices are selected with the corresponding has_forward flag. Resolved
// white xy must be finite and physical (x >= 0, y > 0, x+y < 1), matching the
// surrounding white_temperature API's input domain; this is a caller boundary,
// not an input check recovered from Adobe's inner XY-to-XYZ helper.
struct ColorSpecInput {
    std::array<double, 9> raw_cm1{};
    std::array<double, 9> raw_cm2{};
    std::array<double, 9> cc1{};
    std::array<double, 9> cc2{};
    std::array<double, 3> analog_balance{};
    std::array<double, 9> forward1{};
    std::array<double, 9> forward2{};
    std::array<double, 2> white_xy{};
    double endpoint_kelvin1 = 0.0;
    double endpoint_kelvin2 = 0.0;
    bool has_second_cm = false;
    bool has_forward1 = false;
    bool has_forward2 = false;
};

struct ColorSpecPrepared {
    std::array<double, 3> camera_white{};
    // Camera -> PCS D50 XYZ.
    std::array<double, 9> camera_to_pcs{};
    // PCS D50 XYZ -> camera.
    std::array<double, 9> pcs_to_camera{};
    double mix_weight = 0.0;
};

struct WhiteResolveInput {
    // The profile's white_xy member is ignored. All remaining profile fields
    // carry the same prepared finite RGB inputs used by prepare_acr_colorspec.
    ColorSpecInput profile{};
    // Caller-prepared source-max-normalized and 1e6-quantized neutral. Finite
    // negative components are permitted; at least one component must be > 0.
    std::array<double, 3> camera_neutral{};
    bool output_referred_mode = false;
};

struct WhiteResolvePrepared {
    std::array<double, 2> neutral_xy{};
    std::array<double, 2> adjusted_xy{};
    ColorSpecPrepared colorspec{};
};

// Reconstruct the finite two-illuminant RGB ColorSpec constructor and
// SetWhiteXY chain. On failure output remains byte-for-byte unchanged. Input
// and output storage must not overlap. Matrix products use the recovered
// left-to-right scalar order with contraction disabled by the build; only the
// recovered explicit 3x3 inverse uses std::fma. Exact ARM64 matrix-product
// FMA last-bit parity is not claimed.
ColorSpecStatus prepare_acr_colorspec(const ColorSpecInput &input,
                                      ColorSpecPrepared &output) noexcept;

// Reconstruct the finite two-illuminant RGB NeutralToXY and AdjustResolvedXY
// path. The caller supplies the already prepared camera neutral and mode bit;
// no metadata, automatic-WB, or three-illuminant resolution is performed.
// Failure leaves output unchanged and overlapping input/output is rejected.
ColorSpecStatus resolve_acr_white(const WhiteResolveInput &input,
                                  WhiteResolvePrepared &output) noexcept;

// Build the recovered fixed ProPhoto matrix, normalize it against D50, and
// return its inverse. When requested, normalized_matrix receives Q. Both
// outputs remain unchanged on failure and may not overlap.
ColorSpecStatus prepare_normalized_prophoto_inverse(
    std::array<double, 9> &inverse_output,
    std::array<double, 9> *normalized_matrix = nullptr) noexcept;

}  // namespace imprint::acr
