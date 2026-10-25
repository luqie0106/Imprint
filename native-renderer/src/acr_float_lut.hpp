#pragma once

#include <array>
#include <cstddef>

namespace imprint::acr {

constexpr std::size_t kFloatLutSegments = 4096;
constexpr std::size_t kFloatLutEntryCount = kFloatLutSegments + 2;

struct FloatLutPrepared {
    std::array<float, kFloatLutEntryCount> values;
};

enum class FloatLutStatus : int {
    ok = 0,
    invalid_argument = 1,
    invalid_parameters = 2,
    overlap = 3,
    evaluation_failure = 4,
};

using FloatLutEvaluator = bool (*)(double, const void*, double*) noexcept;

// Recovered 01300490 fixed 4096-segment LUT builder. The evaluator and its
// context are owned by the caller and must remain valid for the duration of
// this call. The context must not overlap `out`; because its extent is unknown,
// that contract cannot be checked here. `out` must be naturally aligned and
// its object span must not wrap the address space. Every failure leaves `out`
// unchanged. Callback failures and non-finite/unrepresentable values return
// evaluation_failure.
FloatLutStatus build_float_lut_01300490(
    FloatLutEvaluator evaluator, const void* context,
    FloatLutPrepared* out) noexcept;

// Recovered 00875edc scalar row application. Only the first `width` samples
// are read or written; counts are capacities and their unused tails are not
// accessed. `source` may equal `destination` for in-place processing, otherwise
// their processed spans must be disjoint. The LUT must not overlap the
// destination span. All pointers must be naturally aligned and every used
// span must not wrap the address space. The LUT must contain finite values and
// bit-identical entries at 4096 and 4097. Unsigned inputs are accepted when
// f32(sample * 4096) lies strictly inside (-1, 4097), matching the recovered
// truncating int32 index domain. Inputs, arithmetic intermediates, and outputs
// are checked before any destination write, so failures are atomic.
// This API implements only the float scalar path; it does not implement the
// separate uint16 LUT/task path.
FloatLutStatus apply_float_lut_row_00875edc(
    const FloatLutPrepared& lut, const float* source,
    std::size_t source_count, float* destination,
    std::size_t destination_count, std::size_t width,
    bool signed_extension) noexcept;

}  // namespace imprint::acr
