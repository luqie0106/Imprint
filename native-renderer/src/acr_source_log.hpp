#pragma once

#include <cstddef>

namespace imprint::acr {

enum class SourceLogStatus : int {
    ok = 0,
    invalid_input = 1,
    allocation_failure = 2,
    numeric_failure = 3,
};

// Applies the source-stage logarithm independently to each consecutive float
// sample. Samples may represent any channel layout; no grayscale guide or
// channel interpretation is used. The input must already be black-prepared
// and source-default-inverse-prepared. Output is committed only on success.
// Input and output ranges must be valid, float-aligned and non-overlapping.
// `epsilon` is the resolved double value in (0, 2^-12], and pin_overrange
// enables the generic executor's finite [0, 1] clamp before the logarithm.
SourceLogStatus apply_acr_source_log(const float *input, float *output,
                                    std::size_t count, double epsilon,
                                    bool pin_overrange) noexcept;

}  // namespace imprint::acr
