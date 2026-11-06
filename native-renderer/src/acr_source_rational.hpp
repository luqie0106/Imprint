#pragma once

#include <cstdint>

namespace imprint::acr {

struct SourceRational {
    std::uint32_t numerator;
    std::uint32_t denominator;
};

enum class SourceRationalStatus : int {
    ok = 0,
    non_finite_factor = 1,
    numeric_overflow = 2,
};

// Multiplies an existing unsigned rational by the finite factor using the
// recovered Source geometry rescale paths. The result is committed only on
// success, and input/output may refer to the same object. This is not a
// general rational normalizer or Adobe's full exception ABI.
SourceRationalStatus scale_source_rational(const SourceRational &input,
                                          double factor,
                                          SourceRational &output) noexcept;

}  // namespace imprint::acr
