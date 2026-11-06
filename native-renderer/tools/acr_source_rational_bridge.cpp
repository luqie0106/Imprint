#include "../src/acr_source_rational.hpp"

#include <cstdint>

extern "C" int acr_source_rational_bridge(std::uint32_t numerator,
                                           std::uint32_t denominator,
                                           double factor,
                                           std::uint32_t *output_pair) {
    if (output_pair == nullptr ||
        reinterpret_cast<std::uintptr_t>(output_pair) %
                alignof(std::uint32_t) !=
            0U) {
        return 3;
    }

    const imprint::acr::SourceRational input{numerator, denominator};
    imprint::acr::SourceRational candidate{};
    const auto status =
        imprint::acr::scale_source_rational(input, factor, candidate);
    if (status != imprint::acr::SourceRationalStatus::ok) {
        return static_cast<int>(status);
    }

    output_pair[0] = candidate.numerator;
    output_pair[1] = candidate.denominator;
    return static_cast<int>(status);
}
