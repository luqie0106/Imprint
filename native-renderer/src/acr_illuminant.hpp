#ifndef IMPRINT_ACR_ILLUMINANT_HPP
#define IMPRINT_ACR_ILLUMINANT_HPP

#include <cstdint>

namespace imprint::acr {

double illuminant_temperature_kelvin(std::uint32_t code) noexcept;

}  // namespace imprint::acr

#endif
