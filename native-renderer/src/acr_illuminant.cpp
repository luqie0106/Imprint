#include "acr_illuminant.hpp"

namespace imprint::acr {

double illuminant_temperature_kelvin(std::uint32_t code) noexcept {
    switch (code) {
        case 1:
        case 4:
        case 9:
        case 18:
        case 20:
            return 5500.0;  // 0x40b57c0000000000
        case 2:
        case 14:
            return 4150.0;  // 0x40b0360000000000
        case 3:
        case 17:
            return 2850.0;  // 0x40a6440000000000
        case 10:
        case 19:
        case 21:
            return 6500.0;  // 0x40b9640000000000
        case 11:
        case 22:
            return 7500.0;  // 0x40bd4c0000000000
        case 12:
            return 6400.0;  // 0x40b9000000000000
        case 13:
            return 5050.0;  // 0x40b3ba0000000000
        case 15:
            return 3525.0;  // 0x40ab8a0000000000
        case 16:
            return 2925.0;  // 0x40a6da0000000000
        case 23:
            return 5000.0;  // 0x40b3880000000000
        case 24:
            return 3200.0;  // 0x40a9000000000000
        default:
            return 0.0;
    }
}

}  // namespace imprint::acr
