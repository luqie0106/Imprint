#include "acr_stage3_balance3_controls.hpp"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>

using namespace imprint::acr;
std::uint32_t bits(float value) {
    std::uint32_t out;
    std::memcpy(&out, &value, 4);
    return out;
}

int main() {
    Balance3FinishLimits out{9, 8, 7, 6};
    if (prepare_balance3_finish_limits_007ed860(1, 1, false, &out) != Balance3ControlsStatus::ok ||
        bits(out.lower_even) != 0 || bits(out.upper_even) != 0x24800000 ||
        bits(out.lower_odd) != 0xa4800000 || bits(out.upper_odd) != 0) return 1;
    if (prepare_balance3_finish_limits_007ed860(-1, 1, true, &out) != Balance3ControlsStatus::ok ||
        out.lower_even != -.5f || out.upper_even != .5f ||
        out.lower_odd != -.5f || out.upper_odd != .5f) return 2;
    if (prepare_balance3_finish_limits_007ed860(-0., 1, true, &out) != Balance3ControlsStatus::ok ||
        bits(out.lower_odd) != 0x80000000 || bits(out.upper_even) != 0) return 3;
    const Balance3FinishLimits before = out;
    const double cases[][2] = {{0, 0}, {-1, -1}, {2, 1}, {-1, 1}, {1e300, 1e300},
                              {std::numeric_limits<double>::quiet_NaN(), 1}};
    const bool direct[] = {false, true, true, false, true, false};
    const int statuses[] = {3, 3, 2, 4, 4, 2};
    for (int i = 0; i < 6; ++i) {
        if (static_cast<int>(prepare_balance3_finish_limits_007ed860(
                cases[i][0], cases[i][1], direct[i], &out)) != statuses[i] ||
            std::memcmp(&before, &out, sizeof(out)) != 0) return 4;
    }
    alignas(Balance3FinishLimits) unsigned char storage[32]{};
    if (prepare_balance3_finish_limits_007ed860(1, 1, false,
            reinterpret_cast<Balance3FinishLimits *>(storage + 1)) != Balance3ControlsStatus::invalid_argument)
        return 5;
    const auto near_end = std::numeric_limits<std::uintptr_t>::max() & ~std::uintptr_t(3);
    if (prepare_balance3_finish_limits_007ed860(1, 1, false,
            reinterpret_cast<Balance3FinishLimits *>(near_end)) != Balance3ControlsStatus::invalid_argument)
        return 6;
    std::cout << "balance3 bounded controls validation passed\n";
}
