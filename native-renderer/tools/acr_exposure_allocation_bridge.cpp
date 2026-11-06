#include "../src/acr_exposure_allocation.hpp"
#include <cstdint>
#include <cstring>
#include <limits>

extern "C" int acr_exposure_allocation_bridge(
    const double *input, double first, double second, unsigned hdr, double *output) {
    using namespace imprint::acr;
    static_assert(sizeof(ModernExposureAllocation) == 13 * sizeof(double));
    constexpr auto bytes = sizeof(ModernExposureAllocation);
    const auto a = reinterpret_cast<std::uintptr_t>(input);
    const auto b = reinterpret_cast<std::uintptr_t>(output);
    if (!input || !output || hdr > 1 || a % alignof(double) || b % alignof(double) ||
        a > std::numeric_limits<std::uintptr_t>::max() - bytes ||
        b > std::numeric_limits<std::uintptr_t>::max() - bytes ||
        (a != b && a < b + bytes && b < a + bytes)) return 3;
    ModernExposureAllocation state{}, result{};
    std::memcpy(&state, input, bytes);
    const auto status = update_modern_exposure_allocation(state, first, second, hdr != 0, result);
    if (status == ExposureAllocationStatus::ok) std::memcpy(output, &result, bytes);
    return static_cast<int>(status);
}
