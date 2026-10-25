#include "../src/acr_resample_axis.hpp"

#include <cstdint>
#include <cstring>

namespace {

std::uint64_t bits_of(std::int64_t value) noexcept {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

std::uint32_t bits_of(std::int32_t value) noexcept {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

}  // namespace

extern "C" int acr_test_prepare_resample_axis(
    double translation, double along_scale, double cross_scale,
    std::int32_t bound_start, std::int32_t bound_end,
    std::int32_t half_taps, std::uint64_t *out6) noexcept {
    if (out6 == nullptr) return 1;

    imprint::acr::ResampleAxisPlan plan{};
    const imprint::acr::ResampleAxisStatus status =
        imprint::acr::prepare_ordinary_resample_axis(
            translation, along_scale, cross_scale, bound_start, bound_end,
            half_taps, plan);
    if (status != imprint::acr::ResampleAxisStatus::ok) {
        return static_cast<int>(status);
    }

    const std::uint64_t values[6] = {
        static_cast<std::uint64_t>(plan.kind),
        bits_of(plan.origin),
        bits_of(plan.step),
        bits_of(plan.lower),
        bits_of(plan.upper),
        bits_of(plan.integer_offset),
    };
    std::memcpy(out6, values, sizeof(values));
    return 0;
}
