#include "../src/acr_exposure_black_prepare.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace {

bool valid_output_range(const void* pointer, std::size_t size,
                        std::size_t alignment) noexcept {
    if (pointer == nullptr || alignment == 0) return false;
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (begin % alignment != 0) return false;
    constexpr auto max_address = std::numeric_limits<std::uintptr_t>::max();
    return size <= static_cast<std::size_t>(max_address - begin);
}

int status_code(imprint::acr::BlackPrepareStatus status) noexcept {
    return static_cast<int>(status);
}

}  // namespace

extern "C" {

int acr_test_black_response(double z, double t0, double t1, double k, double q,
                            double* output) noexcept {
    return status_code(imprint::acr::evaluate_black_control_response(
        z, t0, t1, k, q, output));
}

int acr_test_prepare_black(std::int32_t blacks,
                           std::int32_t profile_gain_param,
                           std::uint32_t gain_adjust_active,
                           std::uint32_t mode, double k, double q,
                           double* output, std::size_t count) noexcept {
    if ((gain_adjust_active != 0 && gain_adjust_active != 1) ||
        (mode != 0 && mode != 1) || count < 3 ||
        count > std::numeric_limits<std::size_t>::max() / sizeof(double) ||
        !valid_output_range(output, count * sizeof(double), alignof(double))) {
        return status_code(imprint::acr::BlackPrepareStatus::invalid_argument);
    }

    imprint::acr::PreparedBlackResponse prepared{};
    const auto status = imprint::acr::prepare_exposure_black_response(
        blacks, profile_gain_param, gain_adjust_active != 0, mode != 0, k, q,
        &prepared);
    if (status != imprint::acr::BlackPrepareStatus::ok) {
        return status_code(status);
    }

    const double values[3] = {prepared.z, prepared.post_strength,
                              prepared.channel_strength};
    std::memcpy(output, values, sizeof(values));
    return status_code(status);
}

}  // extern "C"
