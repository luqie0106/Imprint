#pragma once

#include <array>
#include <cstddef>

namespace imprint::acr {

constexpr std::size_t kProfileSrgbIntervals = 4096;
constexpr std::size_t kProfileSrgbStorage = kProfileSrgbIntervals + 2;

struct ProfileSrgbTables {
    std::array<float, kProfileSrgbStorage> forward;
    std::array<float, kProfileSrgbStorage> inverse;
};

// Built once in C++ from the recovered curve and uniform sampling path.
// These local-host tables are not a captured Adobe runtime resource.
const ProfileSrgbTables &profile_srgb_tables() noexcept;

}  // namespace imprint::acr
