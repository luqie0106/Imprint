#include "../src/acr_profile_encoding.hpp"

#include <algorithm>
#include <cstddef>

extern "C" const void *acr_test_profile_srgb_tables_address() noexcept {
    return &imprint::acr::profile_srgb_tables();
}

extern "C" const float *acr_test_profile_srgb_forward_data() noexcept {
    return imprint::acr::profile_srgb_tables().forward.data();
}

extern "C" const float *acr_test_profile_srgb_inverse_data() noexcept {
    return imprint::acr::profile_srgb_tables().inverse.data();
}

extern "C" int acr_test_copy_profile_srgb_tables(
    float *forward, std::size_t forward_count, float *inverse,
    std::size_t inverse_count) noexcept {
    if (forward == nullptr || inverse == nullptr ||
        forward_count != imprint::acr::kProfileSrgbStorage ||
        inverse_count != imprint::acr::kProfileSrgbStorage) {
        return 1;
    }

    const auto &tables = imprint::acr::profile_srgb_tables();
    std::copy(tables.forward.begin(), tables.forward.end(), forward);
    std::copy(tables.inverse.begin(), tables.inverse.end(), inverse);
    return 0;
}
