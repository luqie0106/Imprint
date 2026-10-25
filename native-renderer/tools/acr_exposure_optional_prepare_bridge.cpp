#include "imprint_renderer.h"

#include <cstddef>

extern "C" im_status im_native_acr_prepare_exposure_optional(
    const double* controls, size_t control_count,
    double* output, size_t output_count);

extern "C" {

int acr_test_prepare_exposure_optional(const double* controls,
                                       std::size_t control_count,
                                       double* output,
                                       std::size_t output_count) noexcept {
    return static_cast<int>(im_native_acr_prepare_exposure_optional(
        controls, control_count, output, output_count));
}

}  // extern "C"
