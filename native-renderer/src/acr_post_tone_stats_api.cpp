#include "imprint_renderer.h"
#include "acr_highlight_stat.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

extern "C" IMPRINT_API im_status im_native_acr_post_tone_stats_prepared_f32(
    const float* log_plane, size_t sample_count, double epsilon,
    double* output, size_t output_count) {
    if (log_plane == nullptr || output == nullptr || sample_count == 0 ||
        sample_count > imprint::acr::kMaxHighlightStatSamples)
        return IM_STATUS_INVALID_ARGUMENT;
    if (output_count < 5) return IM_STATUS_BUFFER_TOO_SMALL;
    const auto in = reinterpret_cast<std::uintptr_t>(log_plane);
    const auto out = reinterpret_cast<std::uintptr_t>(output);
    if (in % alignof(float) != 0 || out % alignof(double) != 0 ||
        output_count > static_cast<size_t>(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(double))
        return IM_STATUS_INVALID_ARGUMENT;
    const size_t in_bytes = sample_count * sizeof(float);
    const size_t out_bytes = output_count * sizeof(double);
    if (in_bytes > std::numeric_limits<std::uintptr_t>::max() - in ||
        out_bytes > std::numeric_limits<std::uintptr_t>::max() - out ||
        (in < out + out_bytes && out < in + in_bytes))
        return IM_STATUS_INVALID_ARGUMENT;
    imprint::acr::HighlightStat stat;
    const auto status = imprint::acr::compute_highlight_stat(log_plane, sample_count, epsilon, stat);
    if (status == imprint::acr::HighlightStatStatus::allocation_failure)
        return IM_STATUS_RUNTIME_ERROR;
    if (status != imprint::acr::HighlightStatStatus::ok)
        return IM_STATUS_INVALID_ARGUMENT;
    const double result[] = {stat.filtered_p0001, stat.filtered_p1,
                             stat.filtered_p99, stat.h,
                             static_cast<double>(stat.filtered_count)};
    std::memcpy(output, result, sizeof(result));
    return IM_STATUS_OK;
}
