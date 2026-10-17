#include "../src/acr_resample_row.hpp"

#include <algorithm>
#include <vector>

extern "C" int acr_test_resample_row(
    const float *source, std::size_t source_count, const float *coefficients,
    std::size_t coefficient_count, std::size_t taps, std::size_t stride,
    std::int64_t origin, std::int64_t step, std::int64_t lower, std::int64_t upper,
    std::int32_t first, int mode, std::size_t count,
    float *output, std::size_t output_count) noexcept {
    if (!output || output_count != count) return 1;
    imprint::acr::PreparedResampleRows rows{coefficients, coefficient_count, taps, stride};
    imprint::acr::RowRequest request{origin, step, lower, upper, first, count,
        static_cast<imprint::acr::ResampleRowMode>(mode)};
    std::vector<float> candidate{31.25f, -12.5f};
    const auto status = imprint::acr::process_resample_row(
        source, source_count, rows, request, candidate);
    if (status != imprint::acr::ResampleRowStatus::ok) {
        if (candidate != std::vector<float>{31.25f, -12.5f}) return 99;
        return static_cast<int>(status);
    }
    std::copy(candidate.begin(), candidate.end(), output);
    return 0;
}

extern "C" int acr_test_resample_row_result_alias() {
    std::vector<float> coefficients(128 * 8, 0.0f);
    for (std::size_t p = 0; p < 128; ++p) coefficients[p * 8] = 0.5f;
    imprint::acr::PreparedResampleRows rows{coefficients.data(), coefficients.size(), 1, 8};
    imprint::acr::RowRequest request{0, std::int64_t{1} << 32, 0,
        std::int64_t{3} << 32, 0, 3, imprint::acr::ResampleRowMode::scalar};
    std::vector<float> values{0.25f, 0.5f, 0.75f, 1.0f};
    auto status = imprint::acr::process_resample_row(values.data(), values.size(), rows, request, values);
    if (status != imprint::acr::ResampleRowStatus::ok ||
        values != std::vector<float>{0.125f, 0.25f, 0.375f}) return 1;
    const auto before = values;
    request.output_count = 4;
    status = imprint::acr::process_resample_row(values.data(), values.size(), rows, request, values);
    return status == imprint::acr::ResampleRowStatus::source_window_out_of_range && values == before ? 0 : 2;
}
