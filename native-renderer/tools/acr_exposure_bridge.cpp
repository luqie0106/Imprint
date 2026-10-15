#include "../src/acr_exposure.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

extern "C" {

struct AcrExposureInputC {
    std::int32_t manual_exposure_x100;
    std::int32_t whites_ui;
    double source_exposure_ev;
    double source_scale;
    double source_linear_response_limit;
    std::uint32_t strongest_gain_q12;
    std::int32_t three_color_planes;
    std::int32_t hdr_edit;
    double limit_ev88;
    double profile_gain_amount;
};

struct AcrExposureOutputC {
    double manual_exposure_ev;
    double whites_ev;
    double source_exposure_ev;
    double source_scale;
    double context_ev50;
    double limit_ev88;
    double profile_gain_amount;
    double first_arg;
    double second_arg;
    double exposure_gain;
    double ev90;
    double ev98;
    double ev100;
    double ev108;
    double ev112;
    double context_ev18_change;
    double camera_scale;
    double rounded_source_limit;
    std::uint32_t source_limit_u16;
};

}  // extern "C"

namespace {

bool overlaps(const void *left, std::size_t left_size,
              const void *right, std::size_t right_size) noexcept {
    const auto left_start = reinterpret_cast<std::uintptr_t>(left);
    const auto right_start = reinterpret_cast<std::uintptr_t>(right);
    if (left_start <= right_start) return right_start - left_start < left_size;
    return left_start - right_start < right_size;
}

}  // namespace

extern "C" int acr_test_prepare_exposure(const AcrExposureInputC *input,
                                           AcrExposureOutputC *output) {
    if (input == nullptr || output == nullptr ||
        overlaps(input, sizeof(*input), output, sizeof(*output))) {
        return 100;
    }
    if ((input->three_color_planes != 0 && input->three_color_planes != 1) ||
        (input->hdr_edit != 0 && input->hdr_edit != 1)) {
        return 101;
    }

    const imprint::acr::ExposureInput cpp_input{
        input->manual_exposure_x100,
        input->whites_ui,
        input->source_exposure_ev,
        input->source_scale,
        input->source_linear_response_limit,
        input->strongest_gain_q12,
        input->three_color_planes != 0,
        input->hdr_edit != 0,
        input->limit_ev88,
        input->profile_gain_amount,
    };
    imprint::acr::ExposurePrepared prepared;
    const auto status = imprint::acr::prepare_exposure(cpp_input, prepared);
    if (status != imprint::acr::ExposureStatus::ok) {
        return static_cast<int>(status);
    }

    const AcrExposureOutputC result{
        prepared.manual_exposure_ev,
        prepared.whites_ev,
        prepared.source_exposure_ev,
        prepared.source_scale,
        prepared.context_ev50,
        prepared.limit_ev88,
        prepared.profile_gain_amount,
        prepared.first_arg,
        prepared.second_arg,
        prepared.exposure_gain,
        prepared.ev90,
        prepared.ev98,
        prepared.ev100,
        prepared.ev108,
        prepared.ev112,
        prepared.context_ev18_change,
        prepared.camera_scale,
        prepared.rounded_source_limit,
        prepared.source_limit_u16,
    };
    std::memcpy(output, &result, sizeof(result));
    return 0;
}
