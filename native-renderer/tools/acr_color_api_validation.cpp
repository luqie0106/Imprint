#include "imprint_renderer.h"

#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>

int main() {
    std::array<double, 2> white{};
    if (im_native_acr_white_temperature(.3457, .3585, white.data(), white.size()) != IM_STATUS_OK ||
        std::abs(white[0]-5000.706605070579) > 1e-5 ||
        std::abs(white[1]-9.562965495976) > 1e-5) {
        std::cerr << "Recovered white projection failed\n";
        return 1;
    }
    for (const auto &xy : std::array<std::array<double, 2>, 5>{{
         {{-.1, .3}}, {{.3, 0.}}, {{.7, .4}},
         {{std::numeric_limits<double>::quiet_NaN(), .3}},
         {{.3, std::numeric_limits<double>::infinity()}}}}) {
        white.fill(-123.);
        if (im_native_acr_white_temperature(xy[0], xy[1], white.data(), 2) != IM_STATUS_INVALID_ARGUMENT ||
            white[0] != -123. || white[1] != -123.) return 1;
    }
    if (im_native_acr_white_temperature(.3, .3, nullptr, 2) != IM_STATUS_INVALID_ARGUMENT ||
        im_native_acr_white_temperature(.3, .3, white.data(), 1) != IM_STATUS_INVALID_ARGUMENT) return 1;
    using Function = im_status (*)(uint32_t, uint32_t, const float *, size_t,
                                  const double *, size_t, float *, size_t);
    std::array<float, 6> input{0.0f, 0.0f, 0.0f, 0.2f, 0.4f, 0.7f};
    std::array<double, 24> hsl{};
    std::array<double, 14> grading{};
    grading[12] = 50.0;
    auto check = [&](Function function, double *controls, size_t count) {
        std::array<float, 6> output{};
        if (function(2, 1, input.data(), 6, controls, count, output.data(), 6) != IM_STATUS_OK ||
            std::memcmp(input.data(), output.data(), sizeof(input)) != 0) return false;
        auto rejects_without_write = [&](const float *source, size_t source_count,
                                          const double *params, size_t param_count,
                                          uint32_t width = 2) {
            output.fill(-123.0f);
            auto status = function(width, 1, source, source_count, params, param_count,
                                   output.data(), output.size());
            if (status != IM_STATUS_INVALID_ARGUMENT) return false;
            for (float value : output) if (value != -123.0f) return false;
            return true;
        };
        if (!rejects_without_write(input.data(), 5, controls, count) ||
            !rejects_without_write(input.data(), 6, controls, count-1) ||
            !rejects_without_write(input.data(), 6, nullptr, count) ||
            !rejects_without_write(input.data(), 6, controls, count, UINT32_MAX)) return false;
        auto bad = input;
        bad[3] = std::numeric_limits<float>::quiet_NaN();
        if (!rejects_without_write(bad.data(), 6, controls, count)) return false;
        bad[3] = 1.01f;
        if (!rejects_without_write(bad.data(), 6, controls, count)) return false;
        const double saved = controls[0];
        controls[0] = std::numeric_limits<double>::infinity();
        bool invalid_control = rejects_without_write(input.data(), 6, controls, count);
        controls[0] = saved;
        if (!invalid_control) return false;
        auto overlap = input;
        if (function(2, 1, overlap.data(), 6, controls, count, overlap.data(), 6) !=
            IM_STATUS_INVALID_ARGUMENT || overlap != input) return false;
        std::array<float, 7> partial{};
        if (function(2, 1, partial.data(), 6, controls, count, partial.data()+1, 6) !=
            IM_STATUS_INVALID_ARGUMENT) return false;
        return true;
    };
    if (!check(im_native_acr_hsl_sdr_rgbf32, hsl.data(), hsl.size()) ||
        !check(im_native_acr_rendered_hsl_sdr_rgbf32, hsl.data(), hsl.size()) ||
        !check(im_native_acr_grading_sdr_rgbf32, grading.data(), grading.size())) {
        std::cerr << "Recovered color ABI contract failed\n";
        return 1;
    }
    std::array<double, 25> with_vibrance{};
    with_vibrance[24] = 100.0;
    std::array<float, 6> output{};
    if (im_native_acr_hsl_sdr_rgbf32(2, 1, input.data(), 6, with_vibrance.data(), 25,
                                    output.data(), 6) != IM_STATUS_OK) return 1;
    for (float value : output) if (!std::isfinite(value)) return 1;
    output.fill(-123.0f);
    with_vibrance[24] = 101.0;
    if (im_native_acr_hsl_sdr_rgbf32(2, 1, input.data(), 6, with_vibrance.data(), 25,
                                    output.data(), 6) != IM_STATUS_INVALID_ARGUMENT) return 1;
    for (float value : output) if (value != -123.0f) return 1;
    std::cout << "Recovered SDR color ABI contract passed\n";
    return 0;
}
