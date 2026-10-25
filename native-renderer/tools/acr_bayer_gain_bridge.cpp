#include "acr_bayer_gain.hpp"

extern "C" int acr_test_gain_prepare(const double *gains, std::size_t count,
    double pivot, int inverse, float *values, int *clip) {
    using namespace imprint::acr;
    if (!values || !clip) return static_cast<int>(BayerGainStatus::null_pointer);
    BayerGainPrepared state{};
    const auto status = prepare_bayer_gain_007ef070(gains, count, pivot, inverse != 0, &state);
    if (status != BayerGainStatus::ok) return static_cast<int>(status);
    for (int i = 0; i < 3; ++i) {
        values[i] = state.gains[i];
        values[i + 3] = state.bias[i];
    }
    *clip = state.clip;
    return 0;
}

extern "C" int acr_test_gain_prepare_source_floor(const double *gains,
    std::size_t count, std::uint16_t source_floor, int inverse,
    float *values, int *clip) {
    using namespace imprint::acr;
    if (!values || !clip) return static_cast<int>(BayerGainStatus::null_pointer);
    BayerGainPrepared state{};
    const auto status = prepare_bayer_gain_from_source_floor_007ef070(
        gains, count, source_floor, inverse != 0, &state);
    if (status != BayerGainStatus::ok) return static_cast<int>(status);
    for (int i = 0; i < 3; ++i) {
        values[i] = state.gains[i];
        values[i + 3] = state.bias[i];
    }
    *clip = state.clip;
    return 0;
}

extern "C" int acr_test_gain_from_neutral(const double *neutral,
    std::size_t count, double *out, std::size_t out_count) {
    return static_cast<int>(imprint::acr::prepare_bayer_gains_from_neutral_007efc64(
        neutral, count, out, out_count));
}

namespace {
imprint::acr::BayerGainPrepared prepared(const float *values, int clip) {
    imprint::acr::BayerGainPrepared state{};
    for (int i = 0; i < 3; ++i) {
        state.gains[i] = values[i];
        state.bias[i] = values[i + 3];
    }
    state.clip = clip != 0;
    return state;
}
}

extern "C" int acr_test_gain_rgb(float *r, std::size_t rc, float *g,
    std::size_t gc, float *b, std::size_t bc, std::size_t width,
    const float *values, int clip) {
    if (!values) return static_cast<int>(imprint::acr::BayerGainStatus::null_pointer);
    return static_cast<int>(imprint::acr::apply_bayer_gain_rgb_row_007ef1f4(
        r, rc, g, gc, b, bc, width, prepared(values, clip)));
}

extern "C" int acr_test_gain_cfa(float *row, std::size_t count,
    std::size_t width, unsigned row_parity, unsigned column_parity,
    const float *values, int clip) {
    if (!values) return static_cast<int>(imprint::acr::BayerGainStatus::null_pointer);
    return static_cast<int>(imprint::acr::apply_bayer_gain_cfa_row_007ef1f4(
        row, count, width, row_parity, column_parity, prepared(values, clip)));
}
