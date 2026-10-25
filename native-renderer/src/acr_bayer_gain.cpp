#include "acr_bayer_gain.hpp"

#include <cmath>
#include <cstdint>
#include <limits>

namespace imprint::acr {
namespace {

struct AddressSpan {
    std::uintptr_t begin;
    std::uintptr_t end;
};

BayerGainStatus make_float_span(float* pointer, std::size_t count,
                                std::size_t width, AddressSpan* span) noexcept {
    if (pointer == nullptr || span == nullptr) return BayerGainStatus::null_pointer;
    if (reinterpret_cast<std::uintptr_t>(pointer) % alignof(float) != 0)
        return BayerGainStatus::misaligned_pointer;
    if (count < width) return BayerGainStatus::invalid_count;
    constexpr std::size_t kMaxSamples =
        static_cast<std::size_t>(PTRDIFF_MAX) / sizeof(float);
    if (count > kMaxSamples || width > kMaxSamples ||
        width > std::numeric_limits<std::size_t>::max() / sizeof(float))
        return BayerGainStatus::size_overflow;

    const std::size_t bytes = width * sizeof(float);
    const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (bytes > std::numeric_limits<std::uintptr_t>::max() - begin)
        return BayerGainStatus::address_overflow;
    span->begin = begin;
    span->end = begin + bytes;
    return BayerGainStatus::ok;
}

bool overlaps(const AddressSpan& a, const AddressSpan& b) noexcept {
    return a.begin < b.end && b.begin < a.end;
}

BayerGainStatus make_object_span(const void* pointer, std::size_t bytes,
                                 AddressSpan* span) noexcept {
    if (pointer == nullptr || span == nullptr) return BayerGainStatus::null_pointer;
    const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (bytes > std::numeric_limits<std::uintptr_t>::max() - begin)
        return BayerGainStatus::address_overflow;
    span->begin = begin;
    span->end = begin + bytes;
    return BayerGainStatus::ok;
}

BayerGainStatus validate_prepared(const BayerGainPrepared& prepared) noexcept {
    for (int channel = 0; channel < 3; ++channel) {
        if (!std::isfinite(prepared.gains[channel]) || prepared.gains[channel] <= 0.0f ||
            !std::isfinite(prepared.bias[channel]))
            return BayerGainStatus::invalid_prepared;
    }
    return BayerGainStatus::ok;
}

float apply_sample(float sample, float gain, float bias, bool clip) noexcept {
    float value = std::fma(sample, gain, bias);
    if (clip) {
        value = std::fmin(value, 1.0f);
        if (value < 0.0f) value = 0.0f;
    }
    return value;
}

bool finite_row(const float* row, std::size_t width) noexcept {
    for (std::size_t i = 0; i < width; ++i)
        if (!std::isfinite(row[i])) return false;
    return true;
}

}  // namespace

BayerGainStatus prepare_bayer_gain_007ef070(
    const double* gains, std::size_t gain_count, double pivot, bool inverse,
    BayerGainPrepared* out) noexcept {
    if (gains == nullptr || out == nullptr) return BayerGainStatus::null_pointer;
    if (gain_count != 3) return BayerGainStatus::invalid_count;
    if (reinterpret_cast<std::uintptr_t>(gains) % alignof(double) != 0 ||
        reinterpret_cast<std::uintptr_t>(out) % alignof(BayerGainPrepared) != 0)
        return BayerGainStatus::misaligned_pointer;

    AddressSpan gain_span{};
    BayerGainStatus status =
        make_object_span(gains, 3 * sizeof(double), &gain_span);
    if (status != BayerGainStatus::ok) return status;
    AddressSpan output_span{};
    status = make_object_span(out, sizeof(*out), &output_span);
    if (status != BayerGainStatus::ok) return status;
    if (overlaps(gain_span, output_span)) return BayerGainStatus::overlap;
    if (!std::isfinite(pivot)) return BayerGainStatus::invalid_pivot;

    BayerGainPrepared candidate{};
    bool clip = false;
    for (int channel = 0; channel < 3; ++channel) {
        const double input_gain = gains[channel];
        if (!std::isfinite(input_gain) || input_gain <= 0.0 || input_gain >= 128.0)
            return BayerGainStatus::invalid_gain;
        const double adjusted = inverse ? 1.0 / input_gain : input_gain;
        if (!std::isfinite(adjusted) || adjusted <= 0.0 ||
            adjusted > std::numeric_limits<float>::max())
            return BayerGainStatus::invalid_gain;
        const float gain = static_cast<float>(adjusted);
        if (!std::isfinite(gain) || gain <= 0.0f)
            return BayerGainStatus::invalid_gain;
        candidate.gains[channel] = gain;
        if (adjusted > 1.0) clip = true;
    }

    const float b = static_cast<float>(pivot);
    if (!std::isfinite(b) || b < 0.0f || b >= 1.0f)
        return BayerGainStatus::invalid_pivot;

    const float den = b + -1.0f;
    const float t = -1.0f / den;
    const float neg = std::fma(-t, b, 0.0f) / t;
    const float pos = std::fma(t, b, -0.0f) / t;
    if (!std::isfinite(den) || !std::isfinite(t) || !std::isfinite(neg) ||
        !std::isfinite(pos))
        return BayerGainStatus::invalid_pivot;
    for (int channel = 0; channel < 3; ++channel) {
        candidate.bias[channel] = std::fma(candidate.gains[channel], neg, pos);
        if (!std::isfinite(candidate.bias[channel]))
            return BayerGainStatus::invalid_gain;
    }
    candidate.clip = clip;
    *out = candidate;
    return BayerGainStatus::ok;
}

BayerGainStatus prepare_bayer_gain_from_source_floor_007ef070(
    const double* gains, std::size_t gain_count, std::uint16_t source_floor,
    bool inverse, BayerGainPrepared* out) noexcept {
    static constexpr double kReciprocal65535 = 0x1.0001000100010p-16;
    const double pivot = static_cast<double>(source_floor) * kReciprocal65535;
    return prepare_bayer_gain_007ef070(gains, gain_count, pivot, inverse, out);
}

BayerGainStatus prepare_bayer_gains_from_neutral_007efc64(
    const double* neutral, std::size_t neutral_count, double* out,
    std::size_t out_count) noexcept {
    if (out == nullptr) return BayerGainStatus::null_pointer;
    if (out_count != 3) return BayerGainStatus::invalid_count;
    if (reinterpret_cast<std::uintptr_t>(out) % alignof(double) != 0)
        return BayerGainStatus::misaligned_pointer;

    AddressSpan output_span{};
    BayerGainStatus status =
        make_object_span(out, 3 * sizeof(double), &output_span);
    if (status != BayerGainStatus::ok) return status;

    static constexpr double kDefaultNeutral[3] = {0.4, 1.0, 0.6};
    const double* source = kDefaultNeutral;
    if (neutral_count == 3) {
        if (neutral == nullptr) return BayerGainStatus::null_pointer;
        if (reinterpret_cast<std::uintptr_t>(neutral) % alignof(double) != 0)
            return BayerGainStatus::misaligned_pointer;

        AddressSpan input_span{};
        status = make_object_span(neutral, 3 * sizeof(double), &input_span);
        if (status != BayerGainStatus::ok) return status;
        if (overlaps(input_span, output_span)) return BayerGainStatus::overlap;

        for (int channel = 0; channel < 3; ++channel)
            if (!std::isfinite(neutral[channel]))
                return BayerGainStatus::invalid_gain;

        double minimum = neutral[0];
        for (int channel = 1; channel < 3; ++channel)
            if (neutral[channel] <= minimum) minimum = neutral[channel];
        if (minimum > 0.0) source = neutral;
    }

    const auto normalize = [](const double* values, double* scaled,
                              double* minimum_scaled) noexcept {
        double maximum = values[0];
        for (int channel = 1; channel < 3; ++channel)
            if (values[channel] >= maximum) maximum = values[channel];

        const double reciprocal = 1.0 / maximum;
        if (!std::isfinite(reciprocal) || reciprocal <= 0.0) return false;

        for (int channel = 0; channel < 3; ++channel) {
            scaled[channel] = values[channel] * reciprocal;
            if (!std::isfinite(scaled[channel])) return false;
        }

        double minimum = scaled[0];
        for (int channel = 1; channel < 3; ++channel)
            if (scaled[channel] <= minimum) minimum = scaled[channel];
        if (!std::isfinite(minimum)) return false;
        *minimum_scaled = minimum;
        return true;
    };

    double scaled[3]{};
    double minimum_scaled = 0.0;
    if (!normalize(source, scaled, &minimum_scaled))
        return BayerGainStatus::invalid_gain;
    if (minimum_scaled < 0.125) {
        if (!normalize(kDefaultNeutral, scaled, &minimum_scaled))
            return BayerGainStatus::invalid_gain;
    }

    double candidate[3]{};
    for (int channel = 0; channel < 3; ++channel) {
        candidate[channel] = minimum_scaled / scaled[channel];
        if (!std::isfinite(candidate[channel]) || candidate[channel] <= 0.0)
            return BayerGainStatus::invalid_gain;
    }

    for (int channel = 0; channel < 3; ++channel)
        out[channel] = candidate[channel];
    return BayerGainStatus::ok;
}

BayerGainStatus apply_bayer_gain_rgb_row_007ef1f4(
    float* r, std::size_t r_count, float* g, std::size_t g_count,
    float* b, std::size_t b_count, std::size_t width,
    const BayerGainPrepared& prepared) noexcept {
    BayerGainStatus status = validate_prepared(prepared);
    if (status != BayerGainStatus::ok) return status;
    if (width == 0) return BayerGainStatus::invalid_width;

    AddressSpan spans[3]{};
    status = make_float_span(r, r_count, width, &spans[0]);
    if (status != BayerGainStatus::ok) return status;
    status = make_float_span(g, g_count, width, &spans[1]);
    if (status != BayerGainStatus::ok) return status;
    status = make_float_span(b, b_count, width, &spans[2]);
    if (status != BayerGainStatus::ok) return status;
    if (overlaps(spans[0], spans[1]) || overlaps(spans[0], spans[2]) ||
        overlaps(spans[1], spans[2]))
        return BayerGainStatus::overlap;

    AddressSpan prepared_span{};
    status = make_object_span(&prepared, sizeof(prepared), &prepared_span);
    if (status != BayerGainStatus::ok) return status;
    for (const AddressSpan& span : spans)
        if (overlaps(span, prepared_span)) return BayerGainStatus::overlap;

    if (!finite_row(r, width) || !finite_row(g, width) || !finite_row(b, width))
        return BayerGainStatus::nonfinite_sample;

    const bool select_r = prepared.gains[0] != 1.0f;
    const bool select_g = !select_r || prepared.gains[1] != 1.0f;
    const int first_channel = select_r ? 0 : 1;
    const int second_channel = select_r && select_g ? 1 : 2;
    float* first = select_r ? r : g;
    float* second = second_channel == 1 ? g : b;
    for (std::size_t i = 0; i < width; ++i) {
        first[i] = apply_sample(first[i], prepared.gains[first_channel],
                                prepared.bias[first_channel], prepared.clip);
        second[i] = apply_sample(second[i], prepared.gains[second_channel],
                                 prepared.bias[second_channel], prepared.clip);
    }
    return BayerGainStatus::ok;
}

BayerGainStatus apply_bayer_gain_cfa_row_007ef1f4(
    float* row, std::size_t count, std::size_t width,
    unsigned row_parity, unsigned col_parity,
    const BayerGainPrepared& prepared) noexcept {
    BayerGainStatus status = validate_prepared(prepared);
    if (status != BayerGainStatus::ok) return status;
    if (width == 0 || (width & 1u) != 0) return BayerGainStatus::invalid_width;
    if (row_parity > 1 || col_parity > 1) return BayerGainStatus::invalid_parity;

    AddressSpan row_span{};
    status = make_float_span(row, count, width, &row_span);
    if (status != BayerGainStatus::ok) return status;
    AddressSpan prepared_span{};
    status = make_object_span(&prepared, sizeof(prepared), &prepared_span);
    if (status != BayerGainStatus::ok) return status;
    if (overlaps(row_span, prepared_span)) return BayerGainStatus::overlap;
    if (!finite_row(row, width)) return BayerGainStatus::nonfinite_sample;

    for (std::size_t i = 0; i < width; ++i) {
        const unsigned column = col_parity ^ static_cast<unsigned>(i & 1u);
        const int channel = row_parity == 0
                                ? (column == 0 ? 0 : 1)
                                : (column == 1 ? 2 : 1);
        row[i] = apply_sample(row[i], prepared.gains[channel],
                              prepared.bias[channel], prepared.clip);
    }
    return BayerGainStatus::ok;
}

}  // namespace imprint::acr
