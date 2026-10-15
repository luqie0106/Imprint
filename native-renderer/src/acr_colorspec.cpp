#include "acr_colorspec.hpp"

#include "acr_color_kernels.hpp"
#include "acr_white_temperature_table.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace imprint::acr {
namespace {

using Matrix = std::array<double, 9>;
using Vector = std::array<double, 3>;

constexpr double kInverseDeterminantThreshold = 1.0e-10;
constexpr double kD50X = 0.3457;
constexpr double kD50Y = 0.3585;
constexpr Matrix kBradford = {
    0.8951, 0.2664, -0.1614,
    -0.7502, 1.7135, 0.0367,
    0.0389, -0.0685, 1.0296,
};

bool ranges_overlap(const void *left, std::size_t left_size,
                    const void *right, std::size_t right_size) noexcept {
    const auto left_start = reinterpret_cast<std::uintptr_t>(left);
    const auto right_start = reinterpret_cast<std::uintptr_t>(right);
    if (left_start <= right_start) return right_start - left_start < left_size;
    return left_start - right_start < right_size;
}

bool finite(const Matrix &matrix) noexcept {
    return std::all_of(matrix.begin(), matrix.end(),
                       [](double value) { return std::isfinite(value); });
}

bool finite(const Vector &vector) noexcept {
    return std::all_of(vector.begin(), vector.end(),
                       [](double value) { return std::isfinite(value); });
}

bool finite_profile_input(const ColorSpecInput &input) noexcept {
    if (!finite(input.raw_cm1) || !finite(input.raw_cm2) ||
        !finite(input.cc1) || !finite(input.cc2) ||
        !finite(input.forward1) || !finite(input.forward2) ||
        !finite(input.analog_balance) ||
        !std::isfinite(input.white_xy[0]) || !std::isfinite(input.white_xy[1]) ||
        !std::isfinite(input.endpoint_kelvin1) ||
        !std::isfinite(input.endpoint_kelvin2)) {
        return false;
    }
    for (const double value : input.analog_balance) {
        if (value <= 0.0) return false;
    }
    return true;
}

bool finite_input(const ColorSpecInput &input) noexcept {
    if (!finite_profile_input(input)) return false;
    const double x = input.white_xy[0];
    const double y = input.white_xy[1];
    return x >= 0.0 && y > 0.0 && x + y < 1.0;
}

bool white_temperature_raw(double x, double y, WhiteTemperature &output) noexcept {
    if (!std::isfinite(x) || !std::isfinite(y)) return false;
    const double denominator = (1.5 - x) + 6.0 * y;
    if (denominator == 0.0 || !std::isfinite(denominator)) return false;
    const double u = 2.0 * x / denominator;
    const double v = 3.0 * y / denominator;
    if (!std::isfinite(u) || !std::isfinite(v)) return false;

    double previous_distance = 0.0;
    double previous_a = 0.0;
    double previous_b = 0.0;
    for (std::size_t j = 1; j < kWhiteTemperatureTable.size(); ++j) {
        const auto &row = kWhiteTemperatureTable[j];
        const double norm = std::sqrt(1.0 + row[3] * row[3]);
        if (norm == 0.0 || !std::isfinite(norm)) return false;
        double a = 1.0 / norm;
        double b = row[3] * a;
        const double distance = (v - row[2]) * a - b * (u - row[1]);
        if (!std::isfinite(distance)) return false;
        if (distance <= 0.0 || j + 1 == kWhiteTemperatureTable.size()) {
            const auto &previous = kWhiteTemperatureTable[j - 1];
            const double clipped_distance = std::min(distance, 0.0);
            double f = 0.0;
            if (j != 1) {
                const double divisor = previous_distance - clipped_distance;
                if (divisor == 0.0 || !std::isfinite(divisor)) return false;
                f = -clipped_distance / divisor;
            }
            const double q = 1.0 - f;
            const double mired = q * row[0] + f * previous[0];
            if (mired == 0.0 || !std::isfinite(mired)) return false;
            const double temperature = 1.0e6 / mired;
            a = q * a + f * previous_a;
            b = q * b + f * previous_b;
            const double direction_norm = std::hypot(a, b);
            if (direction_norm == 0.0 || !std::isfinite(direction_norm)) return false;
            const double u0 = q * row[1] + f * previous[1];
            const double v0 = q * row[2] + f * previous[2];
            const double tint = -3000.0 * (b / direction_norm * (v - v0) +
                                           a / direction_norm * (u - u0));
            if (!std::isfinite(temperature) || !std::isfinite(tint)) return false;
            output = {temperature, tint};
            return true;
        }
        previous_distance = distance;
        previous_a = a;
        previous_b = b;
    }
    return false;
}

bool project_white_temperature(double x, double y,
                               WhiteTemperature &output) noexcept {
    if (!std::isfinite(x) || !std::isfinite(y)) return false;
    if (x >= 0.0 && y > 0.0 && x + y < 1.0) {
        try {
            output = white_temperature(x, y);
            return std::isfinite(output.temperature) && std::isfinite(output.tint);
        } catch (...) {
            return false;
        }
    }
    return white_temperature_raw(x, y, output);
}

bool checked_multiply(double left, double right, double &result) noexcept {
    result = left * right;
    return std::isfinite(result);
}

bool checked_add(double left, double right, double &result) noexcept {
    result = left + right;
    return std::isfinite(result);
}

bool multiply(const Matrix &left, const Matrix &right, Matrix &result) noexcept {
    Matrix candidate{};
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t column = 0; column < 3; ++column) {
            double sum = 0.0;
            for (std::size_t inner = 0; inner < 3; ++inner) {
                double product = 0.0;
                double next = 0.0;
                if (!checked_multiply(left[row * 3 + inner],
                                      right[inner * 3 + column], product) ||
                    !checked_add(sum, product, next)) {
                    return false;
                }
                sum = next;
            }
            candidate[row * 3 + column] = sum;
        }
    }
    result = candidate;
    return true;
}

bool multiply(const Matrix &matrix, const Vector &vector,
              Vector &result) noexcept {
    Vector candidate{};
    for (std::size_t row = 0; row < 3; ++row) {
        double sum = 0.0;
        for (std::size_t column = 0; column < 3; ++column) {
            double product = 0.0;
            double next = 0.0;
            if (!checked_multiply(matrix[row * 3 + column], vector[column], product) ||
                !checked_add(sum, product, next)) {
                return false;
            }
            sum = next;
        }
        candidate[row] = sum;
    }
    result = candidate;
    return true;
}

bool inverse(const Matrix &matrix, Matrix &result,
             ColorSpecStatus &failure) noexcept {
    const double a = matrix[0], b = matrix[1], c = matrix[2];
    const double d = matrix[3], e = matrix[4], f = matrix[5];
    const double g = matrix[6], h = matrix[7], i = matrix[8];
    double hf = 0.0, di = 0.0, ge = 0.0;
    if (!checked_multiply(h, f, hf) || !checked_multiply(d, i, di) ||
        !checked_multiply(g, e, ge)) {
        failure = ColorSpecStatus::non_finite_intermediate;
        return false;
    }
    const double c00 = std::fma(e, i, -hf);
    const double c10 = std::fma(g, f, -di);
    const double c20 = std::fma(d, h, -ge);
    if (!std::isfinite(c00) || !std::isfinite(c10) || !std::isfinite(c20)) {
        failure = ColorSpecStatus::non_finite_intermediate;
        return false;
    }

    double bc10_product = 0.0;
    if (!checked_multiply(b, c10, bc10_product)) {
        failure = ColorSpecStatus::non_finite_intermediate;
        return false;
    }
    const double t = std::fma(a, c00, bc10_product);
    if (!std::isfinite(t)) {
        failure = ColorSpecStatus::non_finite_intermediate;
        return false;
    }
    const double determinant = std::fma(c, c20, t);
    const double negative_determinant = std::fma(-c, c20, -t);
    if (!std::isfinite(determinant) || !std::isfinite(negative_determinant)) {
        failure = ColorSpecStatus::non_finite_intermediate;
        return false;
    }
    const double determinant_magnitude = determinant < 0.0
        ? negative_determinant : determinant;
    if (determinant_magnitude < kInverseDeterminantThreshold) {
        failure = ColorSpecStatus::singular_matrix;
        return false;
    }

    double bi = 0.0, ec = 0.0, c_neg_g = 0.0, af = 0.0;
    double ah = 0.0, bd = 0.0, ad = 0.0;
    if (!checked_multiply(b, i, bi) || !checked_multiply(e, c, ec) ||
        !checked_multiply(c, -g, c_neg_g) || !checked_multiply(a, f, af) ||
        !checked_multiply(a, h, ah) || !checked_multiply(b, d, bd) ||
        !checked_multiply(b, -d, ad)) {
        failure = ColorSpecStatus::non_finite_intermediate;
        return false;
    }
    const std::array<double, 9> adjugate = {
        c00,
        std::fma(h, c, -bi),
        std::fma(b, f, -ec),
        c10,
        std::fma(a, i, c_neg_g),
        std::fma(d, c, -af),
        c20,
        std::fma(g, b, -ah),
        std::fma(a, e, ad),
    };
    if (!finite(adjugate)) {
        failure = ColorSpecStatus::non_finite_intermediate;
        return false;
    }

    Matrix candidate{};
    for (std::size_t index = 0; index < candidate.size(); ++index) {
        candidate[index] = adjugate[index] / determinant;
        if (!std::isfinite(candidate[index])) {
            failure = ColorSpecStatus::non_finite_intermediate;
            return false;
        }
    }
    result = candidate;
    return true;
}

Matrix diagonal(const Vector &values) noexcept {
    return {values[0], 0.0, 0.0,
            0.0, values[1], 0.0,
            0.0, 0.0, values[2]};
}

bool normalize_forward(const Matrix &forward, Matrix &normalized,
                       ColorSpecStatus &failure) noexcept {
    const Vector ones{1.0, 1.0, 1.0};
    Vector row_sums{};
    if (!multiply(forward, ones, row_sums)) {
        failure = ColorSpecStatus::non_finite_intermediate;
        return false;
    }
    Matrix inverse_row_sum{};
    if (!inverse(diagonal(row_sums), inverse_row_sum, failure)) return false;

    // FUN_0127b1d0 returns the D50 white XYZ represented by xy=(.3457,.3585).
    // Keep the same xy-to-XYZ expression/order used by FUN_0127b128.
    const Vector d50_xyz{kD50X / kD50Y, 1.0,
                         ((1.0 - kD50X) - kD50Y) / kD50Y};
    Matrix d50_diagonal{};
    Matrix scaled_rows{};
    Matrix candidate{};
    if (!finite(d50_xyz) ||
        !multiply(diagonal(d50_xyz), inverse_row_sum, d50_diagonal) ||
        !multiply(d50_diagonal, forward, scaled_rows)) {
        failure = ColorSpecStatus::non_finite_intermediate;
        return false;
    }
    candidate = scaled_rows;
    normalized = candidate;
    return true;
}

bool xy_to_xyz(double x, double y, Vector &xyz) noexcept {
    // FUN_0127b128 clips x/y independently to [1e-6,.999999], then rescales
    // their sum to .999999 before forming [x/y, 1, (1-x-y)/y].
    x = std::min(x, 0.999999);
    y = std::min(y, 0.999999);
    if (x < 1.0e-6) x = 1.0e-6;
    if (y < 1.0e-6) y = 1.0e-6;
    const double sum = x + y;
    if (!std::isfinite(sum)) return false;
    if (0.999999 < sum) {
        const double scale = 0.999999 / sum;
        x *= scale;
        y *= scale;
    }
    xyz = {x / y, 1.0, ((1.0 - x) - y) / y};
    return finite(xyz);
}

bool chromatic_adaptation(const Vector &source_xyz, const Vector &target_xyz,
                          Matrix &adaptation,
                          ColorSpecStatus &failure) noexcept {
    Vector source_cone{}, target_cone{};
    if (!multiply(kBradford, source_xyz, source_cone) ||
        !multiply(kBradford, target_xyz, target_cone)) {
        failure = ColorSpecStatus::non_finite_intermediate;
        return false;
    }
    for (double &value : source_cone) value = std::max(value, 0.0);
    for (double &value : target_cone) value = std::max(value, 0.0);

    Vector ratio{};
    for (std::size_t channel = 0; channel < 3; ++channel) {
        double value = 10.0;
        if (source_cone[channel] > 0.0) {
            value = target_cone[channel] / source_cone[channel];
            if (!std::isfinite(value)) {
                failure = ColorSpecStatus::non_finite_intermediate;
                return false;
            }
        }
        value = std::min(value, 10.0);
        value = std::max(value, 0.1);
        ratio[channel] = value;
    }

    Matrix inverse_bradford{};
    if (!inverse(kBradford, inverse_bradford, failure)) return false;
    Matrix inverse_times_scale{};
    Matrix candidate{};
    if (!multiply(inverse_bradford, diagonal(ratio), inverse_times_scale) ||
        !multiply(inverse_times_scale, kBradford, candidate)) {
        failure = ColorSpecStatus::non_finite_intermediate;
        return false;
    }
    adaptation = candidate;
    return true;
}

bool blend(const Matrix &first, const Matrix &second, double weight,
           Matrix &result) noexcept {
    Matrix candidate{};
    const double second_weight = 1.0 - weight;
    for (std::size_t index = 0; index < candidate.size(); ++index) {
        double left = 0.0, right = 0.0;
        if (!checked_multiply(first[index], weight, left) ||
            !checked_multiply(second[index], second_weight, right) ||
            !checked_add(left, right, candidate[index])) {
            return false;
        }
    }
    result = candidate;
    return true;
}

bool multiply_diagonal_left(const Vector &scale, const Matrix &matrix,
                            Matrix &result) noexcept {
    return multiply(diagonal(scale), matrix, result);
}

struct PreparedSlots {
    Matrix color1{};
    Matrix color2{};
    Matrix cc1{};
    Matrix cc2{};
    Matrix forward1{};
    Matrix forward2{};
    double temperature1 = 5000.0;
    double temperature2 = 5000.0;
    bool has_forward1 = false;
    bool has_forward2 = false;
};

struct InterpolatedMatrices {
    Matrix color{};
    Matrix cc{};
    Matrix forward{};
    double weight = 0.0;
    bool has_forward = false;
};

enum class Endpoint { first, second, interior };

ColorSpecStatus prepare_slots(const ColorSpecInput &input,
                              PreparedSlots &output) noexcept {
    if (!finite_profile_input(input)) return ColorSpecStatus::invalid_input;
    PreparedSlots candidate{};
    ColorSpecStatus failure = ColorSpecStatus::non_finite_intermediate;

    candidate.has_forward1 = input.has_forward1;
    if (candidate.has_forward1 &&
        !normalize_forward(input.forward1, candidate.forward1, failure)) {
        return failure;
    }

    Matrix ab_cc1{};
    if (!multiply_diagonal_left(input.analog_balance, input.cc1, ab_cc1) ||
        !multiply(ab_cc1, input.raw_cm1, candidate.color1)) {
        return ColorSpecStatus::non_finite_intermediate;
    }
    candidate.cc1 = input.cc1;

    const bool two_illuminants = input.has_second_cm &&
        input.endpoint_kelvin1 > 0.0 && input.endpoint_kelvin2 > 0.0 &&
        input.endpoint_kelvin1 != input.endpoint_kelvin2;
    if (two_illuminants) {
        candidate.temperature1 = input.endpoint_kelvin1;
        candidate.temperature2 = input.endpoint_kelvin2;
        candidate.has_forward2 = input.has_forward2;
        if (candidate.has_forward2 &&
            !normalize_forward(input.forward2, candidate.forward2, failure)) {
            return failure;
        }
        Matrix ab_cc2{};
        if (!multiply_diagonal_left(input.analog_balance, input.cc2, ab_cc2) ||
            !multiply(ab_cc2, input.raw_cm2, candidate.color2)) {
            return ColorSpecStatus::non_finite_intermediate;
        }
        candidate.cc2 = input.cc2;
    } else {
        candidate.color2 = candidate.color1;
        candidate.cc2 = candidate.cc1;
        candidate.forward2 = candidate.forward1;
        candidate.has_forward2 = candidate.has_forward1;
    }

    if (two_illuminants && candidate.temperature2 < candidate.temperature1) {
        std::swap(candidate.temperature1, candidate.temperature2);
        std::swap(candidate.color1, candidate.color2);
        std::swap(candidate.cc1, candidate.cc2);
        std::swap(candidate.forward1, candidate.forward2);
        std::swap(candidate.has_forward1, candidate.has_forward2);
    }
    output = candidate;
    return ColorSpecStatus::ok;
}

ColorSpecStatus interpolate_at_white(const PreparedSlots &slots,
                                     const std::array<double, 2> &xy,
                                     InterpolatedMatrices &output) noexcept {
    WhiteTemperature projected{};
    if (!project_white_temperature(xy[0], xy[1], projected)) {
        return ColorSpecStatus::invalid_input;
    }
    const double temperature = projected.temperature;
    Endpoint endpoint = Endpoint::interior;
    double weight = 0.0;
    if (temperature <= slots.temperature1) {
        weight = 1.0;
        endpoint = Endpoint::first;
    } else if (temperature >= slots.temperature2) {
        weight = 0.0;
        endpoint = Endpoint::second;
    } else {
        weight = (1.0 / temperature - 1.0 / slots.temperature2) /
                 (1.0 / slots.temperature1 - 1.0 / slots.temperature2);
        if (!std::isfinite(weight)) return ColorSpecStatus::non_finite_intermediate;
        if (weight >= 1.0) weight = 1.0;
        if (weight <= 0.0) weight = 0.0;
    }

    InterpolatedMatrices candidate{};
    candidate.weight = weight;
    if (endpoint == Endpoint::first) {
        candidate.color = slots.color1;
        candidate.cc = slots.cc1;
    } else if (endpoint == Endpoint::second) {
        candidate.color = slots.color2;
        candidate.cc = slots.cc2;
    } else if (!blend(slots.color1, slots.color2, weight, candidate.color) ||
               !blend(slots.cc1, slots.cc2, weight, candidate.cc)) {
        return ColorSpecStatus::non_finite_intermediate;
    }

    if (slots.has_forward1 && slots.has_forward2) {
        if (endpoint == Endpoint::first) {
            candidate.forward = slots.forward1;
        } else if (endpoint == Endpoint::second) {
            candidate.forward = slots.forward2;
        } else if (!blend(slots.forward1, slots.forward2, weight,
                          candidate.forward)) {
            return ColorSpecStatus::non_finite_intermediate;
        }
        candidate.has_forward = true;
    } else if (slots.has_forward1) {
        candidate.forward = slots.forward1;
        candidate.has_forward = true;
    } else if (slots.has_forward2) {
        candidate.forward = slots.forward2;
        candidate.has_forward = true;
    }
    output = candidate;
    return ColorSpecStatus::ok;
}

ColorSpecStatus set_white_xy(const ColorSpecInput &input,
                             const std::array<double, 2> &xy,
                             const InterpolatedMatrices &matrices,
                             ColorSpecPrepared &output) noexcept {
    ColorSpecStatus failure = ColorSpecStatus::non_finite_intermediate;
    Vector target_xyz{};
    if (!xy_to_xyz(xy[0], xy[1], target_xyz)) {
        return ColorSpecStatus::non_finite_intermediate;
    }
    Vector camera_white_raw{};
    if (!multiply(matrices.color, target_xyz, camera_white_raw)) {
        return ColorSpecStatus::non_finite_intermediate;
    }
    const double camera_white_max = *std::max_element(camera_white_raw.begin(),
                                                       camera_white_raw.end());
    if (camera_white_max == 0.0) return ColorSpecStatus::invalid_denominator;
    const double camera_white_reciprocal = 1.0 / camera_white_max;
    if (!std::isfinite(camera_white_reciprocal)) {
        return ColorSpecStatus::non_finite_intermediate;
    }
    Vector camera_white{};
    for (std::size_t channel = 0; channel < 3; ++channel) {
        const double normalized = camera_white_raw[channel] * camera_white_reciprocal;
        if (!std::isfinite(normalized)) return ColorSpecStatus::non_finite_intermediate;
        camera_white[channel] = std::clamp(normalized, 0.001, 1.0);
    }

    Vector d50_xyz{};
    if (!xy_to_xyz(kD50X, kD50Y, d50_xyz)) {
        return ColorSpecStatus::non_finite_intermediate;
    }
    Matrix adaptation{};
    if (!chromatic_adaptation(d50_xyz, target_xyz, adaptation, failure)) return failure;

    Matrix pcs_to_camera{};
    if (!multiply(matrices.color, adaptation, pcs_to_camera)) {
        return ColorSpecStatus::non_finite_intermediate;
    }
    Vector pcs_white{};
    if (!multiply(pcs_to_camera, d50_xyz, pcs_white)) {
        return ColorSpecStatus::non_finite_intermediate;
    }
    const double pcs_white_max = *std::max_element(pcs_white.begin(), pcs_white.end());
    if (pcs_white_max == 0.0) return ColorSpecStatus::invalid_denominator;
    const double pcs_white_reciprocal = 1.0 / pcs_white_max;
    if (!std::isfinite(pcs_white_reciprocal)) {
        return ColorSpecStatus::non_finite_intermediate;
    }
    for (double &value : pcs_to_camera) {
        value *= pcs_white_reciprocal;
        if (!std::isfinite(value)) return ColorSpecStatus::non_finite_intermediate;
    }

    Matrix camera_to_pcs{};
    if (!matrices.has_forward) {
        if (!inverse(pcs_to_camera, camera_to_pcs, failure)) return failure;
    } else {
        Matrix ab_cc{};
        Matrix analog_calibration_inverse{};
        if (!multiply_diagonal_left(input.analog_balance, matrices.cc, ab_cc)) {
            return ColorSpecStatus::non_finite_intermediate;
        }
        if (!inverse(ab_cc, analog_calibration_inverse, failure)) return failure;
        Vector reference_white{};
        if (!multiply(analog_calibration_inverse, camera_white, reference_white)) {
            return ColorSpecStatus::non_finite_intermediate;
        }
        Matrix inverse_reference_diagonal{};
        if (!inverse(diagonal(reference_white), inverse_reference_diagonal, failure)) {
            return failure;
        }
        Matrix forward_times_diagonal{};
        if (!multiply(matrices.forward, inverse_reference_diagonal,
                      forward_times_diagonal) ||
            !multiply(forward_times_diagonal, analog_calibration_inverse,
                      camera_to_pcs)) {
            return ColorSpecStatus::non_finite_intermediate;
        }
    }
    if (!finite(camera_white) || !finite(pcs_to_camera) || !finite(camera_to_pcs) ||
        !std::isfinite(matrices.weight)) {
        return ColorSpecStatus::non_finite_intermediate;
    }

    ColorSpecPrepared candidate{};
    candidate.camera_white = camera_white;
    candidate.camera_to_pcs = camera_to_pcs;
    candidate.pcs_to_camera = pcs_to_camera;
    candidate.mix_weight = matrices.weight;
    output = candidate;
    return ColorSpecStatus::ok;
}

bool round_int32(double value, std::int32_t &result) noexcept {
    if (!std::isfinite(value)) return false;
    const double adjusted = value + (value > 0.0 ? 0.5 : -0.5);
    if (!std::isfinite(adjusted) || adjusted <= -2147483649.0 ||
        adjusted >= 2147483648.0) {
        return false;
    }
    result = static_cast<std::int32_t>(std::trunc(adjusted));
    return true;
}

std::int32_t clamp_int32(std::int32_t value, std::int32_t low,
                         std::int32_t high) noexcept {
    return std::max(low, std::min(value, high));
}

bool project_white(double x, double y, double &temperature,
                   double &tint) noexcept {
    WhiteTemperature projected{};
    if (!project_white_temperature(x, y, projected)) return false;
    temperature = std::clamp(projected.temperature, 2000.0, 50000.0);
    tint = std::clamp(projected.tint, -150.0, 150.0);
    return std::isfinite(temperature) && std::isfinite(tint);
}

bool temperature_tint_to_xy(double temperature, double tint,
                            std::array<double, 2> &xy) noexcept {
    if (!std::isfinite(temperature) || !std::isfinite(tint) ||
        temperature <= 0.0) return false;
    const double mired = 1.0e6 / temperature;
    if (!std::isfinite(mired)) return false;
    std::size_t j = 1;
    for (; j < 30; ++j) {
        if (kWhiteTemperatureTable[j][0] > mired) break;
    }
    const auto &row = kWhiteTemperatureTable[j];
    const auto &previous = kWhiteTemperatureTable[j - 1];
    const double mired_span = row[0] - previous[0];
    if (mired_span == 0.0 || !std::isfinite(mired_span)) return false;
    const double f = (row[0] - mired) / mired_span;
    const double q = 1.0 - f;
    const double current_norm = std::sqrt(1.0 + row[3] * row[3]);
    const double previous_norm = std::sqrt(1.0 + previous[3] * previous[3]);
    if (current_norm == 0.0 || previous_norm == 0.0 ||
        !std::isfinite(current_norm) || !std::isfinite(previous_norm)) return false;
    const double direction_a = q * (1.0 / current_norm) +
                               f * (1.0 / previous_norm);
    const double direction_b = q * (row[3] / current_norm) +
                               f * (previous[3] / previous_norm);
    const double direction_norm = std::sqrt(direction_a * direction_a +
                                             direction_b * direction_b);
    if (direction_norm == 0.0 || !std::isfinite(direction_norm)) return false;
    const double u0 = q * row[1] + f * previous[1];
    const double v0 = q * row[2] + f * previous[2];
    const double tint_scale = -tint / 3000.0;
    const double u = u0 + tint_scale * (direction_a / direction_norm);
    const double v = v0 + tint_scale * (direction_b / direction_norm);
    const double denominator = (u - 4.0 * v) + 2.0;
    if (denominator == 0.0 || !std::isfinite(denominator)) return false;
    xy = {1.5 * u / denominator, v / denominator};
    return std::isfinite(xy[0]) && std::isfinite(xy[1]);
}

bool absolute_temperature_tint(double temperature, double tint,
                               std::array<std::int32_t, 2> &encoded) noexcept {
    temperature = std::clamp(temperature, 2000.0, 50000.0);
    tint = std::clamp(tint, -150.0, 150.0);
    const double z = std::clamp(50000.0 / temperature, 1.0, 25.0);
    const double p = std::clamp(
        100.0 * (12.0 - std::sqrt(144.0 - 12.0 * (10.0 - z))) / 6.0,
        -100.0, 100.0);
    const double q = (std::sqrt((9.562965 - tint) * 38.25186 + 22500.0) - 150.0) /
                     -19.12593;
    const double signed_q = q < 0.0 ? -1.0 : (q > 0.0 ? 1.0 : 0.0);
    const double r = signed_q * 50.0 * (std::sqrt(8.0 * std::abs(q) + 1.0) - 1.0);
    if (!std::isfinite(p) || !std::isfinite(q) || !std::isfinite(r) ||
        !round_int32(p, encoded[0]) || !round_int32(r, encoded[1])) return false;
    encoded[0] = clamp_int32(encoded[0], -100, 100);
    encoded[1] = clamp_int32(encoded[1], -100, 100);
    return true;
}

bool relative_encode(std::int32_t value, std::int32_t base,
                     std::int32_t &encoded) noexcept {
    double result = static_cast<double>(value);
    if (value == base) {
        result = 0.0;
    } else if (base != 0) {
        if (value < base) {
            const double denominator = static_cast<double>(base) + 100.0;
            if (denominator == 0.0) return false;
            result = -100.0 + 100.0 *
                (static_cast<double>(value) + 100.0) / denominator;
        } else {
            const double denominator = 100.0 - static_cast<double>(base);
            if (denominator == 0.0) return false;
            result = ((static_cast<double>(value) - base) / denominator) * 100.0;
        }
        if (!round_int32(result, encoded)) return false;
    } else {
        encoded = value;
    }
    if (value == base) encoded = 0;
    encoded = clamp_int32(encoded, -100, 100);
    return true;
}

bool relative_decode(std::int32_t value, std::int32_t base,
                     std::int32_t &decoded) noexcept {
    if (value == 0) {
        decoded = base;
        return true;
    }
    if (base == 0) {
        decoded = value;
        return true;
    }
    double result = 0.0;
    if (value < 0) {
        result = -100.0 + (static_cast<double>(base) + 100.0) *
                 ((static_cast<double>(value) + 100.0) / 100.0);
    } else {
        result = static_cast<double>(base) + (100.0 - base) *
                 (static_cast<double>(value) / 100.0);
    }
    return round_int32(result, decoded);
}

ColorSpecStatus adjust_resolved_xy(const std::array<double, 2> &input_xy,
                                   bool output_referred_mode,
                                   std::array<double, 2> &output_xy) noexcept {
    double temperature = 0.0, tint = 0.0;
    if (!project_white(input_xy[0], input_xy[1], temperature, tint)) {
        return ColorSpecStatus::invalid_input;
    }
    std::array<std::int32_t, 2> adjusted{};
    if (!output_referred_mode) {
        std::int32_t kelvin = 0, tint_integer = 0;
        if (!round_int32(temperature, kelvin) || !round_int32(tint, tint_integer)) {
            return ColorSpecStatus::invalid_input;
        }
        kelvin = clamp_int32(kelvin, 2000, 50000);
        tint_integer = clamp_int32(tint_integer, -150, 150);

        std::int32_t index = 0;
        const double k = static_cast<double>(kelvin);
        double index_value = 0.0;
        if (k <= 2000.0) {
            index_value = 0.0;
        } else if (k <= 7900.0) {
            index_value = (k - 2000.0) / 50.0;
        } else if (k <= 10000.0) {
            index_value = 118.0 + (k - 7900.0) / 100.0;
        } else if (k <= 17000.0) {
            index_value = 139.0 + (k - 10000.0) / 250.0;
        } else {
            index_value = 167.0 + (k - 17000.0) / 1000.0;
        }
        if (k > 50000.0) index_value = 200.0;
        if (!round_int32(index_value, index)) return ColorSpecStatus::invalid_input;
        double decoded_kelvin = 0.0;
        if (index < 119) {
            decoded_kelvin = 2000.0 + 50.0 * index;
        } else if (index < 140) {
            decoded_kelvin = 7900.0 + 100.0 * (index - 118);
        } else if (index < 168) {
            decoded_kelvin = 10000.0 + 250.0 * (index - 139);
        } else {
            decoded_kelvin = 50000.0 + 1000.0 * (index - 200);
        }
        if (!round_int32(decoded_kelvin, adjusted[0])) {
            return ColorSpecStatus::invalid_input;
        }
        adjusted[1] = tint_integer;
        adjusted[0] = clamp_int32(adjusted[0], 2000, 50000);
        adjusted[1] = clamp_int32(adjusted[1], -150, 150);
        if (!temperature_tint_to_xy(adjusted[0], adjusted[1], output_xy)) {
            return ColorSpecStatus::non_finite_intermediate;
        }
        return ColorSpecStatus::ok;
    }

    std::array<std::int32_t, 2> target_absolute{};
    std::array<std::int32_t, 2> reference_absolute{};
    double reference_temperature = 0.0, reference_tint = 0.0;
    if (!project_white(kD50X, kD50Y, reference_temperature, reference_tint) ||
        !absolute_temperature_tint(temperature, tint, target_absolute) ||
        !absolute_temperature_tint(reference_temperature, reference_tint,
                                   reference_absolute) ||
        !relative_encode(target_absolute[0], reference_absolute[0], adjusted[0]) ||
        !relative_encode(target_absolute[1], reference_absolute[1], adjusted[1])) {
        return ColorSpecStatus::invalid_input;
    }
    if (adjusted[0] == 0 && adjusted[1] == 0) {
        output_xy = {kD50X, kD50Y};
        return ColorSpecStatus::ok;
    }

    std::array<std::int32_t, 2> absolute{};
    if (!relative_decode(adjusted[0], reference_absolute[0], absolute[0]) ||
        !relative_decode(adjusted[1], reference_absolute[1], absolute[1])) {
        return ColorSpecStatus::invalid_input;
    }
    const double temperature_coordinate = std::clamp(absolute[0] / 100.0, -1.0, 1.0);
    const double temperature_curve =
        (temperature_coordinate * 3.0 - 12.0) * temperature_coordinate + 10.0;
    if (temperature_curve == 0.0 || !std::isfinite(temperature_curve)) {
        return ColorSpecStatus::non_finite_intermediate;
    }
    const double decoded_temperature = 50000.0 / temperature_curve;

    const double tint_coordinate = std::clamp(absolute[1] / 100.0, -1.0, 1.0);
    const double tint_shape = tint_coordinate * 0.5 * (std::abs(tint_coordinate) + 1.0);
    const double decoded_tint = (-9.562965 * tint_shape + 150.0) * tint_shape + 9.562965;
    if (!temperature_tint_to_xy(decoded_temperature, decoded_tint, output_xy)) {
        return ColorSpecStatus::non_finite_intermediate;
    }
    return ColorSpecStatus::ok;
}

ColorSpecStatus xyz_to_xy(const Vector &xyz,
                          std::array<double, 2> &xy) noexcept {
    double sum = 0.0;
    if (!checked_add(xyz[0], xyz[1], sum) ||
        !checked_add(sum, xyz[2], sum)) {
        return ColorSpecStatus::non_finite_intermediate;
    }
    if (sum <= 0.0) {
        xy = {kD50X, kD50Y};
        return ColorSpecStatus::ok;
    }
    xy = {xyz[0] / sum, xyz[1] / sum};
    if (!std::isfinite(xy[0]) || !std::isfinite(xy[1])) {
        return ColorSpecStatus::non_finite_intermediate;
    }
    return ColorSpecStatus::ok;
}

ColorSpecStatus neutral_to_xy(const PreparedSlots &slots, const Vector &neutral,
                              std::array<double, 2> &output_xy) noexcept {
    std::array<double, 2> current{kD50X, kD50Y};
    for (std::size_t pass = 0; pass < 30; ++pass) {
        InterpolatedMatrices matrices{};
        const auto interpolation = interpolate_at_white(slots, current, matrices);
        if (interpolation != ColorSpecStatus::ok) return interpolation;
        Matrix inverse_color{};
        ColorSpecStatus failure = ColorSpecStatus::non_finite_intermediate;
        if (!inverse(matrices.color, inverse_color, failure)) return failure;
        Vector xyz{};
        if (!multiply(inverse_color, neutral, xyz)) {
            return ColorSpecStatus::non_finite_intermediate;
        }
        std::array<double, 2> next{};
        const auto conversion = xyz_to_xy(xyz, next);
        if (conversion != ColorSpecStatus::ok) return conversion;
        const double dx = std::abs(next[0] - current[0]);
        const double dy = std::abs(next[1] - current[1]);
        double metric = 0.0;
        if (!checked_add(dx, dy, metric)) return ColorSpecStatus::non_finite_intermediate;
        if (metric < 1.0e-7) {
            output_xy = pass == 29 ? current : next;
            return ColorSpecStatus::ok;
        }
        if (pass == 29) {
            output_xy = {(next[0] + current[0]) * 0.5,
                         (next[1] + current[1]) * 0.5};
            if (!std::isfinite(output_xy[0]) || !std::isfinite(output_xy[1])) {
                return ColorSpecStatus::non_finite_intermediate;
            }
            return ColorSpecStatus::ok;
        }
        current = next;
    }
    return ColorSpecStatus::non_finite_intermediate;
}

ColorSpecStatus build(const ColorSpecInput &input,
                      ColorSpecPrepared &output) noexcept {
    if (!finite_input(input)) return ColorSpecStatus::invalid_input;
    PreparedSlots slots{};
    const auto preparation = prepare_slots(input, slots);
    if (preparation != ColorSpecStatus::ok) return preparation;
    InterpolatedMatrices matrices{};
    const auto interpolation = interpolate_at_white(slots, input.white_xy, matrices);
    if (interpolation != ColorSpecStatus::ok) return interpolation;
    return set_white_xy(input, input.white_xy, matrices, output);
}

}  // namespace

ColorSpecStatus prepare_acr_colorspec(const ColorSpecInput &input,
                                      ColorSpecPrepared &output) noexcept {
    if (ranges_overlap(&input, sizeof(input), &output, sizeof(output))) {
        return ColorSpecStatus::invalid_input;
    }
    return build(input, output);
}

ColorSpecStatus resolve_acr_white(const WhiteResolveInput &input,
                                  WhiteResolvePrepared &output) noexcept {
    if (ranges_overlap(&input, sizeof(input), &output, sizeof(output))) {
        return ColorSpecStatus::invalid_input;
    }
    if (!finite_profile_input(input.profile) || !finite(input.camera_neutral) ||
        *std::max_element(input.camera_neutral.begin(), input.camera_neutral.end()) <= 0.0) {
        return ColorSpecStatus::invalid_input;
    }

    PreparedSlots slots{};
    const auto preparation = prepare_slots(input.profile, slots);
    if (preparation != ColorSpecStatus::ok) return preparation;

    std::array<double, 2> neutral_xy{};
    const auto neutral_status = neutral_to_xy(slots, input.camera_neutral, neutral_xy);
    if (neutral_status != ColorSpecStatus::ok) return neutral_status;

    std::array<double, 2> adjusted_xy{};
    const auto adjustment = adjust_resolved_xy(neutral_xy, input.output_referred_mode,
                                               adjusted_xy);
    if (adjustment != ColorSpecStatus::ok) return adjustment;

    InterpolatedMatrices matrices{};
    const auto interpolation = interpolate_at_white(slots, adjusted_xy, matrices);
    if (interpolation != ColorSpecStatus::ok) return interpolation;

    ColorSpecPrepared colorspec{};
    const auto white_status = set_white_xy(input.profile, adjusted_xy, matrices, colorspec);
    if (white_status != ColorSpecStatus::ok) return white_status;

    WhiteResolvePrepared candidate{};
    candidate.neutral_xy = neutral_xy;
    candidate.adjusted_xy = adjusted_xy;
    candidate.colorspec = colorspec;
    output = candidate;
    return ColorSpecStatus::ok;
}

ColorSpecStatus prepare_normalized_prophoto_inverse(
    std::array<double, 9> &inverse_output,
    std::array<double, 9> *normalized_matrix) noexcept {
    if (normalized_matrix != nullptr &&
        ranges_overlap(&inverse_output, sizeof(inverse_output),
                       normalized_matrix, sizeof(*normalized_matrix))) {
        return ColorSpecStatus::invalid_input;
    }
    constexpr Matrix prophoto = {
        0.7977, 0.1352, 0.0313,
        0.2880, 0.7119, 0.0001,
        0.0, 0.0, 0.8249,
    };
    Matrix normalized{};
    ColorSpecStatus failure = ColorSpecStatus::non_finite_intermediate;
    if (!normalize_forward(prophoto, normalized, failure)) return failure;
    Matrix inverse_candidate{};
    if (!inverse(normalized, inverse_candidate, failure)) return failure;
    if (!finite(normalized) || !finite(inverse_candidate)) {
        return ColorSpecStatus::non_finite_intermediate;
    }
    inverse_output = inverse_candidate;
    if (normalized_matrix != nullptr) *normalized_matrix = normalized;
    return ColorSpecStatus::ok;
}

}  // namespace imprint::acr
