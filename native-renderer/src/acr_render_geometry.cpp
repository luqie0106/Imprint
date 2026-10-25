#include "acr_render_geometry.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace imprint::acr {
namespace {

constexpr std::uint32_t kMaximumLevel = 5;
constexpr std::uint32_t kMaximumIterations = 12;
constexpr double kSqrtTwo = 1.4142135623730951;

using Matrix3 = ResampleMatrix3;

Matrix3 identity_matrix() noexcept {
    return {1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0};
}

bool finite_matrix(const Matrix3 &matrix) noexcept {
    for (const double value : matrix)
        if (!std::isfinite(value)) return false;
    return true;
}

bool multiply_matrix(const Matrix3 &lhs, const Matrix3 &rhs,
                     Matrix3 &product) noexcept {
    Matrix3 candidate{};
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t column = 0; column < 3; ++column) {
            double accumulator = 0.0;
            for (std::size_t k = 0; k < 3; ++k) {
                accumulator = std::fma(lhs[row * 3 + k],
                                       rhs[k * 3 + column], accumulator);
                if (!std::isfinite(accumulator)) return false;
            }
            candidate[row * 3 + column] = accumulator;
        }
    }
    product = candidate;
    return true;
}

bool valid_rect_spans(const AnchorRectI32 &rect) noexcept {
    const std::int64_t height = static_cast<std::int64_t>(rect.bottom) -
                                static_cast<std::int64_t>(rect.top);
    const std::int64_t width = static_cast<std::int64_t>(rect.right) -
                               static_cast<std::int64_t>(rect.left);
    return height >= 1 &&
           height <= static_cast<std::int64_t>(INT32_MAX) &&
           width >= 1 &&
           width <= static_cast<std::int64_t>(INT32_MAX);
}

std::int32_t arm_negate_i32(std::int32_t value) noexcept {
    const std::uint32_t bits = 0u - static_cast<std::uint32_t>(value);
    std::int32_t result = 0;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

bool matrix_norm(const Matrix3 &matrix, double &norm) noexcept {
    const double first_sum = matrix[0] + matrix[1];
    const double second_sum = matrix[3] + matrix[4];
    if (!std::isfinite(first_sum) || !std::isfinite(second_sum)) return false;
    const double magnitude = std::hypot(first_sum, second_sum);
    if (!std::isfinite(magnitude)) return false;
    norm = magnitude / kSqrtTwo;
    return std::isfinite(norm);
}

bool next_level_available(const RenderGeometrySource &source,
                          std::uint32_t level) noexcept {
    return level < kMaximumLevel && source.levels[level + 1].image != nullptr;
}

struct FirstQwordEntry {
    std::uint32_t key;
    std::uint64_t value_bits;
};

constexpr std::array<FirstQwordEntry, 1> kDefaultFirstQwordTable = {{
    {0u, 0x3ff8000000000000ull},
}};

constexpr std::array<FirstQwordEntry, 1> kMode0FirstQwordTable = {{
    {72u, 0x4008000000000000ull},
}};

constexpr std::array<FirstQwordEntry, 5> kMode1FirstQwordTable = {{
    {180u, 0x3ff3333333333333ull},
    {240u, 0x3ffccccccccccccdull},
    {300u, 0x400999999999999aull},
    {360u, 0x400c000000000000ull},
    {480u, 0x400f333333333333ull},
}};

constexpr std::array<FirstQwordEntry, 5> kMode2FirstQwordTable = {{
    {180u, 0x3ff3333333333333ull},
    {240u, 0x3ffccccccccccccdull},
    {300u, 0x400999999999999aull},
    {360u, 0x400c000000000000ull},
    {480u, 0x400f333333333333ull},
}};

double double_from_bits(std::uint64_t bits) noexcept {
    double value = 0.0;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

template <std::size_t N>
double first_qword_for_dpi(
    double dpi, const std::array<FirstQwordEntry, N> &table) noexcept {
    if (dpi <= static_cast<double>(table.front().key))
        return double_from_bits(table.front().value_bits);
    if (dpi >= static_cast<double>(table.back().key))
        return double_from_bits(table.back().value_bits);

    std::size_t upper_index = 1;
    while (upper_index < N &&
           dpi >= static_cast<double>(table[upper_index].key))
        ++upper_index;
    if (upper_index >= N) return double_from_bits(table.back().value_bits);

    const FirstQwordEntry &lower = table[upper_index - 1];
    const FirstQwordEntry &upper = table[upper_index];
    const double lower_value = double_from_bits(lower.value_bits);
    const double upper_value = double_from_bits(upper.value_bits);
    const double key_delta = static_cast<double>(upper.key - lower.key);
    const double reciprocal = 1.0 / key_delta;
    const double dpi_delta = dpi - static_cast<double>(lower.key);
    double t = dpi_delta * reciprocal;
    t = std::fmin(t, 1.0);
    if (t < 0.0) t = 0.0;
    const double value_delta = upper_value - lower_value;
    return std::fma(t, value_delta, lower_value);
}

}  // namespace

RenderGeometryStatus prepare_render_geometry(
    const RenderGeometryInput &input,
    RenderGeometryResult &result) noexcept {
    if (input.extent3[0] <= 0 || input.extent3[1] <= 0 ||
        input.extent4[0] <= 0 || input.extent4[1] <= 0 ||
        input.initial_level < 0 ||
        input.initial_level > static_cast<std::int32_t>(kMaximumLevel))
        return RenderGeometryStatus::invalid_input;

    if (input.optional_present && !finite_matrix(input.optional_matrix))
        return RenderGeometryStatus::non_finite_matrix;

    const std::uint32_t initial_level =
        static_cast<std::uint32_t>(input.initial_level);
    bool mask_290 = false;
    std::uint64_t mask_first_bits = 0;
    bool mask_first_value_present = false;
    const bool mask_eligible =
        input.param6 != 0 && input.context.field_847 != 0 &&
        input.source.field_68a == 0 && initial_level == 0;
    if (mask_eligible) {
        const std::uint8_t initial_mask = static_cast<std::uint8_t>(
            input.context.field_846 ^ 1u);
        mask_290 = initial_mask != 0;
        if (input.context.field_846 == 0 && input.context.field_820 == 0) {
            double first_bound = 0.0;
            mask_first_bits = input.context.field_8a8[0];
            std::memcpy(&first_bound, &mask_first_bits, sizeof(first_bound));
            mask_first_value_present = true;
            mask_290 = first_bound != 0.0;
        } else if (input.context.field_846 == 0) {
            if (!std::isfinite(input.context.field_810))
                return RenderGeometryStatus::unsupported_mask_conversion;

            constexpr std::uint64_t kDpiMultiplierBits =
                0x400451eb851eb852ull;
            volatile double multiplied =
                input.context.field_810 * double_from_bits(kDpiMultiplierBits);
            const double selected_dpi = input.context.field_818 == 3
                                            ? multiplied
                                            : input.context.field_810;

            if (input.feature_5a != 0 && input.context.field_824 > 2)
                return RenderGeometryStatus::unsupported_mask_conversion;

            double first_value = 0.0;
            if (input.feature_5a == 0) {
                first_value = first_qword_for_dpi(
                    selected_dpi, kDefaultFirstQwordTable);
            } else if (input.context.field_824 == 0) {
                first_value = first_qword_for_dpi(
                    selected_dpi, kMode0FirstQwordTable);
            } else if (input.context.field_824 == 1) {
                first_value = first_qword_for_dpi(
                    selected_dpi, kMode1FirstQwordTable);
            } else {
                first_value = first_qword_for_dpi(
                    selected_dpi, kMode2FirstQwordTable);
            }
            std::memcpy(&mask_first_bits, &first_value, sizeof(first_value));
            mask_first_value_present = true;
            mask_290 = first_value != 0.0;
        }
    }

    RenderGeometryContext context = input.context;
    std::uint32_t level = initial_level;
    bool final_flag_0c = false;
    Matrix3 final_e0{};
    Matrix3 final_170{};
    Matrix3 final_200{};
    std::uint32_t completed_iterations = 0;

    for (std::uint32_t iteration = 1;
         iteration <= kMaximumIterations; ++iteration) {
        const RenderGeometryLevel &current = input.source.levels[level];
        if (current.image == nullptr)
            return RenderGeometryStatus::unavailable_level;
        if (!valid_rect_spans(input.source.levels[0].reference_bounds) ||
            !valid_rect_spans(current.reference_bounds))
            return RenderGeometryStatus::invalid_bounds;

        Matrix3 e0 = identity_matrix();
        Matrix3 matrix_170 = identity_matrix();
        Matrix3 matrix_200 = identity_matrix();

        const bool flag_0c = context.field_847 != 0 &&
                             input.source.field_68a == 0 && level == 0;

        const AnchorRectI32 &level_zero =
            input.source.levels[0].reference_bounds;
        const std::int64_t height =
            static_cast<std::int64_t>(level_zero.bottom) - level_zero.top;
        const std::int64_t width =
            static_cast<std::int64_t>(level_zero.right) - level_zero.left;
        double sy = 1.0 / static_cast<double>(height);
        double sx = 1.0 / static_cast<double>(width);
        if (!std::isfinite(sy) || !std::isfinite(sx))
            return RenderGeometryStatus::non_finite_matrix;
        if (level != 0) {
            const double level_factor = std::exp2(static_cast<double>(level));
            if (!std::isfinite(level_factor))
                return RenderGeometryStatus::non_finite_matrix;
            sy *= level_factor;
            sx *= level_factor;
        }
        if (!std::isfinite(sy) || !std::isfinite(sx))
            return RenderGeometryStatus::non_finite_matrix;

        const std::int32_t negative_top = arm_negate_i32(current.reference_bounds.top);
        const std::int32_t negative_left = arm_negate_i32(current.reference_bounds.left);
        const double translation_y = sy * static_cast<double>(negative_top);
        const double translation_x = sx * static_cast<double>(negative_left);
        if (!std::isfinite(translation_y) || !std::isfinite(translation_x))
            return RenderGeometryStatus::non_finite_matrix;

        const Matrix3 level_matrix = {
            sy, 0.0, translation_y,
            0.0, sx, translation_x,
            0.0, 0.0, 1.0};
        if (input.optional_present) {
            if (!multiply_matrix(input.optional_matrix, level_matrix, e0))
                return RenderGeometryStatus::non_finite_matrix;
        } else {
            e0 = level_matrix;
        }

        const Matrix3 extent_matrix = {
            static_cast<double>(input.extent3[0]), 0.0, 0.0,
            0.0, static_cast<double>(input.extent3[1]), 0.0,
            0.0, 0.0, 1.0};
        Matrix3 composed_e0{};
        if (!multiply_matrix(extent_matrix, e0, composed_e0))
            return RenderGeometryStatus::non_finite_matrix;
        e0 = composed_e0;

        const double ratio_x = static_cast<double>(input.extent4[0]) /
                               static_cast<double>(input.extent3[0]);
        const double ratio_y = static_cast<double>(input.extent4[1]) /
                               static_cast<double>(input.extent3[1]);
        if (!std::isfinite(ratio_x) || !std::isfinite(ratio_y))
            return RenderGeometryStatus::non_finite_matrix;
        const Matrix3 ratio_matrix = {
            ratio_x, 0.0, 0.0,
            0.0, ratio_y, 0.0,
            0.0, 0.0, 1.0};
        if (ratio_x < 1.0 || ratio_y < 1.0)
            matrix_170 = ratio_matrix;
        else if (ratio_x > 1.0 || ratio_y > 1.0)
            matrix_200 = ratio_matrix;

        if (!mask_290) {
            Matrix3 folded_e0{};
            if (!multiply_matrix(matrix_170, e0, folded_e0))
                return RenderGeometryStatus::non_finite_matrix;
            e0 = folded_e0;
            matrix_170 = identity_matrix();
        }

        const bool version_limit =
            static_cast<std::uint32_t>(context.field_00 + 1u) <
            0x05070001u;
        const bool b_gate = version_limit && input.source.field_68a == 0 &&
                            flag_0c;
        const bool d_gate = version_limit && flag_0c &&
                            (context.field_911 != 0 || context.field_7c != 0);
        const bool g_gate =
            (context.field_8c == 0 &&
             (context.field_98 == 0 || input.source.field_178 == 1)) ||
            context.field_846 != 0;
        const bool skip_first_gate = b_gate && !g_gate;
        const bool has_next_level = next_level_available(input.source, level);

        if (!skip_first_gate && !d_gate && has_next_level) {
            double first_norm = 0.0;
            if (!matrix_norm(e0, first_norm))
                return RenderGeometryStatus::non_finite_matrix;
            if (first_norm < 0.25) {
                if (flag_0c && context.field_84c == 0)
                    context.field_84c = 5;
                level += 1;
                completed_iterations = iteration;
                final_flag_0c = flag_0c;
                final_e0 = e0;
                final_170 = matrix_170;
                final_200 = matrix_200;
                continue;
            }
        }

        Matrix3 temp_200_170{};
        Matrix3 full_matrix{};
        if (!multiply_matrix(matrix_200, matrix_170, temp_200_170) ||
            !multiply_matrix(temp_200_170, e0, full_matrix))
            return RenderGeometryStatus::non_finite_matrix;
        double late_norm = 0.0;
        if (!matrix_norm(full_matrix, late_norm))
            return RenderGeometryStatus::non_finite_matrix;

        completed_iterations = iteration;
        final_flag_0c = flag_0c;
        final_e0 = e0;
        final_170 = matrix_170;
        final_200 = matrix_200;

        if (late_norm < 0.015625) {
            if (has_next_level) {
                level += 1;
                continue;
            }
            if (context.field_846 == 0) {
                context.field_846 = 1;
                continue;
            }
        }

        if (!valid_rect_spans(current.actual_bounds))
            return RenderGeometryStatus::invalid_bounds;

        RenderGeometryResult candidate{};
        candidate.level = level;
        candidate.image = current.image;
        candidate.reference_bounds = current.reference_bounds;
        candidate.actual_bounds = current.actual_bounds;
        candidate.output_bounds = current.actual_bounds;
        candidate.matrix_e0 = final_e0;
        candidate.matrix_170 = final_170;
        candidate.matrix_200 = final_200;
        candidate.context_after = context;
        candidate.flag_0c = final_flag_0c;
        candidate.mask_290 = mask_290;
        candidate.iterations = completed_iterations;
        if (mask_first_value_present)
            std::memcpy(&candidate.mask_first_value, &mask_first_bits,
                        sizeof(candidate.mask_first_value));
        result = candidate;
        if (mask_first_value_present)
            std::memcpy(&result.mask_first_value, &mask_first_bits,
                        sizeof(result.mask_first_value));
        return RenderGeometryStatus::ok;
    }

    return RenderGeometryStatus::invalid_input;
}

}  // namespace imprint::acr
