// CTest for the recovered dng_linearize_plane LUT build + Process.
//
// The reference below is written from the recovered ARM64 formulas
// (FUN_01304c54 / FUN_01303a18 / FUN_01305598 / FUN_0130312c / FUN_01303c4c)
// and is deliberately independent of the operator's internal table helpers.
// The stronger exhaustive comparison against the independent Python oracle and
// the official CMake object lives in
// research/acr/tools/validate_acr_source_linearization.py.
#include "acr_source_linearization.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using imprint::acr::DestinationFloor;
using imprint::acr::LinearizationInfo;
using imprint::acr::LinearizationInputType;
using imprint::acr::LinearizationOutputType;
using imprint::acr::LinearizationStatus;
using imprint::acr::SourceLinearization;

namespace {

int failures = 0;

#define CHECK(condition)                                                              \
    do {                                                                              \
        if (!(condition)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
            ++failures;                                                               \
        }                                                                             \
    } while (0)

#define CHECK_EQ(a, b)                                                                \
    do {                                                                              \
        auto lhs = (a);                                                               \
        auto rhs = (b);                                                               \
        if (!(lhs == rhs)) {                                                          \
            std::fprintf(stderr, "FAIL %s:%d: %s != %s (%lld vs %lld)\n", __FILE__,   \
                         __LINE__, #a, #b, (long long)lhs, (long long)rhs);           \
            ++failures;                                                               \
        }                                                                             \
    } while (0)

// --- Independent reference -------------------------------------------------

double reference_max_black(const LinearizationInfo& info, std::uint32_t plane) {
    if (info.black_repeat_rows == 0 || info.black_repeat_cols == 0) return 0.0;
    double column_max[8] = {};
    double row_max[8] = {};
    if (info.has_black_level_delta_h) {
        for (std::size_t i = 0; i < info.black_level_delta_h.size(); ++i) {
            std::uint32_t column = static_cast<std::uint32_t>(i % info.black_repeat_cols);
            if (i < info.black_repeat_cols) column_max[column] = info.black_level_delta_h[i];
            else if (column_max[column] <= info.black_level_delta_h[i]) column_max[column] = info.black_level_delta_h[i];
        }
    }
    if (info.has_black_level_delta_v) {
        for (std::size_t i = 0; i < info.black_level_delta_v.size(); ++i) {
            std::uint32_t row = static_cast<std::uint32_t>(i % info.black_repeat_rows);
            if (i < info.black_repeat_rows) row_max[row] = info.black_level_delta_v[i];
            else if (row_max[row] <= info.black_level_delta_v[i]) row_max[row] = info.black_level_delta_v[i];
        }
    }
    double maximum = 0.0;
    bool first = true;
    for (std::uint32_t row = 0; row < info.black_repeat_rows; ++row) {
        for (std::uint32_t col = 0; col < info.black_repeat_cols; ++col) {
            double value = info.black_level[row][col][plane] + column_max[col] + row_max[row];
            if (first) { maximum = value; first = false; }
            else if (maximum <= value) maximum = value;
        }
    }
    return maximum;
}

bool reference_quantize(LinearizationInfo* info) {
    double absolute_max = 0.0;
    for (std::uint32_t row = 0; row < info->black_repeat_rows; ++row)
        for (std::uint32_t col = 0; col < info->black_repeat_cols; ++col)
            for (int plane = 0; plane < 4; ++plane) {
                double value = info->black_level[row][col][plane];
                if (!std::isfinite(value)) return false;
                double magnitude = value <= 0.0 ? -value : value;
                if (absolute_max <= magnitude) absolute_max = magnitude;
            }
    auto scan = [&](const std::vector<double>& values) {
        for (double value : values) {
            if (!std::isfinite(value)) return false;
            double magnitude = value <= 0.0 ? -value : value;
            if (absolute_max <= magnitude) absolute_max = magnitude;
        }
        return true;
    };
    if (!scan(info->black_level_delta_h) || !scan(info->black_level_delta_v)) return false;
    std::uint32_t denominator = 0x100;
    if (1966080000.0 <= absolute_max * 256.0) {
        if (1966080000.0 <= absolute_max * 128.0) {
            if (1966080000.0 <= absolute_max * 64.0) {
                if (1966080000.0 <= absolute_max * 32.0) {
                    if (1966080000.0 <= absolute_max * 16.0) {
                        if (1966080000.0 <= absolute_max * 8.0) {
                            if (1966080000.0 <= absolute_max * 4.0) {
                                denominator = 1966080000.0 <= absolute_max + absolute_max ? 1 : 2;
                            } else denominator = 4;
                        } else denominator = 8;
                    } else denominator = 0x10;
                } else denominator = 0x20;
            } else denominator = 0x40;
        } else denominator = 0x80;
    }
    for (std::uint32_t row = 0; row < info->black_repeat_rows; ++row)
        for (std::uint32_t col = 0; col < info->black_repeat_cols; ++col)
            for (int plane = 0; plane < 4; ++plane) {
                double value = info->black_level[row][col][plane];
                double result = 0.0;
                if (value > 0.0) {
                    double scaled = denominator * value + 0.5;
                    result = std::floor(scaled) / denominator;
                }
                info->black_level[row][col][plane] = result;
            }
    auto quantize_signed = [&](std::vector<double>* values) {
        for (double& value : *values) {
            double scaled = denominator * value;
            double adjusted = scaled <= 0.0 ? scaled - 0.5 : scaled + 0.5;
            if (!(-2147483649.0 < adjusted && adjusted < 2147483648.0)) return false;
            value = static_cast<double>(static_cast<std::int32_t>(adjusted)) / denominator;
        }
        return true;
    };
    return quantize_signed(&info->black_level_delta_h) &&
           quantize_signed(&info->black_level_delta_v);
}

int reference_round(double value) {
    double adjusted = value <= 0.0 ? value - 0.5 : value + 0.5;
    return static_cast<int>(adjusted);
}

std::uint16_t reference_saturate(int value) {
    if (value < 0) return 0;
    if (value > 0xffff) return 0xffff;
    return static_cast<std::uint16_t>(value);
}

struct ReferenceTables {
    bool branch_b = false;
    std::uint32_t rows = 0;
    std::uint32_t cols = 0;
    std::uint32_t row_count = 0;
    std::vector<std::uint16_t> main_u16;
    std::vector<std::int32_t> main_i24;
    std::vector<std::int32_t> black_2d;
    std::vector<std::int32_t> row_lut;
};

bool reference_build(const LinearizationInfo& info, std::uint32_t plane,
                     double scale, std::uint16_t max16, bool clamp_flag,
                     ReferenceTables* tables) {
    std::uint32_t width = static_cast<std::uint32_t>(info.rect_right - info.rect_left);
    std::uint32_t rows = 0;
    std::uint32_t cols = 0;
    if (info.has_black_level_delta_h) { rows = info.black_repeat_rows; cols = width; }
    else if (info.black_repeat_cols >= 2) { rows = info.black_repeat_rows; cols = info.black_repeat_cols; }
    if (rows == 0) cols = 0;
    std::uint32_t row_count = 0;
    if (info.has_black_level_delta_v) row_count = static_cast<std::uint32_t>(info.rect_bottom - info.rect_top);
    else if (rows == 0 && info.black_repeat_rows > 1) row_count = info.black_repeat_rows;
    tables->rows = rows;
    tables->cols = cols;
    tables->row_count = row_count;
    double weight = static_cast<double>(max16 ^ 0xffffu);
    std::uint32_t weight8 = static_cast<std::uint32_t>(max16 ^ 0xffffu) << 8;
    if (rows == 0 && row_count == 0) {
        tables->branch_b = false;
        tables->main_u16.resize(0x10000);
        for (std::uint32_t index = 0; index < 0x10000; ++index) {
            std::uint32_t value = index;
            if (info.has_linearization_table) {
                std::size_t table_index = std::min<std::size_t>(index, info.linearization_table.size() - 1);
                value = info.linearization_table[table_index];
            }
            double normalized = scale * (static_cast<double>(value) - info.black_level[0][0][plane]);
            double mapped = normalized;
            if (clamp_flag) {
                double upper = std::isnan(normalized) ? 1.0 : std::min(normalized, 1.0);
                mapped = 0.0 <= upper ? upper : 0.0;
            }
            // FUN_0130312c @01303844: one fused multiply-add.
            tables->main_u16[index] = reference_saturate(
                reference_round(std::fma(weight, mapped, static_cast<double>(max16))));
        }
    } else {
        tables->branch_b = true;
        tables->main_i24.resize(0x10000);
        for (std::uint32_t index = 0; index < 0x10000; ++index) {
            std::uint32_t value = index;
            if (info.has_linearization_table) {
                std::size_t table_index = std::min<std::size_t>(index, info.linearization_table.size() - 1);
                value = info.linearization_table[table_index];
            }
            // FUN_0130312c @013036d4: fused multiply-add, then exact *256.
            tables->main_i24[index] = reference_round(
                std::fma(weight, scale * static_cast<double>(value),
                         static_cast<double>(max16)) * 256.0);
        }
        if (rows != 0 && cols != 0) {
            tables->black_2d.resize(rows * cols);
            for (std::uint32_t row = 0; row < rows; ++row)
                for (std::uint32_t col = 0; col < cols; ++col) {
                    double value = info.black_level[row][col % info.black_repeat_cols][plane];
                    if (info.has_black_level_delta_h) value += info.black_level_delta_h[col];
                    tables->black_2d[row * cols + col] = reference_round(
                        scale * value * static_cast<double>(static_cast<std::int32_t>(weight8)));
                }
        }
        if (row_count != 0) {
            tables->row_lut.resize(row_count);
            for (std::uint32_t row = 0; row < row_count; ++row) {
                double value = 0.0;
                if (rows == 0) value = info.black_level[row % info.black_repeat_rows][0][plane];
                if (info.has_black_level_delta_v) value += info.black_level_delta_v[row];
                tables->row_lut[row] = reference_round(
                    scale * value * static_cast<double>(static_cast<std::int32_t>(weight8)));
            }
        }
    }
    return true;
}

std::uint16_t reference_pixel(const ReferenceTables& tables, const LinearizationInfo& info,
                              std::uint32_t plane, std::int32_t absolute_row,
                              std::int32_t absolute_col, std::uint32_t pixel) {
    if (!tables.branch_b) return tables.main_u16[pixel];
    // FUN_01303c4c: the row offset and both subtractions wrap mod 2^32, and the
    // stored halfword is selected from the 32-bit bit pattern.
    std::uint32_t row_offset = 0u - 0x80u;
    if (tables.row_count) {
        row_offset = static_cast<std::uint32_t>(
            tables.row_lut[static_cast<std::size_t>(absolute_row) % tables.row_count]) - 0x80u;
    }
    std::uint32_t value = static_cast<std::uint32_t>(tables.main_i24[pixel]) - row_offset;
    if (tables.cols != 0) {
        std::size_t row_index = static_cast<std::size_t>(absolute_row % tables.rows);
        std::size_t col_index = static_cast<std::size_t>(absolute_col % tables.cols);
        value -= static_cast<std::uint32_t>(tables.black_2d[row_index * tables.cols + col_index]);
    }
    if ((value >> 24) == 0) return static_cast<std::uint16_t>(value >> 8);
    return (value >> 31) != 0 ? std::uint16_t{0} : std::uint16_t{0xffff};
}

// --- Fixtures --------------------------------------------------------------

LinearizationInfo make_info(std::uint32_t rows, std::uint32_t cols,
                            std::int32_t width, std::int32_t height) {
    LinearizationInfo info;
    info.rect_top = 0;
    info.rect_left = 0;
    info.rect_bottom = height;
    info.rect_right = width;
    info.black_repeat_rows = rows;
    info.black_repeat_cols = cols;
    info.plane_count = 1;
    for (int plane = 0; plane < 4; ++plane) info.white_level[plane] = 65535.0;
    return info;
}

void compare_tables(const SourceLinearization& stage, const ReferenceTables& reference) {
    CHECK_EQ(stage.branch_b(), reference.branch_b);
    CHECK_EQ(stage.black_2d_rows(), reference.rows);
    CHECK_EQ(stage.black_2d_cols(), reference.cols);
    CHECK_EQ(stage.row_lut_count(), reference.row_count);
    CHECK(stage.main_u16() == reference.main_u16);
    CHECK(stage.main_i24() == reference.main_i24);
    CHECK(stage.black_2d() == reference.black_2d);
    CHECK(stage.row_lut() == reference.row_lut);
}

void test_identity_branch_a() {
    LinearizationInfo info = make_info(1, 1, 256, 256);
    CHECK_EQ(static_cast<int>(imprint::acr::quantize_linearization_info(&info)), 0);
    DestinationFloor floor = imprint::acr::compute_destination_floor(
        info, 1, LinearizationOutputType::u16, true, false, 0, false);
    CHECK_EQ(floor.max16, 0);
    CHECK_EQ(floor.clamp_flag, false);
    SourceLinearization stage;
    CHECK_EQ(static_cast<int>(stage.build(info, 0, LinearizationInputType::u16,
                                          LinearizationOutputType::u16, floor.max16,
                                          floor.clamp_flag)), 0);
    CHECK(!stage.branch_b());
    std::vector<std::uint16_t> pixels(256 * 256);
    for (std::size_t i = 0; i < pixels.size(); ++i) pixels[i] = static_cast<std::uint16_t>(i);
    std::vector<std::uint16_t> output(pixels.size(), 0xA5A5);
    CHECK_EQ(static_cast<int>(stage.process(pixels.data(), 256, output.data(), 256,
                                            0, 0, 256, 256)), 0);
    CHECK(output == pixels);
}

void test_dji_anchor_values() {
    LinearizationInfo info = make_info(2, 2, 4, 2);
    for (int row = 0; row < 2; ++row)
        for (int col = 0; col < 2; ++col) info.black_level[row][col][0] = 4096.0;
    info.white_level[0] = 65472.0;
    CHECK_EQ(static_cast<int>(imprint::acr::quantize_linearization_info(&info)), 0);
    DestinationFloor floor = imprint::acr::compute_destination_floor(
        info, 1, LinearizationOutputType::u16, true, false, 0, true);
    CHECK_EQ(floor.max16, 4100);
    CHECK_EQ(floor.clamp_flag, false);
    SourceLinearization stage;
    CHECK_EQ(static_cast<int>(stage.build(info, 0, LinearizationInputType::u16,
                                          LinearizationOutputType::u16, floor.max16,
                                          floor.clamp_flag)), 0);
    CHECK(stage.branch_b());
    CHECK_EQ(stage.black_2d().size(), 4u);
    CHECK_EQ(stage.black_2d()[0], 1049584);
    CHECK_EQ(stage.main_i24()[0], 1049600);
    CHECK_EQ(stage.main_i24()[4096], 2099184);
    CHECK_EQ(stage.main_i24()[65472], 17826544);
    CHECK_EQ(stage.main_i24()[65535], 17842687);
    std::uint16_t input[8] = {4096, 4097, 0, 65535, 65535, 4096, 65472, 3976};
    std::uint16_t output[8] = {};
    CHECK_EQ(static_cast<int>(stage.process(input, 4, output, 4, 0, 0, 2, 4)), 0);
    const std::uint16_t expected[8] = {4100, 4101, 0, 65535, 65535, 4100, 65535, 3980};
    for (int i = 0; i < 8; ++i) CHECK_EQ(output[i], expected[i]);
}

void test_random_reference() {
    std::mt19937 rng(0x51C0FFEEu);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    for (int case_index = 0; case_index < 200; ++case_index) {
        std::uint32_t rows = std::uniform_int_distribution<int>(1, 8)(rng);
        std::uint32_t cols = std::uniform_int_distribution<int>(1, 8)(rng);
        std::int32_t width = std::uniform_int_distribution<int>(2, 8)(rng);
        std::int32_t height = std::uniform_int_distribution<int>(2, 8)(rng);
        LinearizationInfo info = make_info(rows, cols, width, height);
        double white = 4095.0 + std::floor(unit(rng) * 60000.0);
        info.white_level[0] = white;
        for (std::uint32_t row = 0; row < rows; ++row)
            for (std::uint32_t col = 0; col < cols; ++col)
                info.black_level[row][col][0] = unit(rng) * white * 0.19;
        if (unit(rng) < 0.5) {
            info.has_black_level_delta_h = true;
            info.black_level_delta_h.resize(width);
            for (auto& value : info.black_level_delta_h) value = (unit(rng) - 0.5) * white * 0.01;
        }
        if (unit(rng) < 0.5) {
            info.has_black_level_delta_v = true;
            info.black_level_delta_v.resize(height);
            for (auto& value : info.black_level_delta_v) value = (unit(rng) - 0.5) * white * 0.01;
        }
        if (unit(rng) < 0.4) {
            info.has_linearization_table = true;
            std::uint32_t count = std::uniform_int_distribution<int>(1, 0x10000)(rng);
            info.linearization_table.resize(count);
            for (auto& value : info.linearization_table) value = static_cast<std::uint16_t>(rng());
        }
        LinearizationInfo reference = info;
        CHECK(reference_quantize(&reference));
        CHECK_EQ(static_cast<int>(imprint::acr::quantize_linearization_info(&info)), 0);
        for (std::uint32_t row = 0; row < rows; ++row)
            for (std::uint32_t col = 0; col < cols; ++col)
                CHECK_EQ(info.black_level[row][col][0] == reference.black_level[row][col][0], true);
        if (info.has_black_level_delta_h) CHECK(info.black_level_delta_h == reference.black_level_delta_h);
        if (info.has_black_level_delta_v) CHECK(info.black_level_delta_v == reference.black_level_delta_v);

        bool gate = unit(rng) < 0.8;
        bool mosaic = unit(rng) < 0.7;
        DestinationFloor floor = imprint::acr::compute_destination_floor(
            info, 1, LinearizationOutputType::u16, gate, false, 0, mosaic);
        if (!gate) {
            CHECK_EQ(floor.max16, 0);
            CHECK_EQ(floor.clamp_flag, true);
            continue;
        }
        double black_max = reference_max_black(info, 0);
        double reference_ratio = (0.0 < black_max && black_max < white) ? black_max / white : 0.0;
        double reference_value = reference_ratio * 65535.0 + 0.5;
        if (0.2 <= reference_ratio) reference_value = 13107.5;
        if (!(0.0 <= reference_value)) reference_value = 0.0;
        std::uint32_t truncated = static_cast<std::uint32_t>(reference_value);
        std::uint32_t reference_max16 = truncated;
        if (mosaic) {
            std::uint32_t floored = truncated & 0xffffu;
            if (floored < 0x405u) floored = 0x404;
            reference_max16 = floored;
        }
        CHECK_EQ(floor.max16, static_cast<std::uint16_t>(reference_max16));
        CHECK_EQ(floor.clamp_flag, false);

        double scale = 1.0 / (white - black_max);
        ReferenceTables reference_tables;
        CHECK(reference_build(info, 0, scale, floor.max16, floor.clamp_flag, &reference_tables));
        SourceLinearization stage;
        CHECK_EQ(static_cast<int>(stage.build(info, 0, LinearizationInputType::u16,
                                              LinearizationOutputType::u16, floor.max16,
                                              floor.clamp_flag)), 0);
        compare_tables(stage, reference_tables);

        std::vector<std::uint16_t> input(static_cast<std::size_t>(width) * height);
        for (auto& value : input) value = static_cast<std::uint16_t>(rng());
        std::vector<std::uint16_t> output(input.size(), 0xA5A5);
        CHECK_EQ(static_cast<int>(stage.process(input.data(), width, output.data(), width,
                                                0, 0, height, width)), 0);
        for (std::int32_t row = 0; row < height; ++row)
            for (std::int32_t col = 0; col < width; ++col) {
                std::uint16_t expected = reference_pixel(
                    reference_tables, info, 0, row, col,
                    input[static_cast<std::size_t>(row) * width + col]);
                CHECK_EQ(output[static_cast<std::size_t>(row) * width + col], expected);
            }
    }
}

void test_region_stride_and_alias() {
    LinearizationInfo info = make_info(2, 2, 8, 8);
    for (int row = 0; row < 2; ++row)
        for (int col = 0; col < 2; ++col) info.black_level[row][col][0] = 300.0 + row + col;
    info.white_level[0] = 4095.0;
    CHECK_EQ(static_cast<int>(imprint::acr::quantize_linearization_info(&info)), 0);
    SourceLinearization stage;
    CHECK_EQ(static_cast<int>(stage.build(info, 0, LinearizationInputType::u16,
                                          LinearizationOutputType::u16, 100, true)), 0);

    const std::size_t stride = 11;
    std::vector<std::uint16_t> source(stride * 8);
    for (std::size_t i = 0; i < source.size(); ++i) source[i] = static_cast<std::uint16_t>(i * 37);
    std::vector<std::uint16_t> output(stride * 8, 0xA5A5);
    // The accessors hand Process a pointer to the region's first pixel.
    CHECK_EQ(static_cast<int>(stage.process(source.data() + 2 * stride + 3, stride,
                                            output.data(), stride, 2, 3, 6, 7)), 0);
    CHECK_EQ(output[stride - 1], 0xA5A5);
    // The destination pointer is the region's top-left pixel; the LUT row/col
    // indices still use absolute coordinates, matching FUN_01303c4c.
    std::vector<std::uint16_t> expected(stride * 4, 0xA5A5);
    for (std::int32_t row = 2; row < 6; ++row)
        for (std::int32_t col = 3; col < 7; ++col) {
            std::uint32_t pixel = source[static_cast<std::size_t>(row) * stride + col];
            std::size_t row_index = static_cast<std::size_t>(row % stage.black_2d_rows());
            std::size_t col_index = static_cast<std::size_t>(col % stage.black_2d_cols());
            std::uint32_t v = static_cast<std::uint32_t>(stage.main_i24()[pixel]) -
                (0u - 0x80u);
            v -= static_cast<std::uint32_t>(
                stage.black_2d()[row_index * stage.black_2d_cols() + col_index]);
            expected[static_cast<std::size_t>(row - 2) * stride + (col - 3)] =
                (v >> 24) == 0 ? static_cast<std::uint16_t>(v >> 8)
                               : ((v >> 31) != 0 ? std::uint16_t{0} : std::uint16_t{0xffff});
        }
    CHECK(std::equal(expected.begin(), expected.end(), output.begin()));

    // In place on the same plane.
    std::vector<std::uint16_t> in_place = source;
    CHECK_EQ(static_cast<int>(stage.process(in_place.data(), stride, in_place.data(), stride,
                                            0, 0, 8, 8)), 0);
    for (std::int32_t row = 0; row < 8; ++row)
        for (std::int32_t col = 0; col < 8; ++col) {
            std::uint32_t pixel = source[static_cast<std::size_t>(row) * stride + col];
            std::uint32_t v = static_cast<std::uint32_t>(stage.main_i24()[pixel]) -
                (0u - 0x80u);
            v -= static_cast<std::uint32_t>(
                stage.black_2d()[static_cast<std::size_t>(row % stage.black_2d_rows()) *
                                     stage.black_2d_cols() +
                                 static_cast<std::size_t>(col % stage.black_2d_cols())]);
            std::uint16_t expected_value =
                (v >> 24) == 0 ? static_cast<std::uint16_t>(v >> 8)
                               : ((v >> 31) != 0 ? std::uint16_t{0} : std::uint16_t{0xffff});
            CHECK_EQ(in_place[static_cast<std::size_t>(row) * stride + col], expected_value);
        }
}

void test_failure_atomicity() {
    LinearizationInfo good = make_info(2, 2, 4, 4);
    for (int row = 0; row < 2; ++row)
        for (int col = 0; col < 2; ++col) good.black_level[row][col][0] = 64.0;
    good.white_level[0] = 4095.0;
    CHECK_EQ(static_cast<int>(imprint::acr::quantize_linearization_info(&good)), 0);
    SourceLinearization stage;
    CHECK_EQ(static_cast<int>(stage.build(good, 0, LinearizationInputType::u16,
                                          LinearizationOutputType::u16, 50, true)), 0);
    const std::vector<std::int32_t> before = stage.black_2d();
    const std::vector<std::int32_t> before_main = stage.main_i24();

    LinearizationInfo bad_white = good;
    bad_white.white_level[0] = reference_max_black(bad_white, 0);
    CHECK_EQ(static_cast<int>(stage.build(bad_white, 0, LinearizationInputType::u16,
                                          LinearizationOutputType::u16, 50, true)),
             static_cast<int>(LinearizationStatus::invalid_value));
    CHECK(stage.black_2d() == before);
    CHECK(stage.main_i24() == before_main);

    CHECK_EQ(static_cast<int>(stage.build(good, 0, LinearizationInputType::f32,
                                          LinearizationOutputType::u16, 50, true)),
             static_cast<int>(LinearizationStatus::unsupported_type));
    CHECK(stage.black_2d() == before);
    CHECK(stage.main_i24() == before_main);

    SourceLinearization unbuilt;
    std::uint16_t pixel = 1;
    std::uint16_t out = 0xA5A5;
    CHECK_EQ(static_cast<int>(unbuilt.process(&pixel, 1, &out, 1, 0, 0, 1, 1)),
             static_cast<int>(LinearizationStatus::not_built));
    CHECK_EQ(out, 0xA5A5);

    // Invalid region must not write.
    std::vector<std::uint16_t> input(16, 1);
    std::vector<std::uint16_t> output(16, 0xA5A5);
    const std::vector<std::uint16_t> untouched = output;
    CHECK_EQ(static_cast<int>(stage.process(input.data(), 4, output.data(), 4,
                                            0, 0, 4, 5)),
             static_cast<int>(LinearizationStatus::invalid_region));
    CHECK(output == untouched);
    CHECK_EQ(static_cast<int>(stage.process(input.data(), 2, output.data(), 4,
                                            0, 0, 4, 4)),
             static_cast<int>(LinearizationStatus::invalid_region));
    CHECK(output == untouched);
}

void test_fused_lut_boundaries() {
    // FUN_0130312c @01303844: split multiply-add would land exactly on 6263.5
    // and round to 6264; the recovered instruction is fused and yields 6263.
    LinearizationInfo info = make_info(1, 1, 2, 1);
    for (int plane = 0; plane < 4; ++plane) info.white_level[plane] = 283.96117402357294;
    CHECK_EQ(static_cast<int>(imprint::acr::quantize_linearization_info(&info)), 0);
    SourceLinearization stage;
    CHECK_EQ(static_cast<int>(stage.build(info, 0, LinearizationInputType::u8,
                                          LinearizationOutputType::u16, 4100, true)), 0);
    CHECK_EQ(stage.main_u16()[10], 6263);
    double scale_a = 1.0 / 283.96117402357294;
    CHECK_EQ(reference_round(4100.0 + 61435.0 * (scale_a * 10.0)), 6264);

    // FUN_0130312c @013036d4: same discrimination through the fixed-point arm.
    LinearizationInfo fixed = make_info(1, 2, 2, 2);
    fixed.white_level[0] = 610.2429122790301;
    CHECK_EQ(static_cast<int>(imprint::acr::quantize_linearization_info(&fixed)), 0);
    CHECK_EQ(static_cast<int>(stage.build(fixed, 0, LinearizationInputType::u16,
                                          LinearizationOutputType::u16, 3000, true)), 0);
    CHECK(stage.branch_b());
    CHECK_EQ(stage.main_i24()[10], 1030337);
    double scale_b = 1.0 / 610.2429122790301;
    CHECK_EQ(reference_round((3000.0 + 62535.0 * (scale_b * 10.0)) * 256.0), 1030338);

    // Clamp flag: a pixel below the black pedestal clamps to base16 or saturates.
    LinearizationInfo clamped = make_info(1, 1, 2, 2);
    clamped.black_level[0][0][0] = 1000.0;
    clamped.white_level[0] = 4095.0;
    CHECK_EQ(static_cast<int>(imprint::acr::quantize_linearization_info(&clamped)), 0);
    CHECK_EQ(static_cast<int>(stage.build(clamped, 0, LinearizationInputType::u16,
                                          LinearizationOutputType::u16, 4100, true)), 0);
    CHECK_EQ(stage.main_u16()[0], 4100);
    CHECK_EQ(static_cast<int>(stage.build(clamped, 0, LinearizationInputType::u16,
                                          LinearizationOutputType::u16, 4100, false)), 0);
    CHECK_EQ(stage.main_u16()[0], 0);

    // Short LinearizationTable: every index >= count - 1 reuses the last entry.
    LinearizationInfo short_table = make_info(1, 1, 2, 2);
    short_table.has_linearization_table = true;
    short_table.linearization_table = {1234};
    CHECK_EQ(static_cast<int>(imprint::acr::quantize_linearization_info(&short_table)), 0);
    CHECK_EQ(static_cast<int>(stage.build(short_table, 0, LinearizationInputType::u16,
                                          LinearizationOutputType::u16, 0, true)), 0);
    CHECK(stage.main_u16().size() == 0x10000u);
    CHECK_EQ(stage.main_u16()[0], stage.main_u16().back());
    ReferenceTables reference;
    reference_build(short_table, 0, 1.0 / (65535.0 - 0.0), 0, true, &reference);
    CHECK(stage.main_u16() == reference.main_u16);

    // Exact half-integer boundaries: scale = 1/131070 maps p to p/2, and the
    // recovered sequence is round-half-away-from-zero (013036e4/0130384c).
    LinearizationInfo halves = make_info(1, 1, 2, 2);
    halves.white_level[0] = 131070.0;
    CHECK_EQ(static_cast<int>(imprint::acr::quantize_linearization_info(&halves)), 0);
    CHECK_EQ(static_cast<int>(stage.build(halves, 0, LinearizationInputType::u16,
                                          LinearizationOutputType::u16, 0, true)), 0);
    CHECK_EQ(stage.main_u16()[1], 1);
    CHECK_EQ(stage.main_u16()[3], 2);
    CHECK_EQ(stage.main_u16()[5], 3);
    CHECK_EQ(stage.main_u16()[7], 4);
}

void test_process_32bit_wrap() {
    // The review counterexample: DeltaH = -8380000 puts black2D at
    // -2145280000; the 32-bit sum wraps to 0x80DE5F80 (bit 31 set) and stores 0,
    // while int64 arithmetic would report 0xffff.
    LinearizationInfo info = make_info(1, 2, 4, 2);
    info.has_black_level_delta_h = true;
    info.black_level_delta_h = {-8380000.0, 0.0, -8380000.0, 0.0};
    info.white_level[0] = 65535.0;
    CHECK_EQ(static_cast<int>(imprint::acr::quantize_linearization_info(&info)), 0);
    SourceLinearization stage;
    CHECK_EQ(static_cast<int>(stage.build(info, 0, LinearizationInputType::u16,
                                          LinearizationOutputType::u16, 0, true)), 0);
    CHECK_EQ(stage.black_2d()[0], -2145280000);
    std::uint16_t input[8] = {65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535};
    std::uint16_t output[8] = {};
    CHECK_EQ(static_cast<int>(stage.process(input, 4, output, 4, 0, 0, 2, 4)), 0);
    const std::uint16_t expected[8] = {0, 0xffff, 0, 0xffff, 0, 0xffff, 0, 0xffff};
    for (int i = 0; i < 8; ++i) CHECK_EQ(output[i], expected[i]);

    // The same stack in int64 is what the first implementation did; assert the
    // two semantics really disagree so this stays a regression test.
    std::int64_t legacy = static_cast<std::int64_t>(stage.main_i24()[65535]) + 0x80;
    legacy -= stage.black_2d()[0];
    CHECK(legacy >= 0x1000000);
    CHECK_EQ(static_cast<std::int64_t>(output[0]), 0);

    // Sign-bit boundary on the other side: DeltaH = 0 keeps the value above
    // 0x1000000 with bit 31 clear, so the store is 0xffff.
    LinearizationInfo plain = make_info(1, 2, 4, 2);
    plain.white_level[0] = 65535.0;
    CHECK_EQ(static_cast<int>(imprint::acr::quantize_linearization_info(&plain)), 0);
    CHECK_EQ(static_cast<int>(stage.build(plain, 0, LinearizationInputType::u16,
                                          LinearizationOutputType::u16, 0, true)), 0);
    CHECK_EQ(static_cast<int>(stage.process(input, 4, output, 4, 0, 0, 2, 4)), 0);
    for (int i = 0; i < 8; ++i) CHECK_EQ(output[i], 0xffff);
}

void test_geometry_and_alias_contracts() {
    LinearizationInfo info = make_info(2, 2, 8, 8);
    for (int row = 0; row < 2; ++row)
        for (int col = 0; col < 2; ++col) info.black_level[row][col][0] = 300.0 + row + col;
    info.white_level[0] = 4095.0;
    CHECK_EQ(static_cast<int>(imprint::acr::quantize_linearization_info(&info)), 0);
    SourceLinearization stage;
    CHECK_EQ(static_cast<int>(stage.build(info, 0, LinearizationInputType::u16,
                                          LinearizationOutputType::u16, 100, true)), 0);

    std::vector<std::uint16_t> source(64, 1234);
    std::vector<std::uint16_t> output(64, 0xA5A5);
    const std::vector<std::uint16_t> untouched = output;
    CHECK_EQ(static_cast<int>(stage.process(source.data(), static_cast<std::size_t>(-1),
                                            output.data(), 8, 0, 0, 2, 8)),
             static_cast<int>(LinearizationStatus::invalid_region));
    CHECK(output == untouched);
    CHECK_EQ(static_cast<int>(stage.process(source.data(), 8, output.data(),
                                            static_cast<std::size_t>(-1), 0, 0, 2, 8)),
             static_cast<int>(LinearizationStatus::invalid_region));
    CHECK(output == untouched);
    CHECK_EQ(static_cast<int>(stage.process(source.data(), 0, output.data(), 8, 0, 0, 2, 8)),
             static_cast<int>(LinearizationStatus::invalid_region));
    CHECK_EQ(static_cast<int>(stage.process(source.data(), 7, output.data(), 8, 0, 0, 2, 8)),
             static_cast<int>(LinearizationStatus::invalid_region));
    CHECK(output == untouched);
    // INT32_MIN/INT32_MAX region bounds are outside the built rect but must be
    // rejected without signed overflow.
    CHECK_EQ(static_cast<int>(stage.process(source.data(), 8, output.data(), 8,
                                            -2147483647 - 1, 0, 2147483647,
                                            2147483647)),
             static_cast<int>(LinearizationStatus::invalid_region));
    CHECK(output == untouched);

    // u8 -> u16 at the same address would overwrite the next source byte.
    LinearizationInfo u8_info = make_info(2, 2, 4, 2);
    u8_info.white_level[0] = 65535.0;
    CHECK_EQ(static_cast<int>(imprint::acr::quantize_linearization_info(&u8_info)), 0);
    SourceLinearization u8_stage;
    CHECK_EQ(static_cast<int>(u8_stage.build(u8_info, 0, LinearizationInputType::u8,
                                             LinearizationOutputType::u16, 0, true)), 0);
    std::vector<std::uint8_t> bytes(8, 7);
    const std::vector<std::uint8_t> bytes_before = bytes;
    CHECK_EQ(static_cast<int>(u8_stage.process(bytes.data(), 4, bytes.data(), 4, 0, 0, 2, 4)),
             static_cast<int>(LinearizationStatus::unsupported_aliasing));
    CHECK(bytes == bytes_before);

    // u16 partial overlap and same-base/different-stride are rejected too.
    std::vector<std::uint16_t> overlap(64, 1234);
    const std::vector<std::uint16_t> overlap_before = overlap;
    CHECK_EQ(static_cast<int>(stage.process(overlap.data(), 8, overlap.data() + 1, 8,
                                            0, 0, 2, 8)),
             static_cast<int>(LinearizationStatus::unsupported_aliasing));
    CHECK(overlap == overlap_before);
    CHECK_EQ(static_cast<int>(stage.process(overlap.data(), 8, overlap.data(), 9,
                                            0, 0, 2, 8)),
             static_cast<int>(LinearizationStatus::unsupported_aliasing));
    CHECK(overlap == overlap_before);
    std::vector<std::uint16_t> in_place(64, 1234);
    CHECK_EQ(static_cast<int>(stage.process(in_place.data(), 8, in_place.data(), 8,
                                            0, 0, 8, 8)), 0);

    // A rect spanning the full int32 range must be accepted or rejected
    // explicitly, never through signed-overflow UB.
    LinearizationInfo wide = make_info(1, 2, 2, 2);
    wide.rect_left = -2147483647 - 1;
    wide.rect_right = 2147483647;
    wide.white_level[0] = 65535.0;
    CHECK_EQ(static_cast<int>(imprint::acr::quantize_linearization_info(&wide)), 0);
    LinearizationStatus wide_status = stage.build(wide, 0, LinearizationInputType::u16,
                                                  LinearizationOutputType::u16, 0, true);
    CHECK(wide_status == LinearizationStatus::ok ||
          wide_status == LinearizationStatus::invalid_rect ||
          wide_status == LinearizationStatus::invalid_table ||
          wide_status == LinearizationStatus::resource_limit);
    wide.has_black_level_delta_h = true;
    wide.black_level_delta_h = {0.0, 0.0, 0.0, 0.0};
    CHECK_EQ(static_cast<int>(stage.build(wide, 0, LinearizationInputType::u16,
                                          LinearizationOutputType::u16, 0, true)),
             static_cast<int>(LinearizationStatus::invalid_table));
}

void test_helper_validation() {
    LinearizationInfo bad = make_info(9, 1, 4, 4);
    CHECK(std::isnan(imprint::acr::max_black_level(bad, 0)));
    LinearizationInfo good = make_info(2, 2, 4, 4);
    CHECK(std::isnan(imprint::acr::max_black_level(good, 7)));
    CHECK(std::isfinite(imprint::acr::max_black_level(good, 3)));

    DestinationFloor floor = imprint::acr::compute_destination_floor(
        bad, 1, LinearizationOutputType::u16, true, false, 777, false);
    CHECK_EQ(static_cast<int>(floor.status),
             static_cast<int>(LinearizationStatus::invalid_black_repeat));
    CHECK_EQ(floor.max16, 777);
    floor = imprint::acr::compute_destination_floor(
        good, 5, LinearizationOutputType::u16, true, false, 555, false);
    CHECK_EQ(static_cast<int>(floor.status),
             static_cast<int>(LinearizationStatus::invalid_plane));
    CHECK_EQ(floor.max16, 555);
}

}  // namespace

int main() {
    test_identity_branch_a();
    test_dji_anchor_values();
    test_random_reference();
    test_region_stride_and_alias();
    test_failure_atomicity();
    test_fused_lut_boundaries();
    test_process_32bit_wrap();
    test_geometry_and_alias_contracts();
    test_helper_validation();
    if (failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::printf("source linearization validation passed\n");
    return 0;
}
