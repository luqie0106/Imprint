#include "../src/acr_stage3_core3.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

namespace {

using imprint::acr::Core3Status;

int failures = 0;

void check(bool condition, const std::string &message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message.c_str());
        ++failures;
    }
}

constexpr std::size_t kGuard = 4;  // two required guards + alignment slack

// A contiguous source stack. Column 0 starts kGuard floats into each row, so a
// 16-byte-aligned allocation, kGuard*4 bytes of leading guard and a stride that
// is a multiple of four keep every derived row (center +/- stride) 16-byte
// aligned, which is the aligned path this port implements.
struct Stack {
    std::size_t stride;
    std::size_t height;
    std::size_t width;
    std::vector<float> storage;

    Stack(std::size_t width_in, std::size_t height_in, std::size_t extra = 0)
        : stride(width_in + 2 * kGuard + extra),
          height(height_in),
          width(width_in),
          storage(stride * height_in, 0.0f) {
        check(stride % 4 == 0, "stack: stride must be a multiple of four");
        check(reinterpret_cast<std::uintptr_t>(storage.data()) % 16 == 0,
              "stack: storage must be 16-byte aligned");
    }

    float *row(std::size_t index) {
        return storage.data() + index * stride + kGuard;
    }
    const float *row(std::size_t index) const {
        return storage.data() + index * stride + kGuard;
    }
};

struct Outputs {
    std::vector<std::vector<float>> rows;
    std::size_t stride;

    Outputs(std::size_t count, std::size_t output_stride)
        : rows(count, std::vector<float>(output_stride, 0.0f)),
          stride(output_stride) {
        check(output_stride % 4 == 0,
              "outputs: stride must be a multiple of four");
        for (const auto &row : rows) {
            check(reinterpret_cast<std::uintptr_t>(row.data()) % 16 == 0,
                  "outputs: row must be 16-byte aligned");
        }
    }

    float *ptr(std::size_t index) { return rows[index].data(); }
    std::vector<float *> all() {
        std::vector<float *> pointers;
        for (auto &row : rows) pointers.push_back(row.data());
        return pointers;
    }
};

void fill_constant(Stack &stack, const std::vector<float> &values) {
    for (std::size_t row = 0; row < stack.height; ++row) {
        for (std::ptrdiff_t j = -static_cast<std::ptrdiff_t>(kGuard);
             j < static_cast<std::ptrdiff_t>(stack.stride) -
                     static_cast<std::ptrdiff_t>(kGuard);
             ++j) {
            stack.row(row)[j] = values[row];
        }
    }
}

void test_constant() {
    constexpr std::size_t width = 16;
    Stack stack(width, 5);
    fill_constant(stack, {0.25f, 0.25f, 0.25f, 0.25f, 0.25f});
    Outputs out(6, width);
    auto rows = out.all();
    check(imprint::acr::core3_gradient_rows_0081376c(
              stack.row(2), stack.stride, width, rows.data(), out.stride) ==
              Core3Status::ok,
          "constant: gradient status");
    for (std::size_t row = 0; row < 6; ++row) {
        for (std::size_t column = 0; column < width; ++column) {
            check(out.rows[row][column] == 0.0f,
                  "constant: gradient output must be zero");
        }
    }

    std::vector<float> horizontal_out(width, 123.0f);
    std::vector<float> vertical_out(width, 123.0f);
    check(imprint::acr::core3_local_energy_rows_00813368(
              stack.row(2), stack.row(2), stack.stride, width,
              horizontal_out.data(), vertical_out.data()) == Core3Status::ok,
          "constant: energy status");
    for (std::size_t column = 0; column < width; ++column) {
        check(horizontal_out[column] == 0.0f, "constant: horizontal energy");
        check(vertical_out[column] == 0.0f, "constant: vertical energy");
    }
}

void check_gradient_columns(const Outputs &out, std::size_t width,
                            const std::vector<float> &expected,
                            const std::string &label) {
    for (std::size_t column = 0; column < width; ++column) {
        for (std::size_t row = 0; row < 6; ++row) {
            check(out.rows[row][column] == expected[row],
                  label + ": row " + std::to_string(row) + " column " +
                      std::to_string(column));
        }
    }
}

void test_vertical_step_probe() {
    // Five constant rows 0/10/20/30/40, center row 2:
    //   out2 = |20-40| = 20, out3 = |20-0| + |20-40| = 40, others zero.
    // The pre-fix implementation returned six zero rows for this input.
    constexpr std::size_t width = 4;
    Stack stack(width, 5);
    fill_constant(stack, {0.0f, 10.0f, 20.0f, 30.0f, 40.0f});
    Outputs out(6, width);
    auto rows = out.all();
    check(imprint::acr::core3_gradient_rows_0081376c(
              stack.row(2), stack.stride, width, rows.data(), out.stride) ==
              Core3Status::ok,
          "vertical step: status");
    check_gradient_columns(out, width, {0.0f, 0.0f, 20.0f, 40.0f, 0.0f, 0.0f},
                           "vertical step");
    // 007eaf20 call site: out3 is written to workspace row 5.
    check(out.rows[3][0] == 40.0f, "vertical step: out3 feeds workspace row 5");
    check(out.rows[2][0] == 20.0f, "vertical step: out2 feeds workspace row 2");
}

void test_horizontal_ramp_probe() {
    constexpr std::size_t width = 16;
    Stack stack(width, 5);
    for (std::size_t row = 0; row < stack.height; ++row) {
        for (std::ptrdiff_t j = -static_cast<std::ptrdiff_t>(kGuard);
             j < static_cast<std::ptrdiff_t>(stack.stride) -
                     static_cast<std::ptrdiff_t>(kGuard);
             ++j) {
            stack.row(row)[j] = static_cast<float>(j);
        }
    }
    Outputs out(6, width);
    auto rows = out.all();
    check(imprint::acr::core3_gradient_rows_0081376c(
              stack.row(2), stack.stride, width, rows.data(), out.stride) ==
              Core3Status::ok,
          "horizontal ramp: status");
    check_gradient_columns(out, width, {2.0f, 4.0f, 0.0f, 0.0f, 0.0f, 0.0f},
                           "horizontal ramp");

    std::vector<float> horizontal_out(width, 0.0f);
    std::vector<float> vertical_out(width, 0.0f);
    check(imprint::acr::core3_local_energy_rows_00813368(
              stack.row(2), stack.row(2), stack.stride, width,
              horizontal_out.data(), vertical_out.data()) == Core3Status::ok,
          "horizontal ramp: energy status");
    const float expected = std::sqrt(10.0f);
    for (std::size_t column = 0; column < width; ++column) {
        check(std::fabs(horizontal_out[column] - expected) < 1e-6f,
              "horizontal ramp: horizontal energy");
        check(vertical_out[column] == 0.0f,
              "horizontal ramp: vertical energy must be zero");
    }
}

void test_distinct_slopes_probe() {
    // j + 10*r: horizontal gradient 2/4, vertical gradient 20/40.
    constexpr std::size_t width = 16;
    Stack stack(width, 5);
    for (std::size_t row = 0; row < stack.height; ++row) {
        for (std::ptrdiff_t j = -static_cast<std::ptrdiff_t>(kGuard);
             j < static_cast<std::ptrdiff_t>(stack.stride) -
                     static_cast<std::ptrdiff_t>(kGuard);
             ++j) {
            stack.row(row)[j] = static_cast<float>(j) + 10.0f * row;
        }
    }
    Outputs out(6, width);
    auto rows = out.all();
    check(imprint::acr::core3_gradient_rows_0081376c(
              stack.row(2), stack.stride, width, rows.data(), out.stride) ==
              Core3Status::ok,
          "distinct slopes: status");
    check_gradient_columns(out, width, {2.0f, 4.0f, 20.0f, 40.0f, 0.0f, 0.0f},
                           "distinct slopes");

    std::vector<float> horizontal_out(width, 0.0f);
    std::vector<float> vertical_out(width, 0.0f);
    check(imprint::acr::core3_local_energy_rows_00813368(
              stack.row(2), stack.row(2), stack.stride, width,
              horizontal_out.data(), vertical_out.data()) == Core3Status::ok,
          "distinct slopes: energy status");
    const float expected_h = std::sqrt(10.0f);
    const float expected_v = std::sqrt(1000.0f);
    for (std::size_t column = 0; column < width; ++column) {
        check(std::fabs(horizontal_out[column] - expected_h) < 1e-6f,
              "distinct slopes: horizontal energy");
        check(std::fabs(vertical_out[column] - expected_v) < 1e-6f,
              "distinct slopes: vertical energy");
    }
}

void test_asymmetric_rows_probe() {
    // minus2=100, minus1=-5, center=0, plus1=7, plus2=40:
    //   out2=40, out3=140, out4=2, out5=2.
    constexpr std::size_t width = 8;
    Stack stack(width, 5);
    fill_constant(stack, {100.0f, -5.0f, 0.0f, 7.0f, 40.0f});
    Outputs out(6, width);
    auto rows = out.all();
    check(imprint::acr::core3_gradient_rows_0081376c(
              stack.row(2), stack.stride, width, rows.data(), out.stride) ==
              Core3Status::ok,
          "asymmetric rows: status");
    check_gradient_columns(out, width, {0.0f, 0.0f, 40.0f, 140.0f, 0.0f, 2.0f},
                           "asymmetric rows");
}

void test_vertical_impulses() {
    constexpr std::size_t width = 8;
    {
        // Impulse only in the plus2 row: out2 and out3 see column 5, no others.
        Stack stack(width, 5);
        stack.row(4)[5] = 1.0f;
        Outputs out(6, width);
        auto rows = out.all();
        check(imprint::acr::core3_gradient_rows_0081376c(
                  stack.row(2), stack.stride, width, rows.data(), out.stride) ==
                  Core3Status::ok,
              "plus2 impulse: status");
        for (std::size_t column = 0; column < width; ++column) {
            const float expected = column == 5 ? 1.0f : 0.0f;
            check(out.rows[2][column] == expected, "plus2 impulse: out2");
            check(out.rows[3][column] == expected, "plus2 impulse: out3");
            for (std::size_t row : {0u, 1u, 4u, 5u}) {
                check(out.rows[row][column] == 0.0f,
                      "plus2 impulse: unrelated row must stay zero");
            }
        }
    }
    {
        // Impulse only in the minus2 row: out2 stays zero, out3 sees it.
        Stack stack(width, 5);
        stack.row(0)[5] = -2.0f;
        Outputs out(6, width);
        auto rows = out.all();
        check(imprint::acr::core3_gradient_rows_0081376c(
                  stack.row(2), stack.stride, width, rows.data(), out.stride) ==
                  Core3Status::ok,
              "minus2 impulse: status");
        for (std::size_t column = 0; column < width; ++column) {
            const float expected = column == 5 ? 2.0f : 0.0f;
            check(out.rows[2][column] == 0.0f,
                  "minus2 impulse: out2 must ignore minus2");
            check(out.rows[3][column] == expected, "minus2 impulse: out3");
        }
    }
}

void test_large_values_and_stride() {
    constexpr std::size_t width = 8;
    constexpr std::size_t output_stride = width + 4;
    Stack stack(width, 5, 4);
    for (std::size_t row = 0; row < 5; ++row) {
        for (std::ptrdiff_t j = -static_cast<std::ptrdiff_t>(kGuard);
             j < static_cast<std::ptrdiff_t>(stack.stride) -
                     static_cast<std::ptrdiff_t>(kGuard);
             ++j) {
            const float base = 1.0e6f + static_cast<float>(j) * 1.0e5f;
            stack.row(row)[j] = base * static_cast<float>(row + 1) * 0.25f;
        }
    }
    Outputs out(6, output_stride);
    auto rows = out.all();
    check(imprint::acr::core3_gradient_rows_0081376c(
              stack.row(2), stack.stride, width, rows.data(), out.stride) ==
              Core3Status::ok,
          "large: gradient status");
    for (std::size_t column = 0; column < width; ++column) {
        check(std::isfinite(out.rows[0][column]), "large: out0 finite");
        check(std::isfinite(out.rows[4][column]), "large: out4 finite");
        check(std::isfinite(out.rows[5][column]), "large: out5 finite");
    }
    // Stride padding must stay untouched.
    for (std::size_t row = 0; row < 6; ++row) {
        for (std::size_t column = width; column < output_stride; ++column) {
            check(out.rows[row][column] == 0.0f, "large: output stride padding");
        }
    }
}

void test_nonfinite_contract() {
    // The kernel does not clamp or validate: an infinite center sample
    // propagates to the horizontal rows and produces a NaN where Inf - Inf is
    // evaluated. This is IEEE-754 behaviour, not a supported numeric range.
    constexpr std::size_t width = 8;
    Stack stack(width, 5);
    stack.row(2)[3] = std::numeric_limits<float>::infinity();
    stack.row(2)[5] = std::numeric_limits<float>::quiet_NaN();
    Outputs out(6, width);
    auto rows = out.all();
    check(imprint::acr::core3_gradient_rows_0081376c(
              stack.row(2), stack.stride, width, rows.data(), out.stride) ==
              Core3Status::ok,
          "nonfinite: status");
    check(std::isinf(out.rows[0][1]) && out.rows[0][1] > 0.0f,
          "nonfinite: out0 propagates +Inf");
    check(std::isnan(out.rows[0][3]),
          "nonfinite: out0 propagates NaN through Inf - NaN");
    check(std::isinf(out.rows[1][1]) && out.rows[1][1] > 0.0f,
          "nonfinite: out1 propagates +Inf");
    check(std::isinf(out.rows[4][2]) && out.rows[4][2] > 0.0f,
          "nonfinite: out4 propagates +Inf");
}

void test_failure_atomicity() {
    constexpr std::size_t width = 8;
    Stack stack(width, 5);
    for (std::size_t row = 0; row < 5; ++row) {
        for (std::size_t column = 0; column < width; ++column) {
            stack.row(row)[column] = 1.0f + static_cast<float>(row);
        }
    }
    Outputs out(6, width);
    auto rows = out.all();
    const std::vector<std::vector<float>> before = out.rows;

    check(imprint::acr::core3_gradient_rows_0081376c(
              nullptr, stack.stride, width, rows.data(), out.stride) ==
              Core3Status::invalid_argument,
          "failure: null source");
    check(imprint::acr::core3_gradient_rows_0081376c(
              stack.row(2), stack.stride, width - 1, rows.data(), out.stride) ==
              Core3Status::invalid_dimensions,
          "failure: width not multiple of four");
    check(imprint::acr::core3_gradient_rows_0081376c(
              stack.row(2), width - 1, width, rows.data(), out.stride) ==
              Core3Status::invalid_dimensions,
          "failure: short stride");
    check(imprint::acr::core3_gradient_rows_0081376c(
              stack.row(2), stack.stride + 2, width, rows.data(), out.stride) ==
              Core3Status::invalid_dimensions,
          "failure: stride not multiple of four");
    const float *misaligned = stack.row(2) + 1;
    check(imprint::acr::core3_gradient_rows_0081376c(
              misaligned, stack.stride, width, rows.data(), out.stride) ==
              Core3Status::unsupported_alignment,
          "failure: misaligned source is rejected");
    std::vector<float *> misaligned_out = rows;
    misaligned_out[3] = rows[3] + 1;
    check(imprint::acr::core3_gradient_rows_0081376c(
              stack.row(2), stack.stride, width, misaligned_out.data(),
              out.stride) == Core3Status::unsupported_alignment,
          "failure: misaligned output is rejected");
    check(out.rows == before, "failure: output must not change");

    std::vector<float> horizontal(width, 5.0f);
    std::vector<float> vertical(width, 6.0f);
    check(imprint::acr::core3_local_energy_rows_00813368(
              stack.row(2), stack.row(2), stack.stride, width, nullptr,
              vertical.data()) == Core3Status::invalid_argument,
          "failure: null energy output");
    check(imprint::acr::core3_local_energy_rows_00813368(
              stack.row(2), stack.row(2), width - 1, width,
              horizontal.data(), vertical.data()) ==
              Core3Status::invalid_dimensions,
          "failure: energy short stride");
    check(imprint::acr::core3_local_energy_rows_00813368(
              stack.row(2) + 1, stack.row(2), stack.stride, width,
              horizontal.data(), vertical.data()) ==
              Core3Status::unsupported_alignment,
          "failure: energy misaligned source");
    check(horizontal[0] == 5.0f && vertical[0] == 6.0f,
          "failure: energy output must not change");
}

}  // namespace

int main() {
    test_constant();
    test_vertical_step_probe();
    test_horizontal_ramp_probe();
    test_distinct_slopes_probe();
    test_asymmetric_rows_probe();
    test_vertical_impulses();
    test_large_values_and_stride();
    test_nonfinite_contract();
    test_failure_atomicity();
    if (failures != 0) {
        std::fprintf(stderr, "%d core3 check(s) failed\n", failures);
        return 1;
    }
    std::puts("PASS: core3 0081376c/00813368 row kernels");
    return 0;
}
