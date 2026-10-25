#include "../src/acr_stage3_core3_prefix.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

namespace {

using imprint::acr::Core3PrefixRegion;
using imprint::acr::Core3PrefixStatus;
using imprint::acr::Core3PrefixWorkspace;

int failures = 0;

void check(bool condition, const std::string &message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message.c_str());
        ++failures;
    }
}

float *plane_row(Core3PrefixWorkspace &workspace, std::size_t y,
                 std::size_t plane) {
    return workspace.samples.data() + y * workspace.row_stride +
           plane * workspace.plane_stride;
}

const float *plane_row(const Core3PrefixWorkspace &workspace, std::size_t y,
                       std::size_t plane) {
    return workspace.samples.data() + y * workspace.row_stride +
           plane * workspace.plane_stride;
}

struct InputTile {
    std::size_t width;
    std::size_t height;
    std::size_t stride;
    std::vector<float> samples;

    InputTile(std::size_t w, std::size_t h, std::size_t extra_stride = 0)
        : width(w), height(h), stride(w + 8 + extra_stride),
          samples(stride * h, 0.0f) {
        check((reinterpret_cast<std::uintptr_t>(samples.data()) & 0xFu) == 0,
              "test input allocation is 16-byte aligned");
        check((stride & 3u) == 0, "test input stride is a multiple of four");
    }

    float &at(std::size_t y, std::ptrdiff_t x) {
        return samples[y * stride + static_cast<std::size_t>(x + 4)];
    }

    float at(std::size_t y, std::ptrdiff_t x) const {
        return samples[y * stride + static_cast<std::size_t>(x + 4)];
    }

    void fill(float (*value)(std::size_t, std::ptrdiff_t)) {
        for (std::size_t y = 0; y < height; ++y) {
            for (std::size_t i = 0; i < stride; ++i) {
                const auto x = static_cast<std::ptrdiff_t>(i) - 4;
                samples[y * stride + i] = value(y, x);
            }
        }
    }
};

struct Snapshot {
    Core3PrefixWorkspace value;
};

Core3PrefixWorkspace sentinel_workspace() {
    Core3PrefixWorkspace value;
    value.width = 71;
    value.height = 73;
    value.pitch = 79;
    value.plane_stride = 83;
    value.row_stride = 89;
    value.gradient_valid = {3, 5, 7, 11};
    value.energy_valid = {13, 17, 19, 23};
    value.samples = {31.0f, 37.0f, 41.0f};
    return value;
}

bool same_region(const Core3PrefixRegion &left,
                 const Core3PrefixRegion &right) {
    return left.top == right.top && left.left == right.left &&
           left.bottom == right.bottom && left.right == right.right;
}

bool same_workspace(const Core3PrefixWorkspace &left,
                    const Core3PrefixWorkspace &right) {
    return left.width == right.width && left.height == right.height &&
           left.pitch == right.pitch && left.plane_stride == right.plane_stride &&
           left.row_stride == right.row_stride &&
           same_region(left.gradient_valid, right.gradient_valid) &&
           same_region(left.energy_valid, right.energy_valid) &&
           left.samples == right.samples;
}

float constant_rows(std::size_t y, std::ptrdiff_t) {
    return static_cast<float>(10 * y);
}

float horizontal_ramp(std::size_t, std::ptrdiff_t x) {
    return static_cast<float>(x);
}

float nonlinear_anisotropic(std::size_t y, std::ptrdiff_t x) {
    const float xf = static_cast<float>(x);
    const float yf = static_cast<float>(y);
    const int residue = static_cast<int>((static_cast<std::ptrdiff_t>(y) * x) % 7);
    return 0.125f * yf * yf + 0.0625f * xf * xf + 0.03125f * xf * yf +
           0.5f * static_cast<float>(residue);
}

float scalar_energy(float a, float b, float c, float d, float e) {
    const float mean = (a + b + c + d + e) * 0.2f;
    const float dp2 = e - mean;
    const float dp1 = d - mean;
    const float dc = c - mean;
    const float dm2 = a - mean;
    const float dm1 = b - mean;
    return std::sqrt(dp2 * dp2 + dp1 * dp1 + dc * dc + dm2 * dm2 +
                     dm1 * dm1);
}

bool close_float(float actual, float expected) {
    const float scale = std::max(1.0f, std::fabs(expected));
    return std::fabs(actual - expected) <= 2.0e-5f * scale;
}

void check_metadata(const Core3PrefixWorkspace &workspace, std::size_t width,
                    std::size_t height, const std::string &label) {
    check(workspace.width == width, label + ": width metadata");
    check(workspace.height == height, label + ": height metadata");
    check(workspace.pitch == width + 4, label + ": pitch metadata");
    check(workspace.plane_stride == width + 4,
          label + ": plane stride metadata");
    check(workspace.row_stride == 9 * (width + 4),
          label + ": row stride metadata");
    check(same_region(workspace.gradient_valid, {2, 0, height - 2, width}),
          label + ": gradient valid ROI");
    check(same_region(workspace.energy_valid,
                      {4, 4, height - 4, width - 4}),
          label + ": energy valid ROI");
    check(workspace.samples.size() == height * 9 * (width + 4),
          label + ": sample storage size");
}

void check_defined_regions_and_padding(const Core3PrefixWorkspace &workspace,
                                       const std::string &label) {
    for (std::size_t y = 0; y < workspace.height; ++y) {
        for (std::size_t plane = 0; plane < 9; ++plane) {
            const float *row = plane_row(workspace, y, plane);
            for (std::size_t x = 0; x < workspace.pitch; ++x) {
                bool defined = false;
                if (plane <= 5 && y >= 2 && y < workspace.height - 2 &&
                    x < workspace.width) {
                    defined = true;
                }
                if ((plane == 7 || plane == 8) && y >= 4 &&
                    y < workspace.height - 4 && x >= 4 &&
                    x < workspace.width - 4) {
                    defined = true;
                }
                check(defined ? !std::isnan(row[x]) : std::isnan(row[x]),
                      label + ": defined domain / NaN halo at plane " +
                          std::to_string(plane) + " y " + std::to_string(y) +
                          " x " + std::to_string(x));
            }
        }
    }
}

float expected_gradient_plane(const InputTile &input, std::size_t y,
                              std::size_t x, std::size_t plane) {
    const auto sx = static_cast<std::ptrdiff_t>(x);
    const float c = input.at(y, sx);
    const float xm2 = input.at(y, sx - 2);
    const float xm1 = input.at(y, sx - 1);
    const float xp1 = input.at(y, sx + 1);
    const float xp2 = input.at(y, sx + 2);
    const float twice_c = c + c;
    switch (plane) {
        case 0:
            return std::fabs(c - xp2);
        case 1:
            return std::fabs(c - xm2) + std::fabs(c - xp2);
        case 2:
            return std::fabs(c - input.at(y + 2, sx));
        case 3:
            return std::fabs((xm1 + xp1) - twice_c);
        case 4:
            return std::fabs((input.at(y - 1, sx) + input.at(y + 1, sx)) -
                             twice_c);
        case 5:
            return std::fabs(c - input.at(y - 2, sx)) +
                   std::fabs(c - input.at(y + 2, sx));
        default:
            return std::numeric_limits<float>::quiet_NaN();
    }
}

float expected_energy(const Core3PrefixWorkspace &workspace, std::size_t y,
                      std::size_t x, std::size_t plane) {
    if (plane == 7) {
        return scalar_energy(plane_row(workspace, y, 3)[x - 2],
                             plane_row(workspace, y, 3)[x - 1],
                             plane_row(workspace, y, 3)[x],
                             plane_row(workspace, y, 3)[x + 1],
                             plane_row(workspace, y, 3)[x + 2]);
    }
    return scalar_energy(plane_row(workspace, y - 2, 4)[x],
                         plane_row(workspace, y - 1, 4)[x],
                         plane_row(workspace, y, 4)[x],
                         plane_row(workspace, y + 1, 4)[x],
                         plane_row(workspace, y + 2, 4)[x]);
}

void check_scalar_oracle(const InputTile &input,
                         const Core3PrefixWorkspace &workspace,
                         const std::string &label) {
    for (std::size_t y = 2; y < input.height - 2; ++y) {
        for (std::size_t x = 0; x < input.width; ++x) {
            for (std::size_t plane = 0; plane <= 5; ++plane) {
                check(close_float(plane_row(workspace, y, plane)[x],
                                  expected_gradient_plane(input, y, x, plane)),
                      label + ": scalar gradient plane " +
                          std::to_string(plane) + " y " + std::to_string(y) +
                          " x " + std::to_string(x));
            }
        }
    }
    for (std::size_t y = 4; y < input.height - 4; ++y) {
        for (std::size_t x = 4; x < input.width - 4; ++x) {
            for (std::size_t plane : {7u, 8u}) {
                check(close_float(plane_row(workspace, y, plane)[x],
                                  expected_energy(workspace, y, x, plane)),
                      label + ": scalar energy plane " +
                          std::to_string(plane) + " y " + std::to_string(y) +
                          " x " + std::to_string(x));
            }
        }
    }
}

void test_vertical_constant_rows() {
    InputTile input(16, 11);
    input.fill(constant_rows);
    const auto input_before = input.samples;
    Core3PrefixWorkspace output;
    check(imprint::acr::run_core3_gradient_energy_prefix(
              input.samples.data(), input.samples.size(), input.width,
              input.height, input.stride, &output) == Core3PrefixStatus::ok,
          "vertical rows: status");
    check_metadata(output, input.width, input.height, "vertical rows");
    check_defined_regions_and_padding(output, "vertical rows");
    for (std::size_t y = 2; y < input.height - 2; ++y) {
        for (std::size_t x = 0; x < input.width; ++x) {
            check(plane_row(output, y, 0)[x] == 0.0f,
                  "vertical rows: plane 0 is zero");
            check(plane_row(output, y, 1)[x] == 0.0f,
                  "vertical rows: plane 1 is zero");
            check(plane_row(output, y, 2)[x] == 20.0f,
                  "vertical rows: out2 maps to plane 2");
            check(plane_row(output, y, 3)[x] == 0.0f,
                  "vertical rows: out4 maps to plane 3");
            check(plane_row(output, y, 4)[x] == 0.0f,
                  "vertical rows: out5 maps to plane 4");
            check(plane_row(output, y, 5)[x] == 40.0f,
                  "vertical rows: out3 maps to plane 5");
        }
    }
    check(input.samples == input_before, "vertical rows: input unchanged");
}

void test_horizontal_ramp() {
    InputTile input(20, 9, 4);
    input.fill(horizontal_ramp);
    const auto input_before = input.samples;
    Core3PrefixWorkspace output;
    check(imprint::acr::run_core3_gradient_energy_prefix(
              input.samples.data(), input.samples.size(), input.width,
              input.height, input.stride, &output) == Core3PrefixStatus::ok,
          "horizontal ramp: status");
    check_metadata(output, input.width, input.height, "horizontal ramp");
    check_defined_regions_and_padding(output, "horizontal ramp");
    for (std::size_t y = 2; y < input.height - 2; ++y) {
        for (std::size_t x = 0; x < input.width; ++x) {
            check(plane_row(output, y, 0)[x] == 2.0f,
                  "horizontal ramp: plane 0 is two");
            check(plane_row(output, y, 1)[x] == 4.0f,
                  "horizontal ramp: plane 1 is four");
            for (std::size_t plane : {2u, 3u, 4u, 5u}) {
                check(plane_row(output, y, plane)[x] == 0.0f,
                      "horizontal ramp: other gradient plane is zero");
            }
        }
    }
    for (std::size_t y = 4; y < input.height - 4; ++y) {
        for (std::size_t x = 4; x < input.width - 4; ++x) {
            check(plane_row(output, y, 7)[x] == 0.0f,
                  "horizontal ramp: horizontal energy is zero");
            check(plane_row(output, y, 8)[x] == 0.0f,
                  "horizontal ramp: vertical energy is zero");
        }
    }
    check(input.samples == input_before, "horizontal ramp: input unchanged");
}

void test_nonlinear_oracle_and_minimum_tile() {
    InputTile input(28, 17, 8);
    input.fill(nonlinear_anisotropic);
    const auto input_before = input.samples;
    Core3PrefixWorkspace output;
    check(imprint::acr::run_core3_gradient_energy_prefix(
              input.samples.data(), input.samples.size(), input.width,
              input.height, input.stride, &output) == Core3PrefixStatus::ok,
          "nonlinear tile: status");
    check_metadata(output, input.width, input.height, "nonlinear tile");
    check_defined_regions_and_padding(output, "nonlinear tile");
    check_scalar_oracle(input, output, "nonlinear tile");
    check(input.samples == input_before, "nonlinear tile: input unchanged");

    InputTile minimum(12, 9);
    minimum.fill(nonlinear_anisotropic);
    Core3PrefixWorkspace minimum_output;
    check(imprint::acr::run_core3_gradient_energy_prefix(
              minimum.samples.data(), minimum.samples.size(), minimum.width,
              minimum.height, minimum.stride, &minimum_output) ==
              Core3PrefixStatus::ok,
          "minimum 12x9 tile: status");
    check_metadata(minimum_output, 12, 9, "minimum tile");
    check_defined_regions_and_padding(minimum_output, "minimum tile");
    check_scalar_oracle(minimum, minimum_output, "minimum tile");
}

void check_failure_atomic(Core3PrefixStatus expected, const float *input,
                          std::size_t sample_count, std::size_t width,
                          std::size_t height, std::size_t stride,
                          const std::string &label) {
    Core3PrefixWorkspace output = sentinel_workspace();
    const Snapshot before{output};
    const Core3PrefixStatus status =
        imprint::acr::run_core3_gradient_energy_prefix(
            input, sample_count, width, height, stride, &output);
    check(status == expected, label + ": expected status " +
                                 std::to_string(static_cast<int>(expected)) +
                                 " got " + std::to_string(static_cast<int>(status)));
    check(same_workspace(output, before.value), label + ": output unchanged");
}

void test_failures_are_atomic() {
    InputTile input(12, 9, 4);
    input.fill(nonlinear_anisotropic);
    const std::size_t required = 8 * input.stride + input.width + 8;
    check_failure_atomic(Core3PrefixStatus::buffer_too_small,
                         input.samples.data(), required - 1, 12, 9,
                         input.stride, "short input span");
    check_failure_atomic(Core3PrefixStatus::invalid_dimensions,
                         input.samples.data(), input.samples.size(), 13, 9, 24,
                         "odd width");
    check_failure_atomic(Core3PrefixStatus::invalid_stride,
                         input.samples.data(), input.samples.size(), 12, 9, 19,
                         "short and unaligned stride");
    check_failure_atomic(Core3PrefixStatus::unsupported_alignment,
                         input.samples.data() + 1, input.samples.size() - 1,
                         12, 9, input.stride, "misaligned input pointer");

    const std::size_t overflowing_count =
        std::numeric_limits<std::size_t>::max() / sizeof(float) + 1;
    check_failure_atomic(Core3PrefixStatus::arithmetic_overflow,
                         input.samples.data(), overflowing_count, 12, 9,
                         input.stride, "input sample byte overflow");

    const std::size_t maximum = std::numeric_limits<std::size_t>::max();
    check_failure_atomic(Core3PrefixStatus::arithmetic_overflow,
                         input.samples.data(), input.samples.size(), maximum - 3,
                         9, 0, "width guard arithmetic overflow");

    const std::uintptr_t near_top =
        std::numeric_limits<std::uintptr_t>::max() - 7u;
    const auto *near_top_pointer = reinterpret_cast<const float *>(near_top);
    check_failure_atomic(Core3PrefixStatus::arithmetic_overflow,
                         near_top_pointer, required, 12, 9, input.stride,
                         "input address span overflow");

    constexpr std::size_t large_width = 1024;
    constexpr std::size_t large_height = 10000;
    constexpr std::size_t large_stride = large_width + 8;
    const std::size_t large_input_count = large_height * large_stride;
    const auto *fake_aligned_input =
        reinterpret_cast<const float *>(static_cast<std::uintptr_t>(0x10000000u));
    check_failure_atomic(Core3PrefixStatus::resource_limit,
                         fake_aligned_input, large_input_count, large_width,
                         large_height, large_stride,
                         "workspace resource cap");

    Core3PrefixWorkspace output = sentinel_workspace();
    const Snapshot before{output};
    check(imprint::acr::run_core3_gradient_energy_prefix(
              input.samples.data(), input.samples.size(), 12, 9, input.stride,
              nullptr) == Core3PrefixStatus::invalid_argument,
          "null output: status");
    check(same_workspace(output, before.value), "null output: other output unchanged");
    check(imprint::acr::run_core3_gradient_energy_prefix(
              nullptr, input.samples.size(), 12, 9, input.stride, &output) ==
              Core3PrefixStatus::invalid_argument,
          "null input: status");
    check(same_workspace(output, before.value), "null input: output unchanged");
}

void test_input_aliases_old_output_storage() {
    constexpr std::size_t width = 12;
    constexpr std::size_t height = 9;
    constexpr std::size_t input_stride = width + 8;
    const std::size_t required = (height - 1) * input_stride + width + 8;

    Core3PrefixWorkspace output;
    output.width = 901;
    output.height = 907;
    output.samples.assign(height * 9 * (width + 4), -123.0f);
    check((reinterpret_cast<std::uintptr_t>(output.samples.data()) & 0xFu) == 0,
          "alias source allocation is 16-byte aligned");
    for (std::size_t y = 0; y < height; ++y) {
        for (std::size_t i = 0; i < input_stride; ++i) {
            const auto x = static_cast<std::ptrdiff_t>(i) - 4;
            const float xf = static_cast<float>(x);
            const float yf = static_cast<float>(y);
            output.samples[y * input_stride + i] =
                0.25f * xf * xf + 0.5f * yf + 0.125f * xf * yf;
        }
    }
    const std::vector<float> source_before(output.samples.begin(),
                                           output.samples.begin() + required);
    const float *aliased_input = output.samples.data();
    check(imprint::acr::run_core3_gradient_energy_prefix(
              aliased_input, output.samples.size(), width, height,
              input_stride, &output) == Core3PrefixStatus::ok,
          "output storage input alias: status");
    check_metadata(output, width, height, "output storage input alias");

    InputTile oracle(width, height);
    std::copy(source_before.begin(), source_before.end(), oracle.samples.begin());
    check_scalar_oracle(oracle, output, "output storage input alias");
}

}  // namespace

int main() {
    test_vertical_constant_rows();
    test_horizontal_ramp();
    test_nonlinear_oracle_and_minimum_tile();
    test_failures_are_atomic();
    test_input_aliases_old_output_storage();

    if (failures != 0) {
        std::fprintf(stderr, "%d validation failure(s)\n", failures);
        return 1;
    }
    std::puts("core3 prepared-float prefix validation passed");
    return 0;
}
