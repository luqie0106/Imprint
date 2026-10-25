#include "../src/acr_stage3_core3_green.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>

namespace {

using imprint::acr::Core3GreenParameters;
using imprint::acr::Core3GreenStatus;

constexpr std::size_t kWidth = 8;
constexpr std::size_t kSourceStride = 16;
constexpr std::size_t kWorkspaceStride = 24;
constexpr std::size_t kOutputSize = kWidth + 4;
int failures = 0;

void check(bool condition, const char *message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
}

template <std::size_t Rows, std::size_t Stride, std::size_t CenterRow>
struct alignas(16) GuardedRows {
    std::array<float, Rows * Stride> values{};

    float *row(std::size_t y) { return values.data() + y * Stride + 4; }
    const float *row(std::size_t y) const {
        return values.data() + y * Stride + 4;
    }
    float *center() { return row(CenterRow); }
    const float *center() const { return row(CenterRow); }
};

using SourceRows = GuardedRows<5, kSourceStride, 2>;
using PlaneRows = GuardedRows<7, kWorkspaceStride, 3>;

struct Inputs {
    SourceRows source;
    PlaneRows p0;
    PlaneRows p1;
    PlaneRows p2;
    PlaneRows p5;
    PlaneRows p7;
    PlaneRows p8;
};

struct alignas(16) Output {
    std::array<float, kOutputSize> green{};
    std::array<float, kOutputSize> aux{};

    void set(float green_value, float aux_value) {
        green.fill(green_value);
        aux.fill(aux_value);
    }
};

Core3GreenStatus run(Inputs &in, std::uint32_t phase,
                     Core3GreenParameters parameters, Output &out,
                     float *aux_override = nullptr) {
    return imprint::acr::core3_green_prepared_float_row_0080f2b8(
        in.source.center(), in.p0.center(), in.p1.center(), in.p2.center(),
        in.p5.center(), in.p7.center(), in.p8.center(), kSourceStride,
        kWorkspaceStride, kWidth, phase, parameters, out.green.data(),
        aux_override == nullptr ? out.aux.data() : aux_override);
}

void fill_constant(Inputs &in, float source, float energy) {
    in.source.values.fill(source);
    in.p0.values.fill(energy);
    in.p1.values.fill(energy);
    in.p2.values.fill(energy);
    in.p5.values.fill(energy);
    in.p7.values.fill(energy);
    in.p8.values.fill(energy);
}

void set_directional_source(Inputs &in, std::size_t x) {
    in.source.center()[x] = 0.5f;
    in.source.center()[static_cast<std::ptrdiff_t>(x) - 2] = 0.5f;
    in.source.center()[static_cast<std::ptrdiff_t>(x) - 1] = 0.1875f;
    in.source.center()[static_cast<std::ptrdiff_t>(x) + 1] = 0.1875f;
    in.source.center()[static_cast<std::ptrdiff_t>(x) + 2] = 0.5f;
    in.source.row(1)[x] = 0.5f;
    in.source.row(2)[x] = 0.5f;
    in.source.row(3)[x] = 0.5f;
    in.source.row(4)[x] = 0.5f;
    in.source.row(1)[x] = 0.59375f;
    in.source.row(3)[x] = 0.59375f;
}

bool all_equal(const std::array<float, kOutputSize> &values, float value) {
    for (float sample : values) {
        if (sample != value) return false;
    }
    return true;
}

void check_padding(const Output &out, float green_sentinel,
                   float aux_sentinel, const char *label) {
    for (std::size_t i = kWidth; i < kOutputSize; ++i) {
        check(out.green[i] == green_sentinel, label);
        check(out.aux[i] == aux_sentinel, label);
    }
}

void test_constant_and_phase_raw_copies() {
    Inputs in;
    fill_constant(in, 0.25f, 0.0f);
    const Inputs before = in;

    for (std::uint32_t phase = 0; phase <= 1; ++phase) {
        Output out;
        out.set(-91.0f, -92.0f);
        check(run(in, phase, {-1.0f, 0.0f, 0.5f, 0.0f}, out) ==
                  Core3GreenStatus::ok,
              "constant: status");
        for (std::size_t x = 0; x < kWidth; ++x) {
            check(out.green[x] == 0.25f,
                  "constant: green equals the constant source");
            check(out.aux[x] == 0.5f,
                  "constant: q zero auxiliary equals m");
        }
        check_padding(out, -91.0f, -92.0f,
                      "constant: output padding remains untouched");
    }

    check(in.source.values == before.source.values,
          "constant: source buffer is unchanged");
    check(in.p0.values == before.p0.values && in.p1.values == before.p1.values &&
              in.p2.values == before.p2.values &&
              in.p5.values == before.p5.values &&
              in.p7.values == before.p7.values &&
              in.p8.values == before.p8.values,
          "constant: all prepared plane buffers are unchanged");

    in.source.center()[0] = -2.0f;
    in.source.center()[1] = 3.0f;
    in.source.center()[2] = -4.0f;
    in.source.center()[3] = 5.0f;
    Output phase0;
    Output phase1;
    phase0.set(-71.0f, -72.0f);
    phase1.set(-81.0f, -82.0f);
    check(run(in, 0, {-1.0f, 0.0f, 0.0f, 0.0f}, phase0) ==
              Core3GreenStatus::ok,
          "raw copy: phase 0 status");
    check(run(in, 1, {-1.0f, 0.0f, 0.0f, 0.0f}, phase1) ==
              Core3GreenStatus::ok,
          "raw copy: phase 1 status");
    check(phase0.green[1] == 3.0f && phase0.green[3] == 5.0f,
          "raw copy: phase 0 preserves odd source values verbatim");
    check(phase1.green[0] == -2.0f && phase1.green[2] == -4.0f,
          "raw copy: phase 1 preserves even source values verbatim");
    check_padding(phase0, -71.0f, -72.0f,
                  "raw copy: phase 0 padding remains untouched");
    check_padding(phase1, -81.0f, -82.0f,
                  "raw copy: phase 1 padding remains untouched");
}

void test_horizontal_and_vertical_direction() {
    constexpr std::size_t x = 2;
    Inputs horizontal_input;
    fill_constant(horizontal_input, 0.5f, 0.0f);
    set_directional_source(horizontal_input, x);
    Output horizontal;
    horizontal.set(-1.0f, -2.0f);
    check(run(horizontal_input, 0, {45.0f, 1.0f, 0.0f, 1.0f}, horizontal) ==
              Core3GreenStatus::ok,
          "horizontal direction: status");
    check(horizontal.green[x] == 3.0f / 16.0f,
          "horizontal direction: low-angle switch selects H=3/16");

    Inputs vertical_input;
    fill_constant(vertical_input, 0.5f, 0.0f);
    set_directional_source(vertical_input, x);
    vertical_input.p7.center()[x] = 1.0f;
    vertical_input.p1.row(0)[x] = 1.0f;
    Output vertical;
    vertical.set(-3.0f, -4.0f);
    check(run(vertical_input, 0, {1.0f, 1.0f, 0.0f, 1.0f}, vertical) ==
              Core3GreenStatus::ok,
          "vertical direction: status");
    check(vertical.green[x] == 19.0f / 32.0f,
          "vertical direction: high-angle switch selects V=19/32");
}

float f32_from_bits(std::uint32_t bits) {
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

float reference_polynomial(float ratio) {
    const float a = f32_from_bits(0x4273e173u);
    const float b = f32_from_bits(0xc17f85ccu);
    return ratio * std::fma(ratio, b, a);
}

float reference_angle(float a, float b) {
    if (a == 0.0f && b == 0.0f) return 0.0f;
    const auto inverse = [](float value) {
        return value == 0.0f ? 0.0f : 1.0f / value;
    };
    if (a <= b) return 90.0f - reference_polynomial(a * inverse(b));
    return reference_polynomial(b * inverse(a));
}

void test_asymmetric_plane_probe() {
    Inputs in;
    fill_constant(in, 0.5f, 0.0f);
    constexpr std::size_t x = 3;
    set_directional_source(in, x);

    // Each recovered footprint has distinct samples. The resulting four sums
    // are E_h=31, E_v=121, D_v=84, and D_h=158.
    const std::array<float, 5> eh{1.0f, 2.0f, 4.0f, 8.0f, 16.0f};
    for (std::size_t i = 0; i < eh.size(); ++i) {
        in.p8.center()[static_cast<std::ptrdiff_t>(x) +
                       static_cast<std::ptrdiff_t>(i) - 2] = eh[i];
    }
    const std::array<float, 5> ev{1.0f, 3.0f, 9.0f, 27.0f, 81.0f};
    for (std::size_t i = 0; i < ev.size(); ++i) {
        in.p7.row(i + 1)[x] = ev[i];
    }

    in.p1.row(0)[x] = 1.0f;
    in.p1.row(1)[x - 1] = 2.0f;
    in.p1.row(2)[x] = 3.0f;
    in.p1.row(3)[x - 1] = 4.0f;
    in.p1.row(4)[x] = 5.0f;
    in.p1.row(5)[x - 1] = 6.0f;
    in.p1.row(6)[x] = 7.0f;
    in.p0.row(1)[x + 1] = 8.0f;
    in.p0.row(3)[x + 1] = 16.0f;
    in.p0.row(5)[x + 1] = 32.0f;

    in.p5.row(2)[x - 2] = 3.0f;
    in.p5.row(2)[x] = 5.0f;
    in.p5.row(2)[x + 2] = 7.0f;
    in.p5.row(3)[x - 3] = 11.0f;
    in.p5.row(3)[x - 1] = 13.0f;
    in.p5.row(3)[x + 1] = 17.0f;
    in.p5.row(3)[x + 3] = 19.0f;
    in.p2.row(4)[x - 2] = 23.0f;
    in.p2.row(4)[x] = 29.0f;
    in.p2.row(4)[x + 2] = 31.0f;

    const Inputs before = in;
    Output out;
    out.set(-11.0f, -12.0f);
    // Normalize the degree-valued angles to avoid saturating the base clamp.
    // With k=1 the base is 1, so an unclamped degree mean is not a weight.
    constexpr float scale = 1.0f / 90.0f;
    check(run(in, 1, {0.0f, scale, 0.75f, 1.0f}, out) ==
              Core3GreenStatus::ok,
          "asymmetric planes: status");

    const float theta_e = reference_angle(31.0f, 121.0f);
    const float theta_d = reference_angle(158.0f, 84.0f);
    const float degree_mean = (theta_e + theta_d) * 0.5f;
    const float expected_weight = degree_mean * scale;
    const float expected_green = 3.0f / 16.0f +
                                 (19.0f / 32.0f - 3.0f / 16.0f) *
                                     expected_weight;
    const float aux_factor = std::fmin(
        1.0f, (1.0f - std::fabs(2.0f * expected_weight - 1.0f)) * 2.0f);
    const float expected_aux = aux_factor * 0.75f;
    if (std::fabs(out.green[x] - expected_green) >= 1.0e-6f ||
        std::fabs(out.aux[x] - expected_aux) >= 1.0e-6f) {
        std::fprintf(stderr,
                     "asymmetric actual g=%0.9g aux=%0.9g expected g=%0.9g aux=%0.9g w=%0.9g angles=%0.9g,%0.9g\n",
                     out.green[x], out.aux[x], expected_green, expected_aux,
                     expected_weight, theta_e, theta_d);
    }
    check(std::fabs(out.green[x] - expected_green) < 1.0e-6f,
          "asymmetric planes: selected green follows distinct E/D plane sums");
    check(std::fabs(out.aux[x] - expected_aux) < 1.0e-6f,
          "asymmetric planes: auxiliary follows the same selected weight");

    check(in.source.values == before.source.values &&
              in.p0.values == before.p0.values &&
              in.p1.values == before.p1.values &&
              in.p2.values == before.p2.values &&
              in.p5.values == before.p5.values &&
              in.p7.values == before.p7.values &&
              in.p8.values == before.p8.values,
          "asymmetric planes: all input buffers remain unchanged");
    check_padding(out, -11.0f, -12.0f,
                  "asymmetric planes: output padding remains untouched");
}

void test_q_branches_and_auxiliary_write_rules() {
    Inputs in;
    fill_constant(in, 0.5f, 0.0f);
    constexpr std::size_t x = 2;
    set_directional_source(in, x);

    struct Case {
        float t;
        float q;
        float expected_green;
        float expected_aux;
        const char *label;
    };
    const std::array<Case, 5> cases{{
        {-0.125f, 0.0f, 0.390625f, 2.0f, "q zero"},
        {-0.125f, 0.2f, 0.314453125f, 2.0f, "q lower branch"},
        {-0.875f, 0.2f, 0.466796875f, 2.0f, "q upper branch"},
        {-0.125f, 0.5f, 0.23828125f, 1.0f, "q half branch"},
        {-0.125f, 1.0f, 0.23828125f, 1.0f, "q one branch"},
    }};
    for (const Case &test : cases) {
        Output out;
        out.set(-21.0f, -22.0f);
        check(run(in, 0, {test.t, 1.0f, 2.0f, test.q}, out) ==
                  Core3GreenStatus::ok,
              "q branch: call succeeds");
        check(std::fabs(out.green[x] - test.expected_green) < 1.0e-7f,
              test.label);
        check(std::fabs(out.aux[x] - test.expected_aux) < 1.0e-7f,
              "q branch: auxiliary uses the recovered weight formula");
        check(std::fabs(out.aux[1] - test.expected_aux) < 1.0e-7f,
              "q branch: auxiliary is computed on raw-copy lanes");
        check_padding(out, -21.0f, -22.0f,
                      "q branch: output padding remains untouched");
    }

    Output zero_m;
    zero_m.set(-31.0f, -32.0f);
    check(run(in, 0, {-0.125f, 1.0f, 0.0f, 0.2f}, zero_m) ==
              Core3GreenStatus::ok,
          "m zero: call succeeds with an auxiliary buffer");
    check(all_equal(zero_m.aux, -32.0f),
          "m zero: auxiliary buffer is completely untouched");

    check(imprint::acr::core3_green_prepared_float_row_0080f2b8(
              in.source.center(), in.p0.center(), in.p1.center(),
              in.p2.center(), in.p5.center(), in.p7.center(), in.p8.center(),
              kSourceStride, kWorkspaceStride, kWidth, 0,
              {-0.125f, 1.0f, -0.0f, 0.2f}, zero_m.green.data(), nullptr) ==
              Core3GreenStatus::ok,
          "m negative zero: null auxiliary buffer is accepted");
}

void test_rejections_do_not_write() {
    Inputs in;
    fill_constant(in, 0.5f, 0.0f);
    Output out;

    const auto check_rejected = [&](Core3GreenStatus expected,
                                    std::uint32_t phase,
                                    Core3GreenParameters parameters,
                                    std::size_t width,
                                    std::size_t workspace_stride,
                                    const char *label) {
        out.set(-41.0f, -42.0f);
        const auto status = imprint::acr::core3_green_prepared_float_row_0080f2b8(
            in.source.center(), in.p0.center(), in.p1.center(),
            in.p2.center(), in.p5.center(), in.p7.center(), in.p8.center(),
            kSourceStride, workspace_stride, width, phase, parameters,
            out.green.data(), out.aux.data());
        check(status == expected, label);
        check(all_equal(out.green, -41.0f) && all_equal(out.aux, -42.0f),
              "rejected call: outputs remain untouched");
    };

    check_rejected(Core3GreenStatus::invalid_phase, 2,
                   {-0.125f, 1.0f, 1.0f, 0.2f}, kWidth,
                   kWorkspaceStride, "reject: invalid phase");
    check_rejected(Core3GreenStatus::invalid_dimensions, 0,
                   {-0.125f, 1.0f, 1.0f, 0.2f}, kWidth - 1,
                   kWorkspaceStride, "reject: width not multiple of four");
    check_rejected(Core3GreenStatus::invalid_parameters, 0,
                   {-0.125f, 1.0f, 1.0f,
                    std::numeric_limits<float>::quiet_NaN()},
                   kWidth, kWorkspaceStride, "reject: NaN control");
    check_rejected(Core3GreenStatus::invalid_parameters, 0,
                   {-0.125f, 1.0f, 1.0f, -0.25f}, kWidth,
                   kWorkspaceStride, "reject: negative q");
    check_rejected(Core3GreenStatus::arithmetic_overflow, 0,
                   {-0.125f, 1.0f, 1.0f, 0.2f}, kWidth,
                   std::numeric_limits<std::size_t>::max() & ~std::size_t{3},
                   "reject: workspace stride byte span overflow");
}

}  // namespace

int main() {
    test_constant_and_phase_raw_copies();
    test_horizontal_and_vertical_direction();
    test_asymmetric_plane_probe();
    test_q_branches_and_auxiliary_write_rules();
    test_rejections_do_not_write();
    if (failures != 0) {
        std::fprintf(stderr, "%d core3 green validation failure(s)\n", failures);
        return 1;
    }
    std::puts("core3 prepared-float green validation passed");
    return 0;
}
