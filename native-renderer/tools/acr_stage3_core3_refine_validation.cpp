#include "../src/acr_stage3_core3_refine.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {

using imprint::acr::Core3RefineStatus;
using imprint::acr::RefineInputs;
using imprint::acr::RefineOutputs;

int failures = 0;

void check(bool condition, const std::string &message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message.c_str());
        ++failures;
    }
}

bool same_bits(float left, float right) {
    return std::memcmp(&left, &right, sizeof(left)) == 0;
}

float from_bits(std::uint32_t bits) {
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

float add32(float a, float b) {
    volatile float rounded = a + b;
    return rounded;
}

float sub32(float a, float b) {
    volatile float rounded = a - b;
    return rounded;
}

float mul32(float a, float b) {
    volatile float rounded = a * b;
    return rounded;
}

float div32(float a, float b) {
    volatile float rounded = a / b;
    return rounded;
}

float inv32(float value) {
    return value == 0.0f ? 0.0f : div32(1.0f, value);
}

float fmin_nan(float a, float b) {
    if (std::isnan(a)) return a;
    if (std::isnan(b)) return b;
    return std::fmin(a, b);
}

float fmax_nan(float a, float b) {
    if (std::isnan(a)) return a;
    if (std::isnan(b)) return b;
    return std::fmax(a, b);
}

struct Plane {
    static constexpr std::size_t width = 8;
    static constexpr std::size_t stride = 16;
    static constexpr std::size_t rows = 5;
    std::vector<float> data = std::vector<float>(stride * rows, 0.0f);

    Plane() {
        check((reinterpret_cast<std::uintptr_t>(data.data()) & 0xFu) == 0,
              "plane allocation is 16-byte aligned");
    }

    float *center() { return data.data() + 2 * stride + 4; }
    const float *center() const { return data.data() + 2 * stride + 4; }

    float &at(std::ptrdiff_t y, std::ptrdiff_t x) {
        return center()[y * static_cast<std::ptrdiff_t>(stride) + x];
    }

    float at(std::ptrdiff_t y, std::ptrdiff_t x) const {
        return center()[y * static_cast<std::ptrdiff_t>(stride) + x];
    }
};

struct GuardedOutput {
    static constexpr std::size_t guard = 4;
    std::vector<float> data =
        std::vector<float>(guard + Plane::width + guard, -913.25f);

    float *center() { return data.data() + guard; }
    const float *center() const { return data.data() + guard; }

    bool guards_intact() const {
        for (std::size_t i = 0; i < guard; ++i) {
            if (!same_bits(data[i], -913.25f) ||
                !same_bits(data[guard + Plane::width + i], -913.25f)) {
                return false;
            }
        }
        return true;
    }
};

struct Fixture {
    Plane guide;
    Plane a;
    Plane b;
    Plane mask;
    GuardedOutput out_guide;
    GuardedOutput out_a;
    GuardedOutput out_b;

    RefineInputs inputs(float lower = 0.1f, float upper = 0.2f,
                        std::uint32_t phase = 0) {
        return {guide.center(), a.center(), b.center(), mask.center(),
                Plane::stride, Plane::width, phase, lower, upper};
    }

    RefineOutputs outputs() {
        return {out_guide.center(), out_a.center(), out_b.center()};
    }

    void fill_asymmetric() {
        for (std::ptrdiff_t y = -2; y <= 2; ++y) {
            for (std::ptrdiff_t x = -4; x < 12; ++x) {
                const auto residue = static_cast<float>((y * 7 + x * 11 + 41) % 13);
                guide.at(y, x) = 0.18f + 0.013f * static_cast<float>(x) +
                                 0.021f * static_cast<float>(y) +
                                 residue * 0.003f;
                a.at(y, x) = 0.29f + 0.017f * static_cast<float>(x) -
                             0.009f * static_cast<float>(y) + residue * 0.002f;
                b.at(y, x) = 0.63f - 0.011f * static_cast<float>(x) +
                             0.015f * static_cast<float>(y) - residue * 0.001f;
                mask.at(y, x) = 0.1f +
                                0.02f * static_cast<float>((x + 12) % 7) +
                                0.03f * static_cast<float>((y + 2) % 4);
            }
        }
    }

    void fill_uniform(float guide_value, float channel_value,
                      float mask_value) {
        for (std::ptrdiff_t y = -2; y <= 2; ++y) {
            for (std::ptrdiff_t x = -4; x < 12; ++x) {
                guide.at(y, x) = guide_value;
                a.at(y, x) = channel_value;
                b.at(y, x) = channel_value;
                mask.at(y, x) = mask_value;
            }
        }
    }
};

float oracle_candidate(const RefineInputs &in, const float *channel,
                       std::ptrdiff_t x, float mix) {
    const auto stride = static_cast<std::ptrdiff_t>(in.stride);
    auto at = [stride](const float *p, std::ptrdiff_t y,
                       std::ptrdiff_t col) { return p[y * stride + col]; };
    const float eps = from_bits(0x37800080u);
    const float vbase = add32(std::fabs(sub32(at(channel, -1, x),
                                               at(channel, 1, x))), eps);
    const float hbase = add32(std::fabs(sub32(at(channel, 0, x - 1),
                                               at(channel, 0, x + 1))), eps);
    const float gc = at(in.guide, 0, x);
    const float wu = inv32(add32(std::fabs(sub32(gc, at(in.guide, -2, x))),
                                 vbase));
    const float wd = inv32(add32(std::fabs(sub32(gc, at(in.guide, 2, x))),
                                 vbase));
    const float wl = inv32(add32(std::fabs(sub32(gc, at(in.guide, 0, x - 2))),
                                 hbase));
    const float wr = inv32(add32(std::fabs(sub32(gc, at(in.guide, 0, x + 2))),
                                 hbase));
    const float den = add32(wu, add32(wd, add32(wl, wr)));
    const float tu = mul32(sub32(at(channel, -1, x), at(in.guide, -1, x)), wu);
    const float td = mul32(sub32(at(channel, 1, x), at(in.guide, 1, x)), wd);
    const float tl = mul32(sub32(at(channel, 0, x - 1),
                                 at(in.guide, 0, x - 1)), wl);
    const float tr = mul32(sub32(at(channel, 0, x + 1),
                                 at(in.guide, 0, x + 1)), wr);
    const float numerator = add32(tu, add32(td, add32(tl, tr)));
    const float filtered = std::fma(inv32(den), numerator, gc);
    const float xc = at(channel, 0, x);
    const float low = sub32(xc, in.lower);
    const float high = add32(xc, in.upper);
    const float bounded = fmax_nan(low, fmin_nan(high, filtered));
    return fmax_nan(0.0f,
                    fmin_nan(std::fma(mix, sub32(bounded, xc), xc), 1.0f));
}

float oracle_mask_mean(const RefineInputs &in, std::ptrdiff_t x) {
    const auto stride = static_cast<std::ptrdiff_t>(in.stride);
    const float up = in.mask[-stride + x];
    const float left = in.mask[x - 1];
    const float right = in.mask[x + 1];
    const float down = in.mask[stride + x];
    return mul32(0.25f,
                 add32(up, add32(left, add32(right, down))));
}

void check_guide_copy(const Fixture &fixture, const std::string &label) {
    for (std::size_t x = 0; x < Plane::width; ++x) {
        check(same_bits(fixture.out_guide.center()[x],
                        fixture.guide.center()[x]),
              label + ": guide row copied bit-for-bit at " +
                  std::to_string(x));
    }
}

void check_output_guards(const Fixture &fixture, const std::string &label) {
    check(fixture.out_guide.guards_intact(), label + ": guide padding intact");
    check(fixture.out_a.guards_intact(), label + ": A padding intact");
    check(fixture.out_b.guards_intact(), label + ": B padding intact");
}

void check_expected_14230(const Fixture &fixture, const RefineInputs &in,
                          const std::string &label) {
    const bool phase0 = in.phase == 0;
    for (std::size_t column = 0; column < Plane::width; ++column) {
        const bool refine = (column & 1u) == (phase0 ? 0u : 1u);
        if (refine) {
            const auto x = static_cast<std::ptrdiff_t>(column);
            const float mix = fixture.mask.at(0, x);
            check(same_bits(fixture.out_a.center()[column],
                            oracle_candidate(in, fixture.a.center(), x, mix)),
                  label + ": A candidate at " + std::to_string(column));
            check(same_bits(fixture.out_b.center()[column],
                            oracle_candidate(in, fixture.b.center(), x, mix)),
                  label + ": B candidate at " + std::to_string(column));
        } else {
            check(same_bits(fixture.out_a.center()[column],
                            fixture.a.center()[column]),
                  label + ": A raw-copy parity at " +
                      std::to_string(column));
            check(same_bits(fixture.out_b.center()[column],
                            fixture.b.center()[column]),
                  label + ": B raw-copy parity at " +
                      std::to_string(column));
        }
    }
}

void check_expected_145b4(const Fixture &fixture, const RefineInputs &in,
                          const std::string &label) {
    const bool phase0 = in.phase == 0;
    for (std::size_t column = 0; column < Plane::width; ++column) {
        const bool refine = (column & 1u) == (phase0 ? 1u : 0u);
        if (refine) {
            const auto x = static_cast<std::ptrdiff_t>(column);
            const float mix = oracle_mask_mean(in, x);
            check(same_bits(fixture.out_a.center()[column],
                            oracle_candidate(in, fixture.a.center(), x, mix)),
                  label + ": A candidate at " + std::to_string(column));
            check(same_bits(fixture.out_b.center()[column],
                            oracle_candidate(in, fixture.b.center(), x, mix)),
                  label + ": B candidate at " + std::to_string(column));
        } else {
            check(same_bits(fixture.out_a.center()[column],
                            fixture.a.center()[column]),
                  label + ": A raw-copy parity at " +
                      std::to_string(column));
            check(same_bits(fixture.out_b.center()[column],
                            fixture.b.center()[column]),
                  label + ": B raw-copy parity at " +
                      std::to_string(column));
        }
    }
}

void test_00814230_asymmetric_phases_and_raw_bits() {
    for (std::uint32_t phase = 0; phase <= 1; ++phase) {
        Fixture fixture;
        fixture.fill_asymmetric();
        // Put payload-bearing values only in locations copied raw for each
        // phase; memcpy semantics must preserve their exact object bits.
        const std::size_t raw_column = phase == 0 ? 1 : 0;
        fixture.a.center()[raw_column] = from_bits(0x7fc01234u);
        fixture.b.center()[raw_column] = from_bits(0x80000000u);
        fixture.guide.center()[2] = from_bits(0x7fc05678u);
        const auto in = fixture.inputs(0.07f, 0.13f, phase);
        const auto status = imprint::acr::
            core3_refine_prepared_float_row_00814230(in, fixture.outputs());
        check(status == Core3RefineStatus::ok,
              "00814230 accepts asymmetric phase " + std::to_string(phase));
        check_expected_14230(fixture, in,
                             "00814230 phase " + std::to_string(phase));
        check_guide_copy(fixture, "00814230 guide phase " +
                                      std::to_string(phase));
        check_output_guards(fixture, "00814230 phase " +
                                         std::to_string(phase));
        check(same_bits(fixture.out_a.center()[raw_column],
                        fixture.a.center()[raw_column]),
              "00814230 raw A preserves NaN payload or signed zero");
        check(same_bits(fixture.out_b.center()[raw_column],
                        fixture.b.center()[raw_column]),
              "00814230 raw B preserves NaN payload or signed zero");
    }
}

void test_008145b4_asymmetric_neighborhoods_and_mask_mean() {
    for (std::uint32_t phase = 0; phase <= 1; ++phase) {
        Fixture fixture;
        fixture.fill_asymmetric();
        fixture.guide.center()[5] = from_bits(0x80000000u);
        // Distinct up/left/right/down values ensure the mask is a spatial
        // four-neighbor mean rather than its center or a symmetric average.
        fixture.mask.at(-1, 3) = 0.125f;
        fixture.mask.at(0, 2) = 0.25f;
        fixture.mask.at(0, 4) = 0.625f;
        fixture.mask.at(1, 3) = 0.875f;
        fixture.mask.at(0, 3) = 0.0f;
        const auto in = fixture.inputs(0.03f, 0.19f, phase);
        const auto status = imprint::acr::
            core3_refine_prepared_float_row_008145b4(in, fixture.outputs());
        check(status == Core3RefineStatus::ok,
              "008145b4 accepts asymmetric phase " + std::to_string(phase));
        check_expected_145b4(fixture, in,
                             "008145b4 phase " + std::to_string(phase));
        check_guide_copy(fixture, "008145b4 guide phase " +
                                      std::to_string(phase));
        check_output_guards(fixture, "008145b4 phase " +
                                         std::to_string(phase));

        const float expected_mean =
            mul32(0.25f, add32(0.125f, add32(0.25f, add32(0.625f, 0.875f))));
        check(same_bits(oracle_mask_mean(in, 3), expected_mean),
              "008145b4 mask mean uses explicit right-associated order");
        const float center_weight_candidate =
            oracle_candidate(in, fixture.a.center(), 3, 0.0f);
        const float neighbor_weight_candidate =
            oracle_candidate(in, fixture.a.center(), 3, expected_mean);
        check(!same_bits(center_weight_candidate, neighbor_weight_candidate),
              "008145b4 non-center mask values affect the mixed candidate");
    }
}

void test_hardcoded_epsilon_and_clamps() {
    check(same_bits(from_bits(0x37800080u), 1.5259022e-5f),
          "epsilon uses hard-coded binary32 0x37800080");

    Fixture lower_fixture;
    lower_fixture.fill_uniform(1.0f, 0.5f, 1.0f);
    for (std::size_t x = 0; x < Plane::width; x += 2) {
        lower_fixture.guide.center()[x] = 0.0f;
    }
    auto lower_in = lower_fixture.inputs(0.1f, 0.2f, 0);
    auto status = imprint::acr::core3_refine_prepared_float_row_00814230(
        lower_in, lower_fixture.outputs());
    check(status == Core3RefineStatus::ok,
          "00814230 lower-bound fixture succeeds");
    for (std::size_t x = 0; x < Plane::width; x += 2) {
        check(same_bits(lower_fixture.out_a.center()[x], 0.4f),
              "00814230 hard-coded lower bound is Xc-lower");
    }

    Fixture center_zero_mask;
    center_zero_mask.fill_uniform(0.0f, 0.35f, 0.0f);
    for (std::size_t x = 0; x < Plane::width; x += 2) {
        center_zero_mask.guide.center()[x] = 1.0f;
    }
    auto center_zero_in = center_zero_mask.inputs(1.0f, 1.0f, 0);
    status = imprint::acr::core3_refine_prepared_float_row_00814230(
        center_zero_in, center_zero_mask.outputs());
    check(status == Core3RefineStatus::ok,
          "00814230 zero-center-mask fixture succeeds");
    for (std::size_t x = 0; x < Plane::width; x += 2) {
        check(same_bits(center_zero_mask.out_a.center()[x], 0.35f),
              "00814230 zero center mask preserves the in-range center");
    }

    Fixture upper_fixture;
    upper_fixture.fill_uniform(0.0f, 0.5f, 1.0f);
    for (std::size_t x = 0; x < Plane::width; x += 2) {
        upper_fixture.guide.center()[x] = 1.0f;
    }
    auto upper_in = upper_fixture.inputs(0.1f, 0.1f, 1);
    status = imprint::acr::core3_refine_prepared_float_row_008145b4(
        upper_in, upper_fixture.outputs());
    check(status == Core3RefineStatus::ok,
          "008145b4 upper-bound fixture succeeds");
    for (std::size_t x = 0; x < Plane::width; x += 2) {
        check(same_bits(upper_fixture.out_a.center()[x], 0.6f),
              "008145b4 hard-coded upper bound is Xc+upper");
    }

    Fixture zero_mask;
    zero_mask.fill_uniform(0.0f, 0.35f, 0.0f);
    auto zero_in = zero_mask.inputs(0.0f, 0.0f, 1);
    status = imprint::acr::core3_refine_prepared_float_row_008145b4(
        zero_in, zero_mask.outputs());
    check(status == Core3RefineStatus::ok,
          "008145b4 zero-mask fixture succeeds");
    for (std::size_t x = 0; x < Plane::width; x += 2) {
        check(same_bits(zero_mask.out_a.center()[x], 0.35f),
              "008145b4 zero mask preserves the in-range center");
    }

    Fixture one_mask;
    one_mask.fill_uniform(0.0f, 0.5f, 1.0f);
    for (std::size_t x = 0; x < Plane::width; x += 2) {
        one_mask.guide.center()[x] = 1.0f;
    }
    auto one_in = one_mask.inputs(0.0f, 0.1f, 1);
    status = imprint::acr::core3_refine_prepared_float_row_008145b4(
        one_in, one_mask.outputs());
    check(status == Core3RefineStatus::ok,
          "008145b4 one-mask fixture succeeds");
    for (std::size_t x = 0; x < Plane::width; x += 2) {
        check(same_bits(one_mask.out_a.center()[x], 0.6f),
              "008145b4 one mask reaches the bounded candidate");
    }

    Fixture extrapolated;
    extrapolated.fill_uniform(0.0f, 0.5f, 8.0f);
    for (std::size_t x = 0; x < Plane::width; x += 2) {
        extrapolated.guide.center()[x] = 1.0f;
    }
    auto extrapolated_in = extrapolated.inputs(0.0f, 0.1f, 1);
    status = imprint::acr::core3_refine_prepared_float_row_008145b4(
        extrapolated_in, extrapolated.outputs());
    check(status == Core3RefineStatus::ok,
          "008145b4 extrapolation fixture succeeds");
    for (std::size_t x = 0; x < Plane::width; x += 2) {
        check(same_bits(extrapolated.out_a.center()[x], 1.0f),
              "008145b4 final clamp limits extrapolated mix to one");
    }
}

using Kernel = Core3RefineStatus (*)(const RefineInputs &,
                                     const RefineOutputs &) noexcept;

bool same_bits(const std::vector<float> &left,
               const std::vector<float> &right) {
    if (left.size() != right.size()) return false;
    for (std::size_t i = 0; i < left.size(); ++i) {
        if (!same_bits(left[i], right[i])) return false;
    }
    return true;
}

void check_rejected_atomically(Kernel kernel, const RefineInputs &in,
                               RefineOutputs out, const Fixture &fixture,
                               Core3RefineStatus expected,
                               const std::string &label) {
    const auto before_guide = fixture.out_guide.data;
    const auto before_a = fixture.out_a.data;
    const auto before_b = fixture.out_b.data;
    const auto status = kernel(in, out);
    check(status == expected, label + ": expected rejection status");
    check(same_bits(fixture.out_guide.data, before_guide) &&
              same_bits(fixture.out_a.data, before_a) &&
              same_bits(fixture.out_b.data, before_b),
          label + ": rejection leaves all outputs untouched");
}

void test_rejections_are_atomic_for_both_kernels() {
    const Kernel kernels[] = {
        imprint::acr::core3_refine_prepared_float_row_00814230,
        imprint::acr::core3_refine_prepared_float_row_008145b4,
    };
    const char *names[] = {"00814230", "008145b4"};
    for (std::size_t index = 0; index < 2; ++index) {
        Fixture fixture;
        fixture.fill_asymmetric();
        const auto good = fixture.inputs();
        const auto out = fixture.outputs();
        auto bad = good;

        bad.guide = nullptr;
        check_rejected_atomically(kernels[index], bad, out, fixture,
                                  Core3RefineStatus::invalid_argument,
                                  std::string(names[index]) + " null input");

        bad = good;
        bad.phase = 2;
        check_rejected_atomically(kernels[index], bad, out, fixture,
                                  Core3RefineStatus::invalid_phase,
                                  std::string(names[index]) + " invalid phase");

        bad = good;
        bad.width = 0;
        check_rejected_atomically(kernels[index], bad, out, fixture,
                                  Core3RefineStatus::invalid_dimensions,
                                  std::string(names[index]) + " zero width");

        bad = good;
        bad.stride = 4;
        check_rejected_atomically(kernels[index], bad, out, fixture,
                                  Core3RefineStatus::invalid_dimensions,
                                  std::string(names[index]) + " short stride");

        bad = good;
        bad.lower = -0.01f;
        check_rejected_atomically(kernels[index], bad, out, fixture,
                                  Core3RefineStatus::invalid_parameters,
                                  std::string(names[index]) + " negative lower");

        bad = good;
        bad.upper = std::numeric_limits<float>::infinity();
        check_rejected_atomically(kernels[index], bad, out, fixture,
                                  Core3RefineStatus::invalid_parameters,
                                  std::string(names[index]) + " infinite upper");

        bad = good;
        bad.stride = std::numeric_limits<std::size_t>::max() - 3u;
        check_rejected_atomically(kernels[index], bad, out, fixture,
                                  Core3RefineStatus::arithmetic_overflow,
                                  std::string(names[index]) + " overflow stride");

        bad = good;
        bad.guide = fixture.guide.center() + 1;
        check_rejected_atomically(kernels[index], bad, out, fixture,
                                  Core3RefineStatus::unsupported_alignment,
                                  std::string(names[index]) + " unaligned input");

        auto bad_out = out;
        bad_out.a = fixture.out_a.center() + 1;
        check_rejected_atomically(kernels[index], good, bad_out, fixture,
                                  Core3RefineStatus::unsupported_alignment,
                                  std::string(names[index]) + " unaligned output");

        bad_out = out;
        bad_out.b = nullptr;
        check_rejected_atomically(kernels[index], good, bad_out, fixture,
                                  Core3RefineStatus::invalid_argument,
                                  std::string(names[index]) + " null output");

        bad = good;
        bad.width = std::numeric_limits<std::size_t>::max() - 3u;
        bad.stride = bad.width;
        check_rejected_atomically(kernels[index], bad, out, fixture,
                                  Core3RefineStatus::arithmetic_overflow,
                                  std::string(names[index]) + " overflow width");
    }
}

void test_nonfinite_pixels_propagate() {
    Fixture fixture;
    fixture.fill_asymmetric();
    fixture.a.at(-1, 2) = std::numeric_limits<float>::quiet_NaN();
    auto in = fixture.inputs(0.0f, 1.0f, 0);
    const auto status = imprint::acr::core3_refine_prepared_float_row_00814230(
        in, fixture.outputs());
    check(status == Core3RefineStatus::ok,
          "00814230 accepts non-finite pixel samples");
    check(std::isnan(fixture.out_a.center()[2]),
          "00814230 NaN pixel propagates through candidate and clamp");

    Fixture infinite;
    infinite.fill_asymmetric();
    infinite.a.at(-1, 4) = std::numeric_limits<float>::infinity();
    auto inf_in = infinite.inputs(0.0f, 1.0f, 0);
    const auto inf_status = imprint::acr::
        core3_refine_prepared_float_row_00814230(inf_in, infinite.outputs());
    check(inf_status == Core3RefineStatus::ok,
          "00814230 accepts infinite pixel samples");
    check(std::isnan(infinite.out_a.center()[4]),
          "00814230 IEEE Inf arithmetic propagates generated NaN");
}

}  // namespace

int main() {
    test_00814230_asymmetric_phases_and_raw_bits();
    test_008145b4_asymmetric_neighborhoods_and_mask_mean();
    test_hardcoded_epsilon_and_clamps();
    test_rejections_are_atomic_for_both_kernels();
    test_nonfinite_pixels_propagate();
    if (failures != 0) {
        std::fprintf(stderr, "%d core3 refine validation checks failed\n",
                     failures);
        return 1;
    }
    std::puts("core3 refine validation passed");
    return 0;
}
