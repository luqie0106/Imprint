#include "../src/acr_stage3_core3_colour.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>

namespace {

using imprint::acr::Core3ColourStatus;

constexpr std::size_t kWidth = 8;
constexpr std::size_t kGuard = 4;
constexpr std::size_t kRows = 5;
constexpr std::size_t kSourceStride = 16;
constexpr std::size_t kGreenStride = 20;
constexpr std::size_t kOutputSize = kWidth + 4;

int failures = 0;

void check(bool condition, const char *message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
}

template <std::size_t Stride>
struct Rows {
    alignas(16) std::array<float, kRows * Stride> samples{};

    float *row(std::size_t y) {
        return samples.data() + y * Stride + kGuard;
    }
    const float *row(std::size_t y) const {
        return samples.data() + y * Stride + kGuard;
    }
};

using SourceRows = Rows<kSourceStride>;
using GreenRows = Rows<kGreenStride>;

void fill(SourceRows &rows, float value) {
    rows.samples.fill(value);
}

void fill(GreenRows &rows, float value) {
    rows.samples.fill(value);
}

void init_pair(SourceRows &source, GreenRows &green, float source_value,
               float green_value) {
    fill(source, source_value);
    fill(green, green_value);
}

struct alignas(16) OutputPair {
    std::array<float, kOutputSize> a{};
    std::array<float, kOutputSize> b{};

    void set(float value_a, float value_b) {
        a.fill(value_a);
        b.fill(value_b);
    }
};

bool all_equal(const std::array<float, kOutputSize> &values, float expected) {
    for (float value : values) {
        if (value != expected) return false;
    }
    return true;
}

void check_output_padding(const OutputPair &out, float sentinel_a,
                          float sentinel_b, const char *label) {
    for (std::size_t i = kWidth; i < kOutputSize; ++i) {
        check(out.a[i] == sentinel_a, label);
        check(out.b[i] == sentinel_b, label);
    }
}

void test_constant_and_input_immutability() {
    SourceRows source;
    GreenRows green;
    init_pair(source, green, 0.8f, 0.2f);
    const auto source_before = source.samples;
    const auto green_before = green.samples;

    for (std::uint32_t phase = 0; phase <= 1; ++phase) {
        OutputPair out;
        out.set(-91.0f, -92.0f);
        check(imprint::acr::core3_colour_difference_row_008134b8(
                  source.row(2), green.row(2), kSourceStride, kGreenStride,
                  kWidth, phase, out.a.data(), out.b.data()) ==
                  Core3ColourStatus::ok,
              "constant: status");
        for (std::size_t i = 0; i < kWidth; ++i) {
            check(out.a[i] == 0.8f, "constant: A is 0.8 for every lane");
            check(out.b[i] == 0.8f, "constant: B is 0.8 for every lane");
        }
        check_output_padding(out, -91.0f, -92.0f,
                             "constant: output padding untouched");
    }

    check(source.samples == source_before,
          "constant: source including row padding and guards unchanged");
    check(green.samples == green_before,
          "constant: green including row padding and guards unchanged");
}

void test_phase_lane_interleave() {
    SourceRows source;
    GreenRows green;
    init_pair(source, green, 0.25f, 0.25f);

    // Hand-set two adjacent lanes so their raw, H, V and D candidates differ.
    source.row(2)[1] = 0.25f;
    source.row(2)[2] = 0.375f;
    source.row(2)[3] = 0.75f;
    source.row(2)[4] = 1.0f;

    source.row(1)[1] = 0.875f;
    source.row(1)[2] = 0.5f;
    source.row(1)[3] = 0.875f;
    source.row(1)[4] = 0.875f;
    source.row(3)[1] = 0.875f;
    source.row(3)[2] = 1.0f;
    source.row(3)[3] = 0.875f;
    source.row(3)[4] = 0.875f;

    OutputPair phase0;
    OutputPair phase1;
    phase0.set(-71.0f, -72.0f);
    phase1.set(-81.0f, -82.0f);
    check(imprint::acr::core3_colour_difference_row_008134b8(
              source.row(2), green.row(2), kSourceStride, kGreenStride,
              kWidth, 0, phase0.a.data(), phase0.b.data()) ==
              Core3ColourStatus::ok,
          "phase interleave: phase 0 status");
    check(imprint::acr::core3_colour_difference_row_008134b8(
              source.row(2), green.row(2), kSourceStride, kGreenStride,
              kWidth, 1, phase1.a.data(), phase1.b.data()) ==
              Core3ColourStatus::ok,
          "phase interleave: phase 1 status");

    // At even lane 2: source=.375, H=.5, V=.75, D=.875.
    check(phase0.a[2] == 0.375f && phase0.b[2] == 0.875f,
          "phase interleave: phase 0 even lane selects source and D");
    check(phase1.a[2] == 0.5f && phase1.b[2] == 0.75f,
          "phase interleave: phase 1 even lane selects H and V");

    // At odd lane 3: source=.75, H=.6875, V=.875, D=.8125.
    check(phase0.a[3] == 0.6875f && phase0.b[3] == 0.875f,
          "phase interleave: phase 0 odd lane selects H and V");
    check(phase1.a[3] == 0.75f && phase1.b[3] == 0.8125f,
          "phase interleave: phase 1 odd lane selects source and D");
    check_output_padding(phase0, -71.0f, -72.0f,
                         "phase interleave: phase 0 output padding untouched");
    check_output_padding(phase1, -81.0f, -82.0f,
                         "phase interleave: phase 1 output padding untouched");
}

void test_direction_probes() {
    {
        SourceRows source;
        GreenRows green;
        init_pair(source, green, 0.25f, 0.25f);
        source.row(2)[1] = 0.25f;
        source.row(2)[3] = 0.75f;
        OutputPair out;
        out.set(-1.0f, -2.0f);
        check(imprint::acr::core3_colour_difference_row_008134b8(
                  source.row(2), green.row(2), kSourceStride, kGreenStride,
                  kWidth, 1, out.a.data(), out.b.data()) ==
                  Core3ColourStatus::ok,
              "horizontal-only probe: status");
        check(out.a[2] == 0.5f && out.b[2] == 0.25f,
              "horizontal-only probe: left/right changes H only");
    }
    {
        SourceRows source;
        GreenRows green;
        init_pair(source, green, 0.25f, 0.25f);
        source.row(1)[2] = 0.5f;
        source.row(3)[2] = 1.0f;
        OutputPair out;
        out.set(-1.0f, -2.0f);
        check(imprint::acr::core3_colour_difference_row_008134b8(
                  source.row(2), green.row(2), kSourceStride, kGreenStride,
                  kWidth, 1, out.a.data(), out.b.data()) ==
                  Core3ColourStatus::ok,
              "vertical-only probe: status");
        check(out.a[2] == 0.25f && out.b[2] == 0.75f,
              "vertical-only probe: up/down changes V only");
    }
}

void test_clamp_and_verbatim_source_copy() {
    {
        SourceRows source;
        GreenRows green;
        init_pair(source, green, 0.25f, 0.25f);
        source.row(2)[1] = -2.0f;
        source.row(2)[3] = -2.0f;
        source.row(1)[2] = 2.0f;
        source.row(3)[2] = 2.0f;
        source.row(1)[1] = 2.0f;
        source.row(1)[3] = 2.0f;
        source.row(3)[1] = 2.0f;
        source.row(3)[3] = 2.0f;

        OutputPair phase0;
        OutputPair phase1;
        phase0.set(-1.0f, -2.0f);
        phase1.set(-3.0f, -4.0f);
        check(imprint::acr::core3_colour_difference_row_008134b8(
                  source.row(2), green.row(2), kSourceStride, kGreenStride,
                  kWidth, 0, phase0.a.data(), phase0.b.data()) ==
                  Core3ColourStatus::ok,
              "clamp probe: phase 0 status");
        check(imprint::acr::core3_colour_difference_row_008134b8(
                  source.row(2), green.row(2), kSourceStride, kGreenStride,
                  kWidth, 1, phase1.a.data(), phase1.b.data()) ==
                  Core3ColourStatus::ok,
              "clamp probe: phase 1 status");
        check(phase1.a[2] == 0.0f && phase1.b[2] == 1.0f,
              "clamp probe: H clamps to zero and V clamps to one");
        check(phase0.b[2] == 1.0f,
              "clamp probe: D clamps to one");
    }
    {
        SourceRows source;
        GreenRows green;
        init_pair(source, green, 0.25f, 0.25f);
        source.row(2)[0] = -1.0f;
        source.row(2)[1] = 2.0f;
        OutputPair phase0;
        OutputPair phase1;
        phase0.set(-1.0f, -2.0f);
        phase1.set(-3.0f, -4.0f);
        check(imprint::acr::core3_colour_difference_row_008134b8(
                  source.row(2), green.row(2), kSourceStride, kGreenStride,
                  kWidth, 0, phase0.a.data(), phase0.b.data()) ==
                  Core3ColourStatus::ok,
              "verbatim copy probe: phase 0 status");
        check(imprint::acr::core3_colour_difference_row_008134b8(
                  source.row(2), green.row(2), kSourceStride, kGreenStride,
                  kWidth, 1, phase1.a.data(), phase1.b.data()) ==
                  Core3ColourStatus::ok,
              "verbatim copy probe: phase 1 status");
        check(phase0.a[0] == -1.0f,
              "verbatim copy probe: negative source copy is not clamped");
        check(phase1.a[1] == 2.0f,
              "verbatim copy probe: above-one source copy is not clamped");
    }
}

void test_diagonal_association() {
    SourceRows source;
    GreenRows green;
    init_pair(source, green, 0.0f, 0.0f);
    source.row(2)[2] = 0.5f;

    // a=1e20, b=-1e20, c=3, d=4. The recovered grouping is
    // a + (b + (c + d)): c+d rounds to 7, b+7 rounds to -1e20,
    // then a+(-1e20) is zero. Left association would instead produce 7,
    // making D=1.75. This expected value is calculated independently here.
    source.row(1)[1] = 1.0e20f;
    source.row(1)[3] = -1.0e20f;
    source.row(3)[1] = 3.0f;
    source.row(3)[3] = 4.0f;

    OutputPair out;
    out.set(-1.0f, -2.0f);
    check(imprint::acr::core3_colour_difference_row_008134b8(
              source.row(2), green.row(2), kSourceStride, kGreenStride,
              kWidth, 0, out.a.data(), out.b.data()) == Core3ColourStatus::ok,
          "diagonal association: status");
    check(out.b[2] == 0.0f && !std::signbit(out.b[2]),
          "diagonal association: right grouping yields positive zero");
}

void test_signed_zero_and_nonfinite_values() {
    {
        SourceRows source;
        GreenRows green;
        init_pair(source, green, 0.0f, 0.0f);
        green.row(2)[2] = -0.0f;
        source.row(2)[1] = -0.0f;
        source.row(2)[3] = -0.0f;
        OutputPair out;
        out.set(-1.0f, -2.0f);
        check(imprint::acr::core3_colour_difference_row_008134b8(
                  source.row(2), green.row(2), kSourceStride, kGreenStride,
                  kWidth, 1, out.a.data(), out.b.data()) ==
                  Core3ColourStatus::ok,
              "signed-zero probe: status");
        check(out.a[2] == 0.0f && !std::signbit(out.a[2]),
              "signed-zero probe: lower clamp returns positive zero");
    }
    {
        const float qnan = std::numeric_limits<float>::quiet_NaN();
        SourceRows source;
        GreenRows green;
        init_pair(source, green, 0.25f, 0.25f);
        source.row(2)[1] = qnan;
        source.row(2)[0] = qnan;
        OutputPair phase0;
        OutputPair phase1;
        phase0.set(-1.0f, -2.0f);
        phase1.set(-3.0f, -4.0f);
        check(imprint::acr::core3_colour_difference_row_008134b8(
                  source.row(2), green.row(2), kSourceStride, kGreenStride,
                  kWidth, 0, phase0.a.data(), phase0.b.data()) ==
                  Core3ColourStatus::ok,
              "NaN probe: phase 0 status");
        check(imprint::acr::core3_colour_difference_row_008134b8(
                  source.row(2), green.row(2), kSourceStride, kGreenStride,
                  kWidth, 1, phase1.a.data(), phase1.b.data()) ==
                  Core3ColourStatus::ok,
              "NaN probe: phase 1 status");
        check(std::isnan(phase0.a[0]),
              "NaN probe: source-copy lane preserves NaN classification");
        check(std::isnan(phase1.a[2]),
              "NaN probe: interpolated H propagates NaN");
    }
    {
        SourceRows source;
        GreenRows green;
        init_pair(source, green, 0.25f, 0.25f);
        source.row(2)[1] = std::numeric_limits<float>::infinity();
        source.row(2)[3] = std::numeric_limits<float>::infinity();
        OutputPair out;
        out.set(-1.0f, -2.0f);
        check(imprint::acr::core3_colour_difference_row_008134b8(
                  source.row(2), green.row(2), kSourceStride, kGreenStride,
                  kWidth, 1, out.a.data(), out.b.data()) ==
                  Core3ColourStatus::ok,
              "positive infinity probe: status");
        check(out.a[2] == 1.0f,
              "positive infinity probe: positive infinity clamps to one");
    }
    {
        SourceRows source;
        GreenRows green;
        init_pair(source, green, 0.25f, 0.25f);
        source.row(2)[1] = -std::numeric_limits<float>::infinity();
        source.row(2)[3] = -std::numeric_limits<float>::infinity();
        OutputPair out;
        out.set(-1.0f, -2.0f);
        check(imprint::acr::core3_colour_difference_row_008134b8(
                  source.row(2), green.row(2), kSourceStride, kGreenStride,
                  kWidth, 1, out.a.data(), out.b.data()) ==
                  Core3ColourStatus::ok,
              "negative infinity probe: status");
        check(out.a[2] == 0.0f && !std::signbit(out.a[2]),
              "negative infinity probe: negative infinity clamps to positive zero");
    }
}

void test_rejections_do_not_write() {
    SourceRows source;
    GreenRows green;
    init_pair(source, green, 0.25f, 0.25f);
    OutputPair out;

    auto check_rejected = [&](Core3ColourStatus expected, const char *label,
                              const float *source_ptr,
                              const float *green_ptr,
                              std::size_t source_stride,
                              std::size_t green_stride,
                              std::size_t width, std::uint32_t phase,
                              float *out_a, float *out_b) {
        out.set(123.0f, 456.0f);
        check(imprint::acr::core3_colour_difference_row_008134b8(
                  source_ptr, green_ptr, source_stride, green_stride, width,
                  phase, out_a, out_b) == expected,
              label);
        check(all_equal(out.a, 123.0f), "rejection: A output remains sentinel");
        check(all_equal(out.b, 456.0f), "rejection: B output remains sentinel");
    };

    check_rejected(Core3ColourStatus::invalid_argument,
                   "reject null source", nullptr, green.row(2), kSourceStride,
                   kGreenStride, kWidth, 0, out.a.data(), out.b.data());
    check_rejected(Core3ColourStatus::invalid_argument,
                   "reject null green", source.row(2), nullptr, kSourceStride,
                   kGreenStride, kWidth, 0, out.a.data(), out.b.data());
    check_rejected(Core3ColourStatus::invalid_argument,
                   "reject null A output", source.row(2), green.row(2),
                   kSourceStride, kGreenStride, kWidth, 0, nullptr,
                   out.b.data());
    check_rejected(Core3ColourStatus::invalid_argument,
                   "reject null B output", source.row(2), green.row(2),
                   kSourceStride, kGreenStride, kWidth, 0, out.a.data(),
                   nullptr);
    check_rejected(Core3ColourStatus::invalid_dimensions,
                   "reject width three", source.row(2), green.row(2), 4, 4, 3,
                   0, out.a.data(), out.b.data());
    check_rejected(Core3ColourStatus::invalid_dimensions,
                   "reject undersized source stride", source.row(2),
                   green.row(2), 4, kGreenStride, kWidth, 0, out.a.data(),
                   out.b.data());
    check_rejected(Core3ColourStatus::invalid_dimensions,
                   "reject non-vector green stride", source.row(2),
                   green.row(2), kSourceStride, kGreenStride - 1, kWidth, 0,
                   out.a.data(), out.b.data());
    check_rejected(Core3ColourStatus::invalid_phase,
                   "reject phase two", source.row(2), green.row(2),
                   kSourceStride, kGreenStride, kWidth, 2, out.a.data(),
                   out.b.data());
    check_rejected(Core3ColourStatus::unsupported_alignment,
                   "reject unaligned source", source.row(2) + 1,
                   green.row(2), kSourceStride, kGreenStride, kWidth, 0,
                   out.a.data(), out.b.data());
    check_rejected(Core3ColourStatus::unsupported_alignment,
                   "reject unaligned output", source.row(2), green.row(2),
                   kSourceStride, kGreenStride, kWidth, 0, out.a.data() + 1,
                   out.b.data());

    const auto max_ptrdiff = static_cast<std::size_t>(
        std::numeric_limits<std::ptrdiff_t>::max());
    if (max_ptrdiff < std::numeric_limits<std::size_t>::max()) {
        const std::size_t unrepresentable_stride = max_ptrdiff + 1;
        check_rejected(Core3ColourStatus::invalid_dimensions,
                       "reject stride outside ptrdiff range", source.row(2),
                       green.row(2), unrepresentable_stride, kGreenStride,
                       kWidth, 0, out.a.data(), out.b.data());
    }
}

}  // namespace

int main() {
    test_constant_and_input_immutability();
    test_phase_lane_interleave();
    test_direction_probes();
    test_clamp_and_verbatim_source_copy();
    test_diagonal_association();
    test_signed_zero_and_nonfinite_values();
    test_rejections_do_not_write();

    if (failures != 0) {
        std::fprintf(stderr, "%d colour validation check(s) failed\n", failures);
        return 1;
    }
    std::printf("acr_stage3_core3_colour validation passed\n");
    return 0;
}
