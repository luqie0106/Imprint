#include "../src/acr_warp.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

extern "C" int acr_test_warp(
    const float* source, std::size_t source_sample_count,
    std::size_t source_row_stride, const std::int32_t rect4[4],
    const std::uint32_t controls2[2], const double geometry3[3],
    const double* coefficients, std::size_t coefficient_count,
    std::size_t tile_width, std::size_t tile_height, float* destination,
    std::size_t destination_sample_count) noexcept;

namespace {

using imprint::acr::WarpOutput;
using imprint::acr::WarpSpec;
using imprint::acr::WarpStatus;

bool expect(bool condition, const std::string& message) {
    if (condition) return true;
    std::cerr << "FAIL: " << message << '\n';
    return false;
}

WarpSpec spec_for(std::int32_t top, std::int32_t left,
                  std::int32_t bottom, std::int32_t right,
                  std::uint32_t image_planes, std::uint32_t param_planes,
                  std::size_t tile_width = 64,
                  std::size_t tile_height = 64) {
    WarpSpec spec{};
    spec.image_bounds = {top, left, bottom, right};
    spec.image_planes = image_planes;
    spec.param_planes = param_planes;
    spec.center_x = 0.5;
    spec.center_y = 0.5;
    spec.aspect = 1.0;
    spec.tile_width = tile_width;
    spec.tile_height = tile_height;
    return spec;
}

bool test_hardcoded_nop_bitwise() {
    auto spec = spec_for(-3, 5, -1, 8, 2, 1);
    spec.aspect = std::numeric_limits<double>::max();
    const double coefficients[]{1.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const std::vector<float> source{
        -0.0f, 1.25f, -0.5f, 0.375f, 0.0f, 0.625f, nan,
        0.75f, -1.0f, 0.125f, 2.0f, -0.25f, 0.0f, nan,
    };
    WarpOutput output{};
    const auto status = imprint::acr::apply_warp_prepared_f32(
        source.data(), source.size(), 7, spec, coefficients, 6, &output);
    if (!expect(status == WarpStatus::ok, "radial identity NOP succeeds") ||
        !expect(output.width == 3 && output.height == 2 && output.planes == 2,
                "NOP dimensions and plane count") ||
        !expect(output.pixels.size() == 12, "NOP returns compact HWC output")) {
        return false;
    }
    const std::array<float, 12> expected{{
        -0.0f, 1.25f, -0.5f, 0.375f, 0.0f, 0.625f,
        0.75f, -1.0f, 0.125f, 2.0f, -0.25f, 0.0f,
    }};
    return expect(std::memcmp(output.pixels.data(), expected.data(),
                              sizeof(expected)) == 0,
                  "NOP copies finite active samples bit-for-bit and skips row padding");
}

bool test_scaled_height_int32_conversion_range() {
    auto spec = spec_for(0, 0, 8, 8, 1, 1);
    const double nonidentity[]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    const std::array<float, 64> source{{
        0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f,
        0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f,
        0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f,
        0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f,
        0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f,
        0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f,
        0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f,
        0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f,
    }};
    WarpOutput output{};
    spec.aspect = 2147483647.0 / 8.0;
    auto status = imprint::acr::apply_warp_prepared_f32(
        source.data(), source.size(), 8, spec, nonidentity, 6, &output);
    if (!expect(status == WarpStatus::ok,
                "scaled height plus 0.5 may truncate to INT32_MAX") ||
        !expect(output.pixels.size() == source.size(),
                "largest representable scaled height renders a full candidate")) {
        return false;
    }
    const auto before = output.pixels;
    spec.aspect = 2147483648.0 / 8.0;
    status = imprint::acr::apply_warp_prepared_f32(
        source.data(), source.size(), 8, spec, nonidentity, 6, &output);
    return expect(status == WarpStatus::arithmetic_overflow && output.pixels == before,
                  "scaled height conversion rejects the exclusive int32 upper bound");
}

bool test_nonidentity_tile_edges_and_broadcast() {
    auto spec = spec_for(-4, 7, 5, 16, 3, 1, 4, 3);
    // k0=0 maps every output coordinate to the common radial center. This is
    // a hand-computable border case that still exercises the real consumer.
    const double coefficients[]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    const std::array<float, 3> channel_values{{0.2f, 0.4f, 0.8f}};
    std::vector<float> source(9u * 9u * 3u);
    for (std::size_t i = 0; i < source.size(); ++i) {
        source[i] = channel_values[i % 3u];
    }
    WarpOutput output{};
    const auto status = imprint::acr::apply_warp_prepared_f32(
        source.data(), source.size(), 27, spec, coefficients, 6, &output);
    if (!expect(status == WarpStatus::ok,
                "nonidentity warp with strict multi-tile srcArea succeeds") ||
        !expect(output.width == 9 && output.height == 9 && output.planes == 3,
                "nonidentity output dimensions")) {
        return false;
    }
    for (std::size_t i = 0; i < output.pixels.size(); ++i) {
        const float expected = channel_values[i % 3u];
        if (!expect(std::isfinite(output.pixels[i]) &&
                    std::fabs(output.pixels[i] - expected) < 2.0e-5f,
                    "constant plane survives center-map cubic sampling at tile edges")) {
            return false;
        }
    }
    return true;
}

bool test_aspect_geometry_and_half_rounding() {
    const double coefficients[]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    auto spec = spec_for(0, 0, 8, 8, 1, 1, 4, 3);
    spec.center_y = 0.25;
    spec.aspect = 1.5;
    std::vector<float> source(64);
    for (std::size_t y = 0; y < 8; ++y) {
        for (std::size_t x = 0; x < 8; ++x) {
            source[y * 8 + x] = static_cast<float>(y) / 8.0f;
        }
    }
    WarpOutput output{};
    auto status = imprint::acr::apply_warp_prepared_f32(
        source.data(), source.size(), 8, spec, coefficients, 6, &output);
    if (!expect(status == WarpStatus::ok,
                "aspect geometry with independent scaled-radius center succeeds")) {
        return false;
    }
    for (float value : output.pixels) {
        // The constant center map narrows srcArea to five rows. The recovered
        // max-anchor rule clamps its integer anchor to zero, so the fraction-0
        // kernel samples source row one (the original-height center is row 2).
        if (!expect(std::fabs(value - 0.125f) < 1.0e-6f,
                    "map center uses original image height when aspect changes")) {
            return false;
        }
    }

    // 8 * this aspect is the representable value immediately below 0.5.
    // Adding 0.5 in double rounds to 1.0 before the recovered truncating cast.
    spec = spec_for(0, 0, 8, 8, 1, 1, 8, 8);
    spec.aspect = 0.06249999999999999;
    status = imprint::acr::apply_warp_prepared_f32(
        source.data(), source.size(), 8, spec, coefficients, 6, &output);
    return expect(status == WarpStatus::ok,
                  "double plus-0.5 then truncation accepts the sub-half edge case");
}

bool test_fraction_just_below_one() {
    auto spec = spec_for(0, 0, 16, 16, 1, 1, 16, 16);
    const double coefficients[]{1.0, 1.0, 0.0, 0.0, 0.0, 0.0};
    const std::vector<float> source(16u * 16u, 0.3f);
    WarpOutput output{};
    const auto status = imprint::acr::apply_warp_prepared_f32(
        source.data(), source.size(), 16, spec, coefficients, 6, &output);
    if (!expect(status == WarpStatus::ok,
                "consumer accepts fractional coordinates in [31,32)")) {
        return false;
    }
    for (float value : output.pixels) {
        if (!expect(std::isfinite(value) && std::fabs(value - 0.3f) < 2.0e-5f,
                    "fractional edge table entries preserve a constant image")) {
            return false;
        }
    }
    return true;
}

bool test_failure_atomic_and_unsupported_domain() {
    auto spec = spec_for(0, 0, 4, 8, 1, 1, 4, 4);
    const double coefficients[]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    const std::vector<float> source(32, 0.25f);
    WarpOutput output{};
    output.width = 91;
    output.height = 92;
    output.planes = 93;
    output.pixels = {4.0f, -3.0f};
    const auto before = output.pixels;
    const auto status = imprint::acr::apply_warp_prepared_f32(
        source.data(), source.size(), 8, spec, coefficients, 6, &output);
    if (!expect(status == WarpStatus::unsupported_domain,
                "strict srcArea rejects a source area shorter than five") ||
        !expect(output.width == 91 && output.height == 92 && output.planes == 93 &&
                output.pixels == before,
                "unsupported srcArea leaves vector output unchanged")) {
        return false;
    }

    auto valid_shape = spec_for(0, 0, 8, 8, 1, 1);
    const double invalid_tangent[]{0.0, 0.0, 0.0, 0.0, 0.0, 1.0};
    const auto bad_metadata = imprint::acr::apply_warp_prepared_f32(
        source.data(), source.size(), 8, valid_shape, invalid_tangent, 6, &output);
    return expect(bad_metadata == WarpStatus::invalid_metadata &&
                  output.width == 91 && output.pixels == before,
                  "unsupported tangential coefficient leaves vector output unchanged");
}

bool test_buffer_failures_are_atomic() {
    const std::int32_t rect[]{-2, 4, 2, 8};
    const std::uint32_t controls[]{1, 1};
    const double geometry[]{0.5, 0.5, 1.0};
    const double coefficients[]{1.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    const std::vector<float> source(16, 0.25f);
    std::vector<float> destination(16, -7.0f);
    const auto sentinel = destination;

    const std::int32_t huge_rect[]{
        std::numeric_limits<std::int32_t>::min(), 0,
        std::numeric_limits<std::int32_t>::max(), 4};
    int status = acr_test_warp(source.data(), source.size(), 4, huge_rect,
        controls, geometry, coefficients, 6, 4, 4, destination.data(),
        destination.size());
    if (!expect(status == static_cast<int>(WarpStatus::invalid_dimensions),
                "rect dimension overflow is rejected") ||
        !expect(destination == sentinel, "rect overflow does not write destination")) {
        return false;
    }

    status = acr_test_warp(source.data(), source.size(),
        std::numeric_limits<std::size_t>::max() - 3, rect, controls, geometry,
        coefficients, 6, 4, 4, destination.data(), destination.size());
    if (!expect(status == static_cast<int>(WarpStatus::arithmetic_overflow),
                "row-stride extent overflow is rejected") ||
        !expect(destination == sentinel, "stride overflow does not write destination")) {
        return false;
    }

    const std::int32_t near_limit_rect[]{
        std::numeric_limits<std::int32_t>::max() - 3, 0,
        std::numeric_limits<std::int32_t>::max(), 5};
    const std::uint32_t one_plane[]{1, 1};
    const double tall_geometry[]{0.5, 0.5, 2.0};
    const double nonidentity_coefficients[]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    std::array<float, 15> small_source{};
    status = acr_test_warp(small_source.data(), small_source.size(), 5,
        near_limit_rect, one_plane, tall_geometry, nonidentity_coefficients, 6, 5, 5,
        destination.data(), destination.size());
    if (!expect(status == static_cast<int>(WarpStatus::arithmetic_overflow),
                "scaled rectangle bottom overflow is rejected") ||
        !expect(destination == sentinel,
                "scaled rectangle overflow does not write destination")) {
        return false;
    }

    status = acr_test_warp(source.data(), source.size(), 4, rect, controls,
        geometry, coefficients, 6, 4, 4, destination.data(), 15);
    if (!expect(status == static_cast<int>(WarpStatus::buffer_too_small),
                "short destination is rejected") ||
        !expect(destination == sentinel, "short destination remains unchanged")) {
        return false;
    }

    status = acr_test_warp(source.data(), source.size(), 4, rect, controls,
        geometry, coefficients, 6, 4, 4,
        const_cast<float*>(source.data()), source.size());
    if (!expect(status == static_cast<int>(WarpStatus::overlap),
                "source/destination alias is rejected") ||
        !expect(destination == sentinel, "alias rejection leaves other destination untouched")) {
        return false;
    }

    const float nan = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> bad_source(16, 0.25f);
    bad_source[3] = nan;
    status = acr_test_warp(bad_source.data(), bad_source.size(), 4, rect,
        controls, geometry, coefficients, 6, 4, 4, destination.data(),
        destination.size());
    return expect(status == static_cast<int>(WarpStatus::non_finite_input) &&
                  destination == sentinel,
                  "non-finite active source is rejected atomically");
}

bool test_metadata_overlap_is_rejected() {
    alignas(double) std::array<float, 24> backing{};
    auto* source = backing.data();
    const std::uint32_t controls[]{1, 1};
    const double geometry[]{0.5, 0.5, 1.0};
    const double coefficients[]{1.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    float destination[16];
    std::fill(std::begin(destination), std::end(destination), -9.0f);
    const auto before = std::array<float, 16>{
        -9.0f, -9.0f, -9.0f, -9.0f, -9.0f, -9.0f, -9.0f, -9.0f,
        -9.0f, -9.0f, -9.0f, -9.0f, -9.0f, -9.0f, -9.0f, -9.0f};
    auto* aliased_rect = reinterpret_cast<std::int32_t*>(backing.data());
    const int status = acr_test_warp(source, backing.size(), 4, aliased_rect,
        controls, geometry, coefficients, 6, 4, 4, destination, 16);
    return expect(status == static_cast<int>(WarpStatus::overlap),
                  "source/metadata alias is rejected before reading metadata") &&
           expect(std::memcmp(destination, before.data(), sizeof(before)) == 0,
                  "metadata alias rejection leaves destination bytes unchanged");
}

}  // namespace

int main() {
    if (!test_hardcoded_nop_bitwise() ||
        !test_scaled_height_int32_conversion_range() ||
        !test_nonidentity_tile_edges_and_broadcast() ||
        !test_aspect_geometry_and_half_rounding() ||
        !test_fraction_just_below_one() ||
        !test_failure_atomic_and_unsupported_domain() ||
        !test_buffer_failures_are_atomic() ||
        !test_metadata_overlap_is_rejected()) {
        return 1;
    }
    std::cout << "ACR WarpRectilinear validation passed\n";
    return 0;
}
