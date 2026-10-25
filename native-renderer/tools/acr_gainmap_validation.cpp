#include "../src/acr_gainmap.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

using imprint::acr::GainMapOutput;
using imprint::acr::GainMapSpec;
using imprint::acr::GainMapStatus;

bool expect(bool condition, const std::string& message) {
    if (condition) return true;
    std::cerr << "FAIL: " << message << '\n';
    return false;
}

GainMapSpec base_spec(std::int32_t top, std::int32_t left,
                      std::int32_t bottom, std::int32_t right,
                      std::uint32_t planes, std::uint32_t map_v,
                      std::uint32_t map_h, std::uint32_t map_planes) {
    GainMapSpec spec{};
    spec.image_bounds = {top, left, bottom, right};
    spec.area = {0, 0, 0, 0};
    spec.image_planes = planes;
    spec.first_plane = 0;
    spec.plane_count = planes;
    spec.row_pitch = 1;
    spec.col_pitch = 1;
    spec.map_v = map_v;
    spec.map_h = map_h;
    spec.map_planes = map_planes;
    spec.spacing_v = 1.0;
    spec.spacing_h = 1.0;
    spec.origin_v = 0.0;
    spec.origin_h = 0.0;
    spec.tile_width = 256;
    spec.tile_height = 256;
    return spec;
}

bool run(const std::vector<float>& source, std::size_t stride,
         const GainMapSpec& spec, const std::vector<float>& map,
         GainMapOutput* output) {
    const auto status = imprint::acr::apply_gainmap_prepared_f32(
        source.data(), source.size(), stride, spec, map.data(), map.size(), output);
    if (status != GainMapStatus::ok) {
        std::cerr << "FAIL: gainmap status=" << static_cast<int>(status) << '\n';
        return false;
    }
    return true;
}

bool test_identity_constant_and_clamps() {
    {
        auto spec = base_spec(0, 0, 2, 3, 1, 1, 1, 1);
        const std::vector<float> source{0.0f, 0.25f, 0.75f,
                                        -0.5f, 0.5f, 1.0f};
        const std::vector<float> map{1.0f};
        GainMapOutput out{};
        if (!expect(run(source, 3, spec, map, &out), "identity map succeeds") ||
            !expect(out.width == 3 && out.height == 2 && out.planes == 1,
                    "identity output dimensions") ||
            !expect(out.pixels == source, "identity gain preserves finite in-range values")) {
            return false;
        }
    }
    {
        auto spec = base_spec(0, 0, 1, 2, 1, 1, 1, 1);
        const std::vector<float> source{-0.25f, 0.75f};
        const std::vector<float> map{2.0f};
        GainMapOutput out{};
        if (!expect(run(source, 2, spec, map, &out), "constant map succeeds") ||
            !expect(out.pixels[0] == -0.5f,
                    "negative products are not lower-clamped") ||
            !expect(out.pixels[1] == 1.0f,
                    "positive products are upper-clamped")) {
            return false;
        }
    }
    {
        auto spec = base_spec(-3, 5, -1, 7, 1, 1, 2, 1);
        spec.spacing_h = 0.5;
        spec.origin_h = 0.25;
        const std::vector<float> source{0.2f, 0.2f, 0.2f, 0.2f};
        const std::vector<float> map{1.0f, 3.0f};
        GainMapOutput out{};
        if (!expect(run(source, 2, spec, map, &out),
                    "nonzero bounds and map origin succeed") ||
            !expect(out.pixels[0] == 0.2f,
                    "left edge selects the first map node") ||
            !expect(out.pixels[1] == 0.6f,
                    "right edge selects the last map node")) {
            return false;
        }
    }
    {
        auto spec = base_spec(-8, 11, -7, 12, 1, 2, 2, 1);
        const std::vector<float> source{0.2f};
        const std::vector<float> map{1.0f, 3.0f, 5.0f, 7.0f};
        GainMapOutput out{};
        if (!expect(run(source, 1, spec, map, &out),
                    "center interpolation succeeds") ||
            !expect(out.pixels[0] == 0.8f,
                    "vertical and horizontal center use bilinear map samples")) {
            return false;
        }
    }
    {
        auto spec = base_spec(-8, 11, -7, 12, 1, 2, 2, 1);
        spec.spacing_v = std::numeric_limits<double>::denorm_min();
        spec.spacing_h = std::numeric_limits<double>::denorm_min();
        spec.origin_v = -1.0;
        spec.origin_h = -1.0;
        const std::vector<float> source{0.1f};
        const std::vector<float> map{1.0f, 2.0f, 3.0f, 4.0f};
        GainMapOutput out{};
        if (!expect(run(source, 1, spec, map, &out),
                    "positive infinite coordinates clamp to map edges") ||
            !expect(out.pixels[0] == 0.4f,
                    "positive infinite coordinates select bottom-right map node")) {
            return false;
        }
        spec.origin_v = 1.0;
        spec.origin_h = 1.0;
        if (!expect(run(source, 1, spec, map, &out),
                    "negative infinite coordinates clamp to map edges") ||
            !expect(out.pixels[0] == 0.1f,
                    "negative infinite coordinates select top-left map node")) {
            return false;
        }
    }
    return true;
}

bool test_area_pitch_plane_and_single_axis() {
    auto spec = base_spec(-2, 3, 1, 7, 3, 1, 1, 1);
    spec.area = {-1, 2, 1, 7};
    spec.first_plane = 1;
    spec.plane_count = 2;
    spec.map_planes = 2;
    spec.row_pitch = 9;  // Parser clamps this to the non-empty area's height.
    spec.col_pitch = 2;
    spec.spacing_v = std::numeric_limits<double>::quiet_NaN();
    spec.origin_v = std::numeric_limits<double>::quiet_NaN();
    spec.spacing_h = std::numeric_limits<double>::infinity();
    spec.origin_h = std::numeric_limits<double>::quiet_NaN();
    std::vector<float> source(36);
    for (std::size_t i = 0; i < source.size(); ++i) {
        source[i] = std::array<float, 3>{{0.1f, 0.2f, 0.3f}}[i % 3];
    }
    const std::vector<float> map{2.0f, 3.0f};
    GainMapOutput out{};
    if (!expect(run(source, 12, spec, map, &out),
                "single-map-axis geometry is parser-normalized") ||
        !expect(out.width == 4 && out.height == 3 && out.planes == 3,
                "single-axis output dimensions")) {
        return false;
    }
    // Intersection starts at row -1, column 3. Row pitch was clamped to 2,
    // column pitch remains 2; selected columns are 3 and 5 in that row only.
    for (std::size_t y = 0; y < 3; ++y) {
        for (std::size_t x = 0; x < 4; ++x) {
            for (std::size_t p = 0; p < 3; ++p) {
                const std::size_t i = (y * 4 + x) * 3 + p;
                const bool selected = y == 1 && (x == 1 || x == 3) && p >= 1;
                const float expected = selected ? std::min(source[(y * 4 + x) * 3 + p] * 3.0f, 1.0f)
                                                : source[(y * 4 + x) * 3 + p];
                if (!expect(out.pixels[i] == expected,
                            "area pitch and absolute map-plane selection")) {
                    return false;
                }
            }
        }
    }
    return true;
}

bool test_empty_area_tile_pitches() {
    auto spec = base_spec(-4, 5, -1, 10, 1, 1, 1, 1);
    spec.row_pitch = 2;
    spec.col_pitch = 3;
    spec.tile_width = 3;
    spec.tile_height = 2;
    const std::vector<float> source(15, 0.2f);
    const std::vector<float> map{2.0f};
    GainMapOutput out{};
    if (!expect(run(source, 5, spec, map, &out),
                "empty area uses each tile's full region") ||
        !expect(out.width == 5 && out.height == 3,
                "empty area output dimensions")) {
        return false;
    }
    for (std::size_t y = 0; y < 3; ++y) {
        for (std::size_t x = 0; x < 5; ++x) {
            const bool selected = (y == 0 || y == 2) && (x == 0 || x == 3);
            const float expected = selected ? 0.4f : 0.2f;
            if (!expect(out.pixels[y * 5 + x] == expected,
                        "empty-area row/column pitches restart at tile origin")) {
                return false;
            }
        }
    }
    return true;
}

bool test_black_floor() {
    auto spec = base_spec(0, 0, 1, 1, 1, 1, 1, 1);
    spec.list_stage = 2;
    const std::vector<float> source{0.25f};
    const std::vector<float> map{2.0f};
    GainMapOutput no_wrap{};
    spec.black_floor = 0;
    if (!expect(run(source, 1, spec, map, &no_wrap), "black floor zero succeeds") ||
        !expect(no_wrap.pixels[0] == 0.5f, "zero floor skips black wrapping")) {
        return false;
    }

    spec.black_floor = 4100;
    GainMapOutput wrapped{};
    if (!expect(run(source, 1, spec, map, &wrapped), "4100 floor succeeds")) return false;
    const float z = 4100.0f / 65535.0f;
    const float scale = 1.0f - z;
    const float inv = 1.0f / scale;
    const float bias = 1.0f - inv;
    const float prepared = std::fma(source[0], inv, bias);
    const float mapped = std::fmin(2.0f * prepared, 1.0f);
    const float expected = std::fma(mapped, scale, z);
    return expect(std::memcmp(&wrapped.pixels[0], &expected, sizeof(float)) == 0,
                  "4100 floor follows single-precision pre/post FMA sequence");
}

bool test_padded_source_and_invalid_domains() {
    auto spec = base_spec(-5, 8, -3, 10, 1, 1, 1, 1);
    const std::vector<float> source{
        0.1f, 0.2f, std::numeric_limits<float>::quiet_NaN(),
        0.3f, 0.4f, std::numeric_limits<float>::quiet_NaN(),
    };
    const std::vector<float> map{1.0f};
    GainMapOutput out{};
    if (!expect(run(source, 3, spec, map, &out),
                "active finite samples permit NaN padding") ||
        !expect(out.pixels == std::vector<float>({0.1f, 0.2f, 0.3f, 0.4f}),
                "padded source rows produce compact output")) {
        return false;
    }

    spec.list_stage = 2;
    spec.black_floor = 65535;
    const auto degenerate = imprint::acr::apply_gainmap_prepared_f32(
        source.data(), source.size(), 3, spec, map.data(), map.size(), &out);
    if (!expect(degenerate == GainMapStatus::unsupported_domain,
                "stage two rejects a degenerate full-black domain")) {
        return false;
    }

    spec.black_floor = 0;
    spec.first_plane = std::numeric_limits<std::uint32_t>::max();
    spec.plane_count = 1;
    const auto plane_overflow = imprint::acr::apply_gainmap_prepared_f32(
        source.data(), source.size(), 3, spec, map.data(), map.size(), &out);
    if (!expect(plane_overflow == GainMapStatus::invalid_metadata,
                "overflowing plane end is rejected")) {
        return false;
    }

    spec = base_spec(-5, 8, -3, 10, 1, 1, 2, 1);
    spec.origin_h = 2147483648.0;
    const std::vector<float> horizontal_map{1.0f, 1.0f};
    const auto unrepresentable_end = imprint::acr::apply_gainmap_prepared_f32(
        source.data(), source.size(), 3, spec, horizontal_map.data(),
        horizontal_map.size(), &out);
    return expect(unrepresentable_end == GainMapStatus::unsupported_domain,
                  "unrepresentable segment end is rejected");
}

bool test_tile_state_reset() {
    constexpr std::size_t width = 257;
    auto spec = base_spec(0, 0, 1, static_cast<std::int32_t>(width), 1, 1, 2, 1);
    spec.spacing_h = 1.0;
    const std::vector<float> source(width, 0.73f);
    const std::vector<float> map{0.12345679f, 0.9876543f};
    spec.tile_width = 1;
    GainMapOutput single_pixel_tiles{};
    if (!expect(run(source, width, spec, map, &single_pixel_tiles),
                "single-column tiles succeed")) return false;
    spec.tile_width = 37;
    GainMapOutput wider_tiles{};
    if (!expect(run(source, width, spec, map, &wider_tiles),
                "wide tiles succeed")) return false;
    bool differs = false;
    for (std::size_t i = 0; i < width; ++i) {
        differs = differs || std::memcmp(&single_pixel_tiles.pixels[i],
                                         &wider_tiles.pixels[i], sizeof(float)) != 0;
    }
    return expect(differs,
                  "horizontal state is reinitialized at tile starts (rounding can differ)");
}

bool test_failure_atomicity_alias_and_counts() {
    auto spec = base_spec(0, 0, 1, 2, 1, 1, 1, 1);
    const std::vector<float> source{0.2f, 0.3f};
    const std::vector<float> map{1.0f};
    GainMapOutput out{44, 55, 66, {7.0f, 8.0f}};
    const GainMapOutput old = out;
    spec.map_h = 0;
    const auto bad_spec = imprint::acr::apply_gainmap_prepared_f32(
        source.data(), source.size(), 2, spec, map.data(), map.size(), &out);
    if (!expect(bad_spec != GainMapStatus::ok, "invalid map metadata is rejected") ||
        !expect(out.width == old.width && out.height == old.height &&
                    out.planes == old.planes && out.pixels == old.pixels,
                "invalid metadata leaves vector output unchanged")) {
        return false;
    }

    spec = base_spec(0, 0, 1, 2, 1, 1, 1, 1);
    const std::vector<float> nonfinite_source{
        std::numeric_limits<float>::infinity(), 0.3f};
    const auto nonfinite_status = imprint::acr::apply_gainmap_prepared_f32(
        nonfinite_source.data(), nonfinite_source.size(), 2, spec, map.data(),
        map.size(), &out);
    if (!expect(nonfinite_status == GainMapStatus::non_finite_input,
                "infinite source sample is rejected") ||
        !expect(out.width == old.width && out.height == old.height &&
                    out.planes == old.planes && out.pixels == old.pixels,
                "nonfinite source leaves vector output unchanged")) {
        return false;
    }

    const std::array<std::int32_t, 8> rects{{0, 0, 1, 2, 0, 0, 0, 0}};
    const std::array<std::uint32_t, 9> controls{{1, 0, 1, 1, 1, 1, 1, 1, 0}};
    const std::array<double, 4> geometry{{1.0, 1.0, 0.0, 0.0}};
    std::array<float, 2> destination{{9.0f, 10.0f}};
    const auto alias_status = imprint::acr::apply_gainmap_prepared_f32_to_buffer(
        source.data(), source.size(), 2, rects.data(), controls.data(),
        geometry.data(), source.data(), 1, 0, 256, 256, destination.data(),
        destination.size());
    if (!expect(alias_status == GainMapStatus::overlap,
                "source/map declared-span alias is rejected") ||
        !expect(destination[0] == 9.0f && destination[1] == 10.0f,
                "alias failure leaves destination unchanged")) {
        return false;
    }

    const auto count_status = imprint::acr::apply_gainmap_prepared_f32(
        source.data(), std::numeric_limits<std::size_t>::max(), 2, spec,
        map.data(), map.size(), &out);
    if (!expect(count_status == GainMapStatus::arithmetic_overflow,
                "unrepresentable declared source span is rejected") ||
        !expect(out.width == old.width && out.height == old.height &&
                    out.planes == old.planes && out.pixels == old.pixels,
                "count overflow leaves vector output unchanged")) {
        return false;
    }

    const auto short_status = imprint::acr::apply_gainmap_prepared_f32_to_buffer(
        source.data(), source.size(), 2, rects.data(), controls.data(),
        geometry.data(), map.data(), map.size(), 0, 256, 256,
        destination.data(), 1);
    return expect(short_status == GainMapStatus::buffer_too_small,
                  "short destination is rejected before writing") &&
           expect(destination[0] == 9.0f && destination[1] == 10.0f,
                  "short destination remains unchanged");
}

bool test_vector_alias_contract() {
    auto spec = base_spec(0, 0, 1, 2, 1, 1, 1, 1);
    const std::vector<float> map{2.0f};

    GainMapOutput aliased{};
    aliased.width = 1;
    aliased.height = 2;
    aliased.planes = 1;
    aliased.pixels = {0.25f, 0.5f};
    const auto source_status = imprint::acr::apply_gainmap_prepared_f32(
        aliased.pixels.data(), aliased.pixels.size(), 2, spec, map.data(),
        map.size(), &aliased);
    if (!expect(source_status == GainMapStatus::ok,
                "source may alias old output pixel storage") ||
        !expect(aliased.pixels == std::vector<float>({0.5f, 1.0f}),
                "aliased source is fully read before output commit")) {
        return false;
    }

    GainMapOutput object_alias{12, 13, 14, {7.0f, 8.0f}};
    const GainMapOutput old = object_alias;
    const auto object_status = imprint::acr::apply_gainmap_prepared_f32(
        reinterpret_cast<const float*>(&object_alias), 1, 2, spec,
        map.data(), map.size(), &object_alias);
    return expect(object_status == GainMapStatus::overlap,
                  "source span may not overlap the output object") &&
           expect(object_alias.width == old.width &&
                      object_alias.height == old.height &&
                      object_alias.planes == old.planes &&
                      object_alias.pixels == old.pixels,
                  "output-object alias rejection leaves output unchanged");
}

}  // namespace

int main() {
    if (!test_identity_constant_and_clamps() ||
        !test_area_pitch_plane_and_single_axis() ||
        !test_empty_area_tile_pitches() ||
        !test_black_floor() || !test_padded_source_and_invalid_domains() ||
        !test_tile_state_reset() || !test_vector_alias_contract() ||
        !test_failure_atomicity_alias_and_counts()) {
        return 1;
    }
    std::cout << "GainMap CPU validation passed\n";
    return 0;
}
