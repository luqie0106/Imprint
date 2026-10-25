#include "../src/acr_stage3_dispatch.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

using imprint::acr::Stage3Branch;
using imprint::acr::Stage3Dispatch;
using imprint::acr::Stage3Kernel;
using imprint::acr::Stage3Prepared;
using imprint::acr::Stage3Status;

int failures = 0;

void check(bool condition, const std::string &message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message.c_str());
        ++failures;
    }
}

std::string status_name(Stage3Status status) {
    switch (status) {
        case Stage3Status::ok: return "ok";
        case Stage3Status::invalid_argument: return "invalid_argument";
        case Stage3Status::invalid_dimensions: return "invalid_dimensions";
        case Stage3Status::invalid_cfa: return "invalid_cfa";
        case Stage3Status::unsupported_mode: return "unsupported_mode";
        case Stage3Status::unsupported_pair: return "unsupported_pair";
        case Stage3Status::unsupported_offset: return "unsupported_offset";
        case Stage3Status::unsupported_plane_map: return "unsupported_plane_map";
        case Stage3Status::unsupported_task_transform: return "unsupported_task_transform";
        case Stage3Status::unsupported_full_res_core3: return "unsupported_full_res_core3";
        case Stage3Status::unsupported_kernel: return "unsupported_kernel";
        case Stage3Status::invalid_stride: return "invalid_stride";
        case Stage3Status::buffer_too_small: return "buffer_too_small";
        case Stage3Status::unsupported_aliasing: return "unsupported_aliasing";
        case Stage3Status::allocation_failure: return "allocation_failure";
        case Stage3Status::invalid_rect: return "invalid_rect";
    }
    return "unknown";
}

Stage3Prepared base_prepared() {
    Stage3Prepared prepared;
    prepared.width = 8;
    prepared.height = 4;
    prepared.input_stride = 8;
    prepared.cfa_pattern[0] = 0;
    prepared.cfa_pattern[1] = 1;
    prepared.cfa_pattern[2] = 1;
    prepared.cfa_pattern[3] = 2;
    prepared.plane_count = 3;
    prepared.plane_map[0] = 0;
    prepared.plane_map[1] = 1;
    prepared.plane_map[2] = 2;
    prepared.mode = 1;
    prepared.pair_x = 1;
    prepared.pair_y = 1;
    prepared.rect_top = 0;
    prepared.rect_left = 0;
    prepared.rect_bottom = 4;
    prepared.rect_right = 8;
    return prepared;
}

void check_dispatch_cases() {
    {
        const Stage3Dispatch dispatch = imprint::acr::dispatch_stage3(base_prepared());
        check(dispatch.status == Stage3Status::unsupported_full_res_core3,
              "full-res pair (1,1) must stop at the core3 pipeline");
        check(dispatch.branch == Stage3Branch::direct_00ab3f24,
              "full-res pair (1,1) branch");
        check(dispatch.kernel == Stage3Kernel::bayer_1x1_007f2124,
              "full-res pair (1,1) kernel");
        check(dispatch.output_width == 8 && dispatch.output_height == 4,
              "full-res output geometry");
    }
    {
        Stage3Prepared prepared = base_prepared();
        prepared.pair_x = 2;
        prepared.pair_y = 2;
        const Stage3Dispatch dispatch = imprint::acr::dispatch_stage3(prepared);
        check(dispatch.status == Stage3Status::ok, "preview pair (2,2) status");
        check(dispatch.branch == Stage3Branch::direct_00ab3f24,
              "preview pair (2,2) branch");
        check(dispatch.kernel == Stage3Kernel::bayer_2x2_007f851c,
              "preview pair (2,2) kernel");
        check(dispatch.output_width == 4 && dispatch.output_height == 2,
              "preview pair (2,2) output geometry");
        check(!dispatch.half_pixel_shift, "preview pair (2,2) half-pixel flag");
    }
    {
        Stage3Prepared prepared = base_prepared();
        prepared.pair_x = 2;
        prepared.pair_y = 4;
        const Stage3Dispatch dispatch = imprint::acr::dispatch_stage3(prepared);
        check(dispatch.status == Stage3Status::unsupported_kernel,
              "pair (2,4) must remain explicit");
        check(dispatch.kernel == Stage3Kernel::bayer_2x4_007f9228,
              "pair (2,4) kernel");
    }
    {
        Stage3Prepared prepared = base_prepared();
        prepared.pair_x = 3;
        prepared.pair_y = 3;
        const Stage3Dispatch dispatch = imprint::acr::dispatch_stage3(prepared);
        check(dispatch.status == Stage3Status::unsupported_kernel,
              "other equal pair must remain explicit");
        check(dispatch.kernel == Stage3Kernel::generic_007fb724,
              "other equal pair kernel");
    }
    {
        Stage3Prepared prepared = base_prepared();
        prepared.pair_x = 1;
        prepared.pair_y = 2;
        const Stage3Dispatch dispatch = imprint::acr::dispatch_stage3(prepared);
        check(dispatch.status == Stage3Status::unsupported_pair,
              "non-equal pair must take explicit generic stop");
        check(dispatch.branch == Stage3Branch::generic_012a4a60,
              "non-equal pair branch");
    }
    {
        Stage3Prepared prepared = base_prepared();
        prepared.mode = 2;
        const Stage3Dispatch dispatch = imprint::acr::dispatch_stage3(prepared);
        check(dispatch.status == Stage3Status::unsupported_mode,
              "mode 2 must be rejected");
    }
    {
        Stage3Prepared prepared = base_prepared();
        prepared.offset_80 = 1;
        const Stage3Dispatch dispatch = imprint::acr::dispatch_stage3(prepared);
        check(dispatch.status == Stage3Status::unsupported_offset,
              "nonzero mosaic+0x80 must be rejected");
    }
    {
        Stage3Prepared prepared = base_prepared();
        prepared.plane_map[1] = 2;
        const Stage3Dispatch dispatch = imprint::acr::dispatch_stage3(prepared);
        check(dispatch.status == Stage3Status::unsupported_plane_map,
              "non-aligned plane map must be rejected");
        check(dispatch.branch == Stage3Branch::task_0127cfe0,
              "non-aligned plane map selects the task branch");
    }
    {
        Stage3Prepared prepared = base_prepared();
        prepared.pair_x = 2;
        prepared.pair_y = 2;
        prepared.task_transform = true;
        const Stage3Dispatch dispatch = imprint::acr::dispatch_stage3(prepared);
        check(dispatch.status == Stage3Status::unsupported_task_transform,
              "2x2 task transform must be rejected");
    }
    {
        Stage3Prepared prepared = base_prepared();
        prepared.width = 9;
        prepared.input_stride = 9;
        prepared.rect_right = 9;
        prepared.pair_x = 2;
        prepared.pair_y = 2;
        const Stage3Dispatch dispatch = imprint::acr::dispatch_stage3(prepared);
        check(dispatch.status == Stage3Status::unsupported_kernel,
              "odd 2x2 rect must be rejected");
    }
}

void check_canonical_pattern_and_rejections() {
    // The recovered mode writer writes +0x80/+0x84 = 0 only for identity RGGB.
    // The other standard phases store a packed nonzero offset pair there and
    // then enter the unrecovered FUN_00ab4094 transform.
    const std::uint8_t patterns[4][4] = {
        {0, 1, 1, 2},  // RGGB, offset (0,0)
        {2, 1, 1, 0},  // BGGR, offset (1,1)
        {1, 0, 2, 1},  // GRBG, offset (0,1)
        {1, 2, 0, 1},  // GBRG, offset (1,0)
    };
    const std::int32_t writer_offsets[4][2] = {
        {0, 0}, {1, 1}, {0, 1}, {1, 0},
    };
    const std::vector<std::uint16_t> input = {
        10, 20,
        30, 40,
    };
    for (std::size_t index = 0; index < 4; ++index) {
        Stage3Prepared prepared = base_prepared();
        prepared.width = 2;
        prepared.height = 2;
        prepared.input_stride = 2;
        prepared.rect_bottom = 2;
        prepared.rect_right = 2;
        prepared.pair_x = 2;
        prepared.pair_y = 2;
        for (std::size_t position = 0; position < 4; ++position) {
            prepared.cfa_pattern[position] = patterns[index][position];
        }
        prepared.offset_80 = writer_offsets[index][0];
        prepared.offset_84 = writer_offsets[index][1];
        const Stage3Dispatch dispatch =
            imprint::acr::dispatch_stage3(prepared);
        if (index == 0) {
            check(dispatch.status == Stage3Status::ok,
                  "canonical RGGB with zero offset must be accepted");
            std::vector<std::uint16_t> output;
            const Stage3Status status = imprint::acr::run_stage3_bayer_2x2(
                input.data(), input.size(), prepared, &output);
            check(status == Stage3Status::ok,
                  "canonical RGGB run status " + status_name(status));
            check(output.size() == 3, "canonical RGGB output size");
            check(output[0] == 10 && output[1] == 25 && output[2] == 40,
                  "canonical RGGB pixel [10,25,40]");
        } else {
            check(dispatch.status == Stage3Status::invalid_cfa,
                  "non-RGGB mode-writer state must be rejected");
        }
    }

    // Non-Bayer three-colour permutations must not be accepted even though the
    // independent fast-Bayer helper intentionally allows 36 arrangements.
    const std::uint8_t non_bayer[2][4] = {
        {0, 0, 1, 2},
        {0, 1, 2, 2},
    };
    for (const auto &pattern : non_bayer) {
        Stage3Prepared prepared = base_prepared();
        prepared.pair_x = 2;
        prepared.pair_y = 2;
        for (std::size_t position = 0; position < 4; ++position) {
            prepared.cfa_pattern[position] = pattern[position];
        }
        const Stage3Dispatch dispatch =
            imprint::acr::dispatch_stage3(prepared);
        check(dispatch.status == Stage3Status::invalid_cfa,
              "non-Bayer three-colour pattern must be rejected");
    }

    // Pattern/offset contradictions: canonical RGGB with a nonzero recovered
    // offset means the state did not come from the RGGB writer branch.
    for (int which = 0; which < 2; ++which) {
        Stage3Prepared prepared = base_prepared();
        prepared.pair_x = 2;
        prepared.pair_y = 2;
        if (which == 0) {
            prepared.offset_80 = 1;
        } else {
            prepared.offset_84 = 1;
        }
        const Stage3Dispatch dispatch =
            imprint::acr::dispatch_stage3(prepared);
        check(dispatch.status == Stage3Status::unsupported_offset,
              "canonical RGGB with nonzero offset must be rejected");
    }
}

void check_nonzero_origin_and_stride() {
    Stage3Prepared prepared = base_prepared();
    prepared.width = 8;
    prepared.height = 6;
    prepared.input_stride = 10;
    prepared.rect_top = 2;
    prepared.rect_left = 2;
    prepared.rect_bottom = 6;
    prepared.rect_right = 8;
    prepared.pair_x = 2;
    prepared.pair_y = 2;
    std::vector<std::uint16_t> input(10 * 6, 0xFFFF);
    for (std::size_t y = 0; y < 6; ++y) {
        for (std::size_t x = 0; x < 8; ++x) {
            input[y * 10 + x] = static_cast<std::uint16_t>(y * 100 + x);
        }
    }
    const std::vector<std::uint16_t> before = input;
    std::vector<std::uint16_t> output;
    const Stage3Status status = imprint::acr::run_stage3_bayer_2x2(
        input.data(), input.size(), prepared, &output);
    check(status == Stage3Status::ok, "nonzero origin: status " + status_name(status));
    check(input == before, "nonzero origin: input must not change");
    check(output.size() == 3 * 2 * 3, "nonzero origin: output size");
    check(output[0] == input[2 * 10 + 2], "nonzero origin: red sample");
    check(output[1] == static_cast<std::uint16_t>(
                            (static_cast<std::uint32_t>(input[2 * 10 + 3]) +
                             input[3 * 10 + 2] + 1) /
                            2),
          "nonzero origin: green average");
    check(output[2] == input[3 * 10 + 3], "nonzero origin: blue sample");
}

void check_tile_seam() {
    Stage3Prepared whole = base_prepared();
    whole.width = 4;
    whole.height = 4;
    whole.input_stride = 4;
    whole.rect_bottom = 4;
    whole.rect_right = 4;
    whole.pair_x = 2;
    whole.pair_y = 2;
    const std::vector<std::uint16_t> input = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16,
    };
    std::vector<std::uint16_t> whole_out;
    check(imprint::acr::run_stage3_bayer_2x2(
              input.data(), input.size(), whole, &whole_out) == Stage3Status::ok,
          "tile seam: whole status");

    std::vector<std::uint16_t> tiles;
    for (int tile = 0; tile < 2; ++tile) {
        Stage3Prepared part = whole;
        part.rect_left = tile * 2;
        part.rect_right = part.rect_left + 2;
        std::vector<std::uint16_t> out;
        check(imprint::acr::run_stage3_bayer_2x2(
                  input.data(), input.size(), part, &out) == Stage3Status::ok,
              "tile seam: part status");
        tiles.insert(tiles.end(), out.begin(), out.end());
    }
    const std::vector<std::uint16_t> interleaved = {
        tiles[0], tiles[1], tiles[2],
        tiles[6], tiles[7], tiles[8],
        tiles[3], tiles[4], tiles[5],
        tiles[9], tiles[10], tiles[11],
    };
    check(interleaved == whole_out,
          "tile seam: two 2x4 tiles must match the 4x4 run");
}

void check_failure_atomicity() {
    Stage3Prepared prepared = base_prepared();
    prepared.pair_x = 2;
    prepared.pair_y = 2;
    const std::vector<std::uint16_t> input(8 * 4, 42);
    const std::vector<std::uint16_t> sentinel(9, 0xABCD);
    std::vector<std::uint16_t> output = sentinel;
    const Stage3Status status = imprint::acr::run_stage3_bayer_2x2(
        input.data(), 4, prepared, &output);
    check(status == Stage3Status::buffer_too_small,
          "short buffer: status " + status_name(status));
    check(output == sentinel, "short buffer must not change output");

    Stage3Prepared full = base_prepared();
    std::vector<std::uint16_t> untouched = sentinel;
    const Stage3Status full_status = imprint::acr::run_stage3_bayer_2x2(
        input.data(), input.size(), full, &untouched);
    check(full_status == Stage3Status::unsupported_full_res_core3,
          "full-res run must remain unsupported");
    check(untouched == sentinel, "full-res run must not change output");
}

}  // namespace

int main() {
    check_dispatch_cases();
    check_canonical_pattern_and_rejections();
    check_nonzero_origin_and_stride();
    check_tile_seam();
    check_failure_atomicity();
    if (failures != 0) {
        std::fprintf(stderr, "%d Stage3 check(s) failed\n", failures);
        return 1;
    }
    std::puts("PASS: Stage3 dispatch and closed 2x2 preview branch");
    return 0;
}
