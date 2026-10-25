#include "acr_stage3_dispatch.hpp"

#include <limits>
#include <new>
#include <stdexcept>
#include <vector>

namespace imprint::acr {
namespace {

bool checked_multiply(std::size_t left, std::size_t right,
                      std::size_t *product) noexcept {
    if (product == nullptr) return false;
    if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
        return false;
    }
    *product = left * right;
    return true;
}

bool checked_add(std::size_t left, std::size_t right,
                 std::size_t *sum) noexcept {
    if (sum == nullptr) return false;
    if (left > std::numeric_limits<std::size_t>::max() - right) return false;
    *sum = left + right;
    return true;
}

// FUN_00ab21f0 only leaves mosaic+0x80/+0x84 at zero for the identity RGGB
// arrangement. The other three standard Bayer phases store a nonzero packed
// offset and then enter the unrecovered FUN_00ab4094 transform. FUN_007f851c
// itself is also a fixed-position RGGB body (top-left -> plane 0,
// top-right+bottom-left -> plane 1, bottom-right -> plane 2), so accepting a
// generic three-plane pattern would silently run the wrong kernel.
bool pattern_is_canonical_rggb(const std::uint8_t pattern[4]) noexcept {
    return pattern[0] == 0 && pattern[1] == 1 && pattern[2] == 1 &&
           pattern[3] == 2;
}

bool plane_map_is_aligned(const Stage3Prepared &prepared) noexcept {
    return prepared.plane_count == 3 && prepared.plane_map[0] == 0 &&
           prepared.plane_map[1] == 1 && prepared.plane_map[2] == 2;
}

Stage3Dispatch fail(Stage3Dispatch dispatch, Stage3Status status,
                    const char *reason) noexcept {
    dispatch.status = status;
    dispatch.reason = reason;
    return dispatch;
}

bool rect_is_valid(const Stage3Prepared &prepared) noexcept {
    if (prepared.rect_top < 0 || prepared.rect_left < 0) return false;
    if (prepared.rect_bottom <= prepared.rect_top ||
        prepared.rect_right <= prepared.rect_left) {
        return false;
    }
    const std::int64_t width = static_cast<std::int64_t>(prepared.width);
    const std::int64_t height = static_cast<std::int64_t>(prepared.height);
    return static_cast<std::int64_t>(prepared.rect_right) <= width &&
           static_cast<std::int64_t>(prepared.rect_bottom) <= height;
}

Stage3Dispatch base_dispatch(const Stage3Prepared &prepared) noexcept {
    Stage3Dispatch dispatch;
    dispatch.branch = Stage3Branch::generic_012a4a60;
    dispatch.kernel = Stage3Kernel::none;
    dispatch.status = Stage3Status::invalid_argument;
    dispatch.reason = "uninitialized";
    (void)prepared;
    return dispatch;
}

}  // namespace

Stage3Dispatch dispatch_stage3(const Stage3Prepared &prepared) noexcept {
    Stage3Dispatch dispatch = base_dispatch(prepared);

    if (prepared.width == 0 || prepared.height == 0) {
        return fail(dispatch, Stage3Status::invalid_dimensions,
                    "zero Stage2 CFA dimension");
    }
    if (prepared.input_stride < prepared.width) {
        return fail(dispatch, Stage3Status::invalid_stride,
                    "input stride is shorter than the image width");
    }
    if (prepared.plane_count != 3 ||
        !pattern_is_canonical_rggb(prepared.cfa_pattern)) {
        return fail(dispatch, Stage3Status::invalid_cfa,
                    "only the canonical RGGB pattern {0,1,1,2} is closed");
    }
    if (!rect_is_valid(prepared)) {
        return fail(dispatch, Stage3Status::invalid_rect,
                    "rect is empty, inverted or outside the Stage2 image");
    }
    if (prepared.mode != 1) {
        return fail(dispatch, Stage3Status::unsupported_mode,
                    "only mode 1 (standard Bayer) is closed");
    }
    if (prepared.mosaic_7c != 0) {
        return fail(dispatch, Stage3Status::unsupported_offset,
                    "mosaic+0x7c transform/pair reversal is not closed");
    }
    if (prepared.offset_80 != 0 || prepared.offset_84 != 0) {
        return fail(dispatch, Stage3Status::unsupported_offset,
                    "mosaic+0x80/+0x84 scaling path is not closed");
    }
    if (!plane_map_is_aligned(prepared)) {
        dispatch.branch = Stage3Branch::task_0127cfe0;
        return fail(dispatch, Stage3Status::unsupported_plane_map,
                    "non-[0,1,2] plane map selects the task constructor");
    }

    const bool pair_equal = prepared.pair_x == prepared.pair_y;
    const bool pair_2_4 = prepared.pair_x == 2 && prepared.pair_y == 4;
    if (!pair_equal && !pair_2_4) {
        dispatch.branch = Stage3Branch::generic_012a4a60;
        return fail(dispatch, Stage3Status::unsupported_pair,
                    "mode 1 pair is neither equal nor (2,4)");
    }

    dispatch.branch = Stage3Branch::direct_00ab3f24;
    const std::int64_t rect_width =
        static_cast<std::int64_t>(prepared.rect_right) - prepared.rect_left;
    const std::int64_t rect_height =
        static_cast<std::int64_t>(prepared.rect_bottom) - prepared.rect_top;
    if (rect_width <= 0 || rect_height <= 0) {
        return fail(dispatch, Stage3Status::invalid_rect,
                    "rect has no positive area");
    }

    if (prepared.pair_x == 1 && prepared.pair_y == 1) {
        dispatch.kernel = Stage3Kernel::bayer_1x1_007f2124;
        dispatch.output_width = static_cast<std::size_t>(rect_width);
        dispatch.output_height = static_cast<std::size_t>(rect_height);
        if (!checked_multiply(dispatch.output_width, dispatch.output_height,
                              &dispatch.output_samples) ||
            !checked_multiply(dispatch.output_samples, 3,
                              &dispatch.output_samples)) {
            return fail(dispatch, Stage3Status::invalid_dimensions,
                        "output sample count overflows");
        }
        return fail(dispatch, Stage3Status::unsupported_full_res_core3,
                    "pair (1,1) selects 007f2124; its 007eaf20 CPU executor "
                    "and kernel table are located but not ported");
    }

    if ((rect_width & 1) != 0 || (rect_height & 1) != 0) {
        return fail(dispatch, Stage3Status::unsupported_kernel,
                    "2x2 preview path requires an even rect");
    }
    if (prepared.pair_x == 2 && prepared.pair_y == 2) {
        dispatch.kernel = Stage3Kernel::bayer_2x2_007f851c;
        dispatch.output_width = static_cast<std::size_t>(rect_width / 2);
        dispatch.output_height = static_cast<std::size_t>(rect_height / 2);
        dispatch.half_pixel_shift = prepared.task_transform;
        if (!checked_multiply(dispatch.output_width, dispatch.output_height,
                              &dispatch.output_samples) ||
            !checked_multiply(dispatch.output_samples, 3,
                              &dispatch.output_samples)) {
            return fail(dispatch, Stage3Status::invalid_dimensions,
                        "output sample count overflows");
        }
        if (prepared.task_transform) {
            return fail(dispatch, Stage3Status::unsupported_task_transform,
                        "2x2 preview task half-pixel transform is not closed");
        }
        if ((prepared.rect_left & 1) != 0 || (prepared.rect_top & 1) != 0) {
            return fail(dispatch, Stage3Status::unsupported_kernel,
                        "2x2 preview path requires an even rect origin");
        }
        dispatch.status = Stage3Status::ok;
        dispatch.reason = "direct aligned mode-1 pair (2,2) path";
        return dispatch;
    }

    if (prepared.pair_x == 2 && prepared.pair_y == 4) {
        dispatch.kernel = Stage3Kernel::bayer_2x4_007f9228;
        return fail(dispatch, Stage3Status::unsupported_kernel,
                    "pair (2,4) selects the unrecovered 007f9228 kernel");
    }

    dispatch.kernel = Stage3Kernel::generic_007fb724;
    return fail(dispatch, Stage3Status::unsupported_kernel,
                "other equal mode-1 pairs select the unrecovered generic kernel");
}

Stage3Status run_stage3_bayer_2x2(
    const std::uint16_t *input,
    std::size_t input_sample_count,
    const Stage3Prepared &prepared,
    std::vector<std::uint16_t> *output_rgb) noexcept {
    if (input == nullptr || output_rgb == nullptr) {
        return Stage3Status::invalid_argument;
    }
    const Stage3Dispatch dispatch = dispatch_stage3(prepared);
    if (dispatch.status != Stage3Status::ok ||
        dispatch.kernel != Stage3Kernel::bayer_2x2_007f851c ||
        dispatch.half_pixel_shift) {
        return dispatch.status == Stage3Status::ok
                   ? Stage3Status::unsupported_kernel
                   : dispatch.status;
    }

    const std::size_t rect_width =
        static_cast<std::size_t>(prepared.rect_right - prepared.rect_left);
    const std::size_t rect_height =
        static_cast<std::size_t>(prepared.rect_bottom - prepared.rect_top);
    if ((rect_width & 1) != 0 || (rect_height & 1) != 0) {
        return Stage3Status::unsupported_kernel;
    }

    // Highest sample FUN_007f851c reads: bottom row, last column of the rect.
    // The 2x2 body reads (row, col), (row, col+1), (row+1, col) and
    // (row+1, col+1) for even row/col, so the last read is
    // (rect_bottom - 1, rect_right - 1).
    std::size_t last_row = 0;
    if (!checked_multiply(
            static_cast<std::size_t>(prepared.rect_bottom - 1),
            prepared.input_stride, &last_row)) {
        return Stage3Status::invalid_dimensions;
    }
    std::size_t last_sample = 0;
    if (!checked_add(last_row,
                     static_cast<std::size_t>(prepared.rect_right), &last_sample)) {
        return Stage3Status::invalid_dimensions;
    }
    if (last_sample > input_sample_count) {
        return Stage3Status::buffer_too_small;
    }

    const std::size_t out_width = rect_width / 2;
    const std::size_t out_height = rect_height / 2;
    std::size_t out_samples = 0;
    if (!checked_multiply(out_width, out_height, &out_samples) ||
        !checked_multiply(out_samples, 3, &out_samples)) {
        return Stage3Status::invalid_dimensions;
    }

    std::vector<std::uint16_t> candidate;
    try {
        candidate.resize(out_samples);
    } catch (const std::bad_alloc &) {
        return Stage3Status::allocation_failure;
    } catch (const std::length_error &) {
        return Stage3Status::invalid_dimensions;
    }

    const std::uint16_t *top =
        input + static_cast<std::size_t>(prepared.rect_top) *
                    prepared.input_stride +
        static_cast<std::size_t>(prepared.rect_left);
    const std::size_t row_step = prepared.input_stride;
    std::size_t out = 0;
    for (std::size_t y = 0; y < out_height; ++y) {
        const std::uint16_t *top_row = top + (y * 2) * row_step;
        const std::uint16_t *bottom_row = top_row + row_step;
        for (std::size_t x = 0; x < out_width; ++x) {
            const std::uint32_t red = top_row[x * 2];
            const std::uint32_t green_left = bottom_row[x * 2];
            const std::uint32_t green_right = top_row[x * 2 + 1];
            const std::uint32_t blue = bottom_row[x * 2 + 1];
            // FUN_007f851c scalar tail: (a + b + 1) >> 1 computed in 32-bit
            // lanes, then stored as uint16. The NEON path uses urhadd with the
            // same rounding. Both green samples are <= 65535, so the sum fits.
            candidate[out + 0] = static_cast<std::uint16_t>(red);
            candidate[out + 1] = static_cast<std::uint16_t>(
                (green_right + green_left + 1u) >> 1);
            candidate[out + 2] = static_cast<std::uint16_t>(blue);
            out += 3;
        }
    }

    output_rgb->swap(candidate);
    return Stage3Status::ok;
}

}  // namespace imprint::acr
