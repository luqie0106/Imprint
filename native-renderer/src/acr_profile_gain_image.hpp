#pragma once

#include <cstddef>
#include <cstdint>

namespace imprint::acr {

// Status values for the local prepared ProfileGain image composition.
// This is a caller-parameterized numerical subdomain: map selection,
// profile/source choice, and upstream geometry remain caller supplied. It is
// not an Adobe runtime or complete normalized-request route.
enum class ProfileGainImageStatus : int {
    ok = 0,
    invalid_argument = 1,
    invalid_dimensions = 2,
    invalid_stride = 3,
    buffer_too_small = 4,
    unsupported_alignment = 5,
    overlap = 6,
    arithmetic_overflow = 7,
    resource_limit = 8,
    invalid_prepared = 9,
    non_finite_input = 10,
    non_finite_intermediate = 11,
    allocation_failure = 12,
    unsupported_transform = 13,
    invalid_map = 14,
};

// The four rectangles are [top,left,bottom,right): map_rect, reference_zero,
// state_rect, and tile_rect. rectCount must be exactly 16 and matrixCount
// exactly 9 (a short span returns buffer_too_small; an oversized one returns
// invalid_argument). matrix50 is a row-major 3x3 matrix without any Adobe
// object header. Source and destination are interleaved HWC RGB f32; strides
// and counts are in float elements. Tile height/width are checked in int64
// and must be positive and <= INT32_MAX. Each source/destination stride must
// be >= 3*tile_width and <= INT32_MAX; each declared count must cover at least
// (tile_height-1)*stride + 3*tile_width active elements. Full declared spans
// are checked for size_t, PTRDIFF_MAX, uintptr_t overflow, and natural
// alignment. Destination may not overlap payload, rects, or matrix50. Source
// and destination may overlap only with identical pointer, count, and stride;
// read-only input spans may overlap one another.
//
// The call writes only active destination pixels after all stages succeed;
// destination padding is preserved, and every failure leaves the full
// destination byte-for-byte unchanged. Exact source/destination in-place use
// is supported. has_task=1 returns unsupported_transform after basic pointer,
// count, tile-dimension, and stride validation, before parsing/allocating the
// map or checking matrix/geometry preparation. Other calls parse the source
// payload, prepare the selected source or cache map, and compose the local
// geometry/evaluator/apply operators using their existing numerical formulas.
// They enforce 2*N <= 2^26 coordinate samples,
// 3*N <= 2^26 apply values, and N + table_count <= 2^26 evaluator values.
// The evaluator rejects non-finite RGB inputs and generated coordinates. The
// final scalar apply uses pivot 0 with clipping disabled and IEEE arithmetic;
// finite evaluator inputs do not guarantee finite final RGB if apply
// arithmetic overflows.
// Invalid serialized map payload/header/length/value/datatype returns
// invalid_map (14); map numeric failure returns non_finite_intermediate (11).
// Invalid map geometry and singular/non-finite input matrix return
// invalid_prepared (9); a non-finite coordinate transform result returns
// non_finite_intermediate (11). Arithmetic overflow, resource limit, and
// allocation failure return 7, 8, and 12 respectively.
//
// At amount 100 with predicate bit 0 clear, the parsed source table is
// prepared directly; lookup_flag does not affect this route. Other positive
// amounts use the local generated cache table, whose amount is capped at 200.
// This is a caller-parameterized numerical subdomain: map selection,
// profile/source choice, and upstream geometry remain caller supplied. It is
// not an Adobe runtime or complete normalized-request route.
ProfileGainImageStatus process_profile_gain_image_rgbf32(
    const float *src, std::size_t srcCount, std::size_t srcStride,
    float *dst, std::size_t dstCount, std::size_t dstStride,
    const std::uint8_t *payload, std::size_t bytes, std::uint32_t tag,
    std::uint32_t byte_order, const std::int32_t *rects,
    std::size_t rectCount, const double *matrix50,
    std::size_t matrixCount, double scalar, std::int32_t amount,
    std::uint32_t lookup_flag, std::uint32_t predicate,
    std::uint32_t has_task) noexcept;

}  // namespace imprint::acr
