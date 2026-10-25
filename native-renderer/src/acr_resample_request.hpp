#pragma once

#include "acr_resample_plan.hpp"
#include "acr_resample_separable.hpp"

namespace imprint::acr {

// Logical source-rectangle handoff for an already prepared ordinary plan.
// `intermediate` is the rectangle requested before the horizontal stage;
// `source` is the rectangle requested before the vertical stage. These are
// logical coordinates only and do not prove that halo pixels are materialized.
struct OrdinarySourceRequest {
    DestinationRect source;
    DestinationRect intermediate;
};

enum class OrdinarySourceRequestStatus : int {
    ok = 0,
    invalid_rectangle = 1,
    invalid_axis = 2,
    coordinate_overflow = 3,
};

// Prepare the logical source rectangle for the ordinary vertical-then-
// horizontal pixel pipeline by walking its rectangle in reverse: horizontal
// 0x47 first, then vertical 0x48. This follows the saved +0x28 rectangle
// methods 0x00d342d4 (0x47) and 0x00d358f4 (0x48). It does not materialize
// border/halo storage or claim Adobe runtime dispatch. Input and returned
// rectangles require positive height/width and non-wrapping int32 exclusive
// endpoints and spans no larger than INT32_MAX. A resample axis requires a
// positive step, lower <= upper, and only its coefficient metadata tap_count
// to be positive, even, and <= INT32_MAX; identity and integer-translation
// axes do not inspect coefficient metadata. Resample coordinate multiply/add
// wraps modulo 2^64, clamps as signed 64-bit, extracts signed high words, and
// adds tap_count to the inclusive high word with uint32 wrap before validating
// the returned rectangle span. Integer translation subtracts its stored
// offset in checked int64 coordinates and rejects endpoints outside int32;
// this is the local logical-coordinate contract, not Adobe 0x2f copy-path or
// border behavior. Success returns logical rectangles only: pixel data and
// halo materialization are not performed here. Failure leaves output
// byte-for-byte unchanged.
OrdinarySourceRequestStatus prepare_ordinary_source_request(
    const OrdinaryResamplePlan &plan, const DestinationRect &destination,
    OrdinarySourceRequest &output) noexcept;

}  // namespace imprint::acr
