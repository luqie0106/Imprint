#pragma once

#include "acr_resample_plan.hpp"
#include "acr_resample_separable.hpp"

#include <vector>

namespace imprint::acr {

// Private C++ composition of a prepared ordinary logical source request and
// the float32 ordinary pixel executor. This materializes only the requested
// subdomain with nearest-border copies, as supported by the ordinary
// cr_image/type-11/four-byte mode-2 evidence. It is not a normalized request
// parser, Adobe ABI, full-frame runtime equivalence, raw-object adapter, or
// claim about a completely disjoint Adobe request. The returned logical R0
// must have positive-area intersection with the SourceView rectangle; a fully
// disjoint request is explicitly unsupported. The declared SourceView count
// may exceed its active packed HWC element count, but its complete declared
// byte span must fit PTRDIFF_MAX and must not wrap the address space. Only the
// materialized R0 is capped at kResampleSeparableMaximumElements. Unused source
// pixels (including NaNs) are not scanned; materialized and intermediate
// values remain subject to the existing executor's finite-value checks.
enum class OrdinarySourceMode2Status : int {
    ok = 0,
    invalid_source = 1,
    request_failure = 2,
    unsupported_disjoint_request = 3,
    resource_limit = 4,
    allocation_failure = 5,
    execution_failure = 6,
};

// Prepare the logical source request, materialize that request from a packed
// HWC float source using exact nearest-border bit copies, and run the existing
// ordinary resampler. Only one- and three-channel sources are supported.
// Source coordinates and requested rectangles use non-wrapping int32
// half-open extents. The original source may alias result's current storage;
// it is read to completion before a successful result swap. Every failure
// leaves result, including its capacity and storage, unchanged.
OrdinarySourceMode2Status process_ordinary_source_mode2_f32(
    const SourceView &source, const DestinationRect &destination,
    const OrdinaryResamplePlan &plan,
    std::vector<float> &result) noexcept;

}  // namespace imprint::acr
