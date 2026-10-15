#pragma once

#include <array>
#include <cstddef>

namespace imprint::acr {

constexpr std::size_t kProfileToneLutSegments = 4096;
constexpr std::size_t kProfileToneLutEntries = kProfileToneLutSegments + 2;

using ProfileToneLut = std::array<float, kProfileToneLutEntries>;
using ProfileToneInputTransform = double (*)(double value, const void *context);

// Build the generic natural cubic spline from interleaved (x, y) doubles and
// sample it at i/4096. An optional native scalar transform composes the
// recovered pre-profile branch before evaluating the spline, without adding
// a second interpolated LUT. The spline uses constant endpoint extrapolation.
// clip_curve_output selects the profile-only main-curve branch behavior, which
// clips each sampled curve result to [0, 1] before float32 conversion. At most
// 65536 points are accepted, matching the formal ABI input cap.
ProfileToneLut build_profile_tone_lut(const double *points_xy,
                                      std::size_t point_count,
                                      bool clip_curve_output = true,
                                      ProfileToneInputTransform input_transform = nullptr,
                                      const void *transform_context = nullptr);

// Apply the explicit profile-only RGB subset to interleaved float32 [0, 1]
// pixels. modern=true selects the ordinary MMHG coupling; false selects the
// legacy min/max LUT path. Source, destination, and profile points must not
// overlap where a write could modify an input. All arguments are validated
// before destination is changed. This is not the full ACR main/source/PV path.
void apply_profile_tone(const float *source, float *destination,
                        std::size_t pixel_count, const double *points_xy,
                        std::size_t point_count, bool modern = true,
                        ProfileToneInputTransform input_transform = nullptr,
                        const void *transform_context = nullptr);

}  // namespace imprint::acr
