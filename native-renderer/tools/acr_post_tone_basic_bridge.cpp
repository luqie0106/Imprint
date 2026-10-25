#include "imprint_renderer.h"

#include <cstddef>
#include <cstdint>

// Mirror the public declaration so the test bridge is usable in standalone
// validation builds as well as in the integrated native renderer.
extern "C" IMPRINT_API im_status im_native_acr_normalize_post_tone_basic_i32(
    std::uint32_t process_version, const std::int32_t* input,
    std::size_t count, const std::int32_t* source_defaults,
    std::size_t source_count, std::int32_t* output,
    std::size_t output_count);

extern "C" int acr_test_normalize_post_tone_basic_i32(
    std::uint32_t process_version, const std::int32_t* input,
    std::size_t count, const std::int32_t* source_defaults,
    std::size_t source_count, std::int32_t* output,
    std::size_t output_count) {
    return static_cast<int>(im_native_acr_normalize_post_tone_basic_i32(
        process_version, input, count, source_defaults, source_count,
        output, output_count));
}
