/*
 * Imprint production TicoRAW decoder bridge.
 *
 * The entropy and lifting implementation is derived from the Nikon HE
 * decoding behavior in LibRaw's redlamp-libraw project (LGPL-2.1 or CDDL-1.0).
 * Copyright notices and license terms are retained in the implementation.
 */
#ifndef IMPRINT_TICORAW_DECODE_HPP
#define IMPRINT_TICORAW_DECODE_HPP

#include <cstddef>
#include <cstdint>

namespace nikon_he {

struct HeDecodeResult {
    bool success;
    int tiles_decoded;
    int total_precincts;
};

// ABI-compatible replacement for LibRaw's image-level TicoRAW decoder.
// The LibRaw Nikon loader validates the complete picture header and installs
// its active WGT header before calling this function. Failures are reported by
// success=false; no C++ exception crosses the LibRaw boundary.
HeDecodeResult decode_nikon_he_image(const uint8_t* precinct_data,
                                     size_t precinct_stream_size,
                                     int image_width,
                                     int image_height,
                                     const int32_t* lut,
                                     uint16_t* out_bayer) noexcept;

}  // namespace nikon_he

// Thread-local diagnostic for the backend used by the most recent successful
// decode on this thread: "cpu", "metal", or "none" if decoding did not finish.
extern "C" const char* imprint_tico_last_backend() noexcept;

#endif  // IMPRINT_TICORAW_DECODE_HPP
