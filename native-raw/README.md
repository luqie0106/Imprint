# Native TicoRAW decoding

This library provides the Nikon HE/HE* path used by Imprint's preview,
camera-space rendering and enhanced DNG export. It never reads Python research
arrays or uses Python entropy / inverse-wavelet implementations.

The production stages are:

1. LibRaw reads the NEF container, Nikon MakerNote, WGT and transfer curve.
2. Imprint C++ parses precincts, decodes GCLI/magnitudes/signs, dequantizes,
   performs horizontal and vertical lifting, and carries overlapping tiles.
3. Metal executes the two integer full-frame Bayer reconstruction operators.
   The equivalent C++ operators handle CPU-only execution.
4. LibRaw performs camera-space or linear sRGB demosaicing in C++.
5. `src/raw_decoder.py` only manages handles and copies native buffers.
   Existing Imprint processing and DNG writing receive these buffers.

The public LibRaw TicoRAW entropy, IDWT, tile and Bayer source files are excluded
from the native library build. Format/layout helpers and the NEF metadata/color
rendering implementation use the pinned LGPL-2.1 source. This is source-assisted
engineering, not a claim of complete independent Adobe reverse engineering.

## Build

```bash
cmake -S native-raw -B native-raw/build -DCMAKE_BUILD_TYPE=Release
cmake --build native-raw/build --config Release --target imprint_raw --parallel
```

CMake downloads the fixed `redlamp-libraw` archive with a SHA-256 check; no
system Python or rawpy installation is replaced. The shader is embedded in the
library. A preverified source tree can optionally be provided using
`-DIMPRINT_LIBRAW_SOURCE_DIR=/absolute/source/path` for an offline build.

The developer backend locates `native-raw/build/libimprint_raw.dylib` on macOS,
the corresponding `.so` on Linux, or `imprint_raw.dll` in the Windows build or
Release directory. PyInstaller bundles it beside the sidecar and refuses to
package without it. Restart a running project backend after changing Python
code or rebuilding this library. An already installed release needs repackaging
to include these changes.

## Backend selection and boundaries

- `IMPRINT_TICORAW_BACKEND=auto` (default): Metal on macOS when available,
  otherwise C++; no Python decode fallback.
- `IMPRINT_TICORAW_BACKEND=cpu`: require the C++ reconstruction path.
- `IMPRINT_TICORAW_BACKEND=metal`: require actual Metal; failure is an error.
- `IMPRINT_NATIVE_RAW_LIB`: optional explicit shared-library path.

Ordinary RAWs remain on rawpy. A native metadata-only probe selects Nikon HE
using **NEFCompression 13/14**, not TIFF Compression 34713. Metadata and embedded
thumbnail queries keep their existing fast paths. CFA and RGB results own their
memory after the native handle closes.

The decoder rejects truncated substreams, bad slice order, missing EOC,
unsupported coding modes and unsafe dimensions. Unknown profiles are not
silently approximated. This implementation has real full-image coverage for
the local Z6 III HE* sample; other bodies/profiles still need real fixtures.

## Validation

On Apple M2 Pro, forced CPU and forced Metal decoded all 24,498,560 photosites
of `LCR_3520.NEF` exactly like the fixed reference. CFA SHA-256:
`5c76848dadbd2682b1abd444d32551c0000fe6d9a56c0e638582248cbac59ed2`.
Their linear half-size RGB renders were identical. The actual enhanced-DNG job
completed and its CFA decoded back to the same samples.

See `research/acr/reports/2026-10-12-ticoraw-phase6-imprint-native.md` for the
recorded checks and limitations. Windows execution and installed-package
runtime have not been validated on this Mac. Adobe visual acceptance remains
the user's separate import/export check.

The library and adapted reconstruction files are provided under LGPL-2.1;
source attribution and the full upstream notices are retained in
`THIRD_PARTY_NOTICES.md` and `third_party/licenses`.
