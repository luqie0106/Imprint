# Native GPU renderer prototype

This directory contains an independent C++17 shared library with a narrow C ABI. It is not connected to the Python Sidecar, Vue, Rust, or the current package build. The C ABI is declared in [`include/imprint_renderer.h`](include/imprint_renderer.h).

## Pipeline

The Metal backend (macOS) and optional CUDA backend (Linux/Windows) apply the same stages in one GPU dispatch:

1. Dehaze ported from the Python `natural-global-v8-source-hue-brightness-guard` algorithm using the nine existing UI parameters.
2. Basic exposure, tone, vibrance, and saturation adjustments using the eight existing UI parameters.
3. Runtime filter controls, per-channel curves, then an optional trilinear 3D LUT.

The renderer caches an uploaded RGB16 preview, its L0/L1/L2 pyramid, per-level dehaze statistics, and filter resources. Each level estimates one RGB atmospheric-light vector from the highest 0.1% of dark-channel values (at least 16 candidates), then takes the median of the brightest one eighth of those candidates. Its global haze level comes from that level's dark-channel 75th percentile. `im_renderer_render_full` calculates both statistics again from the newly decoded full-resolution input and bypasses the preview cache. Input buffers are copied and never modified.

Metal and CUDA port the `natural-global-v8-source-hue-brightness-guard` transform in `src/dehaze.py`: one global transmission, natural blend and source-luminance mix, source-hue confidence and chroma recovery, smooth gamut handling, the fixed-endpoint global S-curve, highlight/shadow protection, the global brightness guard, near-saturation caps, and round-to-even RGB16 output. The atmospheric-light candidate ordering can differ for exact ties, and shader floating-point arithmetic can cause small quantization differences.

The brightness guard needs the median of the pre-guard image on source-selected midtones. Before each protected render, the host reproduces the pre-guard pixel transform over the selected cached level (or fresh full input), calculates the two medians, and sends one compensation gain to the shader. This costs one full CPU image pass and median-selection storage for up to two luma arrays on every such render, in addition to the one-time per-level statistics scans at preview upload. It preserves the CPU reference's global behavior without reading a GPU intermediate image, but its latency and memory cost have not been benchmarked and may affect 30 fps preview rendering. A future optimization can compute GPU partial median histograms or reductions and compare their gain against this CPU reference before removing the host scan.

Curve data is three 256-entry RGB16 planes ordered R, G, B. LUT data is an RGB16 cube with edge 2 through 65, ordered `((r * edge + g) * edge + b) * 3 + channel`. Upload updated curves/LUTs with `im_renderer_upload_filter`; the next render uses them.

## Build on macOS

```sh
cmake -S native-renderer -B /tmp/imprint-native-renderer-build -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/imprint-native-renderer-build -j
ctest --test-dir /tmp/imprint-native-renderer-build --output-on-failure
```

The Metal shader text is embedded into the shared library at build time and compiled by Metal at runtime through `newLibraryWithSource`, so installed builds do not depend on the source tree and development builds do not require the standalone `metal` command. A Metal-capable GPU and macOS Metal runtime are required to execute the render tests.

Metal device visibility depends on the process environment. The restricted Codex command environment can return no device, while the ordinary macOS Terminal on the Apple M2 Pro development host exposes Metal. In Terminal, the GPU contract test passed and the default L2 benchmark measured 0.364 ms/render (356 MPix/s) for a 1920x1080 preview rendered at 480x270. This is a native-renderer microbenchmark, not an end-to-end application frame rate.

## Optional CUDA build

CUDA is enabled by default when CMake finds a CUDA compiler. Set `-DIMPRINT_ENABLE_CUDA=OFF` to disable it. CUDA remains optional so a machine without `nvcc` can still configure the shared ABI. This prototype's CUDA source has not been compiled or run on the current macOS development host; use a CUDA-enabled Linux or Windows build to validate that backend.

## Contract tests and local benchmark

`imprint_renderer_tests` covers input validation, RGB16 identity at zero dehaze strength, nonzero dehaze output and range, L0/L1/L2 dimensions, latest-parameter rerendering, the separate full-resolution input path, input immutability, and curve/LUT effects. CTest always runs a null-handle ABI validation check; it marks the GPU contract test skipped when the host has no compiled/available GPU backend.

`tests/test_dehaze_reference.py` compares the no-GPU C++ host reference and Python `apply_dehaze(..., backend="cpu")` on synthetic RGB16 haze, low-saturation sky/sea, a dark foreground with bright sky, near-saturated sun/halo, zero strength, disabled color recovery, local contrast, and brightness protection. It reports MAE, P99, and maximum absolute RGB16 code error. Build the test bridge with CMake, then run the Python comparison with `IMPRINT_NATIVE_REFERENCE_LIB=/tmp/imprint-native-renderer-build/libimprint_dehaze_reference_test.dylib` set to the built library path. This CPU comparison does not execute or validate either GPU shader.

`tests/test_metal_dehaze.py` directly compares the Metal output with Python CPU output on the same nine synthetic RGB16 image/parameter combinations, with basic adjustments and filters neutral. It reports overall and per-channel MAE, P99, maximum error, exact-pixel percentage, and channel bias. Run it in a process that can access Metal:

```sh
IMPRINT_NATIVE_RENDERER_LIB=/tmp/imprint-native-renderer-build/libimprint_renderer.dylib \
  python -m pytest -q -s native-renderer/tests/test_metal_dehaze.py
```

The current M2 Pro run passed all nine cases with maximum channel error of one RGB16 code value. These synthetic tests validate the dehaze stage; they do not compare real-photo color management or the complete basic/filter/LUT pipeline against Python.

Run the repeatable L2 microbenchmark with:

```sh
/tmp/imprint-native-renderer-build/imprint_renderer_benchmark
```

By default it renders a deterministic 1920x1080 preview at L2 (480x270), performs 10 warmups and 100 measured renders, then reports backend, dimensions, elapsed time, milliseconds per render, and megapixels per second. Optional arguments are `preview_width preview_height iterations`. Timing includes this local renderer's output transfer/readback path; it excludes image decode, Sidecar/API work, display presentation, and the rest of the app, so it is not an end-to-end frame-rate result.
