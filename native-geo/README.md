# Imprint Native Geographic Overlay Core

This is a standalone C++17 vector overlay core for Vue/Tauri integration. It accepts an already-decoded RGB8 sRGB image buffer, produces a same-size straight-alpha RGBA8 sRGB layer, and never reads or writes image files. It has no Python, ACR, RAW decoder, network, or DEM dependency. `src/geo.cpp` is intentionally self-contained for direct `cargo cc` compilation.

## Build and native checks

```sh
cmake -S native-geo -B /tmp/imprint-geo-build
cmake --build /tmp/imprint-geo-build
ctest --test-dir /tmp/imprint-geo-build --output-on-failure
```

The library is static (`imprint_geo`). Public headers are `include/imprint_geo.h` and `include/imprint_geo.hpp`.

## C ABI

`GeoConfig` field order and types are fixed in `imprint_geo.h`. `imprint_geo_render` accepts packed RGB pixels (`width * height * 3`) and writes packed RGBA (`width * height * 4`). The output is straight alpha in sRGB code values. A successful disabled render writes an all-zero layer. Invalid configuration, missing GPS, unsupported style/provider, and allocation failure return nonzero and clear the layer where a valid output buffer was supplied. No C++ exception crosses the ABI.

`imprint_geo_composite` performs straight-alpha source-over on sRGB channel values. An overlay pixel with alpha zero copies all three source bytes exactly. Input/output aliasing is supported when `out_rgb == rgb`.

Configuration semantics:

- `has_location` distinguishes missing metadata from valid coordinates `(0,0)`. Latitude and longitude must be finite and in range.
- `opacity` is a single global layer opacity. It is applied once after vector coverage and sky masking, so overlapping strokes cannot exceed the configured layer alpha.
- `radius_km` is the geographic radius represented by the map extent. A local azimuthal-equidistant projection converts geographic coordinates to true metres; east and north use one pixel scale.
- `offset_x` and `offset_y` are normalized to `[-1,1]`; without adaptive sky placement they shift the map center by up to one quarter of the image dimension. With adaptive sky placement they shift within remaining detected-sky space.
- `scale` multiplies the rendered map extent. `rotation` is degrees counterclockwise; the north arrow follows it.
- `line_width` is scaled from a 1000-pixel short edge, so preview and full-size output retain the same normalized appearance.
- `feather` is a fraction of the shorter image edge, in `[0,0.3]`. The default sky mask uses an inward two-dimensional distance transform, including side boundaries around trees and buildings.
- `manual_horizon` is a normalized image y-coordinate measured from the top. Zero means unspecified. A nonzero value is a hard lower protection boundary; feathering fades toward it.
- `coverage` is the preferred fraction of the shorter image edge (default UI preference: `0.8`). With `adaptive` and `auto_sky_fade`, the map extent fits the detected sky bounds and grows with the available sky instead of applying a fixed area cutoff.
- `color_rgb == 0` selects a day/night palette from image brightness. A nonzero `0xRRGGBB` controls all vector paths and primary text.
- `style`: `0=Auto`, `1=Urban`, `2=Coastal`, `3=Terrain`. Auto selects Coastal when the provider has coastline features and Urban otherwise. Terrain returns an explicit unsupported error because this core has no verified DEM.
- `provider`: `0=Mock`, `1=Overpass`. Mock data is synthetic and visibly labeled `MOCK` / `EXAMPLE ONLY`; it is not a real map. Overpass is an explicit unsupported stub and never sends a request.

The default `TopConnectedRgbSkyMaskProvider` is exposed through `ISkyMaskProvider` and can be replaced through `render_with_sky_provider` or `render_with_providers`. The latter also accepts any `IGeoDataProvider`. The default sky mask uses top-connected RGB similarity, blue/gray-blue color evidence, local edge barriers, and a two-dimensional inward distance transform. It accepts low-luma blue-gray skies when they match the top reference, and rejects uncertain neutral gray regions rather than treating arbitrary gray surfaces as sky. If no sky is found, auto-faded rendering returns a transparent layer. Turning `auto_sky_fade` off uses the whole canvas unless `manual_horizon` is supplied.

The default sky provider keeps one thread-local exact-input cache for images up to 16 megapixels. The key includes dimensions and a content hash over every RGB byte; a matching hash is also confirmed against the complete RGB bytes before reuse. Larger images skip caching to bound retained memory. The mock provider also keeps one thread-local generated-dataset entry keyed by exact latitude, longitude, and radius, so opacity, color, labels, and style changes reuse the same mock features.

## Scope limits

This core does not parse JPEG, TIFF, PNG, or DNG files; callers must decode a finished image and pass RGB8. It does not generate terrain or claim that Mock roads, waterways, and coasts are geographically real. The mask is a conservative color/edge heuristic, not semantic segmentation. A scene without confidently detected sky may require an explicit manual horizon or disabled sky fade.
