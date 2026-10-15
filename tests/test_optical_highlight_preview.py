"""Original/effect previews must use the same highlight-safe optical baseline."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import app_api as api
import dng_gainmap
from lens_correction import LensCorrectionResult


@pytest.mark.parametrize("full_resolution", [False, True])
def test_original_matches_zero_strength_effect_after_optical_correction(
    monkeypatch, tmp_path, full_resolution,
):
    path = tmp_path / "scene.dng"
    path.touch()
    session = "optical-highlight-baseline"
    levels = np.linspace(1000, 65535, 400, dtype=np.uint16)
    source = np.broadcast_to(levels[None, :, None], (300, 400, 3)).copy()
    metadata = SimpleNamespace(source_kind="raw", color_space="Linear sRGB", width=400, height=300)
    gain_map = dng_gainmap.DNGGainMap(
        top=0, left=0, bottom=300, right=400, plane=0, planes=3,
        row_pitch=1, col_pitch=1, map_points_v=2, map_points_h=2,
        map_spacing_v=1., map_spacing_h=1., map_origin_v=0., map_origin_h=0.,
        map_planes=3, gains=np.full((2, 2, 3), 2., np.float32),
        active_area=(0, 0, 300, 400), image_width=400, image_height=300, orientation=1,
    )
    monkeypatch.setattr(dng_gainmap, "parse_dng_gain_map", lambda _path: gain_map)
    monkeypatch.setattr(api, "read_image", lambda *_a, **_k: (source.copy(), metadata))
    monkeypatch.setattr(api, "apply_lens_correction", lambda image, *_a, **_k: (
        image.copy(), LensCorrectionResult(False, None, None, False, False, False),
    ))
    api._ENHANCE_SESSIONS[session] = {"files": {"photo": path}}
    for cache in (api._DISPLAY_PREVIEW_CACHE, api._DEHAZED_PREVIEW_CACHE,
                  api._ENHANCE_PREVIEW_CACHE, api._PROCESSING_PREVIEW_CACHE):
        cache.clear()
    api._shutdown_full_resolution_preview_cache()
    api._clear_full_resolution_preview_caches()
    try:
        options = dict(session_id=session, photo_id="photo", preview_level=0,
                       full_resolution=full_resolution, color_manage_srgb=True,
                       render_backend="cpu", basic_backend="python", params={"strength": 0})
        original = api.create_enhance_preview(api.EnhancePreviewRequest(mode="original", **options))
        effect = api.create_enhance_preview(api.EnhancePreviewRequest(mode="dehazed", **options))
        assert original.status_code == effect.status_code == 200
        assert original.body == effect.body
        corrected, _, applied = api._correct_enhanced_raw(source, metadata, path, preview=True)
        assert applied
        assert corrected[100, 40, 0] > source[100, 40, 0]
        np.testing.assert_array_equal(corrected[:, 250:], source[:, 250:])
        # Ricoh's independent rendering/export retains its existing baseline.
        uncorrected = api._cached_display_preview(session, "photo", path, 1800)
        np.testing.assert_array_equal(uncorrected, api._display_rgb8(source, linear=True))
    finally:
        api._ENHANCE_SESSIONS.pop(session, None)
        api._shutdown_full_resolution_preview_cache()
        api._clear_full_resolution_preview_caches()
        for cache in (api._DISPLAY_PREVIEW_CACHE, api._DEHAZED_PREVIEW_CACHE,
                      api._ENHANCE_PREVIEW_CACHE, api._PROCESSING_PREVIEW_CACHE):
            cache.clear()
