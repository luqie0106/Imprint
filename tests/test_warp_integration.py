"""Embedded geometry stays aligned across preview and camera-space export."""
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import app_api as api
import dng_warp
import image_io


@pytest.mark.parametrize(("requested_backend", "warp_backend"), [
    ("native", "auto"),
    ("legacy_gpu", "auto"),
    ("pytorch", "auto"),
    ("cpu", "cpu"),
    ("auto", "auto"),
    ("gpu", "gpu"),
    ("metal", "metal"),
    ("d3d12", "d3d12"),
])
def test_embedded_preview_maps_backend_and_avoids_double_lensfun(
    monkeypatch, requested_backend, warp_backend,
):
    image = np.full((8, 12, 3), 20000, dtype=np.uint16)
    original = image.copy()
    reference = image.copy()
    calls = []

    def warp(value, path, **kwargs):
        calls.append((value.copy(), kwargs))
        kwargs["diagnostics"].update(backend="D3D12", tca_applied=True)
        return value.copy(), True, True

    def unexpected(*args, **kwargs):
        pytest.fail("embedded correction must not also apply Lensfun/GainMap")

    monkeypatch.setattr(api, "apply_dng_warp_correction", warp)
    monkeypatch.setattr(api, "apply_lens_correction", unexpected)
    monkeypatch.setattr(api, "apply_dng_gain_map", unexpected)
    metadata = SimpleNamespace(source_kind="raw", exif={"Model": "L3D-100c"})
    _, result, gain = api._correct_enhanced_raw(
        image, metadata, Path("source.dng"), preview=True,
        backend=requested_backend, highlight_reference=reference,
    )
    assert result.engine == "DNG/WarpRectilinear"
    assert result.backend == "D3D12" and result.tca_applied and gain
    assert len(calls) == 1
    warped_input, kwargs = calls[0]
    np.testing.assert_array_equal(warped_input, original)
    assert kwargs["backend"] == warp_backend
    assert kwargs["preserve_highlights"] is True
    assert kwargs["full_frame"] is True
    assert kwargs["highlight_reference"] is reference
    np.testing.assert_array_equal(image, original)
    np.testing.assert_array_equal(reference, original)


@pytest.mark.parametrize("gain", [False, True])
def test_export_reproduces_embedded_pipeline_once(monkeypatch, gain):
    import lens_correction
    import dng_gainmap
    from test_image_io import _install_enhanced_fake_raw

    rng = np.random.default_rng(82)
    camera = rng.integers(5000, 18000, (256, 256, 3), dtype=np.uint16)
    reference = camera.copy()
    processed = np.ascontiguousarray(np.flip(reference, axis=1))
    calls = []

    def warp(value, path, **kwargs):
        assert kwargs["highlight_reference"] is reference
        calls.append(value.copy())
        if kwargs.get("diagnostics") is not None:
            kwargs["diagnostics"].update(backend="Metal", tca_applied=False)
        return np.ascontiguousarray(np.flip(value, axis=1)), True, gain

    def unexpected(*args, **kwargs):
        pytest.fail("embedded export must not apply a second correction")

    monkeypatch.setattr(dng_warp, "apply_dng_warp_correction", warp)
    monkeypatch.setattr(lens_correction, "apply_lens_correction", unexpected)
    monkeypatch.setattr(dng_gainmap, "apply_dng_gain_map", unexpected)
    _install_enhanced_fake_raw(monkeypatch, camera)
    metadata = {
        "Make": "Hasselblad", "Model": "L3D-100c",
        "LensCorrectionApplied": True, "LensCorrectionEngine": "DNG/WarpRectilinear",
        "LensCorrectionCamera": "L3D-100c", "LensCorrectionLens": "DNG WarpRectilinear",
        "LensCorrectionOperations": "distortion", "DNGGainMapApplied": gain,
    }
    enhanced, *_ = image_io.enhanced_dng_source_data(
        processed, reference, "source.dng", metadata,
    )
    np.testing.assert_array_equal(enhanced, processed)
    assert len(calls) == 2
    np.testing.assert_array_equal(calls[0], camera)
    np.testing.assert_array_equal(calls[1], reference)


def test_malformed_embedded_data_is_not_missing_calibration(monkeypatch):
    def corrupt(*args, **kwargs):
        raise dng_warp.DNGWarpError("truncated opcode")

    monkeypatch.setattr(api, "apply_dng_warp_correction", corrupt)
    with pytest.raises(dng_warp.DNGWarpError):
        api._correct_enhanced_raw_for_export(
            np.zeros((4, 4, 3), dtype=np.uint16),
            SimpleNamespace(source_kind="raw", exif={}), Path("source.dng"),
        )
