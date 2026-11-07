"""Camera-channel metadata remains correct in spatial CFA DNG tags."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import rawpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import dng_writer
import image_io


@pytest.mark.parametrize("pattern", [
    [[0, 1], [3, 2]], [[1, 0], [2, 3]],
    [[3, 2], [0, 1]], [[2, 3], [1, 0]],
])
@pytest.mark.parametrize("orientation", [1, 3, 6, 8])
def test_enhanced_source_maps_channel_black_levels_to_sensor_positions(
    tmp_path, monkeypatch, pattern, orientation,
):
    camera = np.random.default_rng(31).integers(2000, 16000, (192, 224, 3), dtype=np.uint16)
    mosaic = camera[:, :, 0].copy()
    rotated = {1: 0, 3: 2, 6: -1, 8: 1}[orientation]
    display_camera = np.ascontiguousarray(np.rot90(camera, rotated))
    raw = SimpleNamespace(
        raw_image_visible=mosaic, raw_pattern=np.asarray(pattern), color_desc=b"RGBG",
        rgb_xyz_matrix=np.vstack((np.eye(3), np.zeros(3))),
        camera_whitebalance=(2.0, 1.0, 1.5, 1.0),
        black_level_per_channel=(10, 20, 30, 40), white_level=65535,
        sizes=SimpleNamespace(crop_width=0, crop_height=0),
        postprocess=lambda **kwargs: display_camera.copy(),
    )
    monkeypatch.setattr(image_io, "open_raw", lambda path: nullcontext(raw))
    metadata = {"Make": "Imprint", "Model": "Synthetic Camera", "Orientation": orientation}
    rgb, cfa, colors, profile, output_orientation = image_io.enhanced_dng_source_data(
        display_camera, display_camera, "sample.nef", metadata,
    )
    expected_black = tuple(raw.black_level_per_channel[index] for row in pattern for index in row)
    assert profile["DNGBlackLevel"] == expected_black
    np.testing.assert_allclose(profile["DNGAsShotNeutral"], (0.5, 1.0, 2.0 / 3.0))
    np.testing.assert_array_equal(profile["DNGColorMatrix1"], np.eye(3).reshape(-1))
    np.testing.assert_array_equal(rgb, camera)
    np.testing.assert_array_equal(cfa, mosaic)
    assert output_orientation == orientation

    output = dng_writer.write_enhanced_dng(
        rgb, cfa, colors, "sample.nef", tmp_path, {**metadata, **profile},
        orientation=orientation, compression="none",
    )
    with rawpy.imread(str(output)) as decoded:
        np.testing.assert_array_equal(decoded.raw_image_visible, mosaic)
        decoded_levels = decoded.black_level_per_channel
        spatial_black = tuple(decoded_levels[int(index)] for index in decoded.raw_pattern.flat)
        assert spatial_black == expected_black


def test_enhanced_source_rejects_missing_channel_black_level(monkeypatch):
    raw = SimpleNamespace(
        raw_image_visible=np.zeros((64, 64), np.uint16),
        raw_pattern=np.asarray([[0, 1], [3, 2]]), color_desc=b"RGBG",
        rgb_xyz_matrix=np.vstack((np.eye(3), np.zeros(3))),
        camera_whitebalance=(1.0, 1.0, 1.0, 1.0), black_level_per_channel=(0, 0, 0),
    )
    monkeypatch.setattr(image_io, "open_raw", lambda path: nullcontext(raw))
    rgb = np.zeros((64, 64, 3), np.uint16)
    with pytest.raises(image_io.EnhancedDNGColorError, match="CFA black levels"):
        image_io.enhanced_dng_source_data(rgb, rgb, "sample.nef", {})
