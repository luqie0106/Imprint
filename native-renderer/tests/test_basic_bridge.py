from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import native_renderer  # noqa: E402
from native_renderer import NativeRendererError, native_basic  # noqa: E402
from ricoh_filter import apply_basic_preview_effect  # noqa: E402


_FIELDS = ("exposure", "contrast", "highlights", "shadows", "whites",
           "blacks", "vibrance", "saturation")


@pytest.fixture(scope="module", autouse=True)
def require_requested_native_backend():
    requested = os.environ.get("IMPRINT_NATIVE_RENDERER_BACKEND")
    if not requested:
        return
    status = native_renderer.get_native_status()
    if not status["available"] or str(status["backend"]).lower() != requested.lower():
        pytest.fail(f"Requested native backend {requested} is unavailable: {status}")


@pytest.mark.parametrize("changes", [
    {},
    {"exposure": 1.05},
    {"highlights": -100},
    {"highlights": 100},
    {"shadows": -100},
    {"shadows": 100},
    {"whites": -100},
    {"whites": 100},
    {"blacks": -100},
    {"blacks": 100},
    {"exposure": 1.05, "highlights": -26, "shadows": 57},
])
def test_native_basic_tone_matches_python_preview(changes):
    gray = np.arange(256, dtype=np.uint8)
    image = np.broadcast_to(gray[None, :, None], (2, 256, 3)).copy()
    image[1, :3] = [[40, 80, 35], [185, 200, 150], [230, 160, 205]]
    original = image.copy()
    params = {field: 0.0 for field in _FIELDS}
    params.update(changes)
    try:
        actual = native_basic(image, params)
    except NativeRendererError as exc:
        if os.environ.get("IMPRINT_NATIVE_RENDERER_BACKEND"):
            pytest.fail(f"Requested native backend failed: {exc}")
        pytest.skip(f"Native renderer is unavailable: {exc}")
    expected = apply_basic_preview_effect(image, params)
    np.testing.assert_array_equal(image, original)
    assert actual.shape == image.shape and actual.dtype == image.dtype
    assert np.max(np.abs(actual.astype(np.int16) - expected.astype(np.int16))) <= 1
