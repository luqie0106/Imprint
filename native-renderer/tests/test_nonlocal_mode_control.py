"""Per-render experiment selection must never mutate process-wide state."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from dehaze import DehazeParams, resolve_nonlocal_mode
import dehaze_physical as physical


def test_explicit_modes_override_environment_and_stay_isolated(monkeypatch):
    monkeypatch.setenv("IMPRINT_NONLOCAL_RELIEF", "strong")
    air = np.array([.55, .60, .65], np.float32)
    source = np.broadcast_to(air, (160, 240, 3)).copy()
    material = np.array([.06, .08, .10], np.float32)
    depth = np.linspace(.12, .95, 128, dtype=np.float32)
    source[32:] = (air + depth[:, None] * (material - air))[:, None, :]
    monkeypatch.setattr(physical, "_estimate_airlight", lambda rgb, p: (air.copy(), 1.0))
    p = DehazeParams(strength=1)
    modes = ["off", "conservative", "strong"]

    def render(mode):
        return physical.apply_physical_dehaze(source, p, spatial=True, nonlocal_mode=mode)

    expected = [render(mode) for mode in modes]
    assert all(not np.array_equal(expected[0], result) for result in expected[1:])
    with ThreadPoolExecutor(max_workers=3) as pool:
        actual = list(pool.map(render, modes * 2))
    for index, result in enumerate(actual):
        np.testing.assert_array_equal(result, expected[index % 3])
    assert os.environ["IMPRINT_NONLOCAL_RELIEF"] == "strong"
    assert len({p.cache_token(nonlocal_mode=mode) for mode in modes}) == 3
    for mode in modes:
        assert physical.physical_diagnostics(source, p, spatial=True,
                                            nonlocal_mode=mode)["nonlocal_mode"] == mode


def test_legacy_environment_mode_and_explicit_validation(monkeypatch):
    monkeypatch.setenv("IMPRINT_NONLOCAL_RELIEF", "1")
    assert resolve_nonlocal_mode() == "conservative"
    assert resolve_nonlocal_mode("off") == "off"
    assert resolve_nonlocal_mode("strong") == "strong"
    with pytest.raises(ValueError, match="nonlocal_mode"):
        resolve_nonlocal_mode("typo")
