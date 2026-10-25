from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "core3_prefix_reference", ROOT / "tests/acr_support/core3_prefix_oracle.py")
oracle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(oracle)
BRIDGE = ROOT / "native-renderer/build/libimprint_stage3_core3_prefix_test_bridge.dylib"


def test_prefix_workspace_uses_spatial_rows_not_plane_six():
    data = np.broadcast_to(np.arange(9, dtype=np.float32)[:, None] * 10, (9, 20)).copy()
    expected = oracle.reference(data, 12)
    assert np.all(expected[2:7, 2, :12] == 20)
    assert np.all(expected[2:7, 5, :12] == 40)
    assert np.all(np.isnan(expected[:, 6]))
    assert np.all(expected[4, 7:9, 4:8] == 0)


@pytest.mark.skipif(not BRIDGE.exists(), reason="native core3 prefix bridge has not been built")
@pytest.mark.parametrize("width,height,padding", [(12, 9, 0), (16, 17, 4), (64, 20, 12)])
def test_prefix_matches_independent_spatial_reference(width, height, padding):
    rng = np.random.default_rng(731 + width)
    data = rng.uniform(-2, 5, (height, width+8+padding)).astype(np.float32)
    before = data.copy()
    status, actual = oracle.Bridge(BRIDGE).execute(data, width)
    assert status == 0
    oracle.assert_workspace_equal(actual, oracle.reference(data, width))
    assert np.array_equal(data.view(np.uint32), before.view(np.uint32))
