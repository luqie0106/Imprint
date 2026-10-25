"""Stage3 closed-domain regressions.

The oracle lives in ``tests/acr_support/stage3_oracle.py`` so ordinary pytest
does not depend on the ignored ``research/acr`` tree. Bridge output is written
to ``tmp_path``; historical evidence files are never overwritten.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ORACLE_PATH = ROOT / "tests/acr_support/stage3_oracle.py"
BRIDGE = ROOT / "native-renderer/build/libimprint_stage3_test_bridge.dylib"


def load_oracle():
    spec = importlib.util.spec_from_file_location("acr_stage3_oracle", ORACLE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_stage3_oracle_dispatch_conditions():
    oracle = load_oracle()
    full = oracle.Prepared()
    status, branch, kernel, width, height, samples = oracle.expected_dispatch(full)
    assert status == oracle.UNSUPPORTED_FULL_RES_CORE3
    assert branch == oracle.DIRECT
    assert kernel == oracle.KERNEL_1X1
    assert (width, height, samples) == (8, 4, 96)

    preview = oracle.Prepared(pair_x=2, pair_y=2)
    status, branch, kernel, width, height, samples = oracle.expected_dispatch(preview)
    assert status == oracle.OK
    assert branch == oracle.DIRECT
    assert kernel == oracle.KERNEL_2X2
    assert (width, height, samples) == (4, 2, 24)

    generic = oracle.Prepared(pair_x=1, pair_y=2)
    status, branch, _kernel, _w, _h, _s = oracle.expected_dispatch(generic)
    assert status == oracle.UNSUPPORTED_PAIR
    assert branch == oracle.GENERIC


def test_stage3_rejects_non_canonical_patterns():
    oracle = load_oracle()
    # The independent fast-Bayer helper intentionally accepts 36 three-plane
    # arrangements; Stage3's FUN_007f851c is a fixed-position RGGB body and
    # FUN_00ab21f0 only leaves the offset pair at zero for RGGB.
    for pattern in ((0, 0, 1, 2), (0, 1, 2, 2), (0, 1, 1, 1), (1, 0, 2, 1),
                    (2, 1, 1, 0), (1, 2, 0, 1)):
        prepared = oracle.Prepared(
            width=2, height=2, input_stride=2, pattern=pattern,
            rect_bottom=2, rect_right=2, pair_x=2, pair_y=2,
        )
        status, _branch, _kernel, _w, _h, _s = oracle.expected_dispatch(prepared)
        assert status == oracle.INVALID_CFA, pattern


def test_stage3_pattern_offset_contradictions():
    oracle = load_oracle()
    for offset_80, offset_84 in ((1, 0), (0, 1), (1, 1)):
        prepared = oracle.Prepared(
            pair_x=2, pair_y=2, offset_80=offset_80, offset_84=offset_84
        )
        status, _branch, _kernel, _w, _h, _s = oracle.expected_dispatch(prepared)
        assert status == oracle.UNSUPPORTED_OFFSET

    # Every non-RGGB phase with its real mode-writer offset must be rejected as
    # a CFA-phase state, not silently accepted with a zero offset.
    for name, (pattern, offsets) in oracle.MODE_WRITER_OFFSETS.items():
        prepared = oracle.Prepared(
            width=2, height=2, input_stride=2, pattern=pattern,
            offset_80=offsets[0], offset_84=offsets[1],
            rect_bottom=2, rect_right=2, pair_x=2, pair_y=2,
        )
        status, _branch, _kernel, _w, _h, _s = oracle.expected_dispatch(prepared)
        if name == "rggb":
            assert status == oracle.OK
        else:
            assert status == oracle.INVALID_CFA, name


def test_stage3_oracle_2x2_rggb_rounding():
    oracle = load_oracle()
    prepared = oracle.Prepared(width=2, height=2, input_stride=2,
                               rect_bottom=2, rect_right=2,
                               pair_x=2, pair_y=2)
    image = np.array([[2, 3], [4, 5]], dtype=np.uint16)
    expected = oracle.oracle_2x2(image, prepared)
    # Fixed positions: R=top-left, G=(top-right+bottom-left+1)>>1, B=bottom-right.
    assert expected.tolist() == [[[2, 4, 5]]]

    rounding = np.array([[10, 20], [30, 40]], dtype=np.uint16)
    assert oracle.oracle_2x2(rounding, prepared).tolist() == [[[10, 25, 40]]]


@pytest.mark.skipif(not BRIDGE.exists(), reason="Stage3 bridge has not been built")
def test_stage3_bridge_oracle_cases(tmp_path):
    output = tmp_path / "stage3-validation.json"
    oracle = load_oracle()
    payload = oracle.run_bridge(BRIDGE, output)
    assert payload["failed"] == 0
    assert payload["case_count"] == payload["passed"]
    assert output.exists()
    assert "FAIL:" not in output.read_text()

    # A direct subprocess invocation must also stay inside tmp_path and must
    # never touch the checked-in history under research/acr/reports/evidence.
    completed = subprocess.run(
        [
            sys.executable,
            str(ORACLE_PATH),
            "--bridge",
            str(BRIDGE),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "Stage3 oracle cases passed" in completed.stdout
    assert "FAIL:" not in completed.stderr
