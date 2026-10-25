"""core3 row-kernel regressions (FUN_0081376c / FUN_00813368).

These tests cover only the two recovered full-resolution pixel consumers. They
do not claim that the cr_stage_bayer_core3 pipeline or full-resolution RGB is
complete. The direction probes are hardcoded from the saved 0081376c
disassembly and are deliberately not derived from the C++ implementation.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ORACLE_PATH = ROOT / "tests/acr_support/core3_oracle.py"
BRIDGE = ROOT / "native-renderer/build/libimprint_stage3_core3_test_bridge.dylib"


def load_oracle():
    spec = importlib.util.spec_from_file_location("acr_core3_oracle", ORACLE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_core3_gradient_reference_uses_explicit_minus2_row():
    oracle = load_oracle()
    width = 8
    center = np.zeros(width + 8, dtype=np.float32)
    minus2 = np.zeros(width + 8, dtype=np.float32)
    plus2 = np.zeros(width + 8, dtype=np.float32)
    # Only the minus2 row changes: out2 must stay zero, out3 must see it.
    minus2[oracle.GUARD + 5] = -3.0
    rows = oracle.gradient_reference(
        center, minus2, plus2, center, center, width, oracle.GUARD)
    assert np.all(rows[2] == 0.0)
    expected = np.zeros(width, dtype=np.float32)
    expected[5] = 3.0
    assert np.array_equal(rows[3], expected)


def test_core3_gradient_reference_vertical_step():
    oracle = load_oracle()
    width, stride = 4, 12
    rows = oracle.constant_rows([0, 10, 20, 30, 40], width, stride)
    out = oracle.gradient_reference(
        rows[2], rows[0], rows[4], rows[1], rows[3], width, oracle.GUARD)
    assert np.array_equal(out[0], np.zeros(width, dtype=np.float32))
    assert np.array_equal(out[1], np.zeros(width, dtype=np.float32))
    assert np.array_equal(out[2], np.full(width, 20.0, dtype=np.float32))
    assert np.array_equal(out[3], np.full(width, 40.0, dtype=np.float32))
    assert np.array_equal(out[4], np.zeros(width, dtype=np.float32))
    assert np.array_equal(out[5], np.zeros(width, dtype=np.float32))


def test_core3_energy_reference_directions():
    oracle = load_oracle()
    width, stride = 8, 16
    horizontal_rows = np.empty((5, stride), dtype=np.float32)
    for row in range(5):
        horizontal_rows[row, :] = np.arange(
            -oracle.GUARD, stride - oracle.GUARD, dtype=np.float32)
    h_out, v_out = oracle.energy_reference(
        horizontal_rows[2], [horizontal_rows[row] for row in range(5)],
        width, oracle.GUARD)
    assert np.all(h_out == np.float32(np.sqrt(np.float32(10.0))))
    assert np.all(v_out == 0.0)

    vertical_rows = oracle.constant_rows([0, 10, 20, 30, 40], width, stride)
    h_out, v_out = oracle.energy_reference(
        vertical_rows[2], [vertical_rows[row] for row in range(5)],
        width, oracle.GUARD)
    assert np.all(h_out == 0.0)
    assert np.all(v_out == np.float32(np.sqrt(np.float32(1000.0))))


def test_core3_workspace_row_mapping():
    oracle = load_oracle()
    got = [np.full(4, float(index), dtype=np.float32) for index in range(6)]
    workspace = oracle.workspace_rows(got)
    # 007eaf20 call site: out0->0, out1->1, out2->2, out3->5, out4->3, out5->4.
    assert workspace[0] is got[0]
    assert workspace[1] is got[1]
    assert workspace[2] is got[2]
    assert workspace[3] is got[4]
    assert workspace[4] is got[5]
    assert workspace[5] is got[3]


@pytest.mark.skipif(not BRIDGE.exists(), reason="core3 bridge has not been built")
def test_core3_bridge_matches_float32_oracle(tmp_path):
    oracle = load_oracle()
    bridge = oracle.Bridge(BRIDGE)
    records = oracle.case_records(bridge)
    failures = [record for record in records
                if not record["match"] or not record.get("padding_untouched", True)]
    assert not failures, failures
    # Every hardcoded direction probe must be present and pass against its
    # hand-written expectation, not merely against the reference.
    hardcoded = [record for record in records
                 if record["expected_source"] == "hardcoded"]
    assert len(hardcoded) >= 8
    assert all(record["expected_match"] for record in hardcoded)
    names = {record["name"] for record in hardcoded}
    assert "probe_vertical_step_5rows" in names
    assert "probe_plus2_row_impulse" in names
    assert "probe_minus2_row_impulse" in names
    assert "probe_asymmetric_rows" in names
    # Write a fresh artifact under tmp_path; never overwrite historical
    # evidence from a normal pytest run.
    output = tmp_path / "core3.json"
    output.write_text(json.dumps(records, indent=2) + "\n")
    assert output.exists()


@pytest.mark.skipif(not BRIDGE.exists(), reason="core3 bridge has not been built")
def test_core3_bridge_rejects_misaligned_domain():
    oracle = load_oracle()
    bridge = oracle.Bridge(BRIDGE)
    rows = oracle.constant_rows([0, 10, 20, 30, 40], 8, 16)
    center = rows[2][oracle.GUARD + 1:]  # 4-byte offset, not 16-byte aligned
    status, storage = bridge.call_gradient(
        center, rows.shape[1], 8, 8, require_alignment=False)
    assert status == oracle.UNSUPPORTED_ALIGNMENT
    assert np.all(np.isnan(storage))


@pytest.mark.skipif(not BRIDGE.exists(), reason="core3 bridge has not been built")
def test_core3_bridge_nonfinite_contract():
    oracle = load_oracle()
    bridge = oracle.Bridge(BRIDGE)
    width, stride = 8, 16
    rows = np.zeros((5, stride), dtype=np.float32)
    rows[2, oracle.GUARD + 3] = np.inf
    status, storage = bridge.call_gradient(
        rows[2][oracle.GUARD:], stride, width, width)
    assert status == oracle.OK
    assert np.isinf(storage[0][1]) and storage[0][1] > 0
    assert np.isinf(storage[1][1]) and storage[1][1] > 0
    assert np.isinf(storage[4][2]) and storage[4][2] > 0
