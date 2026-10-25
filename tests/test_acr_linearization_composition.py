"""Tag adapter regressions for the real-DNG linearization composition.

The composition script used to parse BlackLevel / BlackLevelDeltaH /
BlackLevelDeltaV with the integer helper. Fractional and negative tag values
were truncated (or raised) before FUN_01304c54 quantization, which the two
integer-only DJI fixtures never exposed. These tests pin the rational parser
and the interaction with the recovered quantizer on synthetic tag payloads.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "research" / "acr" / "tools"

pytestmark = pytest.mark.skipif(
    not (TOOLS / "validate_acr_source_linearization.py").exists(),
    reason="research/acr RE tools are not present in this checkout",
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def oracle():
    return _load("acr_linearization_oracle_for_composition", TOOLS / "validate_acr_source_linearization.py")


@pytest.fixture(scope="module")
def composition(oracle):
    return _load(
        "acr_linearization_composition_for_test",
        TOOLS / "validate_acr_linearization_composition.py",
    )


def _tags(black, delta_h=None, delta_v=None, white=65535, samples=1):
    tags = {
        "BlackLevelRepeatDim": [1, 1],
        "BlackLevel": black,
        "WhiteLevel": [white],
        "SamplesPerPixel": samples,
        "ActiveArea": [0, 0, 4, 4],
    }
    if delta_h is not None:
        tags["BlackLevelDeltaH"] = delta_h
    if delta_v is not None:
        tags["BlackLevelDeltaV"] = delta_v
    return tags


def test_number_list_keeps_fractions_and_signs(composition):
    assert composition.number_list(0.5) == [0.5]
    assert composition.number_list("0.5 0.25 -0.5") == [0.5, 0.25, -0.5]
    assert composition.number_list(["1024.75", "-3.5"]) == [1024.75, -3.5]
    assert composition.number_list("1/4") == [0.25]
    assert composition.number_list("-3/8") == [-0.375]
    assert composition.number_list(None) == []
    # The historical integer helper truncated (or rejected) the same values.
    assert composition.int_list(0.5) == [0]
    with pytest.raises(ValueError):
        composition.int_list("0.5")


def test_fractional_black_level_survives_preparation(composition, oracle):
    info = composition.build_prepared(_tags("1024.75"), 4, 4, oracle)
    assert info["black_level"][0][0][0] == pytest.approx(1024.75)
    quantized = oracle.oracle_quantize(info)[0]
    assert quantized["status"] == "ok"
    # FUN_01304c54 keeps a 1/256 multiple exact.
    assert quantized["black_level"][0][0][0] == pytest.approx(1024.75)


def test_half_black_level_survives_preparation(composition, oracle):
    info = composition.build_prepared(_tags("0.5"), 4, 4, oracle)
    assert info["black_level"][0][0][0] == pytest.approx(0.5)
    quantized = oracle.oracle_quantize(info)[0]
    assert quantized["status"] == "ok"
    assert quantized["black_level"][0][0][0] == pytest.approx(0.5)


def test_negative_deltas_keep_their_sign(composition, oracle):
    info = composition.build_prepared(
        _tags("4096", delta_h="-0.5 0.25", delta_v="-1.5"), 4, 4, oracle
    )
    assert info["has_delta_h"] is True
    assert info["has_delta_v"] is True
    assert info["delta_h"] == pytest.approx([-0.5, 0.25])
    assert info["delta_v"] == pytest.approx([-1.5])
    quantized = oracle.oracle_quantize(info)[0]
    assert quantized["status"] == "ok"
    assert quantized["delta_h"] == pytest.approx([-0.5, 0.25])
    assert quantized["delta_v"] == pytest.approx([-1.5])


def test_integer_tags_still_use_the_integer_parser(composition, oracle):
    tags = _tags("4096")
    tags["BlackLevelRepeatDim"] = "2 2"
    tags["BlackLevel"] = ["4096", "4097", "4098", "4099"]
    tags["SamplesPerPixel"] = 1
    info = composition.build_prepared(tags, 4, 4, oracle)
    assert info["black_repeat_rows"] == 2
    assert info["black_repeat_cols"] == 2
    assert info["black_level"][1][1][0] == pytest.approx(4099.0)


def test_black_level_count_mismatch_is_rejected(composition, oracle):
    with pytest.raises(SystemExit):
        composition.build_prepared(_tags("1 2 3"), 4, 4, oracle)
