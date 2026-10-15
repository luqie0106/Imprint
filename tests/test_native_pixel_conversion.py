from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import native_dense  # noqa: E402
import native_renderer  # noqa: E402
from native_renderer import NativeRendererError  # noqa: E402


_CONVERSION_SYMBOLS = (
    "im_native_rgb8_to_linear_float",
    "im_native_rgb16_to_linear_float",
    "im_native_linear_float_to_rgb16",
    "im_native_expand_rgb8_to_rgb16",
    "im_native_compact_rgb16_to_rgb8",
    "im_native_float_range_valid",
)


@pytest.fixture(scope="module", autouse=True)
def _require_conversion_library():
    try:
        library, _ = native_renderer._get_library()
    except NativeRendererError as exc:
        pytest.skip(f"optional native renderer library unavailable: {exc}")
    missing = [name for name in _CONVERSION_SYMBOLS if getattr(library, name, None) is None]
    if missing:
        pytest.skip(f"native renderer lacks pixel conversion ABI: {', '.join(missing)}")


def _all_codes(dtype, maximum):
    count = maximum + 1
    padded_count = count + (-count % 3)
    samples = np.arange(padded_count, dtype=np.uint32) % count
    return samples.astype(dtype).reshape(1, -1, 3)


def test_integer_to_linear_float_matches_numpy_exhaustively():
    rgb8 = _all_codes(np.uint8, 255)
    original8 = rgb8.copy()
    actual8 = native_dense.rgb8_to_linear_float(rgb8)
    expected8 = rgb8.astype(np.float32) / 255.0
    assert actual8.dtype == np.float32
    assert np.array_equal(actual8, expected8)
    assert np.array_equal(rgb8, original8)

    rgb16 = _all_codes(np.uint16, 65535)
    original16 = rgb16.copy()
    actual16 = native_dense.rgb16_to_linear_float(rgb16)
    expected16 = rgb16.astype(np.float32) / 65535.0
    assert actual16.dtype == np.float32
    assert np.array_equal(actual16, expected16)
    assert np.array_equal(rgb16, original16)


def test_float_to_rgb16_matches_numpy_for_ties_clipping_and_random_values():
    tie_samples = np.arange(33, dtype=np.float32) + np.float32(0.5)
    ties = (tie_samples / np.float32(65535.0)).astype(np.float32)
    edge_values = np.array([-4.0, -0.5 / 65535.0, 0.0, 1.0, 1.01, 9.0], dtype=np.float32)
    random_values = np.random.default_rng(1307).uniform(-0.15, 1.15, size=8193).astype(np.float32)
    flat = np.concatenate((ties, edge_values, random_values))
    image = flat[: flat.size - flat.size % 3].reshape(1, -1, 3)
    saved = image.copy()
    image.flags.writeable = False

    expected = np.clip(np.rint(image * 65535.0), 0, 65535).astype(np.uint16)
    actual = native_dense.linear_float_to_rgb16(image)
    assert actual.dtype == np.uint16
    assert np.array_equal(actual, expected)
    assert np.array_equal(image, saved)
    assert np.array_equal(actual.reshape(-1)[:10], np.array([0, 2, 2, 4, 4, 6, 6, 8, 8, 10], dtype=np.uint16))


def test_byte_expansion_and_compaction_are_exhaustive():
    rgb8 = _all_codes(np.uint8, 255)
    saved8 = rgb8.copy()
    rgb8.flags.writeable = False
    expanded = native_dense.expand_rgb8_to_rgb16(rgb8)
    assert np.array_equal(expanded, rgb8.astype(np.uint16) * 257)
    assert np.array_equal(rgb8, saved8)

    rgb16 = _all_codes(np.uint16, 65535)
    saved16 = rgb16.copy()
    rgb16.flags.writeable = False
    compacted = native_dense.compact_rgb16_to_rgb8(rgb16)
    expected = ((rgb16.astype(np.uint32) + 128) // 257).astype(np.uint8)
    assert np.array_equal(compacted, expected)
    assert np.array_equal(rgb16, saved16)


def test_conversions_accept_noncontiguous_readonly_rgb_inputs():
    rng = np.random.default_rng(75)
    base8 = rng.integers(0, 256, (13, 22, 3), dtype=np.uint8)
    source8 = base8[:, ::2, :]
    source8.flags.writeable = False
    saved8 = source8.copy()
    assert np.array_equal(native_dense.expand_rgb8_to_rgb16(source8), source8.astype(np.uint16) * 257)
    assert np.array_equal(native_dense.rgb8_to_linear_float(source8), source8.astype(np.float32) / 255.0)
    assert np.array_equal(source8, saved8)

    base16 = rng.integers(0, 65536, (13, 22, 3), dtype=np.uint16)
    source16 = base16[:, ::2, :]
    source16.flags.writeable = False
    saved16 = source16.copy()
    assert np.array_equal(native_dense.compact_rgb16_to_rgb8(source16),
                          ((source16.astype(np.uint32) + 128) // 257).astype(np.uint8))
    assert np.array_equal(native_dense.rgb16_to_linear_float(source16),
                          source16.astype(np.float32) / 65535.0)
    assert np.array_equal(source16, saved16)


def test_float_to_rgb16_rejects_nan_and_infinity_before_returning_pixels():
    for bad in (np.nan, np.inf, -np.inf):
        image = np.zeros((2, 2, 3), dtype=np.float32)
        image[0, 0, 1] = bad
        with pytest.raises(NativeRendererError):
            native_dense.linear_float_to_rgb16(image)


def test_wrappers_retain_disable_and_input_validation_contracts(monkeypatch):
    source = np.zeros((2, 2, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        native_dense.rgb8_to_linear_float(source[:, :, :2])
    with pytest.raises(TypeError):
        native_dense.rgb8_to_linear_float(source.astype(np.uint16))

    monkeypatch.setenv("IMPRINT_NATIVE_DENSE", "0")
    with pytest.raises(NativeRendererError):
        native_dense.expand_rgb8_to_rgb16(source)


def test_float_range_scan_checks_finiteness_bounds_and_noncontiguous_inputs():
    base = np.array([
        [-0.25, 0.0, 0.5, 1.0],
        [0.25, 0.75, 0.125, 0.875],
    ], dtype=np.float32)
    source = base[:, 1::2]
    source.flags.writeable = False
    saved = source.copy()

    assert native_dense.float_range_valid(source, 0.0, 1.0)
    assert not native_dense.float_range_valid(source, 0.0, 0.5)
    assert np.array_equal(source, saved)

    for bad in (np.nan, np.inf, -np.inf, -0.0001, 1.0001):
        values = np.array([0.0, 0.5, bad], dtype=np.float32)
        assert not native_dense.float_range_valid(values, 0.0, 1.0)

    assert native_dense.float_range_valid(np.array([-2.0, 0.0, 3.0], dtype=np.float32), -2.0, 3.0)
    with pytest.raises(TypeError):
        native_dense.float_range_valid(np.ones(2, dtype=np.float64))
    with pytest.raises(ValueError):
        native_dense.float_range_valid(np.empty(0, dtype=np.float32))
    with pytest.raises(ValueError):
        native_dense.float_range_valid(np.ones(2, dtype=np.float32), 1.0, 0.0)


def test_float_range_scan_raises_native_error_when_disabled(monkeypatch):
    monkeypatch.setenv("IMPRINT_NATIVE_DENSE", "0")
    with pytest.raises(NativeRendererError):
        native_dense.float_range_valid(np.ones(2, dtype=np.float32))
