from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from src.dng_gainmap import (
    DNGGainMap,
    DNGGainMapError,
    _apply_gain_map,
    apply_dng_gain_map,
    parse_dng_gain_map,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_SAMPLES = (
    PROJECT_ROOT / "test_images" / "DJI_0523.DNG",
    PROJECT_ROOT / "test_images" / "DJI_0539.DNG",
)


def _gainmap_payload(
    *,
    width: int = 3,
    height: int = 3,
    frame_width: int = 5,
    frame_height: int = 5,
    gains: np.ndarray | None = None,
) -> bytes:
    if gains is None:
        gains = np.full((height, width, 3), 2.0, dtype=np.float32)
        gains[height // 2, width // 2] = 1.0
    assert gains.shape == (height, width, 3)
    fixed = struct.pack(
        ">10I4dI",
        0,
        0,
        frame_height,
        frame_width,
        0,
        3,
        1,
        1,
        height,
        width,
        0.5,
        0.5,
        0.0,
        0.0,
        3,
    )
    return fixed + gains.astype(">f4").tobytes()


def _opcode_list(payload: bytes) -> bytes:
    return (
        struct.pack(">I", 1)
        + struct.pack(">4I", 9, 0x00010300, 0, len(payload))
        + payload
    )


def _write_tiny_dng(path: Path, opcode_list: bytes) -> None:
    """Write a small little-endian Classic TIFF with OpcodeList3 in a SubIFD."""
    root_ifd_offset = 8
    sub_ifd_offset = root_ifd_offset + 2 + 12 + 4
    sub_entry_count = 3
    opcode_offset = sub_ifd_offset + 2 + sub_entry_count * 12 + 4
    root = (
        struct.pack("<H", 1)
        + struct.pack("<HHI4s", 330, 4, 1, struct.pack("<I", sub_ifd_offset))
        + struct.pack("<I", 0)
    )
    sub_ifd = (
        struct.pack("<H", sub_entry_count)
        + struct.pack("<HHI4s", 256, 4, 1, struct.pack("<I", 5))
        + struct.pack("<HHI4s", 257, 4, 1, struct.pack("<I", 5))
        + struct.pack("<HHI4s", 51022, 7, len(opcode_list), struct.pack("<I", opcode_offset))
        + struct.pack("<I", 0)
    )
    header = b"II" + struct.pack("<HI", 42, root_ifd_offset)
    path.write_bytes(header + root + sub_ifd + opcode_list)


def _constant_gain_map(gain: float) -> DNGGainMap:
    return DNGGainMap(
        top=0,
        left=0,
        bottom=1,
        right=1,
        plane=0,
        planes=3,
        row_pitch=1,
        col_pitch=1,
        map_points_v=1,
        map_points_h=1,
        map_spacing_v=1.0,
        map_spacing_h=1.0,
        map_origin_v=0.0,
        map_origin_h=0.0,
        map_planes=3,
        gains=np.full((1, 1, 3), gain, dtype=np.float32),
        active_area=(0, 0, 1, 1),
        image_width=1,
        image_height=1,
        orientation=1,
    )


def _write_uniform_dng(path: Path, channel_gains: tuple[float, float, float]) -> None:
    gains = np.empty((3, 3, 3), dtype=np.float32)
    gains[:] = channel_gains
    _write_tiny_dng(path, _opcode_list(_gainmap_payload(gains=gains)))


def test_parses_gainmap_from_both_real_dji_samples() -> None:
    existing = [path for path in REAL_SAMPLES if path.is_file()]
    if not existing:
        pytest.skip("Local ignored test_images DNG samples are unavailable")

    for path in existing:
        gain_map = parse_dng_gain_map(path)
        assert gain_map is not None
        assert gain_map.gains.shape == (32, 32, 3)
        assert gain_map.active_area == (0, 0, 3956, 5280)
        assert gain_map.map_planes == 3
        assert np.isfinite(gain_map.gains).all()
        assert float(gain_map.gains[16, 16].mean()) == pytest.approx(1.0, abs=0.02)


def test_applies_real_dji_gainmap_and_brightens_corners_without_mutating_input() -> None:
    sample = next((path for path in REAL_SAMPLES if path.is_file()), None)
    if sample is None:
        pytest.skip("Local ignored test_images DNG samples are unavailable")

    image = np.full((300, 400, 3), 1000, dtype=np.uint16)
    original = image.copy()
    corrected, applied = apply_dng_gain_map(image, sample)

    assert applied is True
    assert corrected.shape == image.shape
    assert corrected.dtype == np.uint16
    assert np.array_equal(image, original)
    assert corrected[0, 0].mean() > corrected[150, 200].mean()
    assert corrected[-1, -1].mean() > corrected[150, 200].mean()


def test_applies_synthetic_gainmap_and_clips_uint16_overflow(tmp_path: Path) -> None:
    source = tmp_path / "synthetic.dng"
    gains = np.full((3, 3, 3), 2.0, dtype=np.float32)
    gains[1, 1] = 1.0
    _write_tiny_dng(source, _opcode_list(_gainmap_payload(gains=gains)))

    image = np.full((5, 5, 3), 40000, dtype=np.uint16)
    original = image.copy()
    corrected, applied = apply_dng_gain_map(image, source)

    assert applied is True
    assert np.array_equal(image, original)
    assert corrected[2, 2].tolist() == [40000, 40000, 40000]
    assert corrected[0, 0].tolist() == [65535, 65535, 65535]


def test_default_path_keeps_standard_opcode_multiplication(tmp_path: Path) -> None:
    source = tmp_path / "standard-multiply.dng"
    _write_uniform_dng(source, (1.3836, 1.3836, 1.3836))
    image = np.full((5, 5, 3), 40000, dtype=np.uint16)

    standard, applied = apply_dng_gain_map(image, source)
    protected, protected_applied = apply_dng_gain_map(image, source, preserve_highlights=True)

    assert applied is True
    assert protected_applied is True
    assert standard[2, 2].tolist() == [55344, 55344, 55344]
    assert protected[2, 2].tolist() == [40000, 40000, 40000]


def test_protected_colored_sun_does_not_expand_bright_regions(tmp_path: Path) -> None:
    source = tmp_path / "colored-sun.dng"
    _write_uniform_dng(source, (1.18, 1.29, 1.3836))

    height = width = 25
    sky = np.tile(np.linspace(0.08, 0.54, width, dtype=np.float32), (height, 1))
    yy, xx = np.ogrid[:height, :width]
    distance = np.sqrt((xx - 12) ** 2 + (yy - 12) ** 2)
    sky[distance <= 6] = 0.86
    sky[distance <= 3] = 0.98
    color = np.array([0.63, 0.88, 1.0], dtype=np.float32)
    image = np.rint(sky[:, :, None] * color * 65535.0).astype(np.uint16)

    corrected, applied = apply_dng_gain_map(image, source, preserve_highlights=True)

    assert applied is True
    peak_before = np.max(image, axis=2)
    peak_after = np.max(corrected, axis=2)
    for fraction in (0.8, 0.95):
        threshold = round(fraction * 65535)
        assert np.array_equal(peak_after >= threshold, peak_before >= threshold)
    assert np.array_equal(corrected[peak_before >= round(0.55 * 65535)], image[peak_before >= round(0.55 * 65535)])


@pytest.mark.parametrize("gain", [0.5, 1.0, 1.4, 2.2, 4.0, 16.0])
def test_protected_grayscale_response_is_monotonic(gain: float) -> None:
    values = np.rint(np.linspace(0, 65535, 8192, dtype=np.float64)).astype(np.uint16)
    image = np.repeat(values[None, :, None], 3, axis=2)

    corrected = _apply_gain_map(image, _constant_gain_map(gain), preserve_highlights=True)

    assert np.all(np.diff(corrected[0, :, 0].astype(np.int32)) >= 0)


def test_protected_gray_response_has_continuous_slope_at_highlight_boundary() -> None:
    boundary = int(65535 * 0.55)
    values = np.arange(boundary - 512, boundary + 513, dtype=np.uint16)
    image = np.repeat(values[None, :, None], 3, axis=2)

    corrected = _apply_gain_map(image, _constant_gain_map(4.0), preserve_highlights=True)
    output = corrected[0, :, 0].astype(np.float64)
    center = boundary - int(values[0])
    left_slope = (output[center] - output[0]) / 512.0
    right_slope = (output[-1] - output[center]) / 512.0

    assert abs(left_slope - right_slope) < 0.03
    quantized_steps = np.diff(output)
    assert quantized_steps.min() >= 0
    assert quantized_steps.max() <= 2


def test_protection_is_smooth_across_spatial_gain_and_brightness_gradients() -> None:
    gains = np.empty((1, 3, 3), dtype=np.float32)
    gains[:] = np.array([1.0, 1.5, 2.0], dtype=np.float32)[None, :, None]
    gain_map = DNGGainMap(
        top=0,
        left=0,
        bottom=1,
        right=1,
        plane=0,
        planes=3,
        row_pitch=1,
        col_pitch=1,
        map_points_v=1,
        map_points_h=3,
        map_spacing_v=1.0,
        map_spacing_h=0.5,
        map_origin_v=0.0,
        map_origin_h=0.0,
        map_planes=3,
        gains=gains,
        active_area=(0, 0, 1, 1),
        image_width=1,
        image_height=1,
        orientation=1,
    )
    values = np.rint(np.linspace(1000, 55000, 2049)).astype(np.uint16)
    image = np.repeat(values[None, :, None], 3, axis=2)

    corrected = _apply_gain_map(image, gain_map, preserve_highlights=True)
    steps = np.diff(corrected[0, :, 0].astype(np.int32))

    assert np.all(steps >= 0)
    assert int(steps.max()) < 250


def test_reference_applies_shared_protection_strength_without_mutation() -> None:
    gain_map = _constant_gain_map(2.0)
    image_a = np.array([[[8000, 10000, 12000], [12000, 14000, 16000]]], dtype=np.uint16)
    image_b = np.array([[[16000, 18000, 20000], [20000, 22000, 24000]]], dtype=np.uint16)
    reference = np.array([[[15000, 18000, 20000], [50000, 51000, 52000]]], dtype=np.uint16)
    reference_before = reference.copy()

    corrected_a = _apply_gain_map(
        image_a,
        gain_map,
        preserve_highlights=True,
        highlight_reference=reference,
    )
    corrected_b = _apply_gain_map(
        image_b,
        gain_map,
        preserve_highlights=True,
        highlight_reference=reference,
    )
    factor_a = corrected_a[0, 0, 0] / image_a[0, 0, 0]
    factor_b = corrected_b[0, 0, 0] / image_b[0, 0, 0]

    assert factor_a == pytest.approx(factor_b, abs=0.0001)
    assert corrected_a[0, 1].tolist() == image_a[0, 1].tolist()
    assert corrected_b[0, 1].tolist() == image_b[0, 1].tolist()
    assert np.array_equal(reference, reference_before)


@pytest.mark.parametrize(
    ("reference", "error"),
    [
        (np.zeros((5, 4, 3), dtype=np.uint16), ValueError),
        (np.zeros((5, 5, 3), dtype=np.float32), ValueError),
        ("not an array", TypeError),
    ],
)
def test_invalid_highlight_reference_is_rejected(reference: object, error: type[Exception]) -> None:
    image = np.zeros((5, 5, 3), dtype=np.uint16)

    with pytest.raises(error):
        _apply_gain_map(
            image,
            _constant_gain_map(1.5),
            preserve_highlights=True,
            highlight_reference=reference,  # type: ignore[arg-type]
        )


def test_protected_path_lifts_shadows_and_keeps_highlights_unchanged() -> None:
    image = np.full((2, 2, 3), 5000, dtype=np.uint16)
    image[1, 1] = 50000
    original = image.copy()

    corrected = _apply_gain_map(image, _constant_gain_map(2.0), preserve_highlights=True)

    assert corrected[0, 0].mean() > image[0, 0].mean()
    assert corrected[1, 1].tolist() == image[1, 1].tolist()
    assert np.array_equal(image, original)


def test_malformed_opcode_list_fails_with_typed_error(tmp_path: Path) -> None:
    source = tmp_path / "truncated.dng"
    _write_tiny_dng(source, struct.pack(">I", 1))

    with pytest.raises(DNGGainMapError):
        parse_dng_gain_map(source)


def test_non_dng_is_not_applied_and_returns_original_array(tmp_path: Path) -> None:
    source = tmp_path / "plain.jpg"
    source.write_bytes(b"not a TIFF or DNG")
    image = np.zeros((3, 4, 3), dtype=np.uint16)

    corrected, applied = apply_dng_gain_map(image, source)

    assert applied is False
    assert corrected is image
