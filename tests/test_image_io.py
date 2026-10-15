from __future__ import annotations

from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import rawpy



sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from image_io import (
    EnhancedDNGColorError,
    _exif_from_pil,
    _fill_lens_specification,
    _lens_metadata_from_exiftool_record,
    _parse_lens_specification,
    _raw_bit_depth,
    _read_raw,
    _raw_metadata,
    _fit_srgb_to_camera,
    _camera_profile_transfer_numpy,
    build_enhanced_dng_source_cache,
    camera_profile_names,
    enhanced_dng_source_data,
    matching_embedded_profile_dng,
)


def _install_enhanced_fake_raw(monkeypatch, camera: np.ndarray, matrix=None, sizes=None):
    class FakeRaw:
        raw_image_visible = np.full(camera.shape[:2], 1200, dtype=np.uint16)
        raw_pattern = np.array([[0, 1], [3, 2]], dtype=np.uint8)
        color_desc = b"RGBG"
        rgb_xyz_matrix = np.eye(4, 3) if matrix is None else matrix
        camera_whitebalance = [1.4, 1.0, 1.6, 1.0]
        black_level_per_channel = [100, 100, 100, 100]
        white_level = 16383

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def postprocess(self, **_kwargs):
            return camera

    if sizes is not None:
        FakeRaw.sizes = sizes
    monkeypatch.setattr("image_io.rawpy.imread", lambda _path: FakeRaw())


def _pack_fit_samples(source: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    camera = np.zeros((1, source.shape[0] * 16, 3), dtype=np.uint16)
    rendered = np.zeros_like(camera)
    camera[0, ::16] = np.clip(np.rint(source * 65535.0), 0, 65535).astype(np.uint16)
    rendered[0, ::16] = np.clip(np.rint(reference * 65535.0), 0, 65535).astype(np.uint16)
    return camera, rendered


def _make_source_cache_fixture(monkeypatch, tmp_path: Path):
    import lens_correction

    rng = np.random.default_rng(184)
    camera = rng.integers(5000, 16000, (320, 256, 3), dtype=np.uint16)
    camera_to_srgb = np.array(
        [[0.91, 0.04, 0.02], [0.02, 0.89, 0.03], [0.01, 0.02, 0.9]],
        dtype=np.float64,
    )
    reference = np.rint(camera.astype(np.float64) @ camera_to_srgb).astype(np.uint16)
    corrected_camera = np.ascontiguousarray(np.flip(camera, axis=1))
    corrected_reference = np.ascontiguousarray(np.flip(reference, axis=1))
    source_path = tmp_path / "source.nef"
    source_path.write_bytes(b"stable source identity")
    source_exif = {
        "Make": "Nikon", "Model": "Z6_3", "LensCorrectionApplied": True,
        "LensCorrectionEngine": "Lensfun/lensfunpy", "LensCorrectionCamera": "Camera",
        "LensCorrectionLens": "Lens", "LensCorrectionOperations": "distortion",
    }
    lens_calls = []

    def transform(image, metadata, *, require_correction):
        assert require_correction is True
        lens_calls.append(image.copy())
        result = SimpleNamespace(
            applied=True,
            camera_name=metadata["LensCorrectionCamera"],
            lens_name=metadata["LensCorrectionLens"],
            distortion_applied=True, tca_applied=False, vignetting_applied=False,
        )
        return np.ascontiguousarray(np.flip(image, axis=1)), result

    monkeypatch.setattr(lens_correction, "apply_lens_correction", transform)
    result = SimpleNamespace(
        applied=True, camera_name="Camera", lens_name="Lens",
        distortion_applied=True, tca_applied=False, vignetting_applied=False,
    )
    cache = build_enhanced_dng_source_cache(
        source_path, camera, reference, corrected_camera, corrected_reference,
        result, result, correction_version="lens-v1",
    )
    assert cache is not None
    processed = np.clip(
        np.rint(corrected_reference.astype(np.float32) * 0.82), 0, 65535,
    ).astype(np.uint16)
    return {
        "camera": camera,
        "reference": reference,
        "corrected_camera": corrected_camera,
        "corrected_reference": corrected_reference,
        "source_path": source_path,
        "source_exif": source_exif,
        "cache": cache,
        "processed": processed,
        "lens_calls": lens_calls,
    }


def _install_counting_export_raw(monkeypatch, camera: np.ndarray, mosaic_shape=None):
    opens = []
    postprocess_calls = []
    mosaic = np.full(mosaic_shape or camera.shape[:2], 1200, dtype=np.uint16)

    class FakeRaw:
        raw_image_visible = mosaic
        raw_pattern = np.array([[0, 1], [3, 2]], dtype=np.uint8)
        color_desc = b"RGBG"
        rgb_xyz_matrix = np.eye(4, 3)
        camera_whitebalance = [1.4, 1.0, 1.6, 1.0]
        black_level_per_channel = [100, 100, 100, 100]
        white_level = 16383

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def postprocess(self, **_kwargs):
            postprocess_calls.append(True)
            return camera

    def open_raw(_path):
        opens.append(True)
        return FakeRaw()

    monkeypatch.setattr("image_io.rawpy.imread", open_raw)
    return opens, postprocess_calls, mosaic


def test_fit_srgb_to_camera_keeps_the_existing_broad_candidate_first(monkeypatch):
    rng = np.random.default_rng(7201)
    source = rng.uniform(0.04, 0.23, (500, 3))
    camera_to_srgb = np.array(
        [[0.91, 0.03, 0.01], [0.02, 0.88, 0.04], [0.01, 0.02, 0.90]],
    )
    reference = source @ camera_to_srgb
    camera, rendered = _pack_fit_samples(source, reference)
    original_lstsq = np.linalg.lstsq
    fit_counts = []

    def tracked_lstsq(matrix, values, **kwargs):
        fit_counts.append(matrix.shape[0])
        return original_lstsq(matrix, values, **kwargs)

    monkeypatch.setattr(np.linalg, "lstsq", tracked_lstsq)

    fitted = _fit_srgb_to_camera(camera, rendered)

    np.testing.assert_allclose(fitted, np.linalg.inv(camera_to_srgb), atol=2e-4, rtol=2e-4)
    assert fit_counts == [500]


def test_fit_srgb_to_camera_uses_narrower_midtone_after_existing_ranges_fail():
    rng = np.random.default_rng(7202)
    low = rng.uniform(0.04, 0.23, (300, 3))
    high = rng.uniform(0.26, 0.29, (200, 3))
    source = np.concatenate((low, high))
    camera_to_srgb = np.array(
        [[0.82, 0.04, 0.02], [0.03, 0.86, 0.03], [0.02, 0.03, 0.84]],
    )
    reference = source @ camera_to_srgb
    reference[300:] += 0.025
    camera, rendered = _pack_fit_samples(source, reference)

    fitted = _fit_srgb_to_camera(camera, rendered)

    np.testing.assert_allclose(fitted, np.linalg.inv(camera_to_srgb), atol=3e-4, rtol=3e-4)


@pytest.mark.parametrize("sample_count", (99, 400))
def test_fit_srgb_to_camera_still_rejects_insufficient_or_distorted_samples(sample_count):
    rng = np.random.default_rng(7203)
    source = rng.uniform(0.04, 0.07, (sample_count, 3))
    camera_to_srgb = np.array(
        [[0.88, 0.03, 0.01], [0.02, 0.9, 0.02], [0.01, 0.03, 0.87]],
    )
    reference = source @ camera_to_srgb
    if sample_count >= 100:
        reference += rng.uniform(-0.012, 0.012, reference.shape)
    camera, rendered = _pack_fit_samples(source, reference)

    with pytest.raises(EnhancedDNGColorError, match="camera color transform does not fit"):
        _fit_srgb_to_camera(camera, rendered)


def test_raw_preview_uses_the_export_linear_exposure_scale(monkeypatch):
    calls = []

    class FakeRaw:
        camera_whitebalance = None

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def postprocess(self, **kwargs):
            calls.append(kwargs)
            return np.full((4, 6, 3), 32768, dtype=np.uint16)

        def extract_thumb(self):
            raise RuntimeError("no thumbnail")

    monkeypatch.setattr("image_io.rawpy.imread", lambda _path: FakeRaw())
    monkeypatch.setattr("image_io._raw_bit_depth", lambda _raw: 14)
    monkeypatch.setattr("image_io._safe_exif", lambda _path: {
        "LensModel": "Test Lens", "LensSpecification": (35, 35, 2, 2),
    })
    monkeypatch.setattr("image_io._raw_metadata", lambda *_args: None)
    monkeypatch.setattr("image_io._fill_lens_specification", lambda _exif: None)

    image, metadata = _read_raw(Path("sample.nef"), preview=True)
    assert image.dtype == np.uint16
    assert metadata.color_space == "Linear sRGB"
    assert calls[0]["half_size"] is True
    assert calls[0]["no_auto_bright"] is True
    assert calls[0]["output_bps"] == 16
    assert calls[0]["gamma"] == (1.0, 1.0)


class _FakeExif(dict):
    def __init__(self, root: dict, children: dict[int, dict]):
        super().__init__(root)
        self._children = children

    def get_ifd(self, tag: int):
        return self._children.get(tag, {})


class _FakeImage:
    def __init__(self, exif: _FakeExif):
        self._exif = exif

    def getexif(self):
        return self._exif


def test_exif_from_pil_walks_exif_and_gps_ifds_but_skips_makernote():
    exif = _FakeExif(
        {
            271: "NIKON CORPORATION",
            272: "NIKON Z 8",
            34665: 100,
            34853: 200,
        },
        {
            34665: {
                33434: (1, 125),
                33437: 2.8,
                34855: 400,
                36867: "2026:09:22 10:11:12",
                37500: b"vendor-private-offsets",
                42036: "NIKKOR Z 50mm f/1.8 S",
            },
            34853: {
                1: "N",
                2: ((22, 1), (32, 1), (123, 10)),
                3: "E",
                4: ((114, 1), (3, 1), (456, 10)),
            },
        },
    )

    metadata = _exif_from_pil(_FakeImage(exif))

    assert metadata["Make"] == "NIKON CORPORATION"
    assert metadata["Model"] == "NIKON Z 8"
    assert metadata["ExposureTime"] == (1, 125)
    assert metadata["FNumber"] == 2.8
    assert metadata["ISOSpeedRatings"] == 400
    assert metadata["DateTimeOriginal"] == "2026:09:22 10:11:12"
    assert metadata["LensModel"] == "NIKKOR Z 50mm f/1.8 S"
    assert metadata["GPSLatitudeRef"] == "N"
    assert metadata["GPSLatitude"] == ((22, 1), (32, 1), (123, 10))
    assert "MakerNote" not in metadata


def test_raw_metadata_fills_standard_fields_without_overwriting_embedded_exif():
    raw = SimpleNamespace(
        other=SimpleNamespace(
            iso_speed=800,
            shutter_speed=1 / 60,
            aperture=4.0,
            focal_length=85.0,
            timestamp=datetime(2026, 9, 22, 8, 9, 10),
            artist="Photographer",
        ),
        lens=SimpleNamespace(
            make="NIKON",
            model="NIKKOR Z 85mm f/1.8 S",
        ),
        daylight_whitebalance=[2.0, 1.0, 1.5, 1.0],
        auto_whitebalance=None,
    )
    metadata = {"ISO": 200, "LensModel": "Embedded lens name"}

    _raw_metadata(raw, metadata)

    assert metadata["ISO"] == 200
    assert metadata["LensModel"] == "Embedded lens name"
    assert metadata["ExposureTime"] == 1 / 60
    assert metadata["FNumber"] == 4.0
    assert metadata["FocalLength"] == 85.0
    assert metadata["DateTimeOriginal"] == "2026:09:22 08:09:10"
    assert metadata["DateTimeDigitized"] == "2026:09:22 08:09:10"
    assert metadata["Artist"] == "Photographer"
    assert metadata["LensMake"] == "NIKON"
    assert metadata["DaylightWhiteBalance"] == [2.0, 1.0, 1.5, 1.0]


def test_raw_bit_depth_uses_sensor_white_level_and_supported_widths():
    assert _raw_bit_depth(SimpleNamespace(white_level=4095, camera_white_level_per_channel=None)) == 12
    assert _raw_bit_depth(SimpleNamespace(white_level=15520, camera_white_level_per_channel=None)) == 14
    assert _raw_bit_depth(SimpleNamespace(white_level=65535, camera_white_level_per_channel=None)) == 16


def test_private_lens_id_is_mapped_to_portable_lens_fields():
    metadata = _lens_metadata_from_exiftool_record(
        {
            "Make": "NIKON CORPORATION",
            "LensID": "Nikkor Z 14-24mm f/2.8 S",
            "Lens": "14-24mm f/2.8",
            "LensSerialNumber": "12345678",
        }
    )

    assert metadata == {
        "LensModel": "Nikkor Z 14-24mm f/2.8 S",
        "LensMake": "Nikon",
        "LensSerialNumber": "12345678",
        "LensSpecification": (14.0, 24.0, 2.8, 2.8),
    }


def test_lens_name_supplies_missing_lens_specification():
    assert _parse_lens_specification("NIKKOR Z 14-24mm f/2.8 S") == (14.0, 24.0, 2.8, 2.8)
    assert _parse_lens_specification("NIKKOR Z 24-200mm f/4-6.3 VR") == (24.0, 200.0, 4.0, 6.3)
    metadata = {"LensModel": "NIKKOR Z 50mm f/1.8 S"}
    _fill_lens_specification(metadata)
    assert metadata["LensSpecification"] == (50.0, 50.0, 1.8, 1.8)


def test_camera_profile_names_reads_only_whitelisted_names(monkeypatch):
    monkeypatch.setattr("image_io._find_exiftool", lambda: "exiftool")
    monkeypatch.setattr(
        "image_io.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(
            stdout='[{"CameraProfile":"Camera Flexible Color","PictureControlName":"Negative_Cy_01a","SerialNumber":"secret"}]'
        ),
    )
    assert camera_profile_names("sample.nef") == {
        "SourceCameraProfileName": "Camera Flexible Color",
        "NikonPictureControlName": "Negative_Cy_01a",
    }


def test_enhanced_source_data_keeps_cfa_and_rotates_rgb_to_sensor_orientation(monkeypatch):
    rng = np.random.default_rng(34)
    sensor_rgb = rng.integers(8000, 26000, (256, 320, 3), dtype=np.uint16)
    reference = np.ascontiguousarray(np.rot90(sensor_rgb))
    mosaic = np.full((256, 320), 1200, dtype=np.uint16)

    class FakeRaw:
        raw_image_visible = mosaic
        raw_pattern = np.array([[0, 1], [3, 2]], dtype=np.uint8)
        color_desc = b"RGBG"
        rgb_xyz_matrix = np.array([[1.0, 0.1, 0], [0, 1.0, 0.1], [0.1, 0, 1.0], [0, 0, 0]])
        camera_whitebalance = [2.0, 1.0, 1.25, 1.0]
        black_level_per_channel = [100, 100, 100, 100]
        white_level = 16383

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def postprocess(self, *, output_color, **_kwargs):
            assert output_color == rawpy.ColorSpace.raw
            return reference

    monkeypatch.setattr("image_io.rawpy.imread", lambda _path: FakeRaw())
    enhanced, cfa, pattern, profile, orientation = enhanced_dng_source_data(
        reference, reference, "sample.nef", {"Make": "Nikon", "Model": "Z6_3", "Orientation": 8},
    )
    assert enhanced.shape == cfa.shape + (3,)
    assert np.array_equal(enhanced, sensor_rgb)
    assert pattern.tolist() == [[0, 1], [1, 2]]
    assert profile["DNGUniqueCameraModel"] == "Nikon Z 6 3"
    assert profile["DNGBlackLevel"] == (100, 100, 100, 100)
    assert profile["DNGDefaultCropOrigin"] == (0, 0)
    assert profile["DNGDefaultCropSize"] == (320, 256)
    assert orientation == 8


@pytest.mark.parametrize(
    ("left", "top", "expected_size"),
    ((0, 0, (256, 256)), (5, 7, (251, 249))),
)
def test_enhanced_source_data_fills_unspecified_crop_from_visible_cfa(
    monkeypatch, left: int, top: int, expected_size: tuple[int, int],
):
    camera = np.random.default_rng(7210).integers(
        5000, 15000, (256, 256, 3), dtype=np.uint16,
    )
    sizes = SimpleNamespace(
        crop_left_margin=left, crop_top_margin=top,
        crop_width=0, crop_height=0,
    )
    _install_enhanced_fake_raw(monkeypatch, camera, sizes=sizes)

    _enhanced, _cfa, _pattern, profile, _orientation = enhanced_dng_source_data(
        camera.copy(), camera, "unspecified-crop.nef", {"Make": "Nikon", "Model": "Z6_3"},
    )

    assert profile["DNGDefaultCropOrigin"] == (left, top)
    assert profile["DNGDefaultCropSize"] == expected_size


@pytest.mark.parametrize(
    ("left", "top", "width", "height"),
    ((0, 0, -1, 256), (250, 0, 10, 256), (-1, 0, 0, 256)),
)
def test_enhanced_source_data_rejects_invalid_crop_even_when_a_dimension_is_zero(
    monkeypatch, left: int, top: int, width: int, height: int,
):
    camera = np.random.default_rng(7211).integers(
        5000, 15000, (256, 256, 3), dtype=np.uint16,
    )
    sizes = SimpleNamespace(
        crop_left_margin=left, crop_top_margin=top,
        crop_width=width, crop_height=height,
    )
    _install_enhanced_fake_raw(monkeypatch, camera, sizes=sizes)

    with pytest.raises(EnhancedDNGColorError, match="invalid RAW crop"):
        enhanced_dng_source_data(
            camera.copy(), camera, "invalid-crop.nef", {"Make": "Nikon", "Model": "Z6_3"},
        )


def test_enhanced_source_data_identity_preserves_camera_when_highlights_are_nonlinear(monkeypatch):
    rng = np.random.default_rng(8707)
    camera = rng.integers(5000, 18000, (256, 256, 3), dtype=np.uint16)
    camera[128:] = rng.integers(40000, 55000, (128, 256, 3), dtype=np.uint16)
    reference = camera.copy()
    bright = camera.max(axis=2) > 22000
    reference[bright, 0] = np.minimum(
        reference[bright, 0].astype(np.uint32) + 5000, 65535,
    ).astype(np.uint16)

    _install_enhanced_fake_raw(monkeypatch, camera)
    enhanced, _, _, _, _ = enhanced_dng_source_data(
        reference, reference, "bright.nef", {"Make": "Nikon", "Model": "Z6_3"},
    )
    np.testing.assert_array_equal(enhanced, camera)


def test_enhanced_source_data_scalar_darkening_preserves_native_color(monkeypatch):
    rng = np.random.default_rng(641)
    camera = rng.integers(5000, 16000, (256, 256, 3), dtype=np.uint16)
    camera_to_srgb = np.array(
        [[0.82, 0.04, 0.02], [0.03, 0.91, 0.01], [0.01, 0.02, 0.88]],
        dtype=np.float64,
    )
    reference = np.rint(camera.astype(np.float64) @ camera_to_srgb).astype(np.uint16)
    processed = np.rint(reference.astype(np.float64) * 0.42).astype(np.uint16)
    _install_enhanced_fake_raw(monkeypatch, camera)

    enhanced, *_ = enhanced_dng_source_data(
        processed, reference, "scalar.nef", {"Make": "Nikon", "Model": "Z6_3"},
    )

    np.testing.assert_allclose(enhanced, camera.astype(np.float32) * 0.42, atol=3.0, rtol=0.0002)


def test_enhanced_source_data_maps_color_residual_and_suppresses_black_noise(monkeypatch):
    import native_dense
    from native_renderer import NativeRendererError

    def unavailable_native(*_args):
        raise NativeRendererError("native transfer unavailable")

    monkeypatch.setattr(native_dense, "camera_profile_transfer", unavailable_native)
    rng = np.random.default_rng(947)
    camera = rng.integers(5000, 15000, (256, 256, 3), dtype=np.uint16)
    camera_to_srgb = np.array(
        [[0.92, 0.05, 0.01], [0.02, 0.88, 0.04], [0.03, 0.01, 0.90]],
        dtype=np.float64,
    )
    reference = np.rint(camera.astype(np.float64) @ camera_to_srgb).astype(np.uint16)
    processed = reference.copy()
    processed[100:110, 100:110, 0] += np.uint16(1400)
    processed[100:110, 100:110, 2] -= np.uint16(300)
    # A color residual at a black reference pixel is undefined and must not
    # create a colored pixel in the camera-native result.
    camera[0, 0] = (900, 700, 500)
    reference[0, 0] = (0, 0, 0)
    processed[0, 0] = (600, 0, 0)
    _install_enhanced_fake_raw(monkeypatch, camera)
    camera_before = camera.copy()
    reference_before = reference.copy()
    processed_before = processed.copy()

    enhanced, *_ = enhanced_dng_source_data(
        processed, reference, "residual.nef", {"Make": "Nikon", "Model": "Z6_3"},
    )

    y = np.asarray((0.2126, 0.7152, 0.0722))
    p = processed[105, 105].astype(np.float64)
    r = reference[105, 105].astype(np.float64)
    c = camera[105, 105].astype(np.float64)
    ratio = float(p @ y / (r @ y))
    residual = p - r * ratio
    expected = c * ratio + residual @ np.linalg.inv(camera_to_srgb)
    np.testing.assert_allclose(enhanced[105, 105], expected, atol=4.0, rtol=0.0002)
    np.testing.assert_array_equal(enhanced[0, 0], np.zeros(3, dtype=np.uint16))
    np.testing.assert_array_equal(camera, camera_before)
    np.testing.assert_array_equal(reference, reference_before)
    np.testing.assert_array_equal(processed, processed_before)


def test_enhanced_source_data_corrects_camera_and_reference_in_same_geometry(monkeypatch):
    import lens_correction

    rng = np.random.default_rng(111)
    camera = rng.integers(5000, 18000, (256, 256, 3), dtype=np.uint16)
    reference = camera.copy()
    processed = np.ascontiguousarray(np.flip(reference, axis=1))
    calls = []

    def transform(image, _metadata, *, require_correction):
        assert require_correction is True
        calls.append(image.copy())
        result = SimpleNamespace(
            applied=True, camera_name="Camera", lens_name="Lens",
            distortion_applied=True, tca_applied=False, vignetting_applied=False,
        )
        return np.ascontiguousarray(np.flip(image, axis=1)), result

    monkeypatch.setattr(lens_correction, "apply_lens_correction", transform)
    _install_enhanced_fake_raw(monkeypatch, camera)
    source_exif = {
        "Make": "Nikon", "Model": "Z6_3", "LensCorrectionApplied": True,
        "LensCorrectionEngine": "Lensfun/lensfunpy", "LensCorrectionCamera": "Camera",
        "LensCorrectionLens": "Lens", "LensCorrectionOperations": "distortion",
    }
    original_camera = camera.copy()
    original_reference = reference.copy()

    enhanced, *_ = enhanced_dng_source_data(
        processed, reference, "lens.nef", source_exif,
    )

    np.testing.assert_array_equal(enhanced, processed)
    assert len(calls) == 2
    np.testing.assert_array_equal(calls[0], original_camera)
    np.testing.assert_array_equal(calls[1], original_reference)
    np.testing.assert_array_equal(camera, original_camera)
    np.testing.assert_array_equal(reference, original_reference)


def test_enhanced_source_cache_hit_reuses_corrected_rgb_but_reads_cfa_again(monkeypatch, tmp_path: Path):
    fixture = _make_source_cache_fixture(monkeypatch, tmp_path)
    fixture["source_exif"]["Orientation"] = 8
    opens, postprocess_calls, mosaic = _install_counting_export_raw(
        monkeypatch, fixture["camera"], mosaic_shape=(256, 320),
    )
    full_result = enhanced_dng_source_data(
        fixture["processed"], fixture["reference"], fixture["source_path"],
        fixture["source_exif"],
        correction_version="lens-v1",
    )
    assert len(postprocess_calls) == 1
    assert len(fixture["lens_calls"]) == 2

    # Make any attempt to reconstruct or re-correct RGB fail. A cache hit must
    # still open the source and copy its current visible CFA mosaic.
    postprocess_calls.clear()
    fixture["lens_calls"].clear()

    def unexpected_postprocess(**_kwargs):
        raise AssertionError("cache hit should skip raw.postprocess")

    import lens_correction

    monkeypatch.setattr(
        lens_correction, "apply_lens_correction",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("cache hit should skip Lensfun")
        ),
    )
    original_raw = rawpy.imread

    class CacheHitRaw:
        raw_image_visible = mosaic
        raw_pattern = np.array([[0, 1], [3, 2]], dtype=np.uint8)
        color_desc = b"RGBG"
        rgb_xyz_matrix = np.eye(4, 3)
        camera_whitebalance = [1.4, 1.0, 1.6, 1.0]
        black_level_per_channel = [100, 100, 100, 100]
        white_level = 16383

        def __enter__(self):
            opens.append(True)
            return self

        def __exit__(self, *_args):
            return False

        def postprocess(self, **kwargs):
            return unexpected_postprocess(**kwargs)

    monkeypatch.setattr("image_io.rawpy.imread", lambda _path: CacheHitRaw())
    diagnostics = {}
    cached_result = enhanced_dng_source_data(
        fixture["processed"], fixture["reference"], fixture["source_path"],
        fixture["source_exif"], source_cache=fixture["cache"],
        correction_version="lens-v1", cache_diagnostics=diagnostics,
    )

    assert diagnostics == {"camera": "hit"}
    assert len(opens) == 2
    assert postprocess_calls == []
    assert np.array_equal(cached_result[1], mosaic)
    for full, cached in zip(full_result, cached_result):
        if isinstance(full, np.ndarray):
            np.testing.assert_array_equal(cached, full)
        else:
            assert cached == full
    assert fixture["cache"].camera_rgb16 is fixture["corrected_camera"]
    assert fixture["cache"].reference_rgb16 is fixture["corrected_reference"]
    assert not fixture["cache"].camera_rgb16.flags.writeable
    assert not fixture["cache"].reference_rgb16.flags.writeable
    assert not fixture["cache"].srgb_to_camera.flags.writeable
    assert original_raw is not None


@pytest.mark.parametrize(
    "invalidator",
    ("low_resolution", "rgb8", "mutable", "version", "identity", "during_read", "changed_file", "metadata", "gain", "calibration"),
)
def test_invalid_enhanced_source_cache_silently_falls_back(
    monkeypatch, tmp_path: Path, invalidator: str,
):
    fixture = _make_source_cache_fixture(monkeypatch, tmp_path)
    cache = fixture["cache"]
    if invalidator == "low_resolution":
        cache = replace(
            cache,
            source_shape=(128, 128, 3),
            camera_rgb16=np.ascontiguousarray(cache.camera_rgb16[::2, ::2]),
            reference_rgb16=np.ascontiguousarray(cache.reference_rgb16[::2, ::2]),
        )
        cache.camera_rgb16.flags.writeable = False
        cache.reference_rgb16.flags.writeable = False
    elif invalidator == "rgb8":
        cache = replace(
            cache,
            camera_rgb16=cache.camera_rgb16.astype(np.uint8),
        )
        cache.camera_rgb16.flags.writeable = False
    elif invalidator == "mutable":
        cache.camera_rgb16.flags.writeable = True
    elif invalidator == "identity":
        monkeypatch.setattr(
            "image_io.enhanced_dng_source_identity",
            lambda _path: (*cache.source_identity[:-1], cache.source_identity[-1] + 1),
        )
    elif invalidator == "during_read":
        identity_after_read = (*cache.source_identity[:-1], cache.source_identity[-1] + 1)
        identities = iter((cache.source_identity, identity_after_read))
        monkeypatch.setattr(
            "image_io.enhanced_dng_source_identity",
            lambda _path: next(identities),
        )
    elif invalidator == "changed_file":
        fixture["source_path"].write_bytes(b"source file changed after cache construction")
    elif invalidator == "metadata":
        fixture["source_exif"]["LensCorrectionOperations"] = "tca"
    elif invalidator == "gain":
        import dng_gainmap

        fixture["source_exif"]["DNGGainMapApplied"] = True
        monkeypatch.setattr(
            dng_gainmap, "apply_dng_gain_map",
            lambda image, _path, **_kwargs: (image.copy(), True),
        )
    elif invalidator == "calibration":
        import lens_correction

        monkeypatch.setattr(
            lens_correction, "_calibration_signature",
            lambda: (("lensfun.xml", 2, 20),),
        )

    opens, postprocess_calls, _mosaic = _install_counting_export_raw(
        monkeypatch, fixture["camera"],
    )
    diagnostics = {}
    if invalidator == "metadata":
        with pytest.raises(EnhancedDNGColorError, match="lens correction metadata"):
            enhanced_dng_source_data(
                fixture["processed"], fixture["reference"], fixture["source_path"],
                fixture["source_exif"], source_cache=cache,
                correction_version="lens-v1", cache_diagnostics=diagnostics,
            )
    else:
        enhanced_dng_source_data(
            fixture["processed"], fixture["reference"], fixture["source_path"],
            fixture["source_exif"], source_cache=cache,
            correction_version="other-version" if invalidator == "version" else "lens-v1",
            cache_diagnostics=diagnostics,
        )
    assert diagnostics == {"camera": "miss"}
    assert len(opens) == 1
    assert len(postprocess_calls) == 1
    assert len(fixture["lens_calls"]) == 2


def test_enhanced_source_cache_builder_rejects_gain_and_vignette_only_results(tmp_path: Path):
    camera = np.random.default_rng(861).integers(
        5000, 16000, (256, 256, 3), dtype=np.uint16,
    )
    reference = camera.copy()
    result = SimpleNamespace(
        applied=True, camera_name="Camera", lens_name="Lens",
        distortion_applied=False, tca_applied=False, vignetting_applied=True,
    )
    path = tmp_path / "gain.nef"
    path.write_bytes(b"source")
    assert build_enhanced_dng_source_cache(
        path, camera, reference, camera.copy(), reference.copy(), result, result,
        correction_version="lens-v1", camera_gain_applied=True,
    ) is None
    assert build_enhanced_dng_source_cache(
        path, camera, reference, camera.copy(), reference.copy(), result, result,
        correction_version="lens-v1",
    ) is None


def test_enhanced_source_data_applies_dng_gain_map_to_camera_and_reference(monkeypatch):
    import dng_gainmap

    camera = np.random.default_rng(29).integers(5000, 18000, (256, 256, 3), dtype=np.uint16)
    reference = camera.copy()
    gain = np.linspace(0.5, 1.5, camera.shape[1], dtype=np.float32)[None, :, None]
    processed = np.clip(np.rint(reference.astype(np.float32) * gain), 0, 65535).astype(np.uint16)
    calls = []

    def apply_gain_map(image, source_path):
        assert source_path == "gain.dng"
        calls.append(image.copy())
        corrected = np.clip(np.rint(image.astype(np.float32) * gain), 0, 65535).astype(np.uint16)
        return corrected, True

    monkeypatch.setattr(dng_gainmap, "apply_dng_gain_map", apply_gain_map)
    _install_enhanced_fake_raw(monkeypatch, camera)
    original_camera = camera.copy()
    original_reference = reference.copy()

    enhanced, *_ = enhanced_dng_source_data(
        processed, reference, "gain.dng",
        {"Make": "Nikon", "Model": "Z6_3", "DNGGainMapApplied": True},
    )

    np.testing.assert_array_equal(enhanced, processed)
    assert len(calls) == 2
    np.testing.assert_array_equal(calls[0], original_camera)
    np.testing.assert_array_equal(calls[1], original_reference)
    np.testing.assert_array_equal(camera, original_camera)
    np.testing.assert_array_equal(reference, original_reference)


def test_enhanced_gainmap_shoulder_uses_working_rgb_reference_for_camera_layer(monkeypatch):
    import dng_gainmap

    camera = np.random.default_rng(31).integers(5000, 18000, (256, 256, 3), dtype=np.uint16)
    reference = np.rint(camera.astype(np.float32) * 1.25).astype(np.uint16)
    calls = []

    def apply_gain(image, path, *, preserve_highlights, highlight_reference):
        assert preserve_highlights is True
        np.testing.assert_array_equal(highlight_reference, reference)
        calls.append(image.copy())
        return image.copy(), True

    monkeypatch.setattr(dng_gainmap, "apply_dng_gain_map", apply_gain)
    _install_enhanced_fake_raw(monkeypatch, camera)
    enhanced, *_ = enhanced_dng_source_data(
        reference.copy(), reference, "protected.dng",
        {"Make": "Nikon", "Model": "Z6_3", "DNGGainMapApplied": True,
         "DNGGainMapHighlightProtection": True},
    )
    assert len(calls) == 2
    np.testing.assert_array_equal(calls[0], camera)
    np.testing.assert_array_equal(calls[1], reference)
    np.testing.assert_array_equal(enhanced, camera)


def test_enhanced_source_data_rejects_missing_d65_fallback(monkeypatch):
    camera = np.random.default_rng(38).integers(5000, 18000, (256, 256, 3), dtype=np.uint16)
    _install_enhanced_fake_raw(monkeypatch, camera, matrix=np.zeros((4, 3)))
    monkeypatch.setattr("image_io._find_exiftool", lambda: "exiftool")
    monkeypatch.setattr(
        "image_io.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(
            stdout='[{"ColorMatrix2":"1 0 0 0 1 0 0 0 1",'
                   '"CalibrationIlluminant2":17,"AsShotNeutral":"0.5 1 0.8",'
                   '"UniqueCameraModel":"Test DNG"}]'
        ),
    )

    with np.testing.assert_raises_regex(EnhancedDNGColorError, "camera color matrix unavailable"):
        enhanced_dng_source_data(
            camera, camera, "no-d65.dng", {"Make": "Nikon", "Model": "Z6_3"},
        )


def test_enhanced_source_data_uses_dng_d65_profile_string_fields(monkeypatch):
    camera = np.random.default_rng(61).integers(5000, 18000, (256, 256, 3), dtype=np.uint16)
    _install_enhanced_fake_raw(monkeypatch, camera, matrix=np.zeros((4, 3)))
    monkeypatch.setattr("image_io._find_exiftool", lambda: "exiftool")
    calls = []

    def fake_exiftool(arguments, **kwargs):
        calls.append(arguments)
        return SimpleNamespace(stdout='[{"ColorMatrix1":"1 0 0 0 1 0 0 0 1",'
            '"CalibrationIlluminant1":17,'
            '"ColorMatrix2":"0.8575 -0.3219 -0.0868 -0.3351 1.1451 0.1593 0.0207 0.0468 0.4876",'
            '"CalibrationIlluminant2":21,"AsShotNeutral":"0.4466252317 1 0.5235843027",'
            '"UniqueCameraModel":"Hasselblad L2D-20c"}]')

    monkeypatch.setattr("image_io.subprocess.run", fake_exiftool)
    enhanced, _, _, profile, _ = enhanced_dng_source_data(
        camera, camera, "profile.dng", {},
    )

    np.testing.assert_array_equal(enhanced, camera)
    np.testing.assert_allclose(
        profile["DNGColorMatrix1"],
        (0.8575, -0.3219, -0.0868, -0.3351, 1.1451, 0.1593, 0.0207, 0.0468, 0.4876),
    )
    assert profile["DNGAsShotNeutral"] == (0.4466252317, 1.0, 0.5235843027)
    assert profile["DNGUniqueCameraModel"] == "Hasselblad L2D-20c"
    assert "-json" in calls[0] and "-n" in calls[0]
    assert "-ColorMatrix1" in calls[0] and "-ColorMatrix2" in calls[0]


def test_profile_reference_requires_identical_picture_control_bytes(monkeypatch, tmp_path: Path):
    source = tmp_path / "source.nef"
    reference = tmp_path / "reference.dng"
    source.touch()
    reference.touch()
    controls = {str(source): b"nikon-flexible-color", str(reference): b"nikon-flexible-color"}
    curves = {str(source): b"tone-curve", str(reference): b"tone-curve"}

    def fake_run(arguments, **_kwargs):
        path = arguments[-1]
        if "-json" in arguments:
            start = arguments.index("-ProfileEmbedPolicy") + 1
            records = [
                {
                    "SourceFile": source_file,
                    "Make": "NIKON CORPORATION",
                    "Model": "NIKON Z6_3",
                    "ProfileName": "Camera Flexible Color",
                    "ProfileEmbedPolicy": 0,
                }
                for source_file in arguments[start:]
            ]
            return SimpleNamespace(stdout=json.dumps(records))
        if "-Nikon:ContrastCurve" in arguments:
            return SimpleNamespace(stdout=curves[path])
        return SimpleNamespace(stdout=controls[path])

    monkeypatch.setattr("image_io._find_exiftool", lambda: "exiftool")
    monkeypatch.setattr("image_io.subprocess.run", fake_run)
    info = {
        "Make": "NIKON CORPORATION", "Model": "NIKON Z6_3",
        "SourceCameraProfileName": "Camera Flexible Color",
    }
    assert matching_embedded_profile_dng(source, info) == reference
    curves[str(reference)] = b"different tone curve"
    reference.write_bytes(b"candidate changed to invalidate cached metadata")
    assert matching_embedded_profile_dng(source, info) is None
    curves[str(reference)] = curves[str(source)]
    controls[str(reference)] = b"a different camera recipe"
    reference.write_bytes(b"candidate changed again to invalidate cached metadata")
    assert matching_embedded_profile_dng(source, info) is None


def test_enhanced_source_data_uses_fitted_matrix_for_native_transfer(monkeypatch):
    import native_dense

    rng = np.random.default_rng(539)
    camera = rng.integers(5000, 16000, (256, 256, 3), dtype=np.uint16)
    camera_to_srgb = np.array(
        [[0.9, 0.04, 0.02], [0.03, 0.88, 0.02], [0.01, 0.03, 0.92]],
        dtype=np.float64,
    )
    reference = np.rint(camera.astype(np.float64) @ camera_to_srgb).astype(np.uint16)
    processed = np.clip(
        np.rint(reference.astype(np.float32) * 0.8 + (120.0, -45.0, 20.0)),
        0, 65535,
    ).astype(np.uint16)
    _install_enhanced_fake_raw(monkeypatch, camera)
    sentinel = np.full_like(camera, 23456)
    calls = []

    def fake_transfer(camera_base, reference_base, processed_rgb16, srgb_to_camera):
        calls.append((camera_base.copy(), reference_base.copy(), processed_rgb16.copy(), srgb_to_camera.copy()))
        return sentinel.copy()

    monkeypatch.setattr(native_dense, "camera_profile_transfer", fake_transfer)
    enhanced, *_ = enhanced_dng_source_data(
        processed, reference, "native.nef", {"Make": "Nikon", "Model": "Z6_3"},
    )

    np.testing.assert_array_equal(enhanced, sentinel)
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0], camera)
    np.testing.assert_array_equal(calls[0][1], reference)
    np.testing.assert_array_equal(calls[0][2], processed)
    np.testing.assert_allclose(calls[0][3], np.linalg.inv(camera_to_srgb), atol=2e-4, rtol=2e-4)


def test_native_camera_profile_transfer_matches_numpy_reference():
    import native_dense
    from native_renderer import NativeRendererError

    rng = np.random.default_rng(1053)
    camera = rng.integers(500, 35000, (41, 67, 3), dtype=np.uint16)
    reference = np.clip(
        np.rint(camera.astype(np.float32) @ np.array(
            [[0.91, 0.04, 0.01], [0.02, 0.9, 0.03], [0.01, 0.02, 0.89]],
            dtype=np.float32,
        )),
        0, 65535,
    ).astype(np.uint16)
    processed = np.clip(
        np.rint(reference.astype(np.float32) * np.array((0.9, 1.02, 0.94), dtype=np.float32)
                 + np.array((150.0, -80.0, 35.0), dtype=np.float32)),
        0, 65535,
    ).astype(np.uint16)
    srgb_to_camera = np.array(
        [[1.1, -0.04, -0.01], [-0.03, 1.12, -0.04], [-0.01, -0.03, 1.13]],
        dtype=np.float32,
    )
    expected = _camera_profile_transfer_numpy(camera, reference, processed, srgb_to_camera)
    try:
        actual = native_dense.camera_profile_transfer(
            camera, reference, processed, srgb_to_camera,
        )
    except NativeRendererError as exc:
        pytest.skip(f"native dense library unavailable: {exc}")

    np.testing.assert_array_equal(actual, expected)


def _profile_cache_test_setup(tmp_path: Path):
    source = tmp_path / "source.nef"
    candidate = tmp_path / "reference.dng"
    candidate.write_bytes(b"matching recipe")
    info = {
        "Make": "NIKON CORPORATION", "Model": "NIKON Z6_3",
        "SourceCameraProfileName": "Camera Flexible Color",
    }
    calls = []

    def fake_run(arguments, **_kwargs):
        calls.append(tuple(arguments))
        path = Path(arguments[-1])
        if "-json" in arguments:
            start = arguments.index("-ProfileEmbedPolicy") + 1
            records = [
                {
                    "SourceFile": source_file,
                    "Make": "NIKON CORPORATION",
                    "Model": "NIKON Z6_3",
                    "ProfileName": "Camera Flexible Color",
                    "ProfileEmbedPolicy": 0,
                }
                for source_file in arguments[start:]
            ]
            return SimpleNamespace(stdout=json.dumps(records))
        if "-Nikon:ContrastCurve" in arguments:
            return SimpleNamespace(stdout=b"tone curve")
        return SimpleNamespace(stdout=path.read_bytes())

    return source, candidate, info, calls, fake_run


def test_profile_reference_cache_hits_and_invalidates_file_changes(monkeypatch, tmp_path: Path):
    source, candidate, info, calls, fake_run = _profile_cache_test_setup(tmp_path)
    monkeypatch.setattr("image_io._find_exiftool", lambda: "/test/exiftool")
    monkeypatch.setattr("image_io.subprocess.run", fake_run)

    # A source added after a failed lookup must be visible, and a repeated
    # lookup with the same source and candidate identities must use the cache.
    assert matching_embedded_profile_dng(source, info) is None
    source.write_bytes(b"matching recipe")
    assert matching_embedded_profile_dng(source, info) == candidate
    calls_after_miss = len(calls)
    assert matching_embedded_profile_dng(source, info) == candidate
    assert len(calls) == calls_after_miss

    # Source and candidate edits each change the cache key and matching bytes.
    source.write_bytes(b"changed source recipe")
    assert matching_embedded_profile_dng(source, info) is None
    candidate.write_bytes(b"changed source recipe")
    assert matching_embedded_profile_dng(source, info) == candidate

    # Adding and deleting a neighboring candidate changes the full sorted
    # candidate identity list, including for cached positive results.
    added = tmp_path / "aaa_reference.dng"
    added.write_bytes(b"changed source recipe")
    assert matching_embedded_profile_dng(source, info) == added
    added.unlink()
    assert matching_embedded_profile_dng(source, info) == candidate
    candidate.unlink()
    assert matching_embedded_profile_dng(source, info) is None

    # Deleting and recreating the source also invalidates source metadata.
    source.unlink()
    assert matching_embedded_profile_dng(source, info) is None
    source.write_bytes(b"changed source recipe")
    candidate.write_bytes(b"changed source recipe")
    assert matching_embedded_profile_dng(source, info) == candidate


def test_profile_reference_cache_does_not_store_execution_errors(monkeypatch, tmp_path: Path):
    source, candidate, info, calls, fake_run = _profile_cache_test_setup(tmp_path)
    source.write_bytes(b"matching recipe")
    monkeypatch.setattr("image_io._find_exiftool", lambda: "/test/exiftool")
    fail_once = True

    def failing_once(arguments, **kwargs):
        nonlocal fail_once
        if fail_once and "-Nikon:PictureControlData" in arguments:
            fail_once = False
            raise __import__("subprocess").CalledProcessError(1, arguments)
        return fake_run(arguments, **kwargs)

    monkeypatch.setattr("image_io.subprocess.run", failing_once)

    assert matching_embedded_profile_dng(source, info) is None
    assert matching_embedded_profile_dng(source, info) == candidate
    assert calls


def test_profile_reference_reads_metadata_in_bounded_sourcefile_batches(monkeypatch, tmp_path: Path):
    source = tmp_path / "source.nef"
    source.write_bytes(b"matching recipe")
    candidates = [tmp_path / f"reference_{index:03d}.dng" for index in range(65)]
    for candidate in candidates[:-1]:
        candidate.write_bytes(b"different recipe")
    candidates[-1].write_bytes(b"matching recipe")
    info = {
        "Make": "NIKON CORPORATION", "Model": "NIKON Z6_3",
        "SourceCameraProfileName": "Camera Flexible Color",
    }
    batch_sizes = []

    def fake_run(arguments, **_kwargs):
        if "-json" in arguments:
            start = arguments.index("-ProfileEmbedPolicy") + 1
            source_files = arguments[start:]
            batch_sizes.append(len(source_files))
            return SimpleNamespace(stdout=json.dumps([
                {
                    "SourceFile": source_file,
                    "Make": "NIKON CORPORATION",
                    "Model": "NIKON Z6_3",
                    "ProfileName": "Camera Flexible Color",
                    "ProfileEmbedPolicy": 0,
                }
                for source_file in source_files
            ]))
        path = Path(arguments[-1])
        if "-Nikon:ContrastCurve" in arguments:
            return SimpleNamespace(stdout=b"tone curve")
        return SimpleNamespace(stdout=path.read_bytes())

    monkeypatch.setattr("image_io._find_exiftool", lambda: "/test/exiftool")
    monkeypatch.setattr("image_io.subprocess.run", fake_run)

    assert matching_embedded_profile_dng(source, info) == candidates[-1]
    assert batch_sizes == [32, 32, 1]


def test_profile_reference_skips_missing_profile_tag_without_hiding_later_match(
    monkeypatch, tmp_path: Path,
):
    source = tmp_path / "source.nef"
    source.write_bytes(b"matching recipe")
    ordinary = tmp_path / "a_ordinary.dng"
    reference = tmp_path / "z_reference.dng"
    ordinary.write_bytes(b"ordinary recipe")
    reference.write_bytes(b"matching recipe")
    info = {
        "Make": "NIKON CORPORATION", "Model": "NIKON Z6_3",
        "SourceCameraProfileName": "Camera Flexible Color",
    }
    metadata_calls = 0

    def fake_run(arguments, **_kwargs):
        nonlocal metadata_calls
        if "-json" in arguments:
            metadata_calls += 1
            start = arguments.index("-ProfileEmbedPolicy") + 1
            records = []
            for source_file in arguments[start:]:
                record = {"SourceFile": source_file, "Make": "NIKON CORPORATION", "Model": "NIKON Z6_3"}
                if Path(source_file).name == "z_reference.dng":
                    record.update({
                        "ProfileName": "Camera Flexible Color",
                        "ProfileEmbedPolicy": 0,
                    })
                records.append(record)
            records.reverse()
            return SimpleNamespace(stdout=json.dumps(records))
        if "-Nikon:ContrastCurve" in arguments:
            return SimpleNamespace(stdout=b"tone curve")
        return SimpleNamespace(stdout=Path(arguments[-1]).read_bytes())

    monkeypatch.setattr("image_io._find_exiftool", lambda: "/test/exiftool")
    monkeypatch.setattr("image_io.subprocess.run", fake_run)

    assert matching_embedded_profile_dng(source, info) == reference
    # The incomplete ordinary-DNG record makes the result non-cacheable, while
    # its explicit SourceFile still leaves the later exact match usable.
    assert matching_embedded_profile_dng(source, info) == reference
    assert metadata_calls == 2


@pytest.mark.parametrize(
    "bad_response",
    ["missing", "duplicate", "invalid_json", "error", "missing_tag", "record_error"],
)
def test_profile_reference_rejects_bad_batched_metadata_without_caching(
    monkeypatch, tmp_path: Path, bad_response: str,
):
    source, candidate, info, _calls, fake_run = _profile_cache_test_setup(tmp_path)
    source.write_bytes(b"matching recipe")
    metadata_attempts = 0

    def fake_with_bad_first_metadata_batch(arguments, **kwargs):
        nonlocal metadata_attempts
        if "-json" in arguments:
            metadata_attempts += 1
            if metadata_attempts == 1:
                if bad_response == "missing":
                    return SimpleNamespace(stdout="[]")
                if bad_response == "duplicate":
                    start = arguments.index("-ProfileEmbedPolicy") + 1
                    source_file = arguments[start]
                    record = {
                        "SourceFile": source_file,
                        "Make": "NIKON CORPORATION",
                        "Model": "NIKON Z6_3",
                        "ProfileName": "Camera Flexible Color",
                        "ProfileEmbedPolicy": 0,
                    }
                    return SimpleNamespace(stdout=json.dumps([record, record]))
                if bad_response == "invalid_json":
                    return SimpleNamespace(stdout="not-json")
                if bad_response in {"missing_tag", "record_error"}:
                    start = arguments.index("-ProfileEmbedPolicy") + 1
                    source_file = arguments[start]
                    record = {
                        "SourceFile": source_file,
                        "Make": "NIKON CORPORATION",
                        "Model": "NIKON Z6_3",
                        "ProfileName": "Camera Flexible Color",
                        "ProfileEmbedPolicy": 0,
                    }
                    if bad_response == "missing_tag":
                        del record["ProfileName"]
                    else:
                        record["Error"] = "Unable to read file"
                    return SimpleNamespace(stdout=json.dumps([record]))
                raise __import__("subprocess").CalledProcessError(1, arguments)
        return fake_run(arguments, **kwargs)

    monkeypatch.setattr("image_io._find_exiftool", lambda: "/test/exiftool")
    monkeypatch.setattr("image_io.subprocess.run", fake_with_bad_first_metadata_batch)

    assert matching_embedded_profile_dng(source, info) is None
    assert matching_embedded_profile_dng(source, info) == candidate
    assert metadata_attempts == 2


def test_profile_reference_empty_neighbor_cache_avoids_binary_exiftool(monkeypatch, tmp_path: Path):
    source = tmp_path / "source.nef"
    source.write_bytes(b"source")
    info = {
        "Make": "NIKON CORPORATION", "Model": "NIKON Z6_3",
        "SourceCameraProfileName": "Camera Flexible Color",
    }
    calls = []

    def unexpected_exiftool_call(arguments, **_kwargs):
        calls.append(tuple(arguments))
        raise AssertionError("no candidates should require ExifTool")

    monkeypatch.setattr("image_io._find_exiftool", lambda: "/test/exiftool")
    monkeypatch.setattr("image_io.subprocess.run", unexpected_exiftool_call)

    assert matching_embedded_profile_dng(source, info) is None
    assert matching_embedded_profile_dng(source, info) is None
    assert calls == []
