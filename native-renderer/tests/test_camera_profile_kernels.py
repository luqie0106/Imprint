"""Parity checks for the renderer-independent camera profile CPU operators."""
import ctypes as ct
import os
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import camera_profile


_U16 = ct.POINTER(ct.c_uint16)
_U8 = ct.POINTER(ct.c_uint8)
_F32 = ct.POINTER(ct.c_float)


def _ptr(array):
    if array.dtype == np.uint16:
        return array.ctypes.data_as(_U16)
    if array.dtype == np.uint8:
        return array.ctypes.data_as(_U8)
    if array.dtype == np.float32:
        return array.ctypes.data_as(_F32)
    raise TypeError(array.dtype)


@pytest.fixture(scope="module")
def native():
    path = os.environ.get("IMPRINT_NATIVE_RENDERER_LIB")
    if not path:
        pytest.skip("Set IMPRINT_NATIVE_RENDERER_LIB to the freshly built library")
    library = ct.CDLL(path)
    render = library.im_native_camera_profile_render_rgb16_to_rgb8
    render.argtypes = [
        ct.c_uint32, ct.c_uint32, _U16, ct.c_size_t, _F32, ct.c_size_t,
        _F32, ct.c_size_t, ct.c_uint32, ct.c_uint32, ct.c_uint32,
        ct.c_uint32, _F32, ct.c_size_t, _U8, ct.c_size_t,
    ]
    render.restype = ct.c_int
    transfer = library.im_native_camera_profile_transfer_rgb16
    transfer.argtypes = [
        ct.c_uint32, ct.c_uint32, _U16, ct.c_size_t, _U16, ct.c_size_t,
        _U16, ct.c_size_t, _F32, ct.c_size_t, _U16, ct.c_size_t,
    ]
    transfer.restype = ct.c_int
    return render, transfer


def _profile_config(rng):
    identity = np.eye(3, dtype=np.float64)
    look_dims = (8, 7, 9)
    hue, saturation, value = look_dims
    table = np.empty((value, hue, saturation, 3), dtype=np.float32)
    table[..., 0] = rng.uniform(-1.5, 1.5, table.shape[:-1]).astype(np.float32)
    table[..., 1] = rng.uniform(0.88, 1.12, table.shape[:-1]).astype(np.float32)
    table[..., 2] = rng.uniform(0.90, 1.08, table.shape[:-1]).astype(np.float32)
    x = np.linspace(0.0, 1.0, 125, dtype=np.float32)
    curve = np.stack((x, np.power(x.astype(np.float64), 0.91).astype(np.float32)), axis=1)
    profile = camera_profile.Profile(
        name="test", fingerprint=(), color_matrix1=identity, color_matrix2=identity,
        forward_matrix1=identity, forward_matrix2=identity,
        tone_curve=curve, look_table=table, look_dims=look_dims,
        look_encoding=1, baseline_exposure_offset=0.12,
    )
    neutral = np.array([1.0, 0.83, 0.69], dtype=np.float64)
    matrix, white = camera_profile._camera_transform(profile, neutral)
    gain = np.float32(2.0 ** (0.7 + profile.baseline_exposure_offset))
    projection = camera_profile._PROPHOTO_TO_SRGB.astype(np.float32)
    constants = np.concatenate((matrix.reshape(-1), white, [gain], projection.reshape(-1)))
    return profile, neutral, constants.astype(np.float32)


def _render_native(native, image, profile, constants):
    render, _ = native
    h, w = image.shape[:2]
    output = np.empty(image.shape, dtype=np.uint8)
    table = np.ascontiguousarray(profile.look_table)
    curve = np.ascontiguousarray(profile.tone_curve)
    status = render(
        w, h, _ptr(image), image.size, _ptr(constants), constants.size,
        _ptr(table), table.size, *profile.look_dims, profile.look_encoding,
        _ptr(curve), curve.size, _ptr(output), output.size,
    )
    assert status == 0
    return output


def _transfer_reference(camera, reference, processed, inverse):
    output = np.empty_like(camera)
    luma = np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float32)
    for start in range(0, camera.shape[0], 128):
        end = min(start + 128, camera.shape[0])
        p = processed[start:end].astype(np.float32)
        r = reference[start:end].astype(np.float32)
        c = camera[start:end].astype(np.float32)
        pl = np.maximum(p @ luma, 0.0)
        rl = np.maximum(r @ luma, 1.0)
        ratio = pl / rl
        residual = p - r * ratio[:, :, None]
        dark = rl < 1.0
        residual[dark] = 0.0
        ratio[dark] = 0.0
        output[start:end] = np.clip(
            np.rint(c * ratio[:, :, None] + residual @ inverse), 0, 65535,
        ).astype(np.uint16)
    if np.array_equal(reference, processed):
        return camera.copy()
    return output


def _transfer_native(native, camera, reference, processed, inverse):
    _, transfer = native
    output = np.empty_like(camera)
    h, w = camera.shape[:2]
    inverse = np.ascontiguousarray(inverse, dtype=np.float32)
    status = transfer(
        w, h, _ptr(camera), camera.size, _ptr(reference), reference.size,
        _ptr(processed), processed.size, _ptr(inverse), inverse.size,
        _ptr(output), output.size,
    )
    assert status == 0
    return output


def test_render_matches_numpy_reference_on_real_camera_rgb(native):
    fixture = Path(__file__).resolve().parents[2] / "artifacts/acr_tone_dng_20261009/real_photo/working_small.npz"
    if not fixture.is_file():
        pytest.skip("real camera RGB fixture is unavailable")
    with np.load(fixture, allow_pickle=False) as data:
        camera = np.ascontiguousarray(data["enhanced_camera_rgb_u16"])
    saved_camera = camera.copy()
    rng = np.random.default_rng(901)
    profile, neutral, constants = _profile_config(rng)
    expected = camera_profile.render_profile(camera, neutral, profile, 0.7)
    actual = _render_native(native, camera, profile, constants)
    error = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
    assert int(error.max()) <= 1
    assert float(error.mean()) <= 0.01
    np.testing.assert_array_equal(camera, saved_camera)


def test_render_matches_embedded_camera_standard_profile(native):
    root = Path(__file__).resolve().parents[2]
    profile_path = root / "test_images/LCR_0166adobe.DNG"
    fixture = root / "artifacts/acr_tone_dng_20261009/real_photo/working_small.npz"
    if not profile_path.is_file() or not fixture.is_file():
        pytest.skip("real profile or camera RGB fixture is unavailable")
    tags = camera_profile._embedded_profile_tags(profile_path, "Camera Standard")
    profile = camera_profile._parse_profile(tags, "Camera Standard", ())
    neutral = np.array([1.0, 0.83, 0.69], dtype=np.float64)
    matrix, white = camera_profile._camera_transform(profile, neutral)
    exposure_ev = 0.7
    gain = np.float32(2.0 ** (exposure_ev + profile.baseline_exposure_offset))
    projection = camera_profile._PROPHOTO_TO_SRGB.astype(np.float32)
    constants = np.concatenate((matrix.reshape(-1), white, [gain], projection.reshape(-1)))
    constants = np.ascontiguousarray(constants, dtype=np.float32)
    with np.load(fixture, allow_pickle=False) as data:
        camera = np.ascontiguousarray(data["enhanced_camera_rgb_u16"])
    expected = camera_profile.render_profile(camera, neutral, profile, exposure_ev)
    actual = _render_native(native, camera, profile, constants)
    error = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
    assert int(error.max()) <= 1
    assert float(error.mean()) <= 0.01


def test_transfer_matches_numpy_with_real_fixture_and_row_matrix(native):
    fixture = Path(__file__).resolve().parents[2] / "artifacts/acr_tone_dng_20261009/real_photo/working_small.npz"
    if not fixture.is_file():
        pytest.skip("real camera RGB fixture is unavailable")
    with np.load(fixture, allow_pickle=False) as data:
        camera = np.ascontiguousarray(data["enhanced_camera_rgb_u16"])
        reference = np.ascontiguousarray(data["corrected_linear_srgb_u16"])
        processed_srgb = data["reference_srgb_u8"].astype(np.float32) / np.float32(255.0)
    # Feed a non-identity edit into the residual path while retaining real
    # per-pixel camera/reference structure from the recorded comparison.
    processed = np.ascontiguousarray(np.clip(
        np.rint(processed_srgb * 65535.0), 0, 65535,
    ).astype(np.uint16))
    inverse = np.array(
        [[0.91, 0.04, -0.02], [0.07, 1.03, 0.03], [0.02, -0.06, 0.97]],
        dtype=np.float32,
    )
    expected = _transfer_reference(camera, reference, processed, inverse)
    actual = _transfer_native(native, camera, reference, processed, inverse)
    error = np.abs(actual.astype(np.int32) - expected.astype(np.int32))
    assert int(error.max()) <= 1
    assert float(error.mean()) <= 0.01


def test_transfer_rounds_ties_even_and_handles_dark_pixels(native):
    camera = np.array([[[3, 5, 7], [11, 13, 15]]], dtype=np.uint16)
    reference = np.array([[[1000, 1000, 1000], [0, 0, 0]]], dtype=np.uint16)
    processed = np.array([[[500, 500, 500], [300, 200, 100]]], dtype=np.uint16)
    inverse = np.zeros((3, 3), dtype=np.float32)
    actual = _transfer_native(native, camera, reference, processed, inverse)
    np.testing.assert_array_equal(actual, np.array([[[2, 2, 4], [0, 0, 0]]], dtype=np.uint16))


def test_render_rejects_bad_counts_and_nonmonotonic_curve_without_writing(native):
    render, _ = native
    rng = np.random.default_rng(15)
    profile, _, constants = _profile_config(rng)
    image = np.arange(18, dtype=np.uint16).reshape(2, 3, 3)
    table = np.ascontiguousarray(profile.look_table)
    curve = np.ascontiguousarray(profile.tone_curve.copy())
    output = np.full(image.shape, 0xA5, dtype=np.uint8)
    curve[2, 1] = -0.1
    status = render(
        3, 2, _ptr(image), image.size, _ptr(constants), constants.size,
        _ptr(table), table.size, *profile.look_dims, profile.look_encoding,
        _ptr(curve), curve.size, _ptr(output), output.size,
    )
    assert status != 0
    np.testing.assert_array_equal(output, 0xA5)


def test_abi_rejects_wrong_counts_nonfinite_config_and_aliasing(native):
    render, transfer = native
    rng = np.random.default_rng(41)
    profile, _, constants = _profile_config(rng)
    image = np.arange(18, dtype=np.uint16).reshape(2, 3, 3)
    table = np.ascontiguousarray(profile.look_table)
    curve = np.ascontiguousarray(profile.tone_curve)
    output = np.full(image.shape, 0x5A, dtype=np.uint8)

    def call_render(source, source_count, config, look_dims=profile.look_dims,
                    destination=None, destination_ptr=None, look_count=None,
                    curve_count=None):
        target = output if destination is None else destination
        out_ptr = _ptr(target) if destination_ptr is None else destination_ptr
        return render(
            3, 2, _ptr(source), source_count, _ptr(config), config.size,
            _ptr(table), table.size if look_count is None else look_count,
            *look_dims, profile.look_encoding, _ptr(curve),
            curve.size if curve_count is None else curve_count,
            out_ptr, target.size,
        )

    assert call_render(image, image.size - 1, constants) != 0
    assert call_render(image, image.size, constants, look_dims=(0, 7, 9)) != 0
    bad_constants = constants.copy()
    bad_constants[4] = np.nan
    assert call_render(image, image.size, bad_constants) != 0
    backing = np.arange(18, dtype=np.uint16)
    alias_status = call_render(
        backing.reshape(2, 3, 3), backing.size, constants,
        destination=backing.view(np.uint8)[:image.size],
    )
    assert alias_status != 0
    np.testing.assert_array_equal(output, 0x5A)

    reference = np.full((1, 1, 3), 1000, dtype=np.uint16)
    processed = np.full_like(reference, 1200)
    camera = np.full_like(reference, 2400)
    inverse = np.eye(3, dtype=np.float32)
    transfer_output = np.full_like(camera, 0x1234)
    bad_inverse = inverse.copy()
    bad_inverse[0, 0] = np.inf
    status = transfer(
        1, 1, _ptr(camera), camera.size, _ptr(reference), reference.size,
        _ptr(processed), processed.size, _ptr(bad_inverse), bad_inverse.size,
        _ptr(transfer_output), transfer_output.size,
    )
    assert status != 0
    np.testing.assert_array_equal(transfer_output, 0x1234)
    status = transfer(
        1, 1, _ptr(camera), camera.size - 1, _ptr(reference), reference.size,
        _ptr(processed), processed.size, _ptr(inverse), inverse.size,
        _ptr(transfer_output), transfer_output.size,
    )
    assert status != 0


def test_render_rejects_dimensions_whose_product_overflows_u64(native):
    render, _ = native
    rng = np.random.default_rng(54)
    profile, _, constants = _profile_config(rng)
    table = np.ascontiguousarray(profile.look_table)
    curve = np.ascontiguousarray(profile.tone_curve)
    image = np.zeros((1, 1, 3), dtype=np.uint16)
    output = np.full(image.shape, 0x6B, dtype=np.uint8)
    # 2^31 * 2^31 * 4 wraps a uint64 entry count to zero.
    status = render(
        1, 1, _ptr(image), image.size, _ptr(constants), constants.size,
        _ptr(table), 0, 1 << 31, 1 << 31, 4, profile.look_encoding,
        _ptr(curve), curve.size, _ptr(output), output.size,
    )
    assert status != 0
    np.testing.assert_array_equal(output, 0x6B)


def test_render_rejects_tone_curve_count_over_profile_tag_limit(native):
    render, _ = native
    rng = np.random.default_rng(55)
    profile, _, constants = _profile_config(rng)
    table = np.ascontiguousarray(profile.look_table)
    curve = np.ascontiguousarray(profile.tone_curve)
    image = np.zeros((1, 1, 3), dtype=np.uint16)
    output = np.full(image.shape, 0x7C, dtype=np.uint8)
    status = render(
        1, 1, _ptr(image), image.size, _ptr(constants), constants.size,
        _ptr(table), table.size, *profile.look_dims, profile.look_encoding,
        _ptr(curve), 4 * 1024 * 1024 + 2, _ptr(output), output.size,
    )
    assert status != 0
    np.testing.assert_array_equal(output, 0x7C)
