"""Mode4 swap/blend witnesses from retained ARM64, independent of research files."""
import ctypes as C

import numpy as np
import pytest

import test_acr_guide_camera_black as camera
from test_acr_guide_camera_black import native
from acr_support.ordinary_resample_oracle import fma32
from acr_support.log_scene_stats_oracle import LIBM


def _swap(native, values, inverse=0, width=None, height=None, capacity=None):
    image = np.ascontiguousarray(values, np.float32)
    if width is None:
        height, width = image.shape
    output = np.full(image.size, -9123.25, np.float32)
    fn = native.library.acr_test_swap_left_right
    fn.argtypes = [camera.FLOAT_PTR, camera.U32, camera.U32, camera.U32,
                   camera.FLOAT_PTR, camera.SIZE]
    fn.restype = C.c_int
    status = fn(camera._float_ptr(image), width, height, inverse,
                camera._float_ptr(output), output.size if capacity is None else capacity)
    return status, output.reshape(image.shape)


def _blend(native, first, second, width=None, height=None, capacity=None):
    first, second = (np.ascontiguousarray(x, np.float32) for x in (first, second))
    if width is None:
        height, width = first.shape
    output = np.full(first.size, -9123.25, np.float32)
    fn = native.library.acr_test_blend_horizontal
    fn.argtypes = [camera.FLOAT_PTR, camera.FLOAT_PTR, camera.U32, camera.U32,
                   camera.FLOAT_PTR, camera.SIZE]
    fn.restype = C.c_int
    status = fn(camera._float_ptr(first), camera._float_ptr(second), width, height,
                camera._float_ptr(output), output.size if capacity is None else capacity)
    return status, output.reshape(first.shape)


def _blend_oracle(first, second):
    height, width = first.shape
    reciprocal = np.float32(np.float32(1) / np.float32(width - 1))
    output = np.empty_like(first)
    for x in range(width):
        t = np.clip(np.float32(np.float32(x) * reciprocal), np.float32(0), np.float32(1))
        a = abs(fma32(t, np.float32(2), np.float32(-1)))
        weight = np.float32((float(a) * float(a)) * LIBM.fma(-2., float(a), 3.))
        for y in range(height):
            output[y,x] = fma32(weight, np.float32(second[y,x] - first[y,x]), first[y,x])
    return output


@pytest.mark.parametrize("width", [2,3,4,5,6,7,8,9,16,17,31,128,257])
def test_swap_is_cyclic_not_a_flip_and_odd_inverse_differs(native, width):
    image = np.arange(3 * width, dtype=np.float32).reshape(3,width)
    image[0,0] = np.float32(-0.)
    before = image.view(np.uint32).copy()
    for inverse, shift in ((0,(width+1)//2),(1,width//2)):
        status, actual = _swap(native, image, inverse)
        assert status == 0
        expected = image[:, (np.arange(width) + shift) % width]
        camera.assert_bits(actual, expected)
        status, restored = _swap(native, actual, 1-inverse)
        assert status == 0
        camera.assert_bits(restored, image)
    if width >= 3:
        _, actual = _swap(native, image)
        assert not np.array_equal(actual, image[:,::-1])
    np.testing.assert_array_equal(image.view(np.uint32), before)


@pytest.mark.parametrize("width", [2,3,4,5,6,7,8,9,16,17,31,128,257])
def test_blend_matches_original_mixed_precision_and_fma_order(native, width):
    rng = np.random.default_rng(901 + width)
    first, second = (rng.uniform(-20,20,(3,width)).astype(np.float32) for _ in range(2))
    before = first.view(np.uint32).copy(), second.view(np.uint32).copy()
    status, actual = _blend(native, first, second)
    assert status == 0
    camera.assert_bits(actual, _blend_oracle(first, second))
    np.testing.assert_array_equal(first.view(np.uint32), before[0])
    np.testing.assert_array_equal(second.view(np.uint32), before[1])
    if width % 2:
        camera.assert_bits(actual[:,width//2], first[:,width//2])


@pytest.mark.parametrize("bad", ["nan", "inf", "zero", "narrow", "capacity", "inverse"])
def test_swap_and_blend_invalid_leave_destination_unchanged(native, bad):
    first = np.arange(15, dtype=np.float32).reshape(3,5)
    second = first + 7
    if bad in ("nan", "inf"):
        first[-1,-1] = np.nan if bad == "nan" else np.inf
    kwargs = {"width":0,"height":3} if bad == "zero" else {}
    if bad == "narrow":
        kwargs = {"width":1,"height":3}
    if bad == "capacity":
        kwargs = {"capacity":14}
    status, out = _swap(native, first, inverse=2 if bad == "inverse" else 0, **kwargs)
    assert status != 0
    camera.assert_bits(out, np.full_like(out,-9123.25))
    if bad != "inverse":
        status, out = _blend(native, first, second, **kwargs)
        assert status != 0
        camera.assert_bits(out, np.full_like(out,-9123.25))
