"""Fixed-weight guide suffix and ordinary Camera -> black/pin -> log chain."""
from __future__ import annotations

import ctypes as C
import itertools
import math

import numpy as np
import pytest

import test_acr_guide_camera_black as camera
from test_acr_guide_camera_black import native  # shared source/official-object fixtures


_math = C.CDLL(None)
_log2f = _math.log2f
_log2f.argtypes = [C.c_float]
_log2f.restype = C.c_float
_exp2 = _math.exp2
_exp2.argtypes = [C.c_double]
_exp2.restype = C.c_double
_fmaf = camera.camera_oracle._fmaf


def reference(rgb, ev, epsilon):
    """Independent FMUL/FMA/FMA/+0 then optional gain and host log2f."""
    rgb = np.asarray(rgb, dtype=np.float32).reshape(-1, 3)
    coefficients = np.asarray([0x3E99999A, 0x3F170A3D, 0x3DE147B0],
                              dtype=np.uint32).view(np.float32)
    gain = np.float32(_exp2(ev))
    e = np.float32(epsilon)
    result = np.empty(len(rgb), dtype=np.float32)
    for i, (r, g, b) in enumerate(rgb):
        green = np.float32(g * coefficients[1])
        red_green = np.float32(_fmaf(r, coefficients[0], green))
        value = np.float32(_fmaf(b, coefficients[2], red_green))
        value = value if value > 0.0 else np.float32(0.0)
        if ev != 0.0:
            value = np.float32(value * gain)
        result[i] = _log2f(np.float32(e + min(value, np.float32(2.0**120))))
    return result


def suffix(native, rgb, ev, epsilon):
    rgb = np.ascontiguousarray(rgb, dtype=np.float32)
    h, w = rgb.shape[:2]
    out = np.full(h*w, -1234.5, dtype=np.float32)
    fn = native.library.acr_test_guide_suffix
    fn.argtypes = [camera.FLOAT_PTR, camera.U32, camera.U32, camera.DOUBLE,
                   camera.DOUBLE, camera.FLOAT_PTR]
    fn.restype = C.c_int
    status = fn(camera._float_ptr(rgb), w, h, ev, epsilon, camera._float_ptr(out))
    return status, out


def compose(native, source, controls, order, *, height, width, stride,
            selector=1, flags=0, ev=0.0, epsilon=2.0**-12, count=None, reuse=False):
    source = np.ascontiguousarray(source, dtype=np.float32).reshape(-1)
    out = np.full(max(5, source.size), -9123.25, dtype=np.float32)
    snapshot = camera.Snapshot()
    gain = camera.U32()
    fn = native.library.acr_test_guide_gray_log
    fn.argtypes = [camera.FLOAT_PTR, camera.SIZE, camera.SIZE, camera.SIZE, camera.SIZE,
        camera.DOUBLE_PTR, camera.U32_PTR, camera.U32, camera.U32, camera.DOUBLE,
        camera.DOUBLE, C.POINTER(camera.Snapshot), camera.U32_PTR,
        camera.FLOAT_PTR, camera.SIZE, camera.U32]
    fn.restype = C.c_int
    status = fn(camera._float_ptr(source), source.size if count is None else count,
        stride, height, width, camera._double_ptr(controls), camera._u32_ptr(order),
        flags, selector, ev, epsilon, C.byref(snapshot), C.byref(gain),
        camera._float_ptr(out), out.size, int(reuse))
    return status, out[:snapshot.rgb_size].copy(), snapshot, int(gain.value)


def test_suffix_scalar_simd_widths_and_signed_overrange(native):
    rng = np.random.default_rng(0x889340)
    for width in (1, 3, 15, 16, 17, 31, 32, 33):
        rgb = rng.uniform(-0.5, 2.0, size=(2, width, 3)).astype(np.float32)
        for ev, epsilon in itertools.product((-2.5, -0.0, 0.0, 1.25, 8.0), (2**-12, 2**-20)):
            status, actual = suffix(native, rgb, ev, epsilon)
            assert status == 0
            camera.assert_bits(actual, reference(rgb, ev, epsilon))


def test_suffix_keeps_fixed_weights_and_post_gain_over_one(native):
    rgb = np.eye(3, dtype=np.float32).reshape(1, 3, 3)
    status, actual = suffix(native, rgb, 0.0, 2**-12)
    assert status == 0
    camera.assert_bits(actual, reference(rgb, 0.0, 2**-12))
    status, actual = suffix(native, np.ones((1, 1, 3), np.float32), 2.0, 2**-12)
    assert status == 0 and actual[0] > 2.0
    for ev in (0.0, -0.0, 120.0, -120.0):
        rgb = np.asarray([0, -0.0, 0, 1, 1, 1, 2**125, 2**125, 2**125], np.float32).reshape(1,3,3)
        if ev == 120.0:
            rgb[0,2] = 1.0  # keep the scaled value finite
        status, actual = suffix(native, rgb, ev, 2**-12)
        assert status == 0
        camera.assert_bits(actual, reference(rgb, ev, 2**-12))
        assert actual[0] == -12.0


def test_fma_correction_has_a_numeric_witness(native):
    rng = np.random.default_rng(0x889450)
    rgb = rng.random((1, 2048, 3), dtype=np.float32)
    exact = reference(rgb, 0.0, 2**-12)
    old_gray = np.float32(np.float32(rgb[...,1]*np.float32(.59) +
        rgb[...,0]*np.float32(.3)) + rgb[...,2]*np.float32(.110000014))
    old = np.asarray([_log2f(np.float32(x+np.float32(2**-12)))
                      for x in old_gray.flat], np.float32)
    assert np.count_nonzero(exact.view(np.uint32) != old.view(np.uint32)) > 100
    status, actual = suffix(native, rgb, 0.0, 2**-12)
    assert status == 0
    camera.assert_bits(actual, exact)


def test_full_prepared_chain_padded_rows_modes_and_retained_black(native):
    rng = np.random.default_rng(20261108)
    h, w, stride = 3, 17, 55
    source = np.full(h*stride, np.nan, np.float32)
    for y in range(h):
        source[y*stride:y*stride+w*3] = rng.uniform(-0.1, 1.2, w*3)
    before = source.view(np.uint32).copy()
    order = np.asarray([2,0,1], np.uint32)
    for black, flag, selector, ev in itertools.product((0.0,4100.0/65535), (0,1), (1,2), (0.0,1.25)):
        controls = camera.make_controls(direct=False, black=black)
        pixels = camera.expected_pixels(source,controls,order,flag,selector,
            height=h,width=w,source_row_stride=stride)
        expected = reference(pixels,ev,2**-12)
        status, actual, plan, gain = compose(native,source,controls,order,
            height=h,width=w,stride=stride,selector=selector,flags=flag,ev=ev)
        assert status == 0
        assert (plan.height,plan.width,plan.camera_type) == (h,w,2)
        assert plan.black_type == (selector if black else 0)
        assert plan.pin_type == (selector if flag else 0)
        assert gain == int(ev != 0)
        camera.assert_bits(actual,expected)
    camera.assert_bits(source,before.view(np.float32))


def test_gray_log_failure_atomicity_and_old_allocation_alias(native):
    source = np.asarray([.1,.2,.3,.4,.5,.6],np.float32)
    controls = camera.make_controls(direct=True, black=0)
    order = np.asarray([0,1,2],np.uint32)
    failures = [dict(ev=float('nan')),dict(ev=121),dict(epsilon=0),
                dict(epsilon=2**-11),dict(epsilon=math.nextafter(0.0,1.0)),
                dict(stride=5),dict(count=5),dict(width=65536)]
    for changes in failures:
        kwargs = dict(height=1,width=2,stride=6)
        kwargs.update(changes)
        status, actual, plan, gain = compose(native,source,controls,order,**kwargs)
        assert status != 0
        camera.assert_bits(actual,np.full(5,-1234.5,np.float32))
        assert (plan.height,plan.width,plan.camera_type,plan.black_type,
                plan.black_encoding,plan.quantized_before_black,plan.pin_type,gain) == (91,87,31,32,33,1,34,1)
    for reuse in (False,True):
        status, actual, _, _ = compose(native,source,controls,order,
            height=1,width=2,stride=6,reuse=reuse)
        assert status == 0
        camera.assert_bits(actual,reference(source,0,2**-12))
        bad = source.copy(); bad[-1]=np.nan
        status, actual, plan, _ = compose(native,bad,controls,order,
            height=1,width=2,stride=6,reuse=reuse)
        assert status == camera.CAMERA_FAILURE
        camera.assert_bits(actual,bad if reuse else np.full(5,-1234.5,np.float32))
        assert (plan.height,plan.width,plan.pin_type)==(91,87,34)
