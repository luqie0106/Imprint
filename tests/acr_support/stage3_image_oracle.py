"""Independent full-frame reference assembled from instruction models.

No native calls: one padded image is evaluated without a tile scheduler.
The host, profile choice, and Adobe image subclass remain explicit assumptions.
"""
from functools import lru_cache
import importlib.util
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


border = load('image_border_ref', ROOT/'tests/acr_support/bayer_border_oracle.py')
gain = load('image_gain_ref', ROOT/'tests/acr_support/bayer_gain_oracle.py')
curve = load('image_curve_ref', ROOT/'tests/acr_support/stage3_curve_oracle.py')
lut = load('image_u16_ref', ROOT/'tests/acr_support/u16_lut_oracle.py')
hotpixel = load('image_hot_ref', ROOT/'tests/acr_support/hotpixel_oracle.py')
core = load('image_core_ref', ROOT/'tests/acr_support/core3_compose_oracle.py')
controls = load('image_controls_ref', ROOT/'tests/test_acr_core3_controls.py')
F = np.float32
SCALE = np.asarray([0x37800080], dtype=np.uint32).view(F)[0]


@lru_cache(maxsize=8)
def tables(floor, cubic):
    pivot = float(floor) * float.fromhex('0x1.0001000100010p-16')
    prepared = curve.prepare(pivot, cubic)
    result = []
    for inverse in (False, True):
        knots = np.asarray([curve.evaluate(prepared, i/4096., inverse)
                            for i in range(4097)], dtype=F)
        result.append(lut.build(np.concatenate((knots, knots[-1:]))))
    return result


def encode(row):
    values = np.multiply(row, F(65535), dtype=F)
    if not np.isfinite(values).all() or np.any(row < 0) or np.any(row > 1):
        raise ValueError('outside prepared float-to-u16 domain')
    return np.rint(values).astype(np.uint16)


def reference(source, rect, gains=(1., .45550453738287877, .8786001338393127),
              floor=4100, cubic=False, amount=1., suppress=False,
              hot=8000, cold=6000):
    top, left, bottom, right = map(int, rect)
    height, width = bottom-top, right-left
    values = controls.reference(amount, suppress)
    inset = 8 if values[2] == 0 else 12
    padded_width = (width+7)//8*8
    inner_width, inner_height = padded_width+2*inset, height+2*inset
    raw_rect = (top-inset-2, left-inset-2,
                bottom+inset+2, left+padded_width+inset+2)
    raw = border.reference(source, rect, raw_rect)
    pivot = float(floor)*float.fromhex('0x1.0001000100010p-16')
    forward, forward_clip = gain.prepare(gains, pivot)
    backward, backward_clip = gain.prepare(gains, pivot, True)
    p2, inv_p2 = tables(int(floor), bool(cubic))
    before = np.empty_like(raw)
    for row in range(raw.shape[0]):
        normalized = np.multiply(raw[row].astype(F), SCALE, dtype=F)
        adjusted = gain.cfa(normalized, raw.shape[1], forward, forward_clip,
                            (raw_rect[0]+row)&1, raw_rect[1]&1)
        before[row] = p2[encode(adjusted)]
    core_input = np.empty((inner_height, inner_width), dtype=F)
    for row in range(inner_height):
        corrected = hotpixel.reference(before[row:row+5], inner_width,
                                        (top-inset+row+left-inset)&1,
                                        hot, cold)
        core_input[row] = np.multiply(corrected.astype(F), SCALE, dtype=F)
    output = core.reference_f32(
        core_input, width=inner_width, green_parameters=values[:4],
        lower=values[4], upper=values[5], row_origin_parity=(top-inset)&1,
        column_origin_parity=(left-inset)&1)
    rgb = output['rgb_float']
    assert rgb.shape == (height, padded_width, 3)
    result = np.empty_like(rgb, dtype=np.uint16)
    for row in range(height):
        encoded = encode(rgb[row])
        decoded = np.multiply(inv_p2[encoded].astype(F), SCALE, dtype=F)
        adjusted = gain.rgb(decoded.T, padded_width, backward, backward_clip).T
        result[row] = encode(adjusted)
    return result[:, :width].copy()
