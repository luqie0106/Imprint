"""Versioned independent oracle for ACR bulk profile/LookTable SDR mode 1.

This is a frozen transcription of the saved ARM64 float-stage reference plus
the 2026-10-21 unsigned-word wrapper evidence.  It intentionally imports
neither the ignored ``research/acr`` scripts nor native product code.  It is
evidence-bound to the ordinary SDR HSM stage and does not claim Adobe runtime
parity, provider dispatch coverage, or a generic profile API.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import math
from typing import Iterable

import numpy as np


ORACLE_VERSION = "acr-profile-bulk-sdr-u16-arm64-2026-10-21-v1"
ORACLE_SCOPE = "explicit HSM SDR mode-1 wrapper around the recovered bulk float stage"

# Each entry records the original `bulk-curve-constants.json` FMOV address and
# the MOV/MOVK instruction words independently checked in the named raw
# listing.  Bits are embedded here so tests never load ignored research files.
CONSTANT_EVIDENCE = {
    "encode_threshold": ("0x3b4d2e1c", "bulk-multiv-0135f8f8-asm.json", "0135fb14", ("0135f9ec 8dc38552", "0135f9f0 ad69a772")),
    "encode_slope": ("0x414eb852", "bulk-multiv-0135f8f8-asm.json", "0135fbd0", ("0135fa14 500a9752", "0135fa18 d029a872")),
    "e0": ("0xc5375cbc", "bulk-multiv-0135f8f8-asm.json", "0135fb38", ("0135fb30 88978b52", "0135fb34 e8a6b872")),
    "e1": ("0x4686555f", "bulk-multiv-0135f8f8-asm.json", "0135fb2c", ("0135fb24 e8ab8a52", "0135fb28 c8d0a872")),
    "e2": ("0x46d9a6f8", "bulk-multiv-0135f8f8-asm.json", "0135fb48", ("0135fb40 08df9452", "0135fb44 28dba872")),
    "e3": ("0x45308d78", "bulk-multiv-0135f8f8-asm.json", "0135fb58", ("0135fb50 08af9152", "0135fb54 08a6a872")),
    "e4": ("0x41c4250c", "bulk-multiv-0135f8f8-asm.json", "0135fb68", ("0135fb60 88a18452", "0135fb64 8838a872")),
    "e5": ("0xbc9aaf34", "bulk-multiv-0135f8f8-asm.json", "0135fb78", ("0135fb70 88e69552", "0135fb74 4893b772")),
    "d0": ("0x4705a8a5", "bulk-multiv-0135f8f8-asm.json", "0135fb94", ("0135fb8c a8149552", "0135fb90 a8e0a872")),
    "d1": ("0x4622fd1f", "bulk-multiv-0135f8f8-asm.json", "0135fb88", ("0135fb80 e8a39f52", "0135fb84 48c4a872")),
    "d2": ("0x43a4f30a", "bulk-multiv-0135f8f8-asm.json", "0135fba4", ("0135fb9c 48619e52", "0135fba0 8874a872")),
    "decode_threshold": ("0x3d25aed5", "bulk-multiv-0135fc18-asm.json", "0135fe64", ("0135fa0c afda9552", "0135fa10 afa4a772")),
    "decode_slope": ("0x3d9e8391", "bulk-multiv-0135ff38-asm.json", "0135ff78", ("0135ff70 28729052", "0135ff74 c8b3a772")),
    "i0": ("0xbdfcfc81", "bulk-multiv-0135ff38-asm.json", "0135ffa0", ("0135ff98 28909f52", "0135ff9c 88bfb772")),
    "i1": ("0x3ed58f8e", "bulk-multiv-0135ff38-asm.json", "0135ff94", ("0135ff8c c8f19152", "0135ff90 a8daa772")),
    "i2": ("0xbf2169d3", "bulk-multiv-0135ff38-asm.json", "0135ffb0", ("0135ffa8 683a8d52", "0135ffac 28e4b772")),
    "i3": ("0x3f4f6537", "bulk-multiv-0135ff38-asm.json", "0135ffc0", ("0135ffb8 e8a68c52", "0135ffbc e8e9a772")),
    "i4": ("0x3efb441c", "bulk-multiv-0135ff38-asm.json", "0135ffd0", ("0135ffc8 88838852", "0135ffcc 68dfa772")),
    "i5": ("0x3d0fb070", "bulk-multiv-0135ff38-asm.json", "0135ffe0", ("0135ffd8 080e9652", "0135ffdc e8a1a772")),
    "i6": ("0x3a60b5a8", "bulk-multiv-0135ff38-asm.json", "0135fff0", ("0135ffe8 08b59652", "0135ffec 084ca772")),
}

HUE_TO_SECTORS_BITS = 0x3E2AAAAB
HUE_SHIFT_SCALE_BITS = 0x3C888889
U16_TO_F32_SCALE_BITS = 0x37800080  # binary32(1 / 65535), from wrapper fdiv
U16_FROM_F32_SCALE_BITS = 0x477FFF00  # exact binary32(65535)
U16_ROUND_BIAS_BITS = 0x3F000000  # exact binary32(0.5), wrapper FMA addend
UNSIGNED_WORD_TYPE = 3
OFFSET_WORD_TYPE = 8
OFFSET_WORD_XOR = 0x8000

# The corresponding saved wrapper listings live beside the tests under
# research/acr/reports/evidence/2026-10-21-hsm-mode1/. The four slot values were
# recovered by read-only Mach-O VM mapping (the `raw` bytes are file bytes).
WRAPPER_EVIDENCE = {
    "slot_method": "read-only Mach-O LC_SEGMENT_64 VM mapping",
    "slot_file": "wrapper-conversion-slots.json",
    "slots": (
        ("081971c8", "0133d40c", "0cd4330100000000", "type-8 input normalization"),
        ("081971d0", "0133d62c", "2cd6330100000000", "type-3 input normalization"),
        ("081971e8", "0133dbd4", "d4db330100000000", "type-8 output conversion"),
        ("081971f0", "0133de1c", "1cde330100000000", "type-3 output conversion"),
    ),
    "input_listing": "integer-to-float-llvm.txt",
    "type8_input_ops": (
        "0133d47c 78f179e3 ldrsh w3,[x15,x17,lsl #1]",
        "0133d480 11402063 add w3,w3,#0x8000",
        "0133d484 1e220061 scvtf s1,w3",
        "0133d488 1e210801 fmul s1,s0,s1",
    ),
    "type3_input_ops": (
        "0133d69c 7c7179e1 ldr h1,[x15,x17,lsl #1]",
        "0133d6a0 7e21d821 ucvtf s1,s1",
        "0133d6a4 1e210801 fmul s1,s0,s1",
    ),
    "output_listing": "float-to-integer-llvm.txt",
    "type8_output_ops": (
        "0133dc40 bc7179e2 ldr s2,[x15,x17,lsl #2]",
        "0133dc44 1f020402 fmadd s2,s0,s2,s1",
        "0133dc48 1e380043 fcvtzs w3,s2",
        "0133dc4c 52110063 eor w3,w3,#0x8000",
        "0133dc50 78317a03 strh w3,[x16,x17,lsl #1]",
    ),
    "type3_output_ops": (
        "0133de88 bc7179e2 ldr s2,[x15,x17,lsl #2]",
        "0133de8c 1f020402 fmadd s2,s0,s2,s1",
        "0133de90 1e380043 fcvtzs w3,s2",
        "0133de94 78317a03 strh w3,[x16,x17,lsl #1]",
    ),
}


def _f32_from_bits(bits: int) -> np.float32:
    return np.asarray([bits], dtype=np.uint32).view(np.float32)[0]


CONSTANTS = {name: _f32_from_bits(int(record[0], 16))
             for name, record in CONSTANT_EVIDENCE.items()}
HUE_TO_SECTORS = _f32_from_bits(HUE_TO_SECTORS_BITS)
HUE_SHIFT_SCALE = _f32_from_bits(HUE_SHIFT_SCALE_BITS)
U16_TO_F32_SCALE = _f32_from_bits(U16_TO_F32_SCALE_BITS)
U16_FROM_F32_SCALE = _f32_from_bits(U16_FROM_F32_SCALE_BITS)
U16_ROUND_BIAS = _f32_from_bits(U16_ROUND_BIAS_BITS)


def _load_fmaf():
    for name in (ctypes.util.find_library("m"), None):
        try:
            lib = ctypes.CDLL(name) if name else ctypes.CDLL(None)
            fn = lib.fmaf
            fn.argtypes = (ctypes.c_float, ctypes.c_float, ctypes.c_float)
            fn.restype = ctypes.c_float
            return fn
        except (AttributeError, OSError):
            continue
    raise RuntimeError("platform C math library must provide correctly rounded fmaf")


_FMAF = _load_fmaf()


def _load_fmodf():
    for name in (ctypes.util.find_library("m"), None):
        try:
            lib = ctypes.CDLL(name) if name else ctypes.CDLL(None)
            fn = lib.fmodf
            fn.argtypes = (ctypes.c_float, ctypes.c_float)
            fn.restype = ctypes.c_float
            return fn
        except (AttributeError, OSError):
            continue
    raise RuntimeError("platform C math library must provide fmodf")


_FMODF = _load_fmodf()


def f32(value: float | np.floating) -> np.float32:
    return np.float32(value)


def add32(a: float, b: float) -> np.float32:
    with np.errstate(all="ignore"):
        return np.float32(f32(a) + f32(b))


def sub32(a: float, b: float) -> np.float32:
    with np.errstate(all="ignore"):
        return np.float32(f32(a) - f32(b))


def mul32(a: float, b: float) -> np.float32:
    with np.errstate(all="ignore"):
        return np.float32(f32(a) * f32(b))


def div32(a: float, b: float) -> np.float32:
    with np.errstate(all="ignore"):
        return np.float32(f32(a) / f32(b))


def fma32(a: float, b: float, c: float) -> np.float32:
    return np.float32(_FMAF(float(f32(a)), float(f32(b)), float(f32(c))))


def clip01(value: float) -> np.float32:
    value = f32(value)
    if value < 0:
        return f32(0)
    if value > 1:
        return f32(1)
    return value


def transfer(value: float, *, inverse: bool = False) -> np.float32:
    """Evaluate the saved SDR transfer polynomials in ARM instruction order."""
    x = f32(value)
    if inverse:
        if x <= CONSTANTS["decode_threshold"]:
            return mul32(x, CONSTANTS["decode_slope"])
        result = fma32(x, CONSTANTS["i0"], CONSTANTS["i1"])
        for key in ("i2", "i3", "i4", "i5", "i6"):
            result = fma32(result, x, CONSTANTS[key])
        return result
    if x <= CONSTANTS["encode_threshold"]:
        return mul32(x, CONSTANTS["encode_slope"])
    result = fma32(x, CONSTANTS["e0"], CONSTANTS["e1"])
    for key in ("e2", "e3", "e4", "e5"):
        result = fma32(result, x, CONSTANTS[key])
    denominator = fma32(x, CONSTANTS["d0"], CONSTANTS["d1"])
    denominator = fma32(denominator, x, CONSTANTS["d2"])
    denominator = fma32(denominator, x, f32(1))
    return div32(result, denominator)


def hsv_from_rgb(red: float, green: float, blue: float) -> tuple[np.float32, np.float32, np.float32]:
    """Binary32 HSV conversion, retaining the recovered max/min tie order."""
    red, green, blue = f32(red), f32(green), f32(blue)
    max_gb = green if green > blue else blue
    min_gb = green if green < blue else blue
    maximum = red if max_gb < red else max_gb
    minimum = red if min_gb > red else min_gb
    delta = sub32(maximum, minimum)
    hue, saturation = f32(0), f32(0)
    if delta > 0:
        if maximum == red:
            hue = div32(sub32(green, blue), delta)
            if hue < 0:
                hue = add32(hue, f32(6))
        elif maximum == green:
            hue = add32(div32(sub32(blue, red), delta), f32(2))
        else:
            hue = add32(div32(sub32(red, green), delta), f32(4))
        saturation = div32(delta, maximum)
    return hue, saturation, maximum


def _shape_table(table: np.ndarray, dims: tuple[int, int, int]) -> np.ndarray:
    h_count, s_count, v_count = (int(x) for x in dims)
    values = np.asarray(table, dtype=np.float32)
    if values.size != h_count * s_count * v_count * 3:
        raise ValueError("table sample count does not match dimensions")
    return np.ascontiguousarray(values.reshape(v_count, h_count, s_count, 3))


def reference_pixel(rgb: Iterable[float], table: np.ndarray,
                    dims: tuple[int, int, int], encoding: int,
                    lookup_gain: float = 1.0, amount: float = 1.0) -> np.ndarray:
    """Apply one float32 RGB sample through the independent recovered stage."""
    h_count, s_count, v_count = (int(x) for x in dims)
    prepared = _shape_table(table, dims).copy()
    # The table stores hue in degrees; the stage prepares the angular scale.
    with np.errstate(all="ignore"):
        prepared[..., 0] = np.asarray(prepared[..., 0] * HUE_SHIFT_SCALE, dtype=np.float32)
    red, green, blue = (f32(x) for x in rgb)
    hue, saturation, value = hsv_from_rgb(red, green, blue)
    gain, amount = f32(lookup_gain), f32(amount)
    encoded = int(encoding) == 1 and v_count > 1

    hc = mul32(mul32(f32(h_count), HUE_TO_SECTORS), hue) if h_count > 1 else f32(0)
    sc = mul32(saturation, f32(s_count - 1))
    hi_floor, si_floor = math.trunc(float(hc)), math.trunc(float(sc))
    hi, si = min(hi_floor, h_count - 1), min(si_floor, s_count - 2)
    hf, sf = sub32(hc, f32(hi)), sub32(sc, f32(si))

    vi, vf = 0, f32(0)
    if v_count > 1:
        lookup = clip01(mul32(value, gain))
        if encoded:
            lookup = transfer(lookup)
        vc = mul32(lookup, f32(v_count - 1))
        vi_floor = math.trunc(float(vc))
        vi = min(vi_floor, v_count - 2)
        vf = sub32(vc, f32(vi))

    def blend(low: float, high: float, fraction: float) -> np.float32:
        return fma32(sub32(f32(1), fraction), low, mul32(fraction, high))

    def hue_at(v_index: int, s_index: int, channel: int) -> np.float32:
        return blend(prepared[v_index, hi, s_index, channel],
                     prepared[v_index, (hi + 1) % h_count, s_index, channel], hf)

    modifiers = []
    for channel in range(3):
        low = hue_at(vi, si, channel)
        high = hue_at(vi, si + 1, channel)
        if v_count > 1:
            low = blend(low, hue_at(vi + 1, si, channel), vf)
            high = blend(high, hue_at(vi + 1, si + 1, channel), vf)
        modifiers.append(blend(low, high, sf))

    saturation_modifier = max(fma32(amount, sub32(modifiers[1], f32(1)), f32(1)), f32(0))
    value_modifier = max(fma32(amount, sub32(modifiers[2], f32(1)), f32(1)), f32(0))
    mapped_saturation = min(mul32(saturation, saturation_modifier), f32(1))
    mapped_value = clip01(mul32(transfer(value) if encoded else value, value_modifier))
    if encoded:
        mapped_value = transfer(mapped_value, inverse=True)
    if mapped_saturation <= 0:
        return np.asarray([mapped_value] * 3, dtype=np.float32)

    shifted_hue = add32(hue, mul32(modifiers[0], amount))
    wrapped_hue = f32(_FMODF(float(shifted_hue), 6.0))
    if wrapped_hue < 0:
        wrapped_hue = add32(wrapped_hue, f32(6))
    sector = math.trunc(float(wrapped_hue))
    fraction = sub32(wrapped_hue, f32(sector))
    p = mul32(sub32(f32(1), mapped_saturation), mapped_value)
    q = mul32(fma32(-mapped_saturation, fraction, f32(1)), mapped_value)
    t = mul32(fma32(-mapped_saturation, sub32(f32(1), fraction), f32(1)), mapped_value)
    choices = ((mapped_value, t, p), (q, mapped_value, p), (p, mapped_value, t),
               (p, q, mapped_value), (t, p, mapped_value), (mapped_value, p, q))
    return np.asarray(choices[sector if sector < 6 else 0], dtype=np.float32)


def reference_float_image(source: np.ndarray, table: np.ndarray,
                          dims: tuple[int, int, int], encoding: int,
                          lookup_gain: float = 1.0,
                          amount: float = 1.0) -> np.ndarray:
    """Return contiguous float32 RGB results for an HxWx3 or Nx3 source."""
    pixels = np.asarray(source, dtype=np.float32).reshape(-1, 3)
    return np.asarray([reference_pixel(pixel, table, dims, encoding,
                                       lookup_gain, amount)
                       for pixel in pixels], dtype=np.float32).reshape(source.shape)


def decode_u16(source: np.ndarray, pixel_type: int) -> np.ndarray:
    """Model the wrapper's type-3/type-8 word normalization and binary32 scale."""
    words = np.asarray(source, dtype=np.uint16)
    if pixel_type == OFFSET_WORD_TYPE:
        words = np.bitwise_xor(words, np.uint16(OFFSET_WORD_XOR))
    elif pixel_type != UNSIGNED_WORD_TYPE:
        raise ValueError("pixel_type must be 3 or 8")
    return np.asarray(words.astype(np.float32) * U16_TO_F32_SCALE, dtype=np.float32)


def _fcvtzs_i32(value: float) -> int:
    """ARM FCVTZS signed-32 semantics, including saturation and NaN -> zero."""
    x = float(f32(value))
    if math.isnan(x):
        return 0
    if x >= 2147483648.0:
        return 2147483647
    if x <= -2147483648.0:
        return -2147483648
    return math.trunc(x)


def encode_u16(source_float: np.ndarray, pixel_type: int) -> np.ndarray:
    """Model FMA(scale, value, .5), FCVTZS, low-word store, and type-8 XOR."""
    if pixel_type not in (UNSIGNED_WORD_TYPE, OFFSET_WORD_TYPE):
        raise ValueError("pixel_type must be 3 or 8")
    values = np.asarray(source_float, dtype=np.float32).reshape(-1)
    result = np.empty(values.size, dtype=np.uint16)
    for index, value in enumerate(values):
        converted = _fcvtzs_i32(fma32(value, U16_FROM_F32_SCALE, U16_ROUND_BIAS))
        word = converted & 0xFFFF
        if pixel_type == OFFSET_WORD_TYPE:
            word ^= OFFSET_WORD_XOR
        result[index] = word
    return result.reshape(np.asarray(source_float).shape)


def reference_u16(source: np.ndarray, table: np.ndarray,
                  dims: tuple[int, int, int], encoding: int,
                  lookup_gain: float = 1.0, amount: float = 1.0,
                  pixel_type: int = UNSIGNED_WORD_TYPE) -> np.ndarray:
    """Compose the mode-1 word wrapper with the independent float-stage model."""
    words = np.asarray(source, dtype=np.uint16)
    decoded = decode_u16(words, pixel_type)
    mapped = reference_float_image(decoded, table, dims, encoding, lookup_gain, amount)
    return encode_u16(mapped, pixel_type)


def make_table(h_count: int, s_count: int, v_count: int, seed: int,
               *, value_range: tuple[float, float] = (0.35, 0.95),
               saturation_range: tuple[float, float] = (0.0, 1.4),
               hue_range: tuple[float, float] = (-80.0, 80.0)) -> np.ndarray:
    """Create deterministic, finite test data in the original H,S,V,RGB layout."""
    rng = np.random.default_rng(seed)
    count = int(h_count) * int(s_count) * int(v_count)
    table = np.empty((count, 3), dtype=np.float32)
    table[:, 0] = rng.uniform(*hue_range, size=count).astype(np.float32)
    table[:, 1] = rng.uniform(*saturation_range, size=count).astype(np.float32)
    table[:, 2] = rng.uniform(*value_range, size=count).astype(np.float32)
    return np.ascontiguousarray(table.reshape(-1))

