"""Version 1: instruction oracle for prepared 009dcb00..009dcba4 slots.

Input slots are AFTER b69664, defaults are AFTER adf404; this does not build
either input from UI values, a RAW file, a profile, or final rendered pixels.
The dispatch bytes below are the actual 122 bytes at VM 03bedb30 in the
fixed ARM64 slice. No ignored research artifacts are imported at test runtime.
"""
from __future__ import annotations

import operator
import numpy as np

ORACLE_VERSION = "post-tone-basic-arm64-v1"
TARGET_SHA256 = "09978d95dfe76b18495fea9550b051a89f4598ab0fd4734d69d22d330496f40a"
SWITCH_SHA256 = "d4f0d263f455b7c97e2380e779763c08d62503b4c5c8291b01864d162b0ab9d0"
SWITCH_BYTES = bytes.fromhex(
    "0000070707091107070707070707070707070707070707070707070707070707"
    "0707070707070707070707070707070707070707070707070707070707070707"
    "0707070707070707070707070707070707070707070707070707070f07070707"
    "0707070707070700000000000000070707070707070707070000"
)
COUNT = 0x84


def normalize(values, source_defaults, process_version):
    """Interpret original dispatch targets and 32-bit b26298 comparison."""
    pv = operator.index(process_version)
    if not 0 <= pv <= 0xffffffff:
        raise ValueError("process_version must be uint32")
    for array in (values, source_defaults):
        if not isinstance(array, np.ndarray) or array.dtype != np.int32 or array.shape != (COUNT,):
            raise ValueError("prepared basic slots must be int32[132]")
    out = np.empty(COUNT, dtype=np.int32)
    for index in range(COUNT):
        # b5e078's two unsigned index-range checks and b26298's W arithmetic.
        if ((index - 0x1d) & 0xffffffff) < 6 or ((index - 0x19) & 0xffffffff) < 4:
            active = not (((pv + 1) & 0xffffffff) < 0x05070001)
        else:
            active = True
        if not active:
            out[index] = source_defaults[index]  # 9dcb14..1c
            continue
        relative = (index - 9) & 0xffffffff
        target = 0x9dcb58 + 4 * SWITCH_BYTES[relative] if relative <= 0x79 else 0x9dcb74
        if target in (0x9dcb58, 0x9dcb9c):
            result = int(values[index])  # preserved value; key append not modeled
        elif target == 0x9dcb74:
            result = 0
        elif target == 0x9dcb7c:
            # bic w9,w9,w9,asr #31, not a floating-point clamp.
            signed = int(values[index])
            result = signed & ~(signed >> 31)
        elif target == 0x9dcb94:
            result = index  # X24 is 100 here, not the caller's Scale value.
        else:
            raise AssertionError(f"uninterpreted original target {target:x}")
        out[index] = result
    return out
