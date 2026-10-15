#!/usr/bin/env python3
"""Synthetic CPU/GPU camera-profile parity probe; requires a real Metal/D3D12 GPU."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault(
    "IMPRINT_NATIVE_RENDERER_LIB",
    "/private/tmp/imprint-profile-gpu-build/libimprint_renderer.dylib",
)
os.environ["IMPRINT_NATIVE_CAMERA_PROFILE"] = "1"
os.environ["IMPRINT_NATIVE_DENSE"] = "1"

from native_dense import camera_profile_pixels  # noqa: E402
from native_renderer import (  # noqa: E402
    clear_camera_profile_gpu_cache,
    native_camera_profile,
)


def _tone(rng: np.random.Generator, point_count: int = 9) -> np.ndarray:
    x = np.linspace(0.0, 1.0, point_count, dtype=np.float32)
    increments = rng.uniform(0.25, 1.5, point_count - 1).astype(np.float32)
    y = np.concatenate((np.zeros(1, np.float32), np.cumsum(increments, dtype=np.float32)))
    y /= y[-1]
    y[0], y[-1] = 0.0, 1.0
    return np.ascontiguousarray(np.stack((x, y), axis=1), dtype=np.float32)


def _case(source: np.ndarray, rng: np.random.Generator, index: int):
    matrix = np.array(
        [[1.62, -0.34, -0.11], [-0.12, 1.24, -0.08], [-0.05, -0.29, 1.44]],
        dtype=np.float32,
    )
    projection = np.array(
        [[1.45, -0.22, -0.10], [-0.14, 1.32, -0.12], [-0.03, -0.18, 1.21]],
        dtype=np.float32,
    )
    matrix += rng.uniform(-0.03, 0.03, matrix.shape).astype(np.float32)
    projection += rng.uniform(-0.03, 0.03, projection.shape).astype(np.float32)
    white = rng.uniform(0.78, 1.0, 3).astype(np.float32)
    gain = np.float32(2.0 ** (-3.0 + 0.75 * index))
    dims = (36, 7, 5)
    look = np.empty((dims[2], dims[0], dims[1], 3), dtype=np.float32)
    look[..., 0] = rng.uniform(-90.0, 90.0, look.shape[:3]).astype(np.float32)
    look[..., 1] = rng.uniform(0.6, 1.5, look.shape[:3]).astype(np.float32)
    look[..., 2] = rng.uniform(0.6, 1.5, look.shape[:3]).astype(np.float32)
    encoding = index & 1
    tone = _tone(rng, 5 + (index % 5))
    return matrix, white, gain, look, dims, encoding, tone, projection


def _assert_parity(source: np.ndarray, parameters, label: str) -> int:
    reference = camera_profile_pixels(source, *parameters)
    actual, backend = native_camera_profile(source, *parameters)
    difference = np.abs(actual.astype(np.int16) - reference.astype(np.int16))
    maximum = int(difference.max(initial=0))
    if maximum > 1:
        index = np.unravel_index(int(difference.argmax()), difference.shape)
        raise AssertionError(
            f"{label}: {backend} differs from CPU by {maximum} at {index}; "
            f"GPU={actual[index]}, CPU={reference[index]}"
        )
    print(f"{label}: backend={backend}, max_rgb8_difference={maximum}")
    return maximum


def main() -> int:
    rng = np.random.default_rng(0xC4A3E17)
    source = rng.integers(0, 65536, size=(256, 384, 3), dtype=np.uint16)
    source[0, 0] = (0, 65535, 32768)
    source[-1, -1] = (65535, 0, 1)
    source.setflags(write=False)
    try:
        max_difference = 0
        for index in range(12):
            parameters = _case(source, rng, index)
            max_difference = max(max_difference, _assert_parity(
                source, parameters, f"randomized case {index} (EV={-3.0 + 0.75 * index:.2f})"))

        dims = (36, 7, 5)
        extreme_look = np.zeros((dims[2], dims[0], dims[1], 3), dtype=np.float32)
        extreme_look[..., 0] = np.where(np.indices(extreme_look.shape[:3])[1] % 2, -360.0, 360.0)
        extreme_look[..., 1] = np.where(np.indices(extreme_look.shape[:3])[2] % 2, 0.0, 16.0)
        extreme_look[..., 2] = np.where(np.indices(extreme_look.shape[:3])[0] % 2, 0.0, 16.0)
        extreme_look.setflags(write=False)
        identity_tone = np.array([[0.0, 0.0], [0.25, 0.25], [0.5, 0.5],
                                  [0.75, 0.75], [1.0, 1.0]], dtype=np.float32)
        identity_tone.setflags(write=False)
        base = _case(source, rng, 4)
        extreme = (base[0], base[1], np.float32(4.0), extreme_look,
                   dims, 1, identity_tone, base[7])
        max_difference = max(max_difference, _assert_parity(source, extreme, "extreme legal look"))
        print(f"synthetic parity passed: max_rgb8_difference={max_difference} (required <= 1)")
        return 0
    finally:
        clear_camera_profile_gpu_cache()


if __name__ == "__main__":
    raise SystemExit(main())
