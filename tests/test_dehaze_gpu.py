from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import dehaze
from dehaze import DehazeParams, apply_dehaze


def _image() -> np.ndarray:
    return np.full((12, 16, 3), (96, 112, 128), dtype=np.uint8)


def test_explicit_unavailable_backend_falls_back_to_cpu(monkeypatch):
    monkeypatch.setenv("IMPRINT_DEHAZE_BACKEND", "cuda")
    monkeypatch.setattr(dehaze, "_backend_available", lambda backend: False)
    source = _image()
    expected = dehaze._apply_dehaze_cpu(source, DehazeParams())
    actual = apply_dehaze(source, DehazeParams())
    assert np.array_equal(actual, expected)


def test_auto_backend_order_and_environment_are_not_cached(monkeypatch):
    calls: list[str] = []

    def available(backend: str) -> bool:
        calls.append(f"available:{backend}")
        return backend == "mps"

    def run(_image, _params, backend: str):
        calls.append(f"run:{backend}")
        return np.full_like(_image, 7)

    monkeypatch.setattr(dehaze, "_backend_available", available)
    monkeypatch.setattr(dehaze, "_run_gpu_backend", run)
    monkeypatch.setenv("IMPRINT_DEHAZE_BACKEND", "auto")
    assert np.all(apply_dehaze(_image()) == 7)
    assert calls == ["available:cuda", "available:mps", "run:mps"]

    calls.clear()
    monkeypatch.setenv("IMPRINT_DEHAZE_BACKEND", "cpu")
    cpu = apply_dehaze(_image())
    assert calls == []
    assert cpu.dtype == np.uint8


def test_gpu_status_aggregates_backends_and_recommends_stable_order(monkeypatch):
    available = {"cuda": False, "mps": True, "opencl": True}
    monkeypatch.setattr(dehaze, "_backend_available", lambda backend: available.get(backend, False))

    status = dehaze.get_gpu_status()

    assert status == {
        "available": True,
        "backends": ["mps", "opencl"],
        "recommended": "mps",
        "label": "Apple MPS 可用",
    }


def test_gpu_status_reports_clean_unavailable_result(monkeypatch):
    monkeypatch.setattr(dehaze, "_backend_available", lambda _backend: False)

    assert dehaze.get_gpu_status() == {
        "available": False,
        "backends": [],
        "recommended": None,
        "label": "未检测到可用 GPU",
    }


def test_explicit_backend_dispatch_overrides_environment(monkeypatch):
    calls: list[str] = []
    monkeypatch.setenv("IMPRINT_DEHAZE_BACKEND", "cpu")
    monkeypatch.setattr(dehaze, "_backend_available", lambda backend: backend == "cuda")
    monkeypatch.setattr(
        dehaze,
        "_run_gpu_backend",
        lambda image, _params, backend: calls.append(backend) or np.full_like(image, 11),
    )

    result = apply_dehaze(_image(), DehazeParams(), backend="cuda")

    assert calls == ["cuda"]
    assert np.all(result == 11)


def test_gpu_runtime_failure_is_safe(monkeypatch):
    monkeypatch.setenv("IMPRINT_DEHAZE_BACKEND", "opencl")
    monkeypatch.setattr(dehaze, "_backend_available", lambda backend: backend == "opencl")

    def fail(*_args):
        raise RuntimeError("simulated driver failure")

    monkeypatch.setattr(dehaze, "_run_gpu_backend", fail)
    source = _image()
    expected = dehaze._apply_dehaze_cpu(source, DehazeParams())
    actual = apply_dehaze(source, DehazeParams())
    assert np.array_equal(actual, expected)


def test_torch_adapter_matches_cpu_reference_within_quantization():
    torch = pytest.importorskip("torch")
    from dehaze_gpu import _torch_dehaze

    rng = np.random.default_rng(24)
    for dtype in (np.uint8, np.uint16):
        peak = np.iinfo(dtype).max
        source = rng.integers(0, peak + 1, size=(24, 32, 3), dtype=dtype)
        params = DehazeParams(strength=0.8, local_contrast=0.5, color_recovery=0.9)
        cpu = dehaze._apply_dehaze_cpu(source, params)
        result = _torch_dehaze(source, params, "cpu", torch)
        assert result.shape == source.shape
        assert result.dtype == source.dtype
        assert np.isfinite(result).all()
        assert int(np.abs(cpu.astype(np.int64) - result.astype(np.int64)).max()) <= 2


def test_opencl_adapter_matches_cpu_reference_within_quantization():
    pytest.importorskip("cv2")
    from dehaze_gpu import _opencl_dehaze

    rng = np.random.default_rng(42)
    for dtype in (np.uint8, np.uint16):
        peak = np.iinfo(dtype).max
        source = rng.integers(0, peak + 1, size=(24, 32, 3), dtype=dtype)
        params = DehazeParams(strength=0.8, local_contrast=0.5, color_recovery=0.9)
        cpu = dehaze._apply_dehaze_cpu(source, params)
        opencl = _opencl_dehaze(source, params)
        assert opencl.shape == source.shape
        assert opencl.dtype == source.dtype
        assert int(np.abs(cpu.astype(np.int64) - opencl.astype(np.int64)).max()) <= 2
