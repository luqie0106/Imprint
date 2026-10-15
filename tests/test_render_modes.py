from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import app_api
from dehaze import DehazeParams


def test_operators_are_selected_independently(monkeypatch):
    image = np.zeros((4, 5, 3), dtype=np.float32)
    calls: list[str] = []

    def physical(image, params, *, backend, spatial, nonlocal_mode=None, diagnostics=None):
        assert nonlocal_mode is None
        calls.append("physical-dehaze")
        assert image.dtype == np.float32
        assert params.strength == 0.4
        assert backend == "native"
        assert spatial is False
        return image

    monkeypatch.setattr(app_api, "apply_physical_dehaze", physical)
    monkeypatch.setattr(app_api, "native_basic", lambda image, basic: calls.append("basic") or image)
    monkeypatch.setattr(app_api, "native_ricoh", lambda image, preset, basic, **kwargs: calls.append("ricoh") or image)
    monkeypatch.setattr(app_api, "apply_basic_preview_effect", lambda image, basic: calls.append("python-basic") or image)
    monkeypatch.setattr(app_api, "apply_ricoh_preview_effect", lambda image, preset, basic, **kwargs: calls.append("python-ricoh") or image)

    app_api._render_dehaze(image, DehazeParams(strength=0.4), "native")
    app_api._render_basic(image, {"exposure": 1.0}, "python")
    app_api._render_ricoh(image, "gr3_standard", None, "native", use_measured_color=True)
    assert calls == ["physical-dehaze", "python-basic", "ricoh"]


def test_native_ricoh_failure_falls_back_to_python(monkeypatch):
    image = np.zeros((4, 5, 3), dtype=np.uint16)
    def unavailable(*_args, **_kwargs):
        raise RuntimeError("device unavailable")
    monkeypatch.setattr(app_api, "native_ricoh", unavailable)
    monkeypatch.setattr(app_api, "apply_ricoh_preview_effect", lambda image, *_args, **_kwargs: image + 1)
    output = app_api._render_ricoh(image, "gr3_standard", None, "native")
    assert np.all(output == 1)
    assert np.all(image == 0)


def test_new_request_modes_are_validated():
    assert app_api.BurstRequest(input_dir=".", sort_backend="native").sort_backend == "native"
    assert app_api.EnhancePreviewRequest(session_id="s", photo_id="p", basic_backend="python", ricoh_backend="native").ricoh_backend == "native"


def test_physical_status_ignores_legacy_gpu_and_spatial_probes(monkeypatch):
    unavailable = {
        "available": False, "backend": None,
        "gpu_available": False, "cpu_available": False,
    }

    def obsolete_probe(*_args, **_kwargs):
        raise AssertionError("legacy dehaze status must not be queried")

    monkeypatch.setattr(app_api, "get_native_physical_status", lambda: dict(unavailable))
    monkeypatch.setattr(app_api, "get_last_physical_backend", lambda: "Python CPU（线性浮点）")
    monkeypatch.setattr(app_api, "get_native_status", lambda: {"available": True, "backend": "Metal"})
    monkeypatch.setattr(app_api, "get_native_sort_status", lambda: {"available": False})
    monkeypatch.setattr(app_api, "get_gpu_status", obsolete_probe, raising=False)
    monkeypatch.setattr(app_api, "get_native_spatial_status", obsolete_probe, raising=False)

    assert app_api.get_enhance_gpu_status() == {
        "available": False, "backends": [], "recommended": None,
        "label": "未检测到可用 GPU",
    }
    status = app_api.get_enhance_render_status()
    assert status["native"] == {"available": True, "backend": "Metal"}
    assert status["spatial"] == {
        **unavailable, "last_used": "Python CPU（线性浮点）",
    }
    assert status["pytorch"] == {"available": False, "backends": []}


def test_physical_status_reports_only_new_metal_float_backend(monkeypatch):
    physical = {
        "available": True, "backend": "Metal",
        "gpu_available": True, "cpu_available": False,
    }
    monkeypatch.setattr(app_api, "get_native_physical_status", lambda: dict(physical))
    monkeypatch.setattr(app_api, "get_last_physical_backend", lambda: "Metal GPU（线性浮点）")
    monkeypatch.setattr(app_api, "get_native_status", lambda: {"available": False})
    monkeypatch.setattr(app_api, "get_native_sort_status", lambda: {"available": False})

    assert app_api.get_enhance_gpu_status() == {
        "available": True, "backends": ["metal"], "recommended": "metal",
        "label": "Metal GPU 可用",
    }
    status = app_api.get_enhance_render_status()
    assert status["spatial"] == {
        **physical, "last_used": "Metal GPU（线性浮点）",
    }
    assert status["pytorch"] == {"available": False, "backends": []}
