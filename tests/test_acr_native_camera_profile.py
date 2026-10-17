from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import camera_profile as cp


@pytest.fixture(autouse=True)
def _mock_native_illuminants(monkeypatch):
    def resolve(codes):
        assert tuple(codes) == (17, 21)
        return np.asarray([2850.0, 6500.0], dtype=np.float64)
    monkeypatch.setattr(cp, "native_acr_illuminant_temperatures", resolve)


def _profile() -> cp.Profile:
    identity = np.eye(3, dtype=np.float64)
    curve = np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    dims = (2, 2, 2)
    look = np.tile(np.asarray([0.0, 1.0, 1.0]), (2, 2, 2, 1))
    return cp.Profile(
        name="ACR test profile", fingerprint=("native-acr", 1),
        color_matrix1=identity * 1.0, color_matrix2=identity * 2.0,
        forward_matrix1=identity * 3.0, forward_matrix2=identity * 4.0,
        tone_curve=curve, look_table=look, look_dims=dims,
        look_encoding=0, baseline_exposure_offset=0.25,
    )


def _prepared_camera() -> np.ndarray:
    prepared = np.zeros(65, dtype=np.float64)
    prepared[35:44] = np.arange(101.0, 110.0)
    prepared[47:50] = [1.1, 2.2, 3.3]
    prepared[53:56] = [2.0, 0.0, 1.0]
    prepared[56:65] = np.arange(201.0, 210.0)
    return prepared


def test_enabled_experiment_uses_native_camera_pipeline_and_explicit_controls(monkeypatch):
    profile = _profile()
    neutral = np.asarray([0.75, 1.125, 0.5], dtype=np.float64)
    camera = (np.arange(130 * 2 * 3, dtype=np.uint16).reshape(130, 2, 3) * 173)
    before = camera.copy()
    prepared = _prepared_camera()
    observed: dict[str, list] = {name: [] for name in (
        "prepare", "camera", "scale", "look", "hsl", "tone", "grading", "project",
    )}
    event_order: list[str] = []

    def prepare(controls):
        observed["prepare"].append(np.array(controls, copy=True))
        return prepared.copy()

    def camera_float(rgb, controls):
        observed["camera"].append((rgb.copy(), np.array(controls, copy=True)))
        event_order.append("camera")
        return rgb + np.float32(0.1)

    def scale(rgb, gain):
        observed["scale"].append((rgb.copy(), gain))
        event_order.append("scale")
        return rgb * np.float32(2.0)

    def look(rgb, table, dims, encoding):
        assert table is profile.look_table
        assert dims == profile.look_dims
        assert encoding == profile.look_encoding
        observed["look"].append(rgb.copy())
        event_order.append("look")
        return rgb + np.float32(0.01)

    def hsl(rgb, controls):
        observed["hsl"].append((rgb.copy(), np.array(controls, copy=True)))
        event_order.append("hsl")
        return rgb + np.float32(0.02)

    def tone(rgb, points):
        observed["tone"].append((rgb.copy(), np.array(points, copy=True)))
        event_order.append("tone")
        return rgb + np.float32(0.03)

    def grading(rgb, controls):
        observed["grading"].append((rgb.copy(), np.array(controls, copy=True)))
        event_order.append("grading")
        return rgb + np.float32(0.04)

    def project(rgb, matrix):
        observed["project"].append((rgb.copy(), np.array(matrix, copy=True)))
        event_order.append("project")
        return np.full(rgb.shape, 91, dtype=np.uint8)

    monkeypatch.setattr(cp, "_camera_transform", lambda *_args: pytest.fail("legacy camera transform used"))
    monkeypatch.setattr(cp, "_apply_look", lambda *_args: pytest.fail("Python profile LookTable used"))
    monkeypatch.setattr(cp, "native_profile_look", lambda *_args: pytest.fail(
        "compatibility profile LookTable used",
    ), raising=False)
    monkeypatch.setattr(cp, "native_acr_profile_look", lambda *_args: pytest.fail("single-map helper used"), raising=False)
    monkeypatch.setattr(cp, "native_acr_prepare_profile_camera", prepare)
    monkeypatch.setattr(cp, "native_acr_camera_float", camera_float)
    monkeypatch.setattr(cp, "native_acr_scale_sdr", scale)
    monkeypatch.setattr(cp, "native_acr_profile_stage_sdr", look)
    monkeypatch.setattr(cp, "native_acr_hsl_sdr", hsl)
    monkeypatch.setattr(cp, "native_acr_profile_tone", tone)
    monkeypatch.setattr(cp, "native_acr_grading_sdr", grading)
    monkeypatch.setattr(cp, "native_acr_project_srgb", project)

    hsl_controls = np.linspace(-12.0, 12.0, 25)
    grading_controls = np.linspace(-5.0, 8.0, 14)
    output = cp.render_profile(
        camera, neutral, profile, exposure_ev=1.0,
        acr_color={"enabled": True, "hsl": hsl_controls, "grading": grading_controls},
    )

    assert np.array_equal(camera, before)
    assert output.shape == camera.shape
    assert np.all(output == 91)
    assert cp.get_last_profile_backend() == "cpp_cpu_acr"

    controls = observed["prepare"][0]
    identity = np.eye(3, dtype=np.float64).reshape(-1)
    assert controls.shape == (75,)
    assert np.array_equal(controls[0:3], np.ones(3))
    assert np.array_equal(controls[3:12], profile.color_matrix1.reshape(-1))
    assert np.array_equal(controls[12:21], profile.color_matrix2.reshape(-1))
    assert np.array_equal(controls[21:30], identity)
    assert np.array_equal(controls[30:39], identity)
    assert np.array_equal(controls[39:48], profile.forward_matrix1.reshape(-1))
    assert np.array_equal(controls[48:57], profile.forward_matrix2.reshape(-1))
    assert np.array_equal(controls[57:64], [2850.0, 6500.0, 0.3457, 0.3585, 1.0, 1.0, 1.0])
    assert np.array_equal(controls[64:67], neutral)
    assert np.array_equal(controls[67:75], np.zeros(8))

    order = [2, 0, 1]
    camera_float_controls = np.concatenate(([3.3, 1.1, 1.0, 1.0], prepared[35:44]))
    assert len(observed["camera"]) == 2  # The renderer sends bounded 128-row blocks.
    for index, (actual_camera, actual_controls) in enumerate(observed["camera"]):
        start = index * 128
        end = min(start + 128, camera.shape[0])
        expected = camera[start:end].astype(np.float32) / np.float32(65535.0)
        assert np.allclose(actual_camera, expected[..., order])
        assert np.array_equal(actual_controls, camera_float_controls)
    expected_gain = np.float32(2.0 ** (1.0 + profile.baseline_exposure_offset))
    assert all(isinstance(gain, np.float32) and gain == expected_gain
               for _rgb, gain in observed["scale"])
    assert all(np.array_equal(values, hsl_controls) for _rgb, values in observed["hsl"])
    assert all(np.array_equal(points, profile.tone_curve) for _rgb, points in observed["tone"])
    assert all(np.array_equal(values, grading_controls) for _rgb, values in observed["grading"])
    assert all(np.array_equal(matrix, prepared[56:65]) for _rgb, matrix in observed["project"])
    assert event_order == [
        "camera", "scale", "look", "hsl", "tone", "grading", "project",
        "camera", "scale", "look", "hsl", "tone", "grading", "project",
    ]


def test_native_failure_propagates_without_legacy_python_fallback(monkeypatch):
    profile = _profile()
    camera = np.full((3, 4, 3), 12345, dtype=np.uint16)
    before = camera.copy()
    monkeypatch.setattr(cp, "native_acr_prepare_profile_camera", lambda _controls: _prepared_camera())

    def fail_camera(*_args):
        raise cp.NativeRendererError("camera operator unavailable")

    monkeypatch.setattr(cp, "native_acr_camera_float", fail_camera)
    monkeypatch.setattr(cp, "_camera_transform", lambda *_args: pytest.fail("legacy camera transform used"))
    monkeypatch.setattr(cp.native_dense, "camera_profile_pixels", lambda *_args: pytest.fail("legacy native fallback used"))
    with pytest.raises(cp.NativeRendererError, match="camera operator unavailable"):
        cp.render_profile(
            camera, (1.0, 1.0, 1.0), profile, exposure_ev=0.0,
            acr_color={"enabled": True},
        )

    assert cp.get_last_profile_backend() == "python"
    assert np.array_equal(camera, before)


def test_profile_tone_checkbox_false_keeps_legacy_tone_stage(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(cp, "native_acr_prepare_profile_camera", lambda _controls: _prepared_camera())
    monkeypatch.setattr(cp, "native_acr_camera_float", lambda rgb, _controls: rgb.copy())
    monkeypatch.setattr(cp, "native_acr_scale_sdr", lambda rgb, _gain: rgb.copy())
    monkeypatch.setattr(cp, "native_acr_profile_stage_sdr", lambda rgb, *_args: rgb.copy())
    monkeypatch.setattr(cp, "native_acr_hsl_sdr", lambda rgb, _controls: rgb.copy())
    monkeypatch.setattr(cp, "_apply_tone", lambda rgb, _curve: calls.append("legacy-tone") or rgb.copy())
    monkeypatch.setattr(cp, "native_acr_profile_tone", lambda *_args, **_kwargs: pytest.fail(
        "native profile tone used while its checkbox is off",
    ))
    monkeypatch.setattr(cp, "native_acr_grading_sdr", lambda rgb, _controls: rgb.copy())
    monkeypatch.setattr(cp, "native_acr_project_srgb", lambda rgb, _matrix: np.zeros(rgb.shape, dtype=np.uint8))

    output = cp.render_profile(
        np.full((2, 3, 3), 10000, dtype=np.uint16), (1.0, 1.0, 1.0),
        _profile(), exposure_ev=0.0,
        acr_color={"enabled": True, "profile_tone": False},
    )

    assert calls == ["legacy-tone"]
    assert cp.get_last_profile_backend() == "cpp_cpu_acr"
    assert output.dtype == np.uint8


def test_disabled_experiment_keeps_legacy_python_profile_path(monkeypatch):
    camera = np.full((2, 3, 3), 20000, dtype=np.uint16)
    before = camera.copy()
    profile = _profile()
    calls: list[str] = []

    def old_transform(_profile, _neutral):
        calls.append("transform")
        return np.eye(3, dtype=np.float32), np.ones(3, dtype=np.float32)

    def old_look(rgb, _profile):
        calls.append("look")
        return rgb

    def old_tone(rgb, _curve):
        calls.append("tone")
        return rgb

    def old_project(rgb):
        calls.append("project")
        return np.full(rgb.shape, 42, dtype=np.uint8)

    def native_unexpected(*_args, **_kwargs):
        pytest.fail("disabled ACR experiment invoked a new native operator")

    def old_native_dense(*_args, **_kwargs):
        raise cp.NativeRendererError("legacy native profile path unavailable")

    monkeypatch.setattr(cp, "_camera_transform", old_transform)
    monkeypatch.setattr(cp, "_apply_look", old_look)
    monkeypatch.setattr(cp, "_apply_tone", old_tone)
    monkeypatch.setattr(cp, "_project_srgb", old_project)
    monkeypatch.setattr(cp.native_dense, "camera_profile_pixels", old_native_dense)
    monkeypatch.setattr(cp, "native_acr_prepare_profile_camera", native_unexpected)
    monkeypatch.setattr(cp, "native_acr_camera_float", native_unexpected)
    monkeypatch.setattr(cp, "native_acr_scale_sdr", native_unexpected)
    monkeypatch.setattr(cp, "native_acr_project_srgb", native_unexpected)
    monkeypatch.setattr(cp, "native_profile_look", native_unexpected, raising=False)
    monkeypatch.setattr(cp, "native_acr_profile_stage_sdr", native_unexpected)

    output = cp.render_profile(camera, (1.0, 1.0, 1.0), profile, exposure_ev=0.0)

    assert calls == ["transform", "look", "tone", "project"]
    assert cp.get_last_profile_backend() == "python"
    assert np.all(output == 42)
    assert np.array_equal(camera, before)
