"""Profile preview routing: linear EV, cache identity, and safe legacy fallback."""
from pathlib import Path
from types import SimpleNamespace
import io
import threading

import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import app_api as api


@pytest.fixture
def preview_setup(tmp_path, monkeypatch):
    path = tmp_path / 'photo.nef'
    path.write_bytes(b'fixture')
    image = np.full((24, 16, 3), 4096, dtype=np.uint16)
    metadata = SimpleNamespace(source_kind='raw', color_space='Linear sRGB', exif={})
    monkeypatch.setattr(api, 'read_image', lambda *a, **k: (image.copy(), metadata))
    monkeypatch.setattr(api, '_correct_enhanced_raw', lambda a, *args, **kw: (a.copy(), None, False))
    monkeypatch.setattr(api, 'load_camera_rgb', lambda *a: (image.copy(), (1.0, 1.0, 1.0)))
    profile = SimpleNamespace(fingerprint=('fixture', 1), name='Camera Standard')
    monkeypatch.setattr(api, 'resolve_profile', lambda *a: profile)
    monkeypatch.setattr(api, 'transfer_enhancement', lambda a, b, c, **kwargs: c.copy())
    with api._ENHANCE_LOCK:
        api._ENHANCE_SESSIONS['profile-test'] = {'files': {'photo': path}}
        api._ENHANCE_PREVIEW_CACHE.clear()
    with api._DISPLAY_PREVIEW_LOCK:
        api._PROCESSING_PREVIEW_CACHE.clear()
        api._DEHAZED_PREVIEW_CACHE.clear()
        api._DEHAZED_PREVIEW_STATUS_CACHE.clear()
        api._CAMERA_PROFILE_SOURCE_CACHE.clear()
    api._shutdown_full_resolution_preview_cache()
    yield TestClient(api.app), profile, image
    api._shutdown_full_resolution_preview_cache()
    with api._ENHANCE_LOCK:
        api._ENHANCE_SESSIONS.pop('profile-test', None)
        api._ENHANCE_PREVIEW_CACHE.clear()
    with api._DISPLAY_PREVIEW_LOCK:
        api._CAMERA_PROFILE_SOURCE_CACHE.clear()


def request(client, **extra):
    data = dict(session_id='profile-test', photo_id='photo', params={'strength': 0},
                basic_params={'exposure': 2.3}, max_edge=320, render_backend='cpu',
                basic_backend='python')
    data.update(extra)
    response = client.post('/api/enhance/preview', json=data)
    assert response.status_code == 200, response.text
    return response


@pytest.mark.parametrize('full', [False, True])
def test_profile_exposure_consumed_once_and_cached(preview_setup, monkeypatch, full):
    client, profile, image = preview_setup
    calls = []
    def render(camera, neutral, config, ev, *, backend='cpu'):
        calls.append(ev)
        return np.full(camera.shape, 90 + int(ev * 10), dtype=np.uint8)
    monkeypatch.setattr(api, 'render_profile', render)
    first = request(client, full_resolution=full)
    assert first.headers['X-Camera-Profile-Status'] == 'applied'
    assert np.asarray(Image.open(io.BytesIO(first.content))).mean() == pytest.approx(113, abs=1)
    assert calls == [2.3]  # A second display-stage +2.3 would change the pixels.
    assert request(client, full_resolution=full).content == first.content
    assert calls == [2.3]
    changed = request(client, full_resolution=full, basic_params={'exposure': 1})
    assert changed.content != first.content
    assert calls == [2.3, 1]
    profile.fingerprint = ('fixture', 2)
    request(client, full_resolution=full, basic_params={'exposure': 1})
    assert calls == [2.3, 1, 1]


def test_low_resolution_camera_decode_overlaps_dehaze_and_hot_cache_skips_decode(
    preview_setup, monkeypatch,
):
    client, _profile, image = preview_setup
    camera_started = threading.Event()
    dehaze_started = threading.Event()
    release_camera = threading.Event()
    camera_finished = threading.Event()
    camera_calls = []
    render_profile_calls = []
    render_count = 0
    source_identity = api.enhanced_dng_source_identity
    identity_revision = [0]
    monkeypatch.setattr(
        api, "enhanced_dng_source_identity",
        lambda path: (*source_identity(path), identity_revision[0]),
    )

    def load_camera(_path, shape, preview):
        assert preview is True
        camera_calls.append(tuple(shape))
        camera_started.set()
        assert dehaze_started.wait(2)
        assert release_camera.wait(2)
        camera_finished.set()
        return np.full(shape, 4096, dtype=np.uint16), (1.0, 1.0, 1.0)

    def render_dehaze(value, _params, _backend, *, diagnostics, **_kwargs):
        nonlocal render_count
        render_count += 1
        if render_count == 1:
            assert camera_started.wait(2)
            dehaze_started.set()
            assert not camera_finished.is_set()
            release_camera.set()
        diagnostics.update(auto_exposure_ev=0.0, auto_exposure_reason="off")
        return value.copy()

    def render_profile(camera, _neutral, _profile, exposure, *, backend="cpu"):
        render_profile_calls.append(exposure)
        return np.full(camera.shape, 100, dtype=np.uint8)

    monkeypatch.setattr(api, "load_camera_rgb", load_camera)
    monkeypatch.setattr(api, "_render_dehaze", render_dehaze)
    monkeypatch.setattr(api, "render_profile", render_profile)
    first = request(
        client, full_resolution=False, preview_level=1,
        params={"strength": 0.5},
    )
    assert first.headers["x-camera-profile-status"] == "applied"
    assert camera_started.is_set() and dehaze_started.is_set() and camera_finished.is_set()
    assert camera_calls == [(image.shape[0] // 2, image.shape[1] // 2, 3)]

    request(client, full_resolution=False, preview_level=1, params={"strength": 0.7})
    assert camera_calls == [(image.shape[0] // 2, image.shape[1] // 2, 3)]
    key = next(iter(api._CAMERA_PROFILE_SOURCE_CACHE))
    assert key[-1] == api.enhanced_dng_source_identity(
        api._ENHANCE_SESSIONS["profile-test"]["files"]["photo"],
    )

    # A stable legacy stat tuple must not let a camera pair cross a stronger
    # source identity change (for example, same-size replacement with restored mtime).
    identity_revision[0] = 1
    request(client, full_resolution=False, preview_level=1, params={"strength": 0.9})
    assert camera_calls == [
        (image.shape[0] // 2, image.shape[1] // 2, 3),
        (image.shape[0] // 2, image.shape[1] // 2, 3),
    ]


@pytest.mark.parametrize('full', [False, True])
def test_profile_failure_keeps_legacy_exposure(preview_setup, monkeypatch, full):
    client, _, image = preview_setup
    def fail(*args):
        raise ValueError('unsupported camera')
    monkeypatch.setattr(api, 'render_profile', fail)
    failed = request(client, full_resolution=full)
    assert failed.headers['X-Camera-Profile-Status'] == 'fallback'
    monkeypatch.setattr(api, 'resolve_profile', lambda *a: None)
    legacy = request(client, full_resolution=full)
    assert legacy.headers['X-Camera-Profile-Status'] == 'legacy'
    assert failed.content == legacy.content


@pytest.mark.parametrize('full', [False, True])
def test_original_profile_ignores_manual_and_auto_exposure(preview_setup, monkeypatch, full):
    client, _, image = preview_setup
    calls = []
    def render(camera, neutral, config, ev, *, backend='cpu'):
        calls.append(ev)
        return np.full(camera.shape, 80, dtype=np.uint8)
    monkeypatch.setattr(api, 'render_profile', render)
    response = request(client, mode='original', full_resolution=full, auto_exposure=True)
    assert response.headers['X-Camera-Profile-Status'] == 'applied'
    assert calls == [0]


@pytest.mark.parametrize('full', [False, True])
def test_ricoh_profile_and_basic_adjustments_run_in_order(preview_setup, monkeypatch, full):
    client, profile, image = preview_setup
    stages = []

    def render_dehaze(source, *_args, **_kwargs):
        return source.copy()

    def render_ricoh(display16, preset_id, basic, backend, *, use_measured_color=False):
        stages.append('ricoh')
        assert display16.dtype == np.uint16
        assert preset_id == 'gr3_standard'
        assert basic == api.BasicParamsRequest().values()
        assert backend == 'python'
        assert use_measured_color is True
        return np.clip(display16.astype(np.uint32) + 512, 0, 65535).astype(np.uint16)

    def transfer(camera, reference, processed, *, backend="cpu"):
        stages.append('transfer')
        assert camera.shape == reference.shape == processed.shape
        assert processed.dtype == np.uint16
        assert np.all(processed > image)
        return processed.copy()

    def render_profile(camera, neutral, selected_profile, exposure, *, backend='cpu'):
        stages.append('profile')
        assert selected_profile is profile
        assert exposure == 1.25
        return np.full(camera.shape, 90, dtype=np.uint8)

    def render_basic(display, basic, mode):
        stages.append('basic')
        assert basic['exposure'] == 0
        assert basic['contrast'] == 25
        assert mode == 'python'
        return display

    monkeypatch.setattr(api, '_render_dehaze', render_dehaze)
    monkeypatch.setattr(api, '_render_ricoh', render_ricoh)
    monkeypatch.setattr(api, 'transfer_enhancement', transfer)
    monkeypatch.setattr(api, 'render_profile', render_profile)
    monkeypatch.setattr(api, '_render_basic', render_basic)
    monkeypatch.setattr(api, 'list_ricoh_presets', lambda: [{'id': 'gr3_standard'}])

    response = request(
        client,
        full_resolution=full,
        ricoh_preset_id='gr3_standard',
        basic_params={'exposure': 1.25, 'contrast': 25},
    )
    assert response.headers['X-Camera-Profile-Status'] == 'applied'
    assert stages == ['ricoh', 'transfer', 'profile', 'basic']


def test_bake_ricoh_linear_uses_measured_response_and_default_basics(monkeypatch):
    linear = np.full((2, 3, 3), 1200, dtype=np.uint16)
    display = np.full_like(linear, 2400)
    effected = np.full_like(linear, 3600)
    calls = []

    def to_display(source):
        calls.append(('to_display', source))
        return display

    def render(image, preset_id, basic, backend, *, use_measured_color=False):
        calls.append(('ricoh', image, preset_id, basic, backend, use_measured_color))
        return effected

    def to_linear(source):
        calls.append(('to_linear', source))
        return linear + 1

    monkeypatch.setattr(api, '_linear16_to_srgb16', to_display)
    monkeypatch.setattr(api, '_render_ricoh', render)
    monkeypatch.setattr(api, '_srgb16_to_linear16', to_linear)

    result = api._bake_ricoh_linear(linear, 'gr3_standard', 'python')
    assert result.dtype == np.uint16
    np.testing.assert_array_equal(result, linear + 1)
    assert calls[0][0] == 'to_display'
    assert calls[0][1] is linear
    assert calls[1][0] == 'ricoh'
    assert calls[1][1] is display
    assert calls[1][2:] == (
        'gr3_standard', api.BasicParamsRequest().values(), 'python', True,
    )
    assert calls[2][0] == 'to_linear'
    assert calls[2][1] is effected


@pytest.mark.parametrize('full', [False, True])
def test_ricoh_camera_stage_cache_is_preset_specific(preview_setup, monkeypatch, full):
    client, profile, image = preview_setup
    stages = []

    monkeypatch.setattr(api, '_render_dehaze', lambda source, *_args, **_kwargs: source.copy())

    def render_ricoh(display16, preset_id, basic, backend, *, use_measured_color=False):
        stages.append(('ricoh', preset_id))
        assert basic == api.BasicParamsRequest().values()
        assert use_measured_color is True
        amount = 256 if preset_id == 'gr3_standard' else 1024
        return np.clip(display16.astype(np.uint32) + amount, 0, 65535).astype(np.uint16)

    def transfer(camera, reference, processed, *, backend="cpu"):
        stages.append(('transfer',))
        return processed.copy()

    def render_profile(camera, neutral, selected_profile, exposure, *, backend='cpu'):
        stages.append(('profile', exposure))
        assert selected_profile is profile
        return np.full(camera.shape, int(camera.mean()) % 256, dtype=np.uint8)

    monkeypatch.setattr(api, '_render_ricoh', render_ricoh)
    monkeypatch.setattr(api, 'transfer_enhancement', transfer)
    monkeypatch.setattr(api, 'render_profile', render_profile)
    monkeypatch.setattr(api, '_render_basic', lambda display, *_args: display)
    monkeypatch.setattr(api, 'list_ricoh_presets', lambda: [
        {'id': 'gr3_standard'}, {'id': 'gr3_vivid_street'},
    ])

    first = request(
        client, full_resolution=full, ricoh_preset_id='gr3_standard',
        basic_params={'exposure': 1.25},
    )
    repeated = request(
        client, full_resolution=full, ricoh_preset_id='gr3_standard',
        basic_params={'exposure': 1.25},
    )
    assert repeated.content == first.content
    assert stages == [('ricoh', 'gr3_standard'), ('transfer',), ('profile', 1.25)]

    switched = request(
        client, full_resolution=full, ricoh_preset_id='gr3_vivid_street',
        basic_params={'exposure': 1.25},
    )
    assert switched.content != first.content
    assert stages == [
        ('ricoh', 'gr3_standard'), ('transfer',), ('profile', 1.25),
        ('ricoh', 'gr3_vivid_street'), ('transfer',), ('profile', 1.25),
    ]


@pytest.mark.parametrize('full', [False, True])
def test_ricoh_profile_failure_applies_exposure_once(preview_setup, monkeypatch, full):
    client, _, _ = preview_setup
    basic_exposures = []

    def fail_profile(*_args, **_kwargs):
        raise ValueError('unsupported camera')

    def render_basic(image, basic, mode):
        basic_exposures.append(basic['exposure'])
        return image

    monkeypatch.setattr(api, 'render_profile', fail_profile)
    monkeypatch.setattr(api, '_render_basic', render_basic)
    monkeypatch.setattr(api, '_render_ricoh', lambda image, *_args, **_kwargs: image.copy())
    monkeypatch.setattr(api, 'list_ricoh_presets', lambda: [{'id': 'gr3_standard'}])

    response = request(
        client, full_resolution=full, ricoh_preset_id='gr3_standard',
        basic_params={'exposure': 1.25},
    )
    assert response.headers['X-Camera-Profile-Status'] == 'fallback'
    assert basic_exposures == [1.25, 0]
