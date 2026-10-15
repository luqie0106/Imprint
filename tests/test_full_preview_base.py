"""Full preview reuse must preserve final basic parameters and cache boundaries."""
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import app_api as api


@pytest.fixture
def scene(monkeypatch, tmp_path):
    source = tmp_path / 'photo.nef'
    source.touch()
    api._ENHANCE_SESSIONS['full-cache-test'] = {'files': {'photo': source}}
    api._shutdown_full_resolution_preview_cache()
    api._FULL_RESOLUTION_DECODED_CACHE.clear()
    api._FULL_RESOLUTION_BASE_CACHE.clear()
    calls = {'decode': 0, 'dehaze': 0, 'correct': 0}
    image = np.full((16, 23, 3), 16000, np.uint16)
    def decode(*args, **kwargs):
        assert kwargs['preview'] is False
        calls['decode'] += 1
        return image.copy(), SimpleNamespace(source_kind='raw', color_space='Linear sRGB')
    def render(value, params, backend, **kwargs):
        calls['dehaze'] += 1
        kwargs['diagnostics'].update(auto_exposure_ev=.25 if kwargs.get('auto_exposure') else 0,
                                     auto_exposure_reason='applied' if kwargs.get('auto_exposure') else 'off')
        return value * (1 - params.strength * .1)
    def correct(value, *args, **kwargs):
        calls['correct'] += 1
        return value, None, None
    monkeypatch.setattr(api, 'read_image', decode)
    monkeypatch.setattr(api, '_render_dehaze', render)
    monkeypatch.setattr(api, '_correct_enhanced_raw', correct)
    yield source, calls
    api._ENHANCE_SESSIONS.pop('full-cache-test', None)
    api._shutdown_full_resolution_preview_cache()
    api._FULL_RESOLUTION_DECODED_CACHE.clear()
    api._FULL_RESOLUTION_BASE_CACHE.clear()


def request(**kwargs):
    return api.EnhancePreviewRequest(session_id='full-cache-test', photo_id='photo',
                                    full_resolution=True, preview_level=0,
                                    render_backend='cpu', basic_backend='python', **kwargs)


def test_basic_edits_reuse_full_base_but_render_new_pixels(scene):
    source, calls = scene
    first = api.create_enhance_preview(request(basic_params={'exposure': 0}))
    second = api.create_enhance_preview(request(basic_params={'exposure': 1}))
    assert first.status_code == second.status_code == 200
    assert first.body != second.body
    assert second.headers['x-preview-base-cache'] == 'hit'
    assert 'adjustments;dur=' in second.headers['server-timing']
    assert second.headers['x-image-width'] == '23'
    assert calls == {'decode': 1, 'dehaze': 1, 'correct': 1}
    base = next(iter(api._FULL_RESOLUTION_BASE_CACHE.values()))[0]
    assert base.dtype == np.uint8 and not base.flags.writeable


@pytest.mark.parametrize('changes', [
    {'params': {'strength': .8}}, {'auto_mode': True}, {'auto_exposure': True},
    {'nonlocal_mode': 'strong'}, {'render_backend': 'auto'}, {'color_manage_srgb': True},
])
def test_processing_changes_invalidate_base(scene, changes):
    _, calls = scene
    first = request()
    api.create_enhance_preview(first)
    updated = first.model_copy(update=changes)
    if 'params' in changes:
        updated = request(params=changes['params'])
    result = api.create_enhance_preview(updated)
    assert result.status_code == 200
    assert result.headers['x-preview-base-cache'] == 'miss'
    assert calls['decode'] == 1
    assert len(api._FULL_RESOLUTION_BASE_CACHE) == 1


def test_changed_file_and_oversized_base_do_not_reuse(scene, monkeypatch):
    source, calls = scene
    api.create_enhance_preview(request())
    source.write_bytes(b'new source')
    assert api.create_enhance_preview(request()).headers['x-preview-base-cache'] == 'miss'
    monkeypatch.setattr(api, '_MAX_FULL_RESOLUTION_BASE_BYTES', 1)
    api._FULL_RESOLUTION_BASE_CACHE.clear()
    api.create_enhance_preview(request())
    api.create_enhance_preview(request())
    assert not api._FULL_RESOLUTION_BASE_CACHE
    assert calls['decode'] == 2


def test_diagnostics_survive_basic_edits_and_original_bypasses_cache(scene):
    _, calls = scene
    first = api.create_enhance_preview(request(auto_exposure=True))
    second = api.create_enhance_preview(request(auto_exposure=True, basic_params={'contrast': 10}))
    assert second.headers['x-auto-exposure-ev'] == first.headers['x-auto-exposure-ev'] == '0.25'
    # Original must never return the processed display base.
    original = api.create_enhance_preview(request(mode='original'))
    assert original.headers['x-preview-base-cache'] == 'bypass'
    assert calls['decode'] == 1 and calls['dehaze'] == 1
