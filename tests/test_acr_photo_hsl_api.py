"""Explicit photo packet routing: never promote metadata to Adobe consumer state."""
from pathlib import Path
import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import app_api as api
import native_renderer as native
from test_acr_color_preview_api import acr_scene, _color, _request


def _packet(hue=10, *, extra="", look=""):
    return f'''<x:xmpmeta xmlns:x="adobe:ns:meta/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
      xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/">
      <rdf:RDF><rdf:Description rdf:about="" crs:CameraProfile="Adobe Standard"
      crs:HueAdjustmentRed="{hue}" {extra}>{look}</rdf:Description></rdf:RDF></x:xmpmeta>'''.encode()


def _look(camera="Adobe Standard", amount="1.234"):
    return f'''<crs:Look crs:Name="Explicit photo Look" crs:UUID="12345678123456781234567812345678"
      crs:Amount="{amount}"><crs:Parameters crs:CameraProfile="{camera}"
      crs:SaturationAdjustmentOrange="-14"/></crs:Look>'''


@pytest.mark.parametrize("full", [False, True])
def test_photo_source_reads_once_and_invalidates_all_color_caches(acr_scene, monkeypatch, full):
    client, path, calls = acr_scene
    sidecar = path.with_suffix(".xmp")
    sidecar.write_bytes(_packet(10))
    original = api.inspect_photo_hsl
    reads = []

    def inspect(*args):
        reads.append(args)
        return original(*args)

    monkeypatch.setattr(api, "inspect_photo_hsl", inspect)
    color = dict(_color(0), photo_hsl_source="sidecar")
    first = _request(client, color=color, full_resolution=full)
    repeat = _request(client, color=color, full_resolution=full)
    sidecar.write_bytes(_packet(20))
    changed = _request(client, color=color, full_resolution=full)
    # Same control values but different packet digest: not an image-cache hit.
    sidecar.write_bytes(_packet(20, extra='crs:Version="17.2"'))
    digest_changed = _request(client, color=color, full_resolution=full)
    assert first.status_code == repeat.status_code == changed.status_code == digest_changed.status_code == 200
    assert first.content == repeat.content and changed.content != first.content
    assert changed.content == digest_changed.content
    assert len(reads) == 4  # full-resolution route reuses this same request snapshot
    assert [r["hsl"][0] for r in calls["render"]] == [10, 20, 20]
    assert all(r.headers["cache-control"] == "private, no-store" for r in (first, repeat, changed))
    if full:
        assert repeat.headers["x-preview-base-cache"] == "hit"
        assert changed.headers["x-preview-camera-cache"] == "stage-hit"
        assert digest_changed.headers["x-preview-base-cache"] == "miss"


@pytest.mark.parametrize("full", [False, True])
def test_original_and_disabled_ignore_missing_packet(acr_scene, monkeypatch, full):
    client, _path, calls = acr_scene
    monkeypatch.setattr(api, "inspect_photo_hsl", lambda *_: pytest.fail("unused packet read"))
    color = dict(_color(0), photo_hsl_source="sidecar")
    response = _request(client, color=color, mode="original", full_resolution=full)
    disabled = _request(client, color=dict(color, enabled=False), full_resolution=full)
    assert response.status_code == disabled.status_code == 200
    assert response.headers["x-acr-color-status"] == disabled.headers["x-acr-color-status"] == "off"
    assert all(r is None for r in calls["render"])


@pytest.mark.parametrize("extra,look,reason", [
    ("", _look(), "consumer_state_unbound"),
    ('crs:ConvertToGrayscale="True"', "", "color_hsl_required"),
])
def test_unbound_look_and_gray_are_explicitly_rejected(acr_scene, extra, look, reason):
    client, path, calls = acr_scene
    path.with_suffix(".xmp").write_bytes(_packet(extra=extra, look=look))
    response = _request(client, color=dict(_color(0), photo_hsl_source="sidecar"))
    assert response.status_code == 422 and response.json()["reason"] == reason
    assert calls["render"] == [] and calls["resolver"] == []


def test_request_rejects_path_and_mixed_manual_controls():
    for source in ("/private/photo.xmp", "auto", "both"):
        with pytest.raises(ValidationError):
            api.AcrColorPreviewRequest(photo_hsl_source=source)
    with pytest.raises(ValidationError):
        api.AcrColorPreviewRequest(photo_hsl_source="sidecar", hsl=[1.] + [0.] * 24)
    with pytest.raises(ValidationError):
        api.AcrColorPreviewRequest(photo_hsl_source="embedded", hsl_fragment="modern03_hsl")
    assert api.AcrColorPreviewRequest(photo_hsl_source="sidecar", hsl=[0.] * 24 + [12.]).hsl[24] == 12


def test_packet_failure_prevents_stale_image_cache_reuse(acr_scene):
    client, path, calls = acr_scene
    sidecar = path.with_suffix(".xmp")
    sidecar.write_bytes(_packet(10))
    color = dict(_color(0), photo_hsl_source="sidecar")
    assert _request(client, color=color).status_code == 200
    sidecar.write_bytes(b"<invalid")
    rejected = _request(client, color=color)
    assert rejected.status_code == 422
    assert len(calls["render"]) == 1
    assert str(sidecar) not in rejected.text


def test_nonzero_metadata_controls_use_actual_cpp_pixels(tmp_path):
    photo = tmp_path / "input.nef"
    photo.write_bytes(b"synthetic source; no RAW decoder involved")
    photo.with_suffix(".xmp").write_bytes(_packet(57))
    req = api.EnhancePreviewRequest(
        session_id="synthetic", photo_id="photo",
        acr_color={"enabled": True, "photo_hsl_source": "sidecar"},
    )
    color = api._effective_acr_color(req, photo)
    rgb = np.random.default_rng(27).random((9, 31, 3), dtype=np.float32)
    before = rgb.copy()
    actual = native.native_acr_hsl_sdr(rgb, color["hsl"])
    expected = native.native_acr_hsl_sdr(rgb, [57.] + [0.] * 24)
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    np.testing.assert_array_equal(rgb, before)
    assert np.max(np.abs(actual - rgb)) > 0.01


@pytest.mark.parametrize("name,expected", [
    ("Group: Camera ACROS + R", True), ("camera bw suffix", True),
    ("Camera Deep Tone Monochrome 2", True), ("Group: Group: Camera BW", False),
    ("group: Camera BW", False), ("Adobe Monochrome", False),
    ("Camera StandarD", False), ("Group: ", False),
])
def test_exact_recovered_name_gate(name, expected):
    assert api._photo_hsl_monochrome(name) is expected


def test_inspection_uses_real_native_merge_but_never_claims_consumer_binding(acr_scene):
    client, path, _ = acr_scene
    path.with_suffix(".xmp").write_bytes(_packet(10, look=_look()))
    response = client.post("/api/enhance/acr-hsl", json={
        "session_id": "acr-preview-test", "photo_id": "photo", "packet_source": "sidecar",
    })
    assert response.status_code == 200
    data = response.json()
    assert data["consumer_bound"] is False
    assert data["current24"][0] == 10 and data["look"]["defaults24"][9] == -14
    assert data["look"]["amount"] == 1.234
    assert data["local_merge"]["quantized_loader_amount"] == 1.23
    assert data["local_merge"]["wrote_mask"] == 1 << 9
    assert data["local_merge"]["values24"][0] == 10
    assert data["local_merge"]["scope"] == "metadata_projection_only"
    assert data["local_merge_reason"] == "consumer_state_unbound"
    assert response.headers["cache-control"] == "private, no-store"
    assert str(path) not in response.text


def test_local_merge_uses_nested_camera_profile_not_look_display_name(acr_scene):
    client, path, _ = acr_scene
    path.with_suffix(".xmp").write_bytes(_packet(10, look=_look("Group: Camera ACROS + R")))
    response = client.post("/api/enhance/acr-hsl", json={
        "session_id": "acr-preview-test", "photo_id": "photo", "packet_source": "sidecar",
    })
    assert response.status_code == 200
    data = response.json()
    assert data["local_merge"]["wrote_mask"] == 0
    assert data["local_merge"]["values24"] == data["current24"]
    assert data["consumer_bound"] is False


def test_inspection_rechecks_session_after_read(acr_scene, monkeypatch):
    client, _path, _ = acr_scene

    def read(*_):
        with api._ENHANCE_LOCK:
            api._ENHANCE_SESSIONS.pop("acr-preview-test")
        return {"look": None}

    monkeypatch.setattr(api, "inspect_photo_hsl", read)
    response = client.post("/api/enhance/acr-hsl", json={
        "session_id": "acr-preview-test", "photo_id": "photo", "packet_source": "embedded",
    })
    assert response.status_code == 404


def test_inspection_is_scoped_and_errors_are_path_free(monkeypatch):
    monkeypatch.setattr(api, "inspect_photo_hsl", lambda *_: pytest.fail("unauthorized file read"))
    client = TestClient(api.app)
    payload = {"session_id": "not-a-session", "photo_id": "photo", "packet_source": "sidecar"}
    assert client.post("/api/enhance/acr-hsl", json=payload).status_code == 404
    assert client.post("/api/enhance/acr-hsl", json=dict(payload, path="/private/photo.xmp")).status_code == 422


def test_inspection_sanitizes_failed_reader(acr_scene, monkeypatch):
    client, path, _ = acr_scene

    def fail(*_):
        raise OSError(str(path))

    monkeypatch.setattr(api, "inspect_photo_hsl", fail)
    response = client.post("/api/enhance/acr-hsl", json={
        "session_id": "acr-preview-test", "photo_id": "photo", "packet_source": "embedded",
    })
    assert response.status_code == 422 and str(path) not in response.text


def test_native_failure_never_becomes_python_approximation(acr_scene, monkeypatch):
    client, path, _ = acr_scene
    path.with_suffix(".xmp").write_bytes(_packet(look=_look()))

    def fail(*_):
        raise api.NativeRendererError("unavailable")

    monkeypatch.setattr(api, "native_acr_quantize_profile_gain_scalar", fail)
    response = client.post("/api/enhance/acr-hsl", json={
        "session_id": "acr-preview-test", "photo_id": "photo", "packet_source": "sidecar",
    })
    assert response.status_code == 503
    assert "local_merge" not in response.json()
