import ctypes
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import native_renderer as nr


class Call:
    def __init__(self, function):
        self.function = function

    def __call__(self, *args):
        return self.function(*args)


@pytest.fixture
def bridge(monkeypatch):
    state = {"upload": 0, "render": 0, "destroy": 0, "failure": False}
    class Library:
        def im_renderer_destroy(self, _):
            state["destroy"] += 1
        def im_renderer_last_error(self, _):
            return b"test failure"
    library = Library()
    def upload(_, w, h, data, count):
        assert count == w * h * 3
        state["upload"] += 1
        state["pixels"] = np.ctypeslib.as_array(data, shape=(count,)).copy()
        return 0
    def render(*args):
        state["render"] += 1
        if state["failure"]:
            return 1
        destination, count = args[-2:]
        np.ctypeslib.as_array(destination, shape=(count,))[:] = state["pixels"] // 257
        return 0
    library.im_renderer_set_camera_profile_source = Call(upload)
    library.im_renderer_render_camera_profile = Call(render)
    monkeypatch.setattr(nr, "_camera_profile_gpu_state", None)
    monkeypatch.setattr(nr, "_get_library", lambda: (library, "test"))
    monkeypatch.setattr(nr, "_create_renderer", lambda _: (ctypes.c_void_p(1), "Metal"))
    monkeypatch.delenv("IMPRINT_NATIVE_CAMERA_PROFILE", raising=False)
    yield state
    nr.clear_camera_profile_gpu_cache()


def render(source, gain=1):
    table = np.zeros((2, 1, 2, 3), np.float32)
    table[..., 1:] = 1
    return nr.native_camera_profile(source, np.eye(3), np.ones(3), gain,
        table, (1, 2, 2), 0, np.array([[0, 0], [1, 1]]), np.eye(3))


def test_readonly_source_reused_across_ev_and_replacement(bridge):
    image = np.full((2, 3, 3), 32768, np.uint16)
    image.setflags(write=False)
    first, backend = render(image)
    render(image, gain=2)
    assert backend == "metal"
    assert bridge["upload"] == 1
    assert bridge["render"] == 2
    replacement = np.full_like(image, 65535)
    replacement.setflags(write=False)
    second, _ = render(replacement)
    assert bridge["upload"] == 2
    assert np.all(first == 127) and np.all(second == 255)
    nr.clear_camera_profile_gpu_cache()
    assert bridge["destroy"] == 1
    render(replacement)
    assert bridge["upload"] == 3


def test_mutable_source_uploaded_again(bridge):
    image = np.zeros((2, 3, 3), np.uint16)
    render(image)
    image.fill(65535)
    result, _ = render(image)
    assert bridge["upload"] == 2
    assert np.all(result == 255)


def test_render_failure_releases_gpu_resource(bridge):
    bridge["failure"] = True
    with pytest.raises(nr.NativeRendererError):
        render(np.zeros((2, 3, 3), np.uint16))
    assert bridge["destroy"] == 1
    assert nr._camera_profile_gpu_state is None


def test_explicit_disable_avoids_gpu_creation(bridge, monkeypatch):
    monkeypatch.setenv("IMPRINT_NATIVE_CAMERA_PROFILE", "0")
    with pytest.raises(nr.NativeRendererError):
        render(np.zeros((2, 3, 3), np.uint16))
    assert bridge["upload"] == 0 and bridge["render"] == 0


def test_old_library_falls_back_before_gpu_creation(bridge, monkeypatch):
    monkeypatch.setattr(nr, "_get_library", lambda: (object(), "old-library"))
    with pytest.raises(nr.NativeRendererError, match="ABI"):
        render(np.zeros((2, 3, 3), np.uint16))
    assert bridge["upload"] == 0
