import ctypes
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import camera_profile as cp
import native_renderer as nr


class Call:
    def __init__(self, function):
        self.function = function

    def __call__(self, *args):
        return self.function(*args)


@pytest.fixture
def transfer_bridge(monkeypatch):
    state = {"uploads": 0, "renders": 0, "destroys": 0,
             "fail_upload": False, "fail_render": False}

    class Library:
        def im_renderer_destroy(self, _):
            state["destroys"] += 1

        def im_renderer_last_error(self, _):
            return b"transfer test failure"

    library = Library()

    def supports(_):
        return 1

    def upload(_, width, height, camera, reference, count):
        assert count == width * height * 3
        state["uploads"] += 1
        state["camera"] = np.ctypeslib.as_array(camera, shape=(count,)).copy()
        state["reference"] = np.ctypeslib.as_array(reference, shape=(count,)).copy()
        return int(state["fail_upload"])

    def render(_, processed, count, inverse, matrix_count, destination, output_count):
        assert matrix_count == 9 and output_count == count
        state["renders"] += 1
        if state["fail_render"]:
            return 1
        target = np.ctypeslib.as_array(destination, shape=(output_count,))
        target[:] = state["camera"]
        return 0

    library.im_renderer_supports_camera_profile_transfer = Call(supports)
    library.im_renderer_set_camera_profile_transfer_source = Call(upload)
    library.im_renderer_render_camera_profile_transfer = Call(render)
    monkeypatch.setattr(nr, "_camera_profile_transfer_gpu_state", None)
    monkeypatch.setattr(nr, "_get_library", lambda: (library, "fake-transfer"))
    monkeypatch.setattr(nr, "_create_renderer", lambda _: (ctypes.c_void_p(7), "Metal"))
    monkeypatch.setenv("IMPRINT_NATIVE_CAMERA_PROFILE_TRANSFER", "1")
    yield state
    nr.clear_camera_profile_gpu_cache()


def _small_images(*, readonly=False):
    camera = np.arange(18, dtype=np.uint16).reshape(2, 3, 3) + 100
    reference = camera.copy()
    processed = reference.copy()
    processed[..., 0] += 50
    if readonly:
        camera.setflags(write=False)
        reference.setflags(write=False)
    return camera, reference, processed


def test_readonly_transfer_pair_is_cached_and_result_owns_storage(transfer_bridge):
    camera, reference, processed = _small_images(readonly=True)
    first, backend = nr.native_camera_profile_transfer(
        camera, reference, processed, np.eye(3, dtype=np.float32),
    )
    second, _ = nr.native_camera_profile_transfer(
        camera, reference, processed, np.eye(3, dtype=np.float32),
    )
    assert backend == "metal"
    assert transfer_bridge["uploads"] == 1
    assert transfer_bridge["renders"] == 2
    assert np.array_equal(first, camera) and np.array_equal(second, camera)
    assert first.flags.owndata and not np.shares_memory(first, camera)


def test_mutable_transfer_pair_is_reuploaded_after_source_changes(transfer_bridge):
    camera, reference, processed = _small_images()
    first, _ = nr.native_camera_profile_transfer(camera, reference, processed, np.eye(3))
    camera += 1000
    second, _ = nr.native_camera_profile_transfer(camera, reference, processed, np.eye(3))
    assert transfer_bridge["uploads"] == 2
    assert np.array_equal(first, _small_images()[0])
    assert np.array_equal(second, camera)


@pytest.mark.parametrize("failure", ["upload", "render"])
def test_gpu_failure_destroys_transfer_renderer(transfer_bridge, failure):
    camera, reference, processed = _small_images()
    transfer_bridge[f"fail_{failure}"] = True
    with pytest.raises(nr.NativeRendererError, match="failed"):
        nr.native_camera_profile_transfer(camera, reference, processed, np.eye(3))
    assert nr._camera_profile_transfer_gpu_state is None
    assert transfer_bridge["destroys"] == 1


def test_profile_cache_clear_also_releases_transfer_renderer(transfer_bridge):
    camera, reference, processed = _small_images()
    nr.native_camera_profile_transfer(camera, reference, processed, np.eye(3))
    nr.clear_camera_profile_gpu_cache()
    assert nr._camera_profile_transfer_gpu_state is None
    assert transfer_bridge["destroys"] == 1


def test_missing_transfer_abi_fails_before_renderer_creation(monkeypatch):
    created = []
    monkeypatch.setattr(nr, "_camera_profile_transfer_gpu_state", None)
    monkeypatch.setattr(nr, "_get_library", lambda: (object(), "old-library"))
    monkeypatch.setattr(nr, "_create_renderer", lambda _: created.append(True))
    camera, reference, processed = _small_images()
    with pytest.raises(nr.NativeRendererError, match="ABI is unavailable"):
        nr.native_camera_profile_transfer(camera, reference, processed, np.eye(3))
    assert created == []


@pytest.mark.parametrize(
    ("backend", "pixels", "setting", "expected"),
    [
        ("native", 240_000, None, "cpp_cpu"),
        ("native", 1_048_576, None, "metal"),
        ("native", 240_000, "1", "metal"),
        ("native", 1_048_576, "0", "cpp_cpu"),
        ("cpu", 1_048_576, "1", "cpp_cpu"),
    ],
)
def test_profile_transfer_backend_gate(monkeypatch, backend, pixels, setting, expected):
    monkeypatch.setattr(cp.sys, "platform", "darwin")
    if setting is None:
        monkeypatch.delenv("IMPRINT_NATIVE_CAMERA_PROFILE_TRANSFER", raising=False)
    else:
        monkeypatch.setenv("IMPRINT_NATIVE_CAMERA_PROFILE_TRANSFER", setting)
    height = 400 if pixels == 240_000 else 1024
    width = pixels // height
    camera = np.full((height, width, 3), 12000, dtype=np.uint16)
    reference = camera.copy()
    processed = reference.copy()
    processed[..., 0] += 200
    monkeypatch.setattr(cp, "_fit_transfer_matrix", lambda *_: np.eye(3, dtype=np.float32))
    calls = {"gpu": 0, "cpu": 0}

    def gpu(*args):
        calls["gpu"] += 1
        return np.full_like(camera, 111), "metal"

    def cpu(*args):
        calls["cpu"] += 1
        return np.full_like(camera, 222)

    monkeypatch.setattr(cp, "native_camera_profile_transfer", gpu)
    monkeypatch.setattr(cp.native_dense, "camera_profile_transfer", cpu)
    result = cp.transfer_enhancement(camera, reference, processed, backend=backend)
    assert cp.get_last_transfer_backend() == expected
    assert calls["gpu"] == (expected == "metal")
    assert calls["cpu"] == (expected == "cpp_cpu")
    assert np.all(result == (111 if expected == "metal" else 222))


def test_gpu_runtime_error_falls_back_to_cpp_cpu(monkeypatch):
    monkeypatch.setattr(cp.sys, "platform", "darwin")
    monkeypatch.setenv("IMPRINT_NATIVE_CAMERA_PROFILE_TRANSFER", "1")
    camera = np.full((400, 600, 3), 12000, dtype=np.uint16)
    reference = camera.copy()
    processed = reference.copy()
    processed[..., 0] += 200
    monkeypatch.setattr(cp, "_fit_transfer_matrix", lambda *_: np.eye(3, dtype=np.float32))

    def unavailable(*_):
        raise nr.NativeRendererError("GPU runtime failure")

    monkeypatch.setattr(cp, "native_camera_profile_transfer", unavailable)
    monkeypatch.setattr(cp.native_dense, "camera_profile_transfer", lambda *_: np.full_like(camera, 333))
    output = cp.transfer_enhancement(camera, reference, processed, backend="native")
    assert cp.get_last_transfer_backend() == "cpp_cpu"
    assert np.all(output == 333)
