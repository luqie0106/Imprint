from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import ricoh_filter


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
@pytest.mark.parametrize("preset_id", ["gr3_standard", "gr3_high_contrast_bw"])
@pytest.mark.parametrize("measured", [False, True])
def test_parallel_rows_match_serial_for_color_and_grayscale_presets(
    monkeypatch, dtype, preset_id: str, measured: bool,
):
    source_max = np.iinfo(dtype).max
    source = np.random.default_rng(913).integers(
        0, source_max + 1, size=(130, 79, 3), dtype=dtype,
    )
    original = source.copy()
    # Exercise the same 64-row map path with a compact fixture. A separate
    # test below checks the production one-million-pixel threshold.
    monkeypatch.setattr(ricoh_filter, "_RICOH_PARALLEL_PIXEL_THRESHOLD", 1)

    monkeypatch.setenv("IMPRINT_RICOH_ROW_WORKERS", "1")
    serial = ricoh_filter.apply_ricoh_preview_effect(
        source, preset_id, use_measured_color=measured,
    )
    monkeypatch.setenv("IMPRINT_RICOH_ROW_WORKERS", "4")
    parallel = ricoh_filter.apply_ricoh_preview_effect(
        source, preset_id, use_measured_color=measured,
    )

    assert np.array_equal(parallel, serial)
    assert np.array_equal(source, original)
    assert parallel.dtype == dtype
    assert parallel.shape == source.shape


def test_parallel_rows_run_at_the_one_million_pixel_threshold(monkeypatch):
    source = np.random.default_rng(152).integers(
        0, 256, size=(1000, 1000, 3), dtype=np.uint8,
    )
    monkeypatch.setenv("IMPRINT_RICOH_ROW_WORKERS", "1")
    serial = ricoh_filter.apply_ricoh_preview_effect(source, "gr3_standard")
    monkeypatch.setenv("IMPRINT_RICOH_ROW_WORKERS", "4")
    parallel = ricoh_filter.apply_ricoh_preview_effect(source, "gr3_standard")

    assert np.array_equal(parallel, serial)


def test_small_preview_does_not_create_a_thread_pool(monkeypatch):
    source = np.full((8, 12, 3), 96, dtype=np.uint8)
    monkeypatch.setenv("IMPRINT_RICOH_ROW_WORKERS", "4")

    def unexpected_pool(*_args, **_kwargs):
        raise AssertionError("small preview should stay serial")

    monkeypatch.setattr(ricoh_filter, "ThreadPoolExecutor", unexpected_pool)
    result = ricoh_filter.apply_ricoh_preview_effect(source, "gr3_standard")

    assert result.shape == source.shape


@pytest.mark.parametrize(
    ("configured", "expected"),
    [(None, 4), ("invalid", 4), ("0", 1), ("1", 1), ("4", 4), ("12", 4), ("-2", 1)],
)
def test_row_worker_setting_uses_default_and_bounds(monkeypatch, configured, expected: int):
    if configured is None:
        monkeypatch.delenv("IMPRINT_RICOH_ROW_WORKERS", raising=False)
    else:
        monkeypatch.setenv("IMPRINT_RICOH_ROW_WORKERS", configured)

    assert ricoh_filter._ricoh_row_worker_count() == expected


def test_worker_exception_waits_for_other_rows_before_propagating(monkeypatch):
    monkeypatch.setattr(ricoh_filter, "_RICOH_PARALLEL_PIXEL_THRESHOLD", 1)
    monkeypatch.setenv("IMPRINT_RICOH_ROW_WORKERS", "2")
    source = np.full((130, 67, 3), 80, dtype=np.uint8)
    first_at_curves = threading.Event()
    second_at_curves = threading.Event()
    release_second = threading.Event()
    finished = threading.Event()
    calls_lock = threading.Lock()
    calls = 0
    errors: list[BaseException] = []

    def controlled_curves(rgb, *_args, **_kwargs):
        nonlocal calls
        with calls_lock:
            calls += 1
            call_number = calls
        if call_number == 1:
            first_at_curves.set()
            assert second_at_curves.wait(timeout=5)
            raise RuntimeError("row rendering failed")
        if call_number == 2:
            second_at_curves.set()
            assert release_second.wait(timeout=5)
        return rgb

    monkeypatch.setattr(ricoh_filter, "_apply_curves", controlled_curves)

    def run_effect():
        try:
            ricoh_filter.apply_ricoh_preview_effect(source, "gr3_standard")
        except BaseException as exc:
            errors.append(exc)
        finally:
            finished.set()

    caller = threading.Thread(target=run_effect)
    caller.start()
    assert first_at_curves.wait(timeout=5)
    assert second_at_curves.wait(timeout=5)
    assert not finished.wait(timeout=0.05)
    release_second.set()
    assert finished.wait(timeout=5)
    caller.join(timeout=5)

    assert not caller.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert str(errors[0]) == "row rendering failed"
