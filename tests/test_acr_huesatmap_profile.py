from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import acr_profile_metadata
import camera_profile as cp
from dng_writer import TYPE_SIZES, _Tag


PROFILE_NAME = "HSM-only test profile"


def _tag(code: int, values, *, kind: int = 11) -> _Tag:
    array = np.asarray(values)
    if kind == 11:
        data = array.astype("<f4").tobytes()
    elif kind == 4:
        data = array.astype("<u4").tobytes()
    elif kind == 3:
        data = array.astype("<u2").tobytes()
    else:
        raise AssertionError(kind)
    return _Tag(code, kind, int(array.size), data)


def _hsm_tags() -> list[_Tag]:
    identity = np.eye(3, dtype=np.float32).reshape(-1)
    curve = np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32).reshape(-1)
    dims = np.asarray([3, 2, 1], dtype=np.uint32)
    first = np.asarray(
        [[[[0.0, 1.0, 1.0], [2.0, 0.9, 1.0]],
          [[4.0, 1.1, 1.0], [6.0, 1.0, 0.9]],
          [[8.0, 1.0, 1.1], [10.0, 0.8, 1.0]]]],
        dtype=np.float32,
    )
    second = np.asarray(
        [[[[1.0, 1.0, 1.0], [3.0, 1.0, 0.9]],
          [[5.0, 0.9, 1.0], [7.0, 1.0, 1.1]],
          [[9.0, 1.1, 1.0], [11.0, 1.0, 0.8]]]],
        dtype=np.float32,
    )
    return [
        _Tag(50936, 2, len(PROFILE_NAME) + 1, PROFILE_NAME.encode("ascii") + b"\0"),
        _tag(50721, identity), _tag(50722, identity),
        _tag(50778, [17], kind=3), _tag(50779, [21], kind=3),
        _tag(50940, curve), _tag(50937, dims, kind=4),
        _tag(50938, first), _tag(50939, second),
    ]


def _tag_map(tags: list[_Tag]) -> dict[int, _Tag]:
    return {tag.code: tag for tag in tags}


def _minimal_profile_dng(tags: list[_Tag]) -> bytes:
    ordered_tags = sorted(tags, key=lambda tag: tag.code)
    payload = bytearray()
    entries = bytearray()
    payload_offset = 8 + 2 + 12 * len(ordered_tags) + 4
    for tag in ordered_tags:
        data = bytes(tag.data)
        assert len(data) == tag.count * TYPE_SIZES[tag.kind]
        if len(data) <= 4:
            value_field = data.ljust(4, b"\0")
        else:
            value_field = np.asarray([payload_offset + len(payload)], dtype="<u4").tobytes()
            payload.extend(data)
        entries.extend(np.asarray([tag.code, tag.kind], dtype="<u2").tobytes())
        entries.extend(np.asarray([tag.count], dtype="<u4").tobytes())
        entries.extend(value_field)
    return (
        b"II*\0" + np.asarray([8], dtype="<u4").tobytes()
        + np.asarray([len(ordered_tags)], dtype="<u2").tobytes()
        + bytes(entries) + b"\0\0\0\0" + bytes(payload)
    )


def _profile(tags: list[_Tag] | None = None) -> cp.Profile:
    return cp._parse_profile(
        list(_hsm_tags() if tags is None else tags), PROFILE_NAME,
        ("synthetic-hsm",), _acr_extended=True,
    )


def _append_full_profile_parts(tags: list[_Tag]) -> list[_Tag]:
    result = list(tags)
    identity = np.eye(3, dtype=np.float32).reshape(-1)
    result.extend((_tag(50964, identity), _tag(50965, identity)))
    dims = np.asarray([4, 2, 2], dtype=np.uint32)
    look = np.tile(np.asarray([0.0, 1.0, 1.0], dtype=np.float32), 4 * 2 * 2)
    result.extend((
        _tag(50981, dims, kind=4), _tag(50982, look),
        _tag(51108, [0], kind=4),
    ))
    return result


def test_acr_parser_accepts_hsm_without_forward_matrices_or_look_table() -> None:
    tags = _hsm_tags()
    profile = _profile(tags)

    assert profile.acr_only is True
    assert profile.has_forward1 is False
    assert profile.has_forward2 is False
    assert profile.forward_matrix1.shape == (3, 3)
    assert not np.any(profile.forward_matrix1)
    assert not np.any(profile.forward_matrix2)
    assert profile.look_table is None
    assert profile.hue_sat_dims == (3, 2, 1)
    assert profile.hue_sat_encoding == 0
    assert profile.hue_sat_map1 is not None
    assert profile.hue_sat_map2 is not None
    assert profile.hue_sat_map1.shape == (1, 3, 2, 3)
    assert not profile.hue_sat_map1.flags.writeable
    assert not profile.hue_sat_map2.flags.writeable
    assert np.array_equal(profile.hue_sat_map1, np.frombuffer(tags[7].data, dtype="<f4").reshape(1, 3, 2, 3))

    with pytest.raises(ValueError):
        cp._parse_profile(tags, PROFILE_NAME, ("legacy",))

    # The legacy parser also rejects non-identity HSM when the otherwise
    # required ForwardMatrices and LookTable are present.
    with pytest.raises(ValueError, match="non-identity HueSatMap"):
        cp._parse_profile(_append_full_profile_parts(tags), PROFILE_NAME, ("legacy-hsm",))


def test_acr_parser_accepts_single_color_matrix_and_marks_second_matrix_missing() -> None:
    tags = [tag for tag in _hsm_tags() if tag.code != 50722]
    profile = _profile(tags)

    assert profile.has_color_matrix2 is False
    assert np.array_equal(profile.color_matrix1, np.eye(3))
    assert np.array_equal(profile.color_matrix2, np.zeros((3, 3)))
    assert not profile.color_matrix2.flags.writeable


@pytest.mark.parametrize("missing", [(50778,), (50779,), (50778, 50779)])
def test_acr_parser_defaults_absent_illuminant_tags_to_zero(missing) -> None:
    tags = [tag for tag in _hsm_tags() if tag.code not in missing]
    profile = _profile(tags)
    expected = tuple(0 if code in missing else (17 if code == 50778 else 21)
                     for code in (50778, 50779))

    assert profile.calibration_illuminants == expected


@pytest.mark.parametrize("matrix_code", [50721, 50722])
@pytest.mark.parametrize("invalid", ["wrong-count", "singular"])
def test_acr_parser_rejects_invalid_present_color_matrices(matrix_code, invalid) -> None:
    tags = _tag_map(_hsm_tags())
    if invalid == "wrong-count":
        tags[matrix_code] = _tag(matrix_code, np.ones(8, dtype=np.float32))
    else:
        tags[matrix_code] = _tag(matrix_code, np.zeros(9, dtype=np.float32))

    with pytest.raises(ValueError, match="profile tag 5072[12]|singular"):
        _profile(list(tags.values()))


@pytest.mark.parametrize("third_illuminant_tag", [52529, 52530, 52531, 52532, 52535, 52537])
def test_acr_parser_rejects_third_illuminant_markers_even_without_cm2(third_illuminant_tag) -> None:
    tags = [tag for tag in _hsm_tags() if tag.code != 50722]
    tags.append(_tag(third_illuminant_tag, [1], kind=4))

    with pytest.raises(ValueError, match="three-illuminant"):
        _profile(tags)


@pytest.mark.parametrize("missing", [50722, 50778, 50779])
def test_legacy_parser_still_requires_second_matrix_and_both_illuminants(missing) -> None:
    tags = _append_full_profile_parts(_hsm_tags())
    if missing in (50778, 50779):
        tags = [tag for tag in tags if tag.code != missing]
    else:
        tags = [tag for tag in tags if tag.code != 50722]
    # Legacy parsing accepts only an identity HueSatMap so the missing
    # two-illuminant metadata is the reason these profiles must be rejected.
    tags_map = _tag_map(tags)
    identity_hsm = np.tile(np.asarray([0.0, 1.0, 1.0], dtype=np.float32), 6)
    tags_map[50938] = _tag(50938, identity_hsm)
    tags_map[50939] = _tag(50939, identity_hsm)

    with pytest.raises(ValueError):
        cp._parse_profile(list(tags_map.values()), PROFILE_NAME, ("legacy-missing",))


@pytest.mark.parametrize(
    "invalid_case",
    [
        "dimension-count", "dimension-range", "data-count", "nan-data",
        "encoding-kind", "encoding-value", "missing-dimensions", "missing-map1",
        "data3", "multi-value",
    ],
)
def test_acr_parser_rejects_unsupported_or_partial_huesatmaps(invalid_case: str) -> None:
    tags = _tag_map(_hsm_tags())
    if invalid_case == "dimension-count":
        tags[50937] = _tag(50937, [3, 2], kind=4)
    elif invalid_case == "dimension-range":
        tags[50937] = _tag(50937, [3, 1, 1], kind=4)
    elif invalid_case == "data-count":
        tags[50938] = _tag(50938, np.zeros(17, dtype=np.float32))
    elif invalid_case == "nan-data":
        data = np.frombuffer(tags[50938].data, dtype="<f4").copy()
        data[0] = np.nan
        tags[50938] = _tag(50938, data)
    elif invalid_case == "encoding-kind":
        tags[51107] = _tag(51107, [0], kind=3)
    elif invalid_case == "encoding-value":
        tags[51107] = _tag(51107, [2], kind=4)
    elif invalid_case == "missing-dimensions":
        del tags[50937]
    elif invalid_case == "missing-map1":
        del tags[50938]
    elif invalid_case == "data3":
        tags[52537] = _tag(52537, [3, 2, 1], kind=4)
    elif invalid_case == "multi-value":
        tags[50937] = _tag(50937, [3, 2, 2], kind=4)

    with pytest.raises(ValueError):
        cp._parse_profile(list(tags.values()), PROFILE_NAME, ("bad-hsm",), _acr_extended=True)


def _prepared_camera() -> np.ndarray:
    prepared = np.zeros(65, dtype=np.float64)
    prepared[2:4] = [0.3125, 0.329]
    prepared[35:44] = np.arange(101.0, 110.0)
    prepared[47:50] = [1.1, 2.2, 3.3]
    prepared[53:56] = [2.0, 0.0, 1.0]
    prepared[56:65] = np.arange(201.0, 210.0)
    return prepared


def _patch_forbidden_legacy_paths(monkeypatch) -> None:
    def fail(*_args, **_kwargs):
        pytest.fail("ACR HueSatMap rendering entered a legacy/Python helper")

    for name in (
        "_camera_transform", "_apply_look", "_apply_tone", "_project_srgb",
        "native_camera_profile", "native_profile_look", "native_acr_profile_look",
    ):
        monkeypatch.setattr(cp, name, fail, raising=False)
    monkeypatch.setattr(cp.native_dense, "camera_profile_pixels", fail)


def _patch_native_render_stages(
    monkeypatch, profile: cp.Profile, *, with_look: bool,
    endpoints=(2850.0, 6500.0),
):
    prepared = _prepared_camera()
    prepared_hsm = np.full((1, 3, 2, 3), 0.375, dtype=np.float32)
    calls: list[tuple] = []
    events: list[str] = []

    def prepare_camera(controls):
        calls.append(("prepare-camera", np.array(controls, copy=True)))
        return prepared.copy()

    def illuminant_temperatures(codes):
        calls.append(("illuminants", tuple(codes)))
        return np.asarray(endpoints, dtype=np.float64)

    def prepare_hsm(first, second, dims, white_xy, *, endpoint_kelvin):
        calls.append((
            "prepare-hsm", np.array(first, copy=True), np.array(second, copy=True),
            tuple(dims), np.array(white_xy, copy=True), np.array(endpoint_kelvin, copy=True),
        ))
        return prepared_hsm, {"selected": True}

    def camera_float(rgb, controls):
        events.append("camera")
        calls.append(("camera", rgb.copy(), np.array(controls, copy=True)))
        return rgb.copy()

    def stage(rgb, table, dims, encoding):
        if table is prepared_hsm:
            name = "hsm"
        elif with_look and table is profile.look_table:
            name = "look"
        else:
            pytest.fail("unexpected profile resource passed to the bulk stage")
        events.append(name)
        calls.append((name, rgb.copy(), table, tuple(dims), encoding))
        return rgb.copy()

    def scale(rgb, gain):
        events.append("scale")
        calls.append(("scale", rgb.copy(), gain))
        return rgb.copy()

    def hsl(rgb, controls):
        events.append("hsl")
        calls.append(("hsl", rgb.copy(), np.array(controls, copy=True)))
        return rgb.copy()

    def tone(rgb, points):
        events.append("tone")
        calls.append(("tone", rgb.copy(), np.array(points, copy=True)))
        return rgb.copy()

    def grading(rgb, controls):
        events.append("grading")
        calls.append(("grading", rgb.copy(), np.array(controls, copy=True)))
        return rgb.copy()

    def project(rgb, matrix):
        events.append("project")
        calls.append(("project", rgb.copy(), np.array(matrix, copy=True)))
        return np.full(rgb.shape, 73, dtype=np.uint8)

    monkeypatch.setattr(cp, "native_acr_prepare_profile_camera", prepare_camera)
    monkeypatch.setattr(cp, "native_acr_illuminant_temperatures", illuminant_temperatures)
    monkeypatch.setattr(cp, "native_acr_prepare_huesatmap", prepare_hsm)
    monkeypatch.setattr(cp, "native_acr_camera_float", camera_float)
    monkeypatch.setattr(cp, "native_acr_profile_stage_sdr", stage)
    monkeypatch.setattr(cp, "native_acr_scale_sdr", scale)
    monkeypatch.setattr(cp, "native_acr_hsl_sdr", hsl)
    monkeypatch.setattr(cp, "native_acr_profile_tone", tone)
    monkeypatch.setattr(cp, "native_acr_grading_sdr", grading)
    monkeypatch.setattr(cp, "native_acr_project_srgb", project)
    _patch_forbidden_legacy_paths(monkeypatch)
    return prepared, prepared_hsm, calls, events


def test_hsm_render_blends_once_and_runs_before_exposure_for_each_stripe(monkeypatch) -> None:
    profile = _profile()
    prepared, prepared_hsm, calls, events = _patch_native_render_stages(
        monkeypatch, profile, with_look=False,
    )
    camera = np.full((129, 2, 3), 12000, dtype=np.uint16)
    result = cp.render_profile(
        camera, (0.75, 1.125, 0.5), profile, exposure_ev=0.5,
        acr_color={"enabled": True},
    )

    hsm_preparations = [entry for entry in calls if entry[0] == "prepare-hsm"]
    assert len(hsm_preparations) == 1
    first, second, dims, white_xy, endpoints = hsm_preparations[0][1:]
    assert np.array_equal(first, profile.hue_sat_map1)
    assert np.array_equal(second, profile.hue_sat_map2)
    assert dims == profile.hue_sat_dims
    assert np.array_equal(white_xy, prepared[2:4])
    assert np.array_equal(endpoints, [2850.0, 6500.0])
    assert sum(entry[0] == "hsm" for entry in calls) == 2
    assert not any(entry[0] == "look" for entry in calls)
    assert events == [
        "camera", "hsm", "scale", "hsl", "tone", "grading", "project",
        "camera", "hsm", "scale", "hsl", "tone", "grading", "project",
    ]

    controls = next(entry[1] for entry in calls if entry[0] == "prepare-camera")
    assert np.array_equal(controls[39:57], np.zeros(18))
    assert np.array_equal(controls[61:64], [1.0, 0.0, 0.0])
    assert np.array_equal(controls[64:67], [0.75, 1.125, 0.5])
    rendered_camera = [entry for entry in calls if entry[0] == "camera"]
    assert [entry[1].shape[0] for entry in rendered_camera] == [128, 1]
    expected_camera_controls = np.concatenate(([3.3, 1.1, 1.0, 1.0], prepared[35:44]))
    assert all(np.array_equal(entry[2], expected_camera_controls) for entry in rendered_camera)
    assert all(entry[2] is prepared_hsm for entry in calls if entry[0] == "hsm")
    assert all(entry[2] == np.float32(2.0 ** 0.5) for entry in calls if entry[0] == "scale")
    assert result.shape == camera.shape
    assert result.dtype == np.uint8
    assert np.all(result == 73)


def test_acr_render_keeps_optional_look_after_hsm_and_exposure(monkeypatch) -> None:
    tags = _append_full_profile_parts(_hsm_tags())
    profile = _profile(tags)
    assert profile.look_table is not None
    assert profile.has_forward1 and profile.has_forward2
    _prepared, _prepared_hsm, calls, events = _patch_native_render_stages(
        monkeypatch, profile, with_look=True,
    )
    result = cp.render_profile(
        np.full((3, 4, 3), 25000, dtype=np.uint16), (1.0, 1.0, 1.0),
        profile, exposure_ev=0.0, acr_color={"enabled": True},
    )

    assert events == ["camera", "hsm", "scale", "look", "hsl", "tone", "grading", "project"]
    assert sum(entry[0] == "look" for entry in calls) == 1
    controls = next(entry[1] for entry in calls if entry[0] == "prepare-camera")
    assert np.array_equal(controls[39:57], np.concatenate((
        profile.forward_matrix1.reshape(-1), profile.forward_matrix2.reshape(-1),
    )))
    assert np.array_equal(controls[61:64], [1.0, 1.0, 1.0])
    assert result.dtype == np.uint8 and np.all(result == 73)


def test_acr_only_profile_cannot_render_when_experiment_is_disabled(monkeypatch) -> None:
    profile = _profile()

    def unexpected(*_args, **_kwargs):
        pytest.fail("disabled ACR-only profile invoked a renderer")

    monkeypatch.setattr(cp, "native_acr_prepare_profile_camera", unexpected)
    monkeypatch.setattr(cp, "_camera_transform", unexpected)
    monkeypatch.setattr(cp.native_dense, "camera_profile_pixels", unexpected)
    with pytest.raises(ValueError, match="requires the native ACR preview experiment"):
        cp.render_profile(
            np.full((2, 2, 3), 1, dtype=np.uint16), (1.0, 1.0, 1.0),
            profile, exposure_ev=0.0,
        )


def test_resolver_routes_acr_profiles_through_preview_reader_with_isolated_fingerprint(
    tmp_path, monkeypatch,
) -> None:
    source = tmp_path / "source.nef"
    candidate = tmp_path / "camera-profile.dng"
    source.write_bytes(b"synthetic raw metadata fixture")
    candidate.write_bytes(b"synthetic dng metadata fixture")
    tags = _hsm_tags()
    reader_calls: list[str] = []

    monkeypatch.setattr(cp, "camera_profile_names", lambda _source: {})
    monkeypatch.setattr(cp, "matching_embedded_profile_dng", lambda *_args: candidate)

    def acr_reader(path, name):
        reader_calls.append("acr")
        assert Path(path) == candidate
        assert name == PROFILE_NAME
        return tags

    def legacy_reader(path, name):
        reader_calls.append("legacy")
        assert Path(path) == candidate
        assert name == PROFILE_NAME
        return tags

    monkeypatch.setattr(cp, "read_acr_profile_tags", acr_reader)
    monkeypatch.setattr(cp, "_embedded_profile_tags", legacy_reader)
    info = {"SourceCameraProfileName": PROFILE_NAME, "Make": "Example", "Model": "Camera"}

    acr_profile = cp.resolve_acr_profile(source, info)
    legacy_profile = cp.resolve_profile(source, info)

    assert acr_profile is not None and acr_profile.acr_only
    assert acr_profile.fingerprint[0] == "acr-huesatmap-v4-source-calibration"
    assert legacy_profile is None
    assert reader_calls == ["acr", "legacy"]


def test_acr_resolver_uses_root_profile_name_for_dng_without_selected_name(
    tmp_path, monkeypatch,
) -> None:
    source = tmp_path / "embedded-profile.dng"
    source.write_bytes(_minimal_profile_dng(_hsm_tags()))
    embedded_calls: list[Path] = []
    named_calls: list[tuple[Path, str]] = []
    read_embedded = acr_profile_metadata.read_embedded_acr_profile_tags
    read_named = acr_profile_metadata.read_acr_profile_tags

    monkeypatch.setattr(cp, "camera_profile_names", lambda _path: {})

    def observe_embedded(path):
        embedded_calls.append(Path(path))
        return read_embedded(Path(path))

    def observe_named(path, name):
        named_calls.append((Path(path), name))
        return read_named(Path(path), name)

    monkeypatch.setattr(cp, "read_embedded_acr_profile_tags", observe_embedded)
    monkeypatch.setattr(cp, "read_acr_profile_tags", observe_named)

    profile = cp.resolve_acr_profile(source, {})

    assert profile is not None
    assert profile.name == PROFILE_NAME
    assert profile.acr_only
    assert profile.fingerprint[0] == "acr-huesatmap-v4-source-calibration"
    assert embedded_calls == [source.resolve()]
    assert named_calls == []


def test_acr_resolver_rejects_explicit_profile_name_that_conflicts_with_dng_root(
    tmp_path, monkeypatch,
) -> None:
    source = tmp_path / "conflicting-profile.dng"
    source.write_bytes(_minimal_profile_dng(_hsm_tags()))
    embedded_calls: list[Path] = []
    named_calls: list[tuple[Path, str]] = []
    read_embedded = acr_profile_metadata.read_embedded_acr_profile_tags
    read_named = acr_profile_metadata.read_acr_profile_tags

    monkeypatch.setattr(cp, "camera_profile_names", lambda _path: {})

    def observe_embedded(path):
        embedded_calls.append(Path(path))
        return read_embedded(Path(path))

    def observe_named(path, name):
        named_calls.append((Path(path), name))
        return read_named(Path(path), name)

    monkeypatch.setattr(cp, "read_embedded_acr_profile_tags", observe_embedded)
    monkeypatch.setattr(cp, "read_acr_profile_tags", observe_named)

    profile = cp.resolve_acr_profile(
        source, {"SourceCameraProfileName": "Explicit conflicting profile"},
    )

    assert profile is None
    assert embedded_calls == []
    assert named_calls == [(source.resolve(), "Explicit conflicting profile")]


def test_legacy_resolver_still_rejects_dng_without_selected_profile_name(
    tmp_path, monkeypatch,
) -> None:
    source = tmp_path / "legacy-no-selected-name.dng"
    source.write_bytes(_minimal_profile_dng(_hsm_tags()))
    monkeypatch.setattr(cp, "camera_profile_names", lambda _path: {})

    def unexpected(*_args):
        pytest.fail("legacy no-name resolution must stop before reading root profile metadata")

    monkeypatch.setattr(cp, "read_embedded_acr_profile_tags", unexpected)
    monkeypatch.setattr(cp, "_embedded_profile_tags", unexpected)

    assert cp.resolve_profile(source, {}) is None


@pytest.mark.parametrize("codes", [(2, 22), (21, 17), (1, 4), (0, 21), (254, 17)])
def test_acr_parser_keeps_fixed_illuminants_for_native_resolution(codes):
    tags = [tag for tag in _hsm_tags() if tag.code not in (50778, 50779)]
    tags += [_tag(50778, [codes[0]], kind=3), _tag(50779, [codes[1]], kind=3)]
    profile = _profile(tags)
    assert profile.calibration_illuminants == codes
    with pytest.raises(ValueError, match="only calibration illuminants"):
        cp._parse_profile(_append_full_profile_parts(tags), PROFILE_NAME, ("legacy",))


@pytest.mark.parametrize("code,kind", [(255, 3), (65536, 4), (17.5, 11)])
def test_acr_parser_rejects_unprepared_custom_or_invalid_illuminants(code, kind):
    tags = [tag for tag in _hsm_tags() if tag.code != 50778]
    tags.append(_tag(50778, [code], kind=kind))
    with pytest.raises(ValueError, match="calibration illuminant"):
        _profile(tags)


def test_render_passes_one_native_endpoint_pair_to_camera_and_hsm(monkeypatch):
    from dataclasses import replace
    profile = replace(_profile(), calibration_illuminants=(2, 22))
    _, _, calls, _ = _patch_native_render_stages(
        monkeypatch, profile, with_look=False, endpoints=(4150.0, 7500.0),
    )
    cp.render_profile(np.full((129, 2, 3), 12000, dtype=np.uint16),
                      (1.0, 1.0, 1.0), profile, 0.0, acr_color={"enabled": True})
    assert [call for call in calls if call[0] == "illuminants"] == [("illuminants", (2, 22))]
    controls = next(call[1] for call in calls if call[0] == "prepare-camera")
    assert np.array_equal(controls[57:59], [4150.0, 7500.0])
    hsm_calls = [call for call in calls if call[0] == "prepare-hsm"]
    assert len(hsm_calls) == 1
    assert np.array_equal(hsm_calls[0][5], controls[57:59])


def test_render_passes_missing_second_color_matrix_flag_to_native(monkeypatch):
    tags = [tag for tag in _hsm_tags() if tag.code != 50722]
    profile = _profile(_append_full_profile_parts(tags))
    _, _, calls, _ = _patch_native_render_stages(
        monkeypatch, profile, with_look=True,
    )

    cp.render_profile(
        np.full((2, 3, 3), 12000, dtype=np.uint16), (1.0, 1.0, 1.0),
        profile, 0.0, acr_color={"enabled": True},
    )

    controls = next(call[1] for call in calls if call[0] == "prepare-camera")
    assert np.array_equal(controls[12:21], np.zeros(9))
    assert np.array_equal(controls[61:64], [0.0, 1.0, 1.0])


def test_illuminant_native_failure_has_no_python_fallback(monkeypatch):
    def unavailable(_codes):
        raise cp.NativeRendererError("illuminant operator unavailable")
    monkeypatch.setattr(cp, "native_acr_illuminant_temperatures", unavailable)
    _patch_forbidden_legacy_paths(monkeypatch)
    with pytest.raises(cp.NativeRendererError, match="illuminant operator unavailable"):
        cp.render_profile(np.ones((1, 1, 3), dtype=np.uint16),
                          (1.0, 1.0, 1.0), _profile(), 0.0,
                          acr_color={"enabled": True})
