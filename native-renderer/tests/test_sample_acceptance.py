"""Read-only RAW acceptance parameter, scope and output safety tests."""

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import physical_linear_acceptance as acceptance


def arguments(tmp_path, *extra):
    samples = tmp_path / "sample"
    samples.mkdir(exist_ok=True)
    return ["--baseline", str(tmp_path / "baseline"), "--samples-dir", str(samples),
            "--output", str(tmp_path / "output"), *extra]


@pytest.mark.parametrize("contents", [b"", b"read-only RAW" * 200000])
def test_digest_streams_read_only_files_and_preserves_contents(tmp_path, contents):
    path = tmp_path / "sample.NEF"
    path.write_bytes(contents)
    path.chmod(0o444)
    before_mode = path.stat().st_mode

    assert acceptance.digest(path) == hashlib.sha256(contents).hexdigest()
    assert path.read_bytes() == contents
    assert path.stat().st_mode == before_mode


def test_digest_missing_file_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        acceptance.digest(tmp_path / "missing.NEF")


def test_default_prefers_sample_then_test_images(tmp_path, monkeypatch):
    monkeypatch.setattr(acceptance, "ROOT", tmp_path)
    fallback = tmp_path / "test_images"
    fallback.mkdir()
    argv = ["--baseline", str(tmp_path / "baseline"), "--output", str(tmp_path / "output")]
    assert acceptance.parse_args(argv).samples_dir == fallback
    preferred = tmp_path / "sample"
    preferred.mkdir()
    parsed = acceptance.parse_args(argv)
    assert parsed.samples_dir == preferred
    assert parsed.max_edge is None


@pytest.mark.parametrize("extra, message", [
    (("--max-edge", "0"), "positive integer"),
    (("--max-edge", "-5"), "positive integer"),
    (("--max-edge", "1.5"), "invalid int value"),
    (("--strength", "nan"), "--strength must be"),
    (("--strength", "1.1"), "--strength must be"),
    (("--only",), "at least one sample"),
    (("--only", "unknown.NEF"), "unknown sample names"),
])
def test_invalid_options_fail_before_writing(tmp_path, capsys, extra, message):
    with pytest.raises(SystemExit) as error:
        acceptance.parse_args(arguments(tmp_path, *extra))
    assert error.value.code == 2
    assert message in capsys.readouterr().err
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("relative", ["sample", ".", "sample/other-output"])
def test_output_must_be_separate_from_input_tree(tmp_path, relative, capsys):
    with pytest.raises(SystemExit):
        acceptance.parse_args(arguments(tmp_path, "--output", str(tmp_path / relative)))
    assert "isolated from the input photo directory" in capsys.readouterr().err


def test_input_output_directory_errors_and_allowed_output_subtree(tmp_path, capsys):
    base = arguments(tmp_path)
    with pytest.raises(SystemExit):
        acceptance.parse_args([*base, "--samples-dir", str(tmp_path / "missing")])
    assert "existing directory" in capsys.readouterr().err
    file_output = tmp_path / "file-output"
    file_output.write_text("existing file")
    with pytest.raises(SystemExit):
        acceptance.parse_args([*base, "--output", str(file_output)])
    assert "must be a directory" in capsys.readouterr().err
    allowed = tmp_path / "sample" / "去朦胧输出" / "run"
    assert acceptance.parse_args([*base, "--output", str(allowed)]).output == allowed
    assert not allowed.exists()


def test_output_symlink_cannot_bypass_input_isolation(tmp_path):
    base = arguments(tmp_path)
    alias = tmp_path / "output-alias"
    alias.symlink_to(tmp_path / "sample", target_is_directory=True)
    with pytest.raises(SystemExit):
        acceptance.parse_args([*base, "--output", str(alias)])


def test_existing_outputs_require_explicit_resume_and_reject_symlinks(tmp_path):
    args = acceptance.parse_args(arguments(tmp_path))
    args.output.mkdir()
    path = args.samples_dir / acceptance.SAMPLE_NAMES[0]
    result = args.output / "验证结果.json"
    artifact = acceptance.sample_outputs(args.output, path)[0]
    artifact.write_text("existing output")
    with pytest.raises(FileExistsError):
        acceptance.refuse_existing_outputs(args.output, [path], args, result)
    assert artifact.read_text() == "existing output"

    args.only = [path.name]
    assert acceptance.refuse_existing_outputs(args.output, [path], args, result) == []
    protected = args.samples_dir / "protected.xmp"
    protected.write_text("source metadata")
    result.symlink_to(protected)
    with pytest.raises(ValueError, match="symbolic links"):
        acceptance.refuse_existing_outputs(args.output, [path], args, result)
    assert protected.read_text() == "source metadata"


def test_resume_cannot_mix_preview_resolution_or_source_directory(tmp_path):
    args = acceptance.parse_args(arguments(tmp_path, "--only", acceptance.SAMPLE_NAMES[0]))
    args.output.mkdir()
    result = args.output / "验证结果.json"
    rows = [{"file": acceptance.SAMPLE_NAMES[1], "max_edge": 640}]
    result.write_text(json.dumps(rows))
    paths = [args.samples_dir / name for name in acceptance.SAMPLE_NAMES]
    with pytest.raises(ValueError, match="different --max-edge"):
        acceptance.refuse_existing_outputs(args.output, paths, args, result)
    args.max_edge = 640
    assert acceptance.refuse_existing_outputs(args.output, paths, args, result) == rows
    (args.output / "验证范围.json").write_text(json.dumps({"samples_directory": "/different/sample"}))
    with pytest.raises(ValueError, match="different sample directory"):
        acceptance.refuse_existing_outputs(args.output, paths, args, result)


def test_all_seven_samples_are_required_before_creating_outputs(tmp_path, monkeypatch):
    args = acceptance.parse_args(arguments(tmp_path))
    for name in acceptance.SAMPLE_NAMES[:-1]:
        (args.samples_dir / name).write_bytes(b"test placeholder")
    monkeypatch.setattr(acceptance, "parse_args", lambda: args)
    with pytest.raises(FileNotFoundError, match="LCR_8538.NEF"):
        acceptance.main()
    assert not args.output.exists()


@pytest.mark.parametrize("max_edge", [None, 24])
@pytest.mark.parametrize("changed_hash", [False, True])
def test_seven_sample_run_preserves_sources_and_records_actual_resolution(
    tmp_path, monkeypatch, max_edge, changed_hash,
):
    extra = () if max_edge is None else ("--max-edge", str(max_edge))
    args = acceptance.parse_args(arguments(tmp_path, "--backend", "cpu", *extra))
    source_bytes = {}
    for name in acceptance.SAMPLE_NAMES:
        path = args.samples_dir / name
        path.write_bytes(name.encode())
        path.chmod(0o444)
        source_bytes[path] = path.read_bytes()
    xmp = args.samples_dir / "DJI_0523.xmp"
    xmp.write_bytes(b"original metadata")
    xmp.chmod(0o444)
    source_bytes[xmp] = xmp.read_bytes()
    baseline_source = tmp_path / "baseline.py"
    baseline_source.write_text("baseline fixture")
    monkeypatch.setattr(acceptance, "parse_args", lambda: args)
    calls = []
    full_image = np.linspace(1000, 60000, 32 * 48 * 3, dtype=np.uint16).reshape(32, 48, 3)

    def read_image(path, *, preview, max_edge):
        calls.append((path.name, preview, max_edge))
        image = full_image.copy()
        if preview:
            scale = max_edge / max(image.shape[:2])
            image = acceptance.cv2.resize(image, None, fx=scale, fy=scale,
                                           interpolation=acceptance.cv2.INTER_AREA)
        return image, SimpleNamespace(color_space="Linear sRGB")

    def identity(image, _params, **_kwargs):
        return image.copy()

    monkeypatch.setattr(acceptance, "read_image", read_image)
    monkeypatch.setattr(acceptance, "load_baseline", lambda *_args: (
        SimpleNamespace(apply_physical_dehaze=identity), baseline_source))
    monkeypatch.setattr(acceptance.current_physical, "apply_physical_dehaze", identity)
    monkeypatch.setattr(acceptance.current_physical, "get_last_physical_backend", lambda: "CPU test double")
    monkeypatch.setattr(acceptance.current_physical, "physical_diagnostics", lambda *_args, **_kwargs: {})

    if changed_hash:
        original_digest = acceptance.digest
        hash_calls = []

        def changed_digest(path):
            if path == args.samples_dir / acceptance.SAMPLE_NAMES[0]:
                hash_calls.append(path)
                if len(hash_calls) > 1:
                    return "changed-source-hash"
            return original_digest(path)

        monkeypatch.setattr(acceptance, "digest", changed_digest)
        with pytest.raises(AssertionError, match="source photo or XMP hash changed"):
            acceptance.main()
    else:
        acceptance.main()

    assert calls == [(name, max_edge is not None, max_edge or 2048) for name in acceptance.SAMPLE_NAMES]
    for path, contents in source_bytes.items():
        assert path.read_bytes() == contents
        assert path.stat().st_mode & 0o222 == 0
    scope = json.loads((args.output / "验证范围.json").read_text())
    assert scope["source_and_xmp_unchanged"] is (not changed_hash)
    assert scope["full_resolution_acceptance"] is (max_edge is None)
    assert scope["max_edge"] == max_edge
    assert set(scope["hashes"]) == {path.name for path in source_bytes}
    assert scope["completed_samples"] == sorted(acceptance.SAMPLE_NAMES)
    assert scope["dng_export"] is False and scope["external_import"] is False
    rows = json.loads((args.output / "验证结果.json").read_text())
    assert len(rows) == 7
    solar_names = {"DJI_0523.DNG", "DJI_0539.DNG", "LCR_0166.NEF"}
    assert {row["file"] for row in rows if "solar_luminance_profiles" in row} == solar_names
    expected_shape = [32, 48, 3] if max_edge is None else [16, 24, 3]
    for row in rows:
        assert row["shape"] == expected_shape
        assert row["resolution"] == ("full-resolution" if max_edge is None else "preview")
        assert row["max_edge"] == max_edge
        assert len(row["strength_sweep"]) == 6
        assert all(item["finite"] for item in row["strength_sweep"])
        if row["file"] in solar_names:
            assert len(row["solar_luminance_profiles"]["horizontal"]["original"]) == expected_shape[1]
            assert len(row["solar_luminance_profiles"]["vertical"]["original"]) == expected_shape[0]
        if row["file"] in acceptance.SAMPLE_NAMES[4:]:
            assert row["contrast_roi"]["roi_name"] == "generic center/lower region"
        artifacts = acceptance.sample_outputs(args.output, Path(row["file"]), max_edge)
        assert all(path.is_file() for path in artifacts)
        assert ("太阳" in artifacts[3].name) is (row["file"] in solar_names)
        assert ("预览尺寸" if max_edge is not None else "原尺寸") in artifacts[3].name
