"""Read-only RAW acceptance of the linear physical dehaze stage.

Rendered comparisons and diagnostics are written to an isolated output folder.
This does not export DNG or test lens correction, creative adjustments or an
external RAW editor. Full-resolution processing is the default; --max-edge
selects a preview-only comparison that cannot establish full-resolution quality.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_NAMES = ("DJI_0523.DNG", "DJI_0539.DNG", "LCR_9472.NEF", "LCR_1132.NEF",
                "LCR_9453.NEF", "LCR_0166.NEF", "LCR_8538.NEF")
SUN_SAMPLE_STEMS = {"DJI_0523", "DJI_0539", "LCR_0166"}
LUMA_WEIGHTS = np.array([.2126, .7152, .0722], dtype=np.float32)
SOLAR_STRIP_HALF_WIDTH = 2

# Fixed, composition-based ROIs in normalized (x0, y0, x1, y1) coordinates.
# The masks inside each ROI are then selected once from the original linear
# luminance at P10/P90 and reused unchanged for every processed version.
CONTRAST_ROIS = {
    "DJI_0523": ("DJI building", (.16, .42, .84, .92)),
    "DJI_0539": ("DJI building", (.16, .42, .84, .92)),
    "LCR_9472": ("middle sea turbines", (.24, .30, .78, .74)),
    "LCR_1132": ("building", (.45, .34, .88, .84)),
    # 新增样片仅使用通用构图区间，不推断场景语义。
    "LCR_9453": ("generic center/lower region", (.20, .35, .80, .85)),
    "LCR_0166": ("generic center/lower region", (.20, .35, .80, .85)),
    "LCR_8538": ("generic center/lower region", (.20, .35, .80, .85)),
}

sys.path.insert(0, str(ROOT / "src"))
from dehaze import DehazeParams
import dehaze_physical as current_physical
from image_io import read_image


def digest(path: Path) -> str:
    sha256 = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def linear_rgb(source: np.ndarray) -> np.ndarray:
    rgb = source.astype(np.float32, copy=False)
    if source.dtype == np.uint16:
        rgb /= 65535
    elif source.dtype == np.uint8:
        rgb /= 255
    return rgb


def linear_luma(source: np.ndarray) -> np.ndarray:
    return linear_rgb(source) @ LUMA_WEIGHTS


def display(source: np.ndarray) -> np.ndarray:
    rgb = linear_rgb(source)
    rgb = np.where(rgb <= .0031308, 12.92 * rgb,
                   1.055 * np.maximum(rgb, 0) ** (1 / 2.4) - .055)
    return np.clip(np.rint(rgb * 255), 0, 255).astype(np.uint8)


def comparison(images, labels, output: Path, edge: int = 1600) -> None:
    panels = []
    for source, label in zip(images, labels):
        scale = min(1, edge / max(source.shape[:2]))
        sample = (cv2.resize(source, (round(source.shape[1] * scale), round(source.shape[0] * scale)),
                             interpolation=cv2.INTER_AREA) if scale < 1 else source)
        panel = Image.fromarray(display(sample))
        canvas = Image.new("RGB", (panel.width, panel.height + 32), "#202020")
        canvas.paste(panel, (0, 32))
        ImageDraw.Draw(canvas).text((12, 10), label, fill="white")
        panels.append(canvas)
    output_image = Image.new("RGB", (sum(p.width for p in panels), max(p.height for p in panels)))
    left = 0
    for panel in panels:
        output_image.paste(panel, (left, 0))
        left += panel.width
    output_image.save(output)


def metrics(source: np.ndarray, result: np.ndarray) -> dict:
    # 色彩和像素变化使用固定步长采样；饱和计数与最大亮度变化覆盖当前验收尺寸。
    before = linear_rgb(source)[::4, ::4]
    after = linear_rgb(result)[::4, ::4]
    y, new_y = before @ LUMA_WEIGHTS, after @ LUMA_WEIGHTS
    chroma, new_chroma = before - y[..., None], after - new_y[..., None]
    norm, new_norm = np.linalg.norm(chroma, axis=2), np.linalg.norm(new_chroma, axis=2)
    valid = (norm > .015) & (new_norm > .005) & (y > .025) & (y < .9)
    cos = np.sum(chroma * new_chroma, axis=2) / np.maximum(norm * new_norm, 1e-8)
    angle = np.degrees(np.arccos(np.clip(cos[valid], -1, 1)))
    ratio_mask = (y > .015) & (y < .9)

    source_near_saturated = result_near_saturated = new_black_pixels = 0
    max_luma_increase = -np.inf
    maximum_output_luma = -np.inf
    for start in range(0, source.shape[0], 128):
        stop = min(source.shape[0], start + 128)
        source_tile = linear_rgb(source[start:stop])
        result_tile = linear_rgb(result[start:stop])
        source_tile_luma = source_tile @ LUMA_WEIGHTS
        result_tile_luma = result_tile @ LUMA_WEIGHTS
        source_near_saturated += int(np.count_nonzero(source_tile_luma >= .97))
        result_near_saturated += int(np.count_nonzero(result_tile_luma >= .97))
        new_black_pixels += int(np.count_nonzero(np.all(result_tile == 0, axis=2) & (source_tile_luma > .001)))
        max_luma_increase = max(max_luma_increase, float(np.max(result_tile_luma - source_tile_luma)))
        maximum_output_luma = max(maximum_output_luma, float(np.max(result_tile_luma)))
    return {
        "median_luma_ratio": float(np.median(new_y[ratio_mask] / y[ratio_mask])) if np.any(ratio_mask) else 1.0,
        "luma_ratio_p01": float(np.percentile(new_y[ratio_mask] / y[ratio_mask], 1)) if np.any(ratio_mask) else 1.0,
        "hue_angle_p95_degrees": float(np.percentile(angle, 95)) if angle.size else 0,
        "new_all_black_pixels": new_black_pixels,
        "source_near_saturated_pixels": source_near_saturated,
        "result_near_saturated_pixels": result_near_saturated,
        "max_luma_increase": max_luma_increase if np.isfinite(max_luma_increase) else 0.0,
        "maximum_output_luma": maximum_output_luma if np.isfinite(maximum_output_luma) else 0.0,
    }


def contrast_roi(path: Path, source: np.ndarray, baseline: np.ndarray, current: np.ndarray) -> dict:
    label, normalized = CONTRAST_ROIS[path.stem]
    height, width = source.shape[:2]
    x0, y0, x1, y1 = normalized
    left, top = int(round(width * x0)), int(round(height * y0))
    right, bottom = int(round(width * x1)), int(round(height * y1))
    left, top = max(0, min(width - 1, left)), max(0, min(height - 1, top))
    right, bottom = max(left + 1, min(width, right)), max(top + 1, min(height, bottom))
    roi_slices = (slice(top, bottom), slice(left, right))
    def roi_luma(image: np.ndarray) -> np.ndarray:
        return linear_rgb(image[roi_slices]) @ LUMA_WEIGHTS

    original_luma = roi_luma(source)
    p10, p90 = np.percentile(original_luma, (10, 90))
    low_mask, high_mask = original_luma <= p10, original_luma >= p90

    def value(image: np.ndarray) -> dict:
        luma = roi_luma(image)
        low = float(np.median(luma[low_mask]))
        high = float(np.median(luma[high_mask]))
        denominator = high + low
        ratio = (high - low) / denominator if denominator > 1e-12 else None
        return {"median_low": low, "median_high": high, "structure_contrast": ratio}

    return {
        "roi_name": label,
        "roi_normalized_xyxy": list(normalized),
        "roi_pixels_xyxy": [left, top, right, bottom],
        "luminance_space": "linear RGB Rec.709",
        "mask_selection": "original ROI luminance P10 and P90; identical pixel masks for all versions",
        "original_p10": float(p10),
        "original_p90": float(p90),
        "low_sample_count": int(np.count_nonzero(low_mask)),
        "high_sample_count": int(np.count_nonzero(high_mask)),
        "interpretation": "relative structure contrast; not a measurement of physical haze",
        "original": value(source),
        "baseline": value(baseline),
        "current": value(current),
    }


def solar_profiles(source: np.ndarray, baseline: np.ndarray, current: np.ndarray) -> dict:
    peak_value = -np.inf
    peak_row = peak_col = 0
    for start in range(0, source.shape[0], 128):
        tile_luma = linear_luma(source[start:start + 128])
        tile_index = int(np.argmax(tile_luma))
        value = float(tile_luma.flat[tile_index])
        if value > peak_value:
            local_row, peak_col = np.unravel_index(tile_index, tile_luma.shape)
            peak_row = start + int(local_row)
            peak_value = value
    row, col = int(peak_row), int(peak_col)
    half = SOLAR_STRIP_HALF_WIDTH

    def profiles(image: np.ndarray) -> tuple[list[float], list[float]]:
        horizontal_rgb = linear_rgb(image[max(0, row - half):min(image.shape[0], row + half + 1), :, :])
        vertical_rgb = linear_rgb(image[:, max(0, col - half):min(image.shape[1], col + half + 1), :])
        horizontal = (horizontal_rgb @ LUMA_WEIGHTS).mean(axis=0, dtype=np.float64).tolist()
        vertical = (vertical_rgb @ LUMA_WEIGHTS).mean(axis=1, dtype=np.float64).tolist()
        return horizontal, vertical

    original_horizontal, original_vertical = profiles(source)
    baseline_horizontal, baseline_vertical = profiles(baseline)
    current_horizontal, current_vertical = profiles(current)

    return {
        "luminance_space": "linear RGB Rec.709 normalized to [0, 1]",
        "reference_point_xy": [int(col), int(row)],
        "reference_point_source_luma": peak_value,
        "strip_half_width_pixels": half,
        "horizontal": {
            "center_y": int(row), "sampled_rows_inclusive": [max(0, row - half), min(source.shape[0] - 1, row + half)],
            "x_start": 0, "x_end_exclusive": int(source.shape[1]),
            "original": original_horizontal, "baseline": baseline_horizontal, "current": current_horizontal,
        },
        "vertical": {
            "center_x": int(col), "sampled_columns_inclusive": [max(0, col - half), min(source.shape[1] - 1, col + half)],
            "y_start": 0, "y_end_exclusive": int(source.shape[0]),
            "original": original_vertical, "baseline": baseline_vertical, "current": current_vertical,
        },
    }


def sample_outputs(output: Path, path: Path, max_edge: int | None = None) -> list[Path]:
    names = [f"{path.stem}_全图对照.png", f"{path.stem}_结构ROI.png", f"{path.stem}_强度对照.png"]
    region = "太阳" if path.stem in SUN_SAMPLE_STEMS else (
        "边缘" if path.stem in ("LCR_9472", "LCR_1132") else "局部")
    size = "原尺寸" if max_edge is None else "预览尺寸"
    names.append(f"{path.stem}_{region}{size}.png")
    return [output / name for name in names]


def refuse_existing_outputs(output: Path, paths: list[Path], args, result_path: Path) -> list[dict]:
    rows = []
    selected = [path for path in paths if not args.only or path.name in args.only]
    artifacts = [artifact for path in selected for artifact in sample_outputs(output, path, args.max_edge)]
    artifacts += [result_path, output / "验证范围.json"]
    if any(artifact.is_symlink() for artifact in artifacts):
        raise ValueError("Acceptance outputs must not be symbolic links")
    if args.only and result_path.exists():
        rows = json.loads(result_path.read_text())
        if any(row.get("max_edge") != args.max_edge for row in rows):
            raise ValueError("Cannot resume with a different --max-edge; use an isolated output directory")
        scope_path = output / "验证范围.json"
        if scope_path.exists():
            scope = json.loads(scope_path.read_text())
            previous_directory = scope.get("samples_directory")
            if previous_directory is not None and previous_directory != str(args.samples_dir):
                raise ValueError("Cannot resume from a different sample directory; use an isolated output directory")
    if args.only is None:
        existing = [artifact for artifact in artifacts if artifact.exists()]
        if existing:
            raise FileExistsError("Acceptance outputs already exist; use an isolated directory or --only to resume. "
                                  + ", ".join(p.name for p in existing))
    else:
        unknown = sorted(set(args.only) - set(SAMPLE_NAMES))
        if unknown:
            raise ValueError(f"--only contains unknown sample names: {unknown}")
        if not args.only:
            raise ValueError("--only requires at least one sample filename")
        rows = [row for row in rows if row.get("file") not in args.only]
    return rows


def load_baseline(directory: Path, stage: str):
    source_path = directory / "src" / ("dehaze_physical.py" if stage == "physical" else "dehaze_spatial.py")
    if not source_path.is_file():
        raise FileNotFoundError(f"baseline source not found: {source_path}")
    spec = importlib.util.spec_from_file_location(f"acceptance_baseline_{stage}", source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load baseline source: {source_path}")
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    if stage == "physical" and not hasattr(baseline, "apply_physical_dehaze"):
        raise AttributeError("physical baseline lacks apply_physical_dehaze")
    if stage == "spatial" and not hasattr(baseline, "apply_spatial_dehaze"):
        raise AttributeError("spatial baseline lacks apply_spatial_dehaze")
    return baseline, source_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only seven-RAW comparison; full resolution by default.")
    parser.add_argument("--baseline", type=Path, required=True, help="directory containing baseline src/")
    parser.add_argument("--baseline-stage", choices=("physical", "spatial"), default="physical",
                        help="baseline module to load (default: physical)")
    parser.add_argument("--baseline-label", default="v11", help="label shown for the baseline")
    parser.add_argument("--current-label", default="v12", help="label shown for the current source")
    parser.add_argument("--strength", type=float, default=.7, help="comparison strength in [0, 1] (default: 0.7)")
    parser.add_argument("--samples-dir", type=Path,
                        default=ROOT / ("sample" if (ROOT / "sample").is_dir() else "test_images"),
                        help="RAW input directory (default: sample/ if present, otherwise test_images/)")
    parser.add_argument("--max-edge", type=int,
                        help="preview-only maximum edge in pixels; omit for full-resolution acceptance")
    parser.add_argument("--output", type=Path, required=True, help="isolated output directory")
    parser.add_argument("--backend", choices=("native", "cpu"), default="native",
                        help="current comparison backend (default: native)")
    parser.add_argument("--only", nargs="*", help="explicitly resume selected sample filenames")
    args = parser.parse_args(argv)
    if not 0 <= args.strength <= 1:
        parser.error("--strength must be in [0, 1]")

    if args.max_edge is not None and args.max_edge < 1:
        parser.error("--max-edge must be a positive integer")
    if args.only is not None:
        if not args.only:
            parser.error("--only requires at least one sample filename")
        unknown = sorted(set(args.only) - set(SAMPLE_NAMES))
        if unknown:
            parser.error(f"--only contains unknown sample names: {unknown}")
    args.output = args.output.expanduser().resolve()
    args.samples_dir = args.samples_dir.expanduser().resolve()
    if not args.samples_dir.is_dir():
        parser.error("--samples-dir must be an existing directory")
    default_output_root = (args.samples_dir / "去朦胧输出").resolve()
    inside_input_tree = args.output == args.samples_dir or args.samples_dir in args.output.parents
    allowed_output_subtree = args.output == default_output_root or default_output_root in args.output.parents
    if (args.output == args.samples_dir or args.output in args.samples_dir.parents
            or (inside_input_tree and not allowed_output_subtree)):
        parser.error("--output must be isolated from the input photo directory")
    if args.output.exists() and not args.output.is_dir():
        parser.error("--output must be a directory")
    return args


def main() -> None:
    args = parse_args()
    output = args.output
    paths = [args.samples_dir / name for name in SAMPLE_NAMES]
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing required RAW samples in {args.samples_dir}: {missing}")
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "验证结果.json"
    rows = refuse_existing_outputs(output, paths, args, result_path)
    baseline, baseline_source = load_baseline(args.baseline.expanduser().resolve(), args.baseline_stage)
    current_source = Path(current_physical.__file__).resolve()
    code_labels = {
        "baseline": {"label": args.baseline_label, "stage": args.baseline_stage,
                     "source_file": baseline_source.name, "sha256": digest(baseline_source)},
        "current": {"label": args.current_label, "stage": "physical", "source_file": current_source.name,
                    "sha256": digest(current_source)},
    }
    protected = paths + [p.with_suffix(".xmp") for p in paths if p.with_suffix(".xmp").exists()]
    hashes = {p.name: digest(p) for p in protected}
    resolution = "full-resolution" if args.max_edge is None else "preview"
    pixel_space = "native decoded pixels" if args.max_edge is None else "preview pixels"

    for path in paths:
        if args.only and path.name not in args.only:
            continue
        image, metadata = read_image(path, preview=args.max_edge is not None,
                                     max_edge=args.max_edge if args.max_edge is not None else 2048)
        assert metadata.color_space == "Linear sRGB"
        linear = linear_rgb(image)
        params = DehazeParams(strength=args.strength)
        t0 = time.perf_counter()
        if args.baseline_stage == "physical":
            old = baseline.apply_physical_dehaze(linear, params, backend="cpu", spatial=True)
        else:
            old = baseline.apply_spatial_dehaze(image, params, backend="cpu")
        old_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        result = current_physical.apply_physical_dehaze(
            linear, params, backend=args.backend, spatial=True,
        )
        seconds = time.perf_counter() - t0
        backend = current_physical.get_last_physical_backend()
        if args.backend == "native" and args.strength > 1e-6:
            assert "GPU" in backend, f"Acceptance requires the native GPU backend, got {backend}"

        percent = f"{args.strength * 100:g}%"
        baseline_label = f"{args.baseline_label} {args.baseline_stage} / {percent} / {resolution}"
        current_label = f"{args.current_label} physical / {percent} / {resolution}"
        roi_data = contrast_roi(path, image, old, result)
        comparison([image, old, result], [f"Original / {resolution}", baseline_label, current_label],
                   output / f"{path.stem}_全图对照.png")
        if path.stem in SUN_SAMPLE_STEMS:
            # 在当前验收尺寸中裁切最亮点；预览模式不声称原图像素级验收。
            y = linear @ LUMA_WEIGHTS
            row, col = np.unravel_index(int(np.argmax(y)), y.shape)
            x0 = max(0, min(image.shape[1] - 1400, col - 700))
            y0 = max(0, min(image.shape[0] - 1000, row - 500))
            slices = (slice(y0, min(image.shape[0], y0 + 1000)),
                      slice(x0, min(image.shape[1], x0 + 1400)))
            comparison([image[slices], old[slices], result[slices]],
                       [f"Original solar / {pixel_space}", baseline_label, current_label],
                       sample_outputs(output, path, args.max_edge)[3], edge=2000)
            del y
        else:
            # 原有样片保持边缘区域；新增样片只采用通用中央下部区域。
            center = {"LCR_9472": (.81, .48), "LCR_1132": (.70, .60)}.get(path.stem, (.50, .60))
            cx, cy = round(image.shape[1] * center[0]), round(image.shape[0] * center[1])
            x0 = max(0, min(image.shape[1] - 1400, cx - 700))
            y0 = max(0, min(image.shape[0] - 1000, cy - 500))
            slices = (slice(y0, min(image.shape[0], y0 + 1000)),
                      slice(x0, min(image.shape[1], x0 + 1400)))
            comparison([image[slices], old[slices], result[slices]],
                       [f"Original crop / {pixel_space}", baseline_label, current_label],
                       sample_outputs(output, path, args.max_edge)[3], edge=2000)

        row_data = {
            "file": path.name, "shape": list(image.shape), "params": params.__dict__,
            "resolution": resolution, "max_edge": args.max_edge, "crop_pixel_space": pixel_space,
            "baseline_stage": args.baseline_stage, "source_code_labels": code_labels,
            "backend": backend, "seconds_excluding_decode": seconds, "baseline_cpu_seconds": old_seconds,
            "baseline_metrics": metrics(image, old), "new_metrics": metrics(image, result),
            "contrast_roi": contrast_roi(path, image, old, result),
            "estimate": current_physical.physical_diagnostics(linear, params, spatial=True),
        }
        if path.stem in SUN_SAMPLE_STEMS:
            row_data["solar_luminance_profiles"] = solar_profiles(image, old, result)

        # 强度扫描只用于低分辨率诊断，不放大已经缩小的预览图。
        sweep_width = min(720, linear.shape[1])
        sample = cv2.resize(linear, (sweep_width, max(1, round(linear.shape[0] * sweep_width / linear.shape[1]))),
                            interpolation=cv2.INTER_AREA)
        np.clip(sample, 0, 1, out=sample)
        sweep, previous = [], sample
        for strength in (0, .001, .10, .35, .7, 1):
            out = current_physical.apply_physical_dehaze(
                sample, DehazeParams(strength=strength), backend="cpu", spatial=True,
            )
            sweep.append({"strength": strength, "mean_abs_change": float(np.mean(np.abs(out - sample))),
                          "mean_step": float(np.mean(np.abs(out - previous))),
                          "finite": bool(np.isfinite(out).all())})
            previous = out
        row_data["strength_sweep"] = sweep
        row_data["strength_sweep_shape"] = list(sample.shape)
        comparison([sample] + [current_physical.apply_physical_dehaze(
            sample, DehazeParams(strength=s), backend="cpu", spatial=True,
        ) for s in (.35, .7, 1)],
                   ["Original", f"{args.current_label} 35%", f"{args.current_label} 70%",
                    f"{args.current_label} 100%"], output / f"{path.stem}_强度对照.png", edge=900)
        roi_left, roi_top, roi_right, roi_bottom = roi_data["roi_pixels_xyxy"]
        roi_slices = (slice(roi_top, roi_bottom), slice(roi_left, roi_right))
        comparison([image[roi_slices], old[roi_slices], result[roi_slices]],
                   [f"Original / {resolution}", baseline_label, current_label],
                   output / f"{path.stem}_结构ROI.png", edge=1600)

        rows = [r for r in rows if r.get("file") != path.name] + [row_data]
        result_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
        print(json.dumps({"file": path.name, "strength": args.strength,
                          "resolution": resolution, "shape": list(image.shape),
                          "seconds": round(seconds, 3), "backend": backend,
                          "metrics": row_data["new_metrics"], "contrast_roi": row_data["contrast_roi"],
                          "estimate": row_data["estimate"]}, ensure_ascii=False), flush=True)
        del image, old, linear, result, sample

    unchanged = all(digest(p) == hashes[p.name] for p in protected)
    completed_names = sorted(row.get("file") for row in rows)
    if args.only is None:
        assert completed_names == sorted(SAMPLE_NAMES), f"Expected all seven samples, got {completed_names}"
    scope = {"source_and_xmp_unchanged": unchanged, "hashes": hashes,
             "completed_samples": completed_names, "selected_samples": args.only or list(SAMPLE_NAMES),
             "strength": args.strength, "source_code_labels": code_labels,
             "samples_directory": str(args.samples_dir), "resolution": resolution, "max_edge": args.max_edge,
             "full_resolution_acceptance": args.max_edge is None, "crop_pixel_space": pixel_space,
             "linear_stage_only": True, "dng_export": False, "external_import": False}
    (output / "验证范围.json").write_text(json.dumps(scope, ensure_ascii=False, indent=2))
    assert unchanged, "source photo or XMP hash changed during read-only acceptance"


if __name__ == "__main__":
    main()
