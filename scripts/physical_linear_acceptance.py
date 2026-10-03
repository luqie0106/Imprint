"""Read-only, full-resolution RAW acceptance of the linear physical dehaze stage.

Rendered comparisons and diagnostics are written to an isolated output folder.
This does not export DNG or test lens correction, creative adjustments or an
external RAW editor.
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
SAMPLE_NAMES = ("DJI_0523.DNG", "DJI_0539.DNG", "LCR_9472.NEF", "LCR_1132.NEF")
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
}

sys.path.insert(0, str(ROOT / "src"))
from dehaze import DehazeParams
import dehaze_physical as current_physical
from image_io import read_image


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


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
    # Colour and pixel-change metrics use a regular sample; saturation counts
    # and maximum luminance change are computed across the full-resolution image.
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


def sample_outputs(output: Path, path: Path) -> list[Path]:
    names = [f"{path.stem}_全图对照.png", f"{path.stem}_结构ROI.png", f"{path.stem}_强度对照.png"]
    names.append(f"{path.stem}_太阳原尺寸.png" if path.stem.startswith("DJI")
                 else f"{path.stem}_边缘原尺寸.png")
    return [output / name for name in names]


def refuse_existing_outputs(output: Path, paths: list[Path], args, result_path: Path) -> list[dict]:
    rows = []
    if args.only and result_path.exists():
        rows = json.loads(result_path.read_text())
    selected = [path for path in paths if not args.only or path.name in args.only]
    if args.only is None:
        existing = [artifact for path in selected for artifact in sample_outputs(output, path) if artifact.exists()]
        if result_path.exists():
            existing.append(result_path)
        scope_path = output / "验证范围.json"
        if scope_path.exists():
            existing.append(scope_path)
        if existing:
            raise FileExistsError("验收输出已存在；请指定独立输出目录，或用 --only 明确恢复选中的照片。 "
                                  + ", ".join(p.name for p in existing))
    else:
        unknown = sorted(set(args.only) - set(SAMPLE_NAMES))
        if unknown:
            raise ValueError(f"--only contains unknown sample names: {unknown}")
        if not args.only:
            raise ValueError("--only requires at least one sample filename")
        rows = [row for row in rows if row.get("file") not in args.only]
    selected_existing = [artifact for path in selected for artifact in sample_outputs(output, path)
                         if artifact.exists()]
    if selected_existing and args.only is None:
        raise FileExistsError("验收输出已存在：" + ", ".join(p.name for p in selected_existing))
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only full-resolution four-RAW comparison.")
    parser.add_argument("--baseline", type=Path, required=True, help="directory containing baseline src/")
    parser.add_argument("--baseline-stage", choices=("physical", "spatial"), default="physical",
                        help="baseline module to load (default: physical)")
    parser.add_argument("--baseline-label", default="v11", help="label shown for the baseline")
    parser.add_argument("--current-label", default="v12", help="label shown for the current source")
    parser.add_argument("--strength", type=float, default=.7, help="full-resolution strength in [0, 1] (default: 0.7)")
    parser.add_argument("--output", type=Path, required=True, help="isolated output directory")
    parser.add_argument("--backend", choices=("native", "cpu"), default="native",
                        help="current full-resolution backend (default: native)")
    parser.add_argument("--only", nargs="*", help="explicitly resume selected sample filenames")
    args = parser.parse_args()
    if not 0 <= args.strength <= 1:
        parser.error("--strength must be in [0, 1]")

    output = args.output.expanduser().resolve()
    test_images = (ROOT / "test_images").resolve()
    default_output_root = (test_images / "去朦胧输出").resolve()
    inside_input_tree = output == test_images or test_images in output.parents
    allowed_output_subtree = output == default_output_root or default_output_root in output.parents
    if output in test_images.parents or (inside_input_tree and not allowed_output_subtree):
        parser.error("--output must be isolated from the input photo directory")
    paths = [ROOT / "test_images" / name for name in SAMPLE_NAMES]
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing required RAW samples in test_images: {missing}")
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
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in protected}

    for path in paths:
        if args.only and path.name not in args.only:
            continue
        image, metadata = read_image(path, preview=False)
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
        baseline_label = f"{args.baseline_label} {args.baseline_stage} / {percent}"
        current_label = f"{args.current_label} physical / {percent}"
        roi_data = contrast_roi(path, image, old, result)
        comparison([image, old, result], ["Original", baseline_label, current_label],
                   output / f"{path.stem}_全图对照.png")
        if path.stem.startswith("DJI"):
            # Native-pixel crop around the brightest point in the source.
            y = linear @ LUMA_WEIGHTS
            row, col = np.unravel_index(int(np.argmax(y)), y.shape)
            x0 = max(0, min(image.shape[1] - 1400, col - 700))
            y0 = max(0, min(image.shape[0] - 1000, row - 500))
            slices = (slice(y0, min(image.shape[0], y0 + 1000)),
                      slice(x0, min(image.shape[1], x0 + 1400)))
            comparison([image[slices], old[slices], result[slices]],
                       ["Original solar / native pixels", baseline_label, current_label],
                       output / f"{path.stem}_太阳原尺寸.png", edge=2000)
            del y
        else:
            # Same normalized scene area as the existing visual edge review.
            cx = round(image.shape[1] * (.81 if path.stem == "LCR_9472" else .70))
            cy = round(image.shape[0] * (.48 if path.stem == "LCR_9472" else .60))
            x0 = max(0, min(image.shape[1] - 1400, cx - 700))
            y0 = max(0, min(image.shape[0] - 1000, cy - 500))
            slices = (slice(y0, min(image.shape[0], y0 + 1000)),
                      slice(x0, min(image.shape[1], x0 + 1400)))
            comparison([image[slices], old[slices], result[slices]],
                       ["Original edges / native pixels", baseline_label, current_label],
                       output / f"{path.stem}_边缘原尺寸.png", edge=2000)

        row_data = {
            "file": path.name, "shape": list(image.shape), "params": params.__dict__,
            "baseline_stage": args.baseline_stage, "source_code_labels": code_labels,
            "backend": backend, "seconds_excluding_decode": seconds, "baseline_cpu_seconds": old_seconds,
            "baseline_metrics": metrics(image, old), "new_metrics": metrics(image, result),
            "contrast_roi": contrast_roi(path, image, old, result),
            "estimate": current_physical.physical_diagnostics(linear, params, spatial=True),
        }
        if path.stem.startswith("DJI"):
            row_data["solar_luminance_profiles"] = solar_profiles(image, old, result)

        # Low-resolution strength sweep remains a diagnostic; the principal
        # baseline/current comparison and all reported full-size metrics use
        # the unscaled decoded image.
        sample = cv2.resize(linear, (720, max(1, round(linear.shape[0] * 720 / linear.shape[1]))),
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
        comparison([sample] + [current_physical.apply_physical_dehaze(
            sample, DehazeParams(strength=s), backend="cpu", spatial=True,
        ) for s in (.35, .7, 1)],
                   ["Original", f"{args.current_label} 35%", f"{args.current_label} 70%",
                    f"{args.current_label} 100%"], output / f"{path.stem}_强度对照.png", edge=900)
        roi_left, roi_top, roi_right, roi_bottom = roi_data["roi_pixels_xyxy"]
        roi_slices = (slice(roi_top, roi_bottom), slice(roi_left, roi_right))
        comparison([image[roi_slices], old[roi_slices], result[roi_slices]],
                   ["Original", baseline_label, current_label],
                   output / f"{path.stem}_结构ROI.png", edge=1600)

        rows = [r for r in rows if r.get("file") != path.name] + [row_data]
        result_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
        print(json.dumps({"file": path.name, "strength": args.strength,
                          "seconds": round(seconds, 3), "backend": backend,
                          "metrics": row_data["new_metrics"], "contrast_roi": row_data["contrast_roi"],
                          "estimate": row_data["estimate"]}, ensure_ascii=False), flush=True)
        del image, old, linear, result, sample

    unchanged = all(digest(p) == hashes[str(p.relative_to(ROOT))] for p in protected)
    completed_names = sorted(row.get("file") for row in rows)
    if args.only is None:
        assert completed_names == sorted(SAMPLE_NAMES), f"Expected all four samples, got {completed_names}"
    scope = {"source_and_xmp_unchanged": unchanged, "hashes": hashes,
             "completed_samples": completed_names, "selected_samples": args.only or list(SAMPLE_NAMES),
             "strength": args.strength, "source_code_labels": code_labels,
             "linear_stage_only": True, "dng_export": False, "external_import": False}
    (output / "验证范围.json").write_text(json.dumps(scope, ensure_ascii=False, indent=2))
    assert unchanged, "source photo or XMP hash changed during read-only acceptance"


if __name__ == "__main__":
    main()
