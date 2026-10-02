"""Read-only RAW acceptance of the linear physical dehaze stage.

Outputs are isolated from photos/XMP. This does not export DNG or test lens
correction, creative adjustments or an external RAW editor.
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
sys.path.insert(0, str(ROOT / "src"))
from dehaze import DehazeParams
from dehaze_physical import apply_physical_dehaze, physical_diagnostics, get_last_physical_backend
from image_io import read_image


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def display(source):
    rgb = source.astype(np.float32)
    if source.dtype == np.uint16:
        rgb /= 65535
    elif source.dtype == np.uint8:
        rgb /= 255
    rgb = np.where(rgb <= .0031308, 12.92 * rgb, 1.055 * np.maximum(rgb, 0) ** (1 / 2.4) - .055)
    return np.clip(np.rint(rgb * 255), 0, 255).astype(np.uint8)


def comparison(images, labels, output, edge=1600):
    panels = []
    for source, label in zip(images, labels):
        scale = min(1, edge / max(source.shape[:2]))
        sample = cv2.resize(source, (round(source.shape[1] * scale), round(source.shape[0] * scale)),
                            interpolation=cv2.INTER_AREA) if scale < 1 else source
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


def metrics(source, result):
    # Use a regular sample for colour statistics; saturation counts are native.
    weights = np.array([.2126, .7152, .0722], dtype=np.float32)
    before = source[::4, ::4].astype(np.float32) / 65535
    after = result[::4, ::4].astype(np.float32) / 65535 if result.dtype == np.uint16 else result[::4, ::4]
    y, new_y = before @ weights, after @ weights
    chroma, new_chroma = before - y[..., None], after - new_y[..., None]
    norm, new_norm = np.linalg.norm(chroma, axis=2), np.linalg.norm(new_chroma, axis=2)
    valid = (norm > .015) & (new_norm > .005) & (y > .025) & (y < .9)
    cos = np.sum(chroma * new_chroma, axis=2) / np.maximum(norm * new_norm, 1e-8)
    angle = np.degrees(np.arccos(np.clip(cos[valid], -1, 1)))
    ratio_mask = (y > .015) & (y < .9)
    def satcount(a):
        count = 0
        for start in range(0, a.shape[0], 256):
            tile = a[start:start + 256].astype(np.float32)
            if a.dtype == np.uint16:
                tile /= 65535
            count += int(np.count_nonzero(tile @ weights >= .97))
        return count
    return {"median_luma_ratio": float(np.median(new_y[ratio_mask] / y[ratio_mask])),
            "luma_ratio_p01": float(np.percentile(new_y[ratio_mask] / y[ratio_mask], 1)),
            "hue_angle_p95_degrees": float(np.percentile(angle, 95)) if angle.size else 0,
            "new_all_black_pixels": int(np.count_nonzero(np.all(after == 0, axis=2) & (y > .001))),
            "source_near_saturated_pixels": satcount(source), "result_near_saturated_pixels": satcount(result)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", default="cpu")
    parser.add_argument("--edge", type=int, default=0)
    parser.add_argument("--only", nargs="*", help="Resume selected sample filenames")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("approved_solar_baseline", args.baseline / "src/dehaze_spatial.py")
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    args.output.mkdir(parents=True, exist_ok=True)
    paths = [ROOT / "test_images" / f for f in ("DJI_0523.DNG", "DJI_0539.DNG", "LCR_9472.NEF", "LCR_1132.NEF")]
    protected = paths + [p.with_suffix(".xmp") for p in paths if p.with_suffix(".xmp").exists()]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in protected}
    result_path = args.output / "验证结果.json"
    rows = json.loads(result_path.read_text()) if args.only and result_path.exists() else []
    for path in paths:
        if args.only and path.name not in args.only:
            continue
        image, metadata = read_image(path, preview=False)
        assert metadata.color_space == "Linear sRGB"
        if args.edge and max(image.shape[:2]) > args.edge:
            scale = args.edge / max(image.shape[:2])
            image = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        params = DehazeParams(strength=.7)
        t0 = time.perf_counter()
        old = baseline.apply_spatial_dehaze(image, params, backend="cpu")
        old_seconds = time.perf_counter() - t0
        linear = image.astype(np.float32) / 65535
        t0 = time.perf_counter()
        result = apply_physical_dehaze(linear, params, backend=args.backend, spatial=True)
        seconds = time.perf_counter() - t0
        backend = get_last_physical_backend()
        if args.backend == "native":
            assert backend == "Metal GPU（线性浮点）", f"Acceptance requires real Metal, got {backend}"
        comparison([image, old, result], ["Original", "Approved solar v10 / 70%", "Linear physical v11 / 70%"],
                   args.output / f"{path.stem}_全图对照.png")
        if path.stem.startswith("DJI"):
            # Native-pixel crop containing the brightest solar region.
            y = image.astype(np.float32) @ np.array([.2126, .7152, .0722], dtype=np.float32)
            row, col = np.unravel_index(np.argmax(y), y.shape)
            x0 = max(0, min(image.shape[1] - 1400, col - 700))
            y0 = max(0, min(image.shape[0] - 1000, row - 500))
            slices = (slice(y0, y0 + 1000), slice(x0, x0 + 1400))
            comparison([image[slices], old[slices], result[slices]],
                       ["Original solar / native pixels", "Approved solar v10", "Linear physical v11"],
                       args.output / f"{path.stem}_太阳原尺寸.png", edge=2000)
            del y
        else:
            # Consistent native-size edge crop: large turbine and skyline.
            cx = round(image.shape[1] * (.81 if path.stem == "LCR_9472" else .70))
            cy = round(image.shape[0] * (.48 if path.stem == "LCR_9472" else .60))
            x0 = max(0, min(image.shape[1] - 1400, cx - 700))
            y0 = max(0, min(image.shape[0] - 1000, cy - 500))
            slices = (slice(y0, y0 + 1000), slice(x0, x0 + 1400))
            comparison([image[slices], old[slices], result[slices]],
                       ["Original edges / native pixels", "Approved solar v10", "Linear physical v11"],
                       args.output / f"{path.stem}_边缘原尺寸.png", edge=2000)
        row = {"file": path.name, "shape": list(image.shape), "params": params.__dict__,
               "backend": backend, "seconds_excluding_decode": seconds, "baseline_cpu_seconds": old_seconds,
               "baseline_metrics": metrics(image, old), "new_metrics": metrics(image, result),
               "estimate": physical_diagnostics(linear, params, spatial=True)}
        # Strength sweep on the same decoded source sample, fixed colour mode.
        sample = cv2.resize(linear, (720, max(1, round(linear.shape[0] * 720 / linear.shape[1]))), interpolation=cv2.INTER_AREA)
        # INTER_AREA can exceed an exact endpoint by one float32 ULP.
        np.clip(sample, 0, 1, out=sample)
        sweep = []
        previous = sample
        for strength in (0, .001, .10, .35, .7, 1):
            out = apply_physical_dehaze(sample, DehazeParams(strength=strength), spatial=True)
            sweep.append({"strength": strength, "mean_abs_change": float(np.mean(np.abs(out - sample))),
                          "mean_step": float(np.mean(np.abs(out - previous))), "finite": bool(np.isfinite(out).all())})
            previous = out
        row["strength_sweep"] = sweep
        comparison([sample] + [apply_physical_dehaze(sample, DehazeParams(strength=s), spatial=True) for s in (.35, .7, 1)],
                   ["Original", "v11 35%", "v11 70%", "v11 100%"], args.output / f"{path.stem}_强度对照.png", edge=900)
        rows = [r for r in rows if r["file"] != path.name] + [row]
        result_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
        print(json.dumps({"file": path.name, "seconds": round(seconds, 3), "backend": backend,
                          "metrics": row["new_metrics"], "estimate": row["estimate"]}, ensure_ascii=False), flush=True)
        del image, old, linear, result
    unchanged = all(digest(p) == hashes[str(p.relative_to(ROOT))] for p in protected)
    (args.output / "验证范围.json").write_text(json.dumps({"source_and_xmp_unchanged": unchanged,
        "hashes": hashes, "linear_stage_only": True, "dng_export": False, "external_import": False}, ensure_ascii=False, indent=2))
    assert unchanged


if __name__ == "__main__":
    main()
