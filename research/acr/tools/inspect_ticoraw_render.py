#!/usr/bin/env python3
"""Create a review JPEG, and optionally a crop, from a LibRaw RGB16 PPM."""

import argparse
from pathlib import Path

from PIL import Image


def resolve(path: Path) -> Path:
    return path.expanduser().resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ppm", type=Path, help="RGB16 P6 PPM produced by the CFA renderer")
    parser.add_argument("--thumbnail", required=True, type=Path, help="new JPEG output path")
    parser.add_argument("--max-dimension", type=int, default=1800)
    parser.add_argument("--roi", help="optional crop as x,y,width,height in rendered-image pixels")
    parser.add_argument("--roi-output", type=Path, help="new JPEG output path for --roi")
    args = parser.parse_args()

    source = resolve(args.ppm)
    outputs = [args.thumbnail]
    roi = None
    if args.roi:
        if args.roi_output is None:
            parser.error("--roi-output is required when --roi is set")
        try:
            roi = tuple(int(value) for value in args.roi.split(","))
        except ValueError:
            parser.error("--roi must contain four integers: x,y,width,height")
        if len(roi) != 4 or roi[0] < 0 or roi[1] < 0 or roi[2] <= 0 or roi[3] <= 0:
            parser.error("--roi must be nonnegative x,y and positive width,height")
        outputs.append(args.roi_output)
    elif args.roi_output is not None:
        parser.error("--roi requires --roi-output")

    resolved_outputs = [resolve(path) for path in outputs]
    if len(set(resolved_outputs)) != len(resolved_outputs):
        parser.error("output paths must be distinct")
    for path, resolved in zip(outputs, resolved_outputs):
        if resolved == source:
            parser.error(f"refusing to overwrite input: {path}")
        if path.exists():
            parser.error(f"refusing to overwrite existing output: {path}")
    if args.max_dimension <= 0:
        parser.error("--max-dimension must be positive")

    with Image.open(source) as opened:
        if opened.format != "PPM" or opened.width <= 0 or opened.height <= 0:
            parser.error("input must be a readable PPM image")
        width, height = opened.size
        if roi is not None:
            x, y, roi_width, roi_height = roi
            if x + roi_width > width or y + roi_height > height:
                parser.error("ROI extends beyond the rendered image")
            crop = opened.crop((x, y, x + roi_width, y + roi_height)).convert("RGB")
        else:
            crop = None
        review = opened.convert("RGB")
        review.thumbnail((args.max_dimension, args.max_dimension), Image.Resampling.LANCZOS)

    with args.thumbnail.open("xb") as output:
        review.save(output, format="JPEG", quality=92, optimize=True)
    if crop is not None:
        with args.roi_output.open("xb") as output:
            crop.save(output, format="JPEG", quality=95, optimize=True)

    print(f"source_ppm={source.name} source_dimensions={width}x{height}")
    print(f"thumbnail={args.thumbnail} thumbnail_dimensions={review.width}x{review.height}")
    if crop is not None:
        print(f"roi={roi} roi_output={args.roi_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
