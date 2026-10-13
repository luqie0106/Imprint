#!/usr/bin/env python3
"""Measure an unpacked uint16 CFA; never demosaic or apply camera colour processing."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cfa", type=Path)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--planes", type=Path)
    args = parser.parse_args()
    inputs = {args.cfa.resolve(), args.metadata.resolve()}
    outputs = [p for p in (args.output, args.planes) if p is not None]
    if len({p.resolve() for p in outputs}) != len(outputs):
        parser.error("outputs must be distinct")
    for path in outputs:
        if path.resolve() in inputs or path.exists():
            parser.error(f"refusing to overwrite: {path}")
    meta = json.loads(args.metadata.read_text())
    if meta["open"] != 0 or meta["unpack"] != 0 or meta["cfa"] != "RGGB":
        parser.error("requires a successful RGGB unpack")
    w, h = int(meta["raw_width"]), int(meta["raw_height"])
    if w <= 0 or h <= 0 or w % 2 or h % 2:
        parser.error("requires positive even dimensions")
    data = args.cfa.read_bytes()
    if len(data) != w * h * 2:
        parser.error("CFA size does not match metadata")
    a = np.frombuffer(data, dtype="<u2").reshape(h, w)
    report = {
        "format": "little-endian uint16 RGGB CFA, no demosaic",
        "width": w, "height": h, "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "minimum": int(a.min()), "maximum": int(a.max()),
        "zero_count": int((a == 0).sum()),
        "above_white_count": int((a > int(meta["maximum"])).sum()),
        "percentile_levels": [0, 1, 50, 99, 100], "planes": {},
        "row_mean_minimum": float(a.mean(axis=1).min()),
        "row_mean_maximum": float(a.mean(axis=1).max()),
        "last_two_row_means": a[-2:].mean(axis=1).tolist(),
    }
    canvas = Image.new("L", (1000, 708), 0) if args.planes else None
    for name, y, x, dx, dy in [("R", 0, 0, 0, 0), ("G1", 0, 1, 500, 0),
                              ("G2", 1, 0, 0, 354), ("B", 1, 1, 500, 354)]:
        b = a[y::2, x::2]
        report["planes"][name] = {
            "mean": float(b.mean()),
            "percentiles": np.percentile(b, report["percentile_levels"]).tolist(),
        }
        if canvas is not None:
            lo, hi = np.percentile(b, [1, 99])
            scaled = np.zeros(b.shape, dtype=np.uint8) if hi <= lo else np.uint8(
                np.clip((b.astype(np.float64) - lo) / (hi - lo) * 255, 0, 255))
            canvas.paste(Image.fromarray(scaled).resize((500, 333)), (dx, dy + 21))
            ImageDraw.Draw(canvas).text((dx + 5, dy + 4),
                                       f"{name}: independent 1-99 percentile grayscale", fill=255)
    g1, g2 = report["planes"]["G1"]["mean"], report["planes"]["G2"]["mean"]
    report["green_mean_difference_percent"] = abs(g1 - g2) / ((g1 + g2) / 2) * 100
    with args.output.open("x") as out:
        json.dump(report, out, indent=2)
        out.write("\n")
    if canvas is not None:
        with args.planes.open("xb") as out:
            canvas.save(out, format="PNG")
    print(json.dumps({"sha256": report["sha256"], "dimensions": [w, h],
                      "range": [report["minimum"], report["maximum"]]}))


if __name__ == "__main__":
    main()
