#!/usr/bin/env python3
"""Independent continuous entropy model and scatter comparison for LCR_3520.

The Python path reads the real NEF strip, decodes every precinct call using
the supplied ACR-extracted tables, and scatters coefficients into buffers
whose geometry is computed here. It compares those bytes against the pinned
public LibRaw harness. It neither executes Adobe code nor reconstructs CFA
pixels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from probe_ticoraw_first_precinct import (
    LB_BANDS,
    PUBLIC_SOURCE_COMMIT,
    PUBLIC_SOURCE_PATH,
    MSBBitReader,
    ProbeError,
    apply_signs,
    compute_geometry,
    compute_gtli,
    decode_gcli_band,
    decode_magnitudes,
    dequantize,
    load_entropy_tables,
    parse_lb_payloads,
    sb_to_wgt_index,
)
from probe_ticoraw_header import (
    EXPECTED_RAW_HEIGHT,
    EXPECTED_RAW_WIDTH,
    ClassicTiff,
    find_raw_ifd,
    read_strip_bounds,
    sha256_file,
)
from probe_ticoraw_packets import parse_header, parse_precinct_stream


class FastStrictBitReader(MSBBitReader):
    """MSB-first reader with the same hard boundary and a chunked bit read."""

    def read_bits(self, width: int) -> int:
        if width < 0:
            raise ProbeError(f"{self.label}: negative read width {width}")
        remaining = self.total_bits - self.bit_position
        if width > remaining:
            raise ProbeError(
                f"{self.label}: truncated bitstream at bit {self.bit_position}; "
                f"requested {width}, have {remaining}"
            )
        if width == 0:
            return 0
        byte_start = self.bit_position >> 3
        bit_in_first = self.bit_position & 7
        byte_end = (self.bit_position + width + 7) >> 3
        chunk = int.from_bytes(self.data[byte_start:byte_end], "big")
        shift = ((byte_end - byte_start) * 8) - bit_in_first - width
        value = (chunk >> shift) & ((1 << width) - 1)
        self.bit_position += width
        return value


def round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def compute_scatter_layout(half_width: int) -> tuple[list[dict[str, int]], int, int]:
    """Compute scatter addresses independently from the public C++ config."""
    layout, ng_by_sb = compute_geometry(half_width)
    ng_max = (half_width + 7) // 8
    ng_ll = half_width // 4
    memcpy_lb = round_up(ng_ll, 32)
    lift_lb = memcpy_lb + round_up((ng_max + 15) // 16, 32)
    lb_bases = {
        0: 0,
        1: lift_lb,
        2: 2 * lift_lb,
        3: 2 * lift_lb + memcpy_lb,
        4: 0,
        5: lift_lb,
        6: 2 * lift_lb,
        7: 2 * lift_lb + memcpy_lb,
    }
    group_cursor = [0] * 8
    sb_to_lb = [0] * 26
    for lb, bands in enumerate(LB_BANDS):
        for sb in bands:
            sb_to_lb[sb] = lb

    subbands: list[dict[str, int]] = []
    for sb in range(26):
        lb = sb_to_lb[sb]
        buffer_idx = 1 if lb >= 4 else 0
        x24 = lb_bases[lb] + group_cursor[lb]
        group_cursor[lb] += round_up(ng_by_sb[sb], 8)
        subbands.append(
            {
                "sb": sb,
                "lb": lb,
                "ng": ng_by_sb[sb],
                "x24": x24,
                "buffer_idx": buffer_idx,
            }
        )
    pass_a_stride = 3 * lift_lb + memcpy_lb
    stripe_ints = 4 * pass_a_stride
    if any((sb["x24"] + sb["ng"]) * 4 > stripe_ints for sb in subbands):
        raise ProbeError("Python-computed sub-band scatter exceeds full stripe buffer")
    return subbands, stripe_ints, pass_a_stride


def read_sample_stream(sample_path: Path) -> tuple[bytes, dict[str, Any], list[dict[str, Any]], dict[str, Any], list[dict[str, int]]]:
    tiff = ClassicTiff(sample_path)
    try:
        directories = tiff.walk_ifds()
        selected = find_raw_ifd(directories)
        strips = read_strip_bounds(tiff, selected["metadata"])
        if len(strips) != 1:
            raise ProbeError(f"expected one RAW strip; found {len(strips)}")
        strip_info = strips[0]
        strip = tiff.read_at(
            strip_info["offset"], strip_info["byte_count"], "complete RAW strip"
        )
    finally:
        tiff.close()

    stream_offset, segments, header, slice_boundaries = parse_header(strip)
    precincts, termination = parse_precinct_stream(strip, stream_offset, slice_boundaries)
    if not precincts:
        raise ProbeError("packet parser found no precincts")
    return strip, strip_info, precincts, {"segments": segments, "parsed": header, "termination": termination}, slice_boundaries


def zero_state(ng_by_sb: list[int]) -> list[list[int]]:
    return [[0] * ng_by_sb[sb] for sb in range(26)]


def decode_call(
    precinct: bytes,
    call: dict[str, int],
    layout: list[dict[str, int]],
    subbands: list[dict[str, int]],
    ng_by_sb: list[int],
    prediction: bytes,
    scale: list[int],
    wgt: list[dict[str, int]],
    previous_by_sb: list[list[int]],
    previous_sb23: list[int],
    padding: dict[str, Any],
) -> tuple[bytearray, bytearray, dict[str, Any]]:
    lbs, bp, br, dpb, tail_report = parse_lb_payloads(precinct, layout)
    if (bp, br) != (call["Bp"], call["Br"]):
        raise ProbeError(
            f"call {call['call']} Bp/Br={bp}/{br} differ from parsed packet header "
            f"{call['Bp']}/{call['Br']}"
        )

    by_sb = {item["sb"]: item for item in subbands}
    buf_a = bytearray(call["stripe_ints"] * 4)
    buf_b = bytearray(call["stripe_ints"] * 4)
    gcli_by_sb: dict[int, list[int]] = {}
    gtli_by_sb: dict[int, int] = {}
    mode_by_sb: dict[str, str] = {}
    lb_reports: dict[str, Any] = {}
    raw_by_lb: dict[str, bool] = {}

    for lb, payload in enumerate(lbs):
        sig_reader = FastStrictBitReader(payload["sig"], f"call {call['call']} LB {lb} significance")
        gcli_reader = FastStrictBitReader(payload["gcli"], f"call {call['call']} LB {lb} GCLI")
        data_reader = FastStrictBitReader(payload["data"], f"call {call['call']} LB {lb} magnitude")
        sign_reader = FastStrictBitReader(payload["sign"], f"call {call['call']} LB {lb} sign")
        raw_by_lb[str(lb)] = bool(payload["raw"])

        for sb in payload["bands"]:
            dpb_index = sb if sb < 16 else 16 + (sb - 16)
            mode = 0x70 | dpb[dpb_index]
            gtli = compute_gtli(bp, br, sb, wgt)
            if sb == 12:
                previous = previous_sb23
            elif sb == 23:
                previous = gcli_by_sb[12]
            else:
                previous = previous_by_sb[sb]

            gcli = decode_gcli_band(
                sb,
                ng_by_sb[sb],
                gtli,
                mode,
                bool(payload["raw"]),
                sig_reader,
                gcli_reader,
                prediction,
                previous,
            )
            magnitudes = decode_magnitudes(gcli, gtli, data_reader)
            signed = apply_signs(magnitudes, sign_reader)
            coefficients = dequantize(signed, gcli, gtli, scale)
            gcli_by_sb[sb] = gcli
            gtli_by_sb[sb] = gtli
            previous_by_sb[sb] = gcli
            mode_by_sb[str(sb)] = f"0x{mode:02x}"

            base_int = by_sb[sb]["x24"] * 4
            target = buf_a if by_sb[sb]["buffer_idx"] == 0 else buf_b
            try:
                struct.pack_into(f"<{len(coefficients)}i", target, base_int * 4, *coefficients)
            except struct.error as exc:
                raise ProbeError(
                    f"call {call['call']} sb {sb} coefficient does not fit int32: {exc}"
                ) from exc

        reports = {
            "sig": sig_reader.report(),
            "gcli": gcli_reader.report(),
            "data": data_reader.report(),
            "sign": sign_reader.report(),
        }
        lb_reports[str(lb)] = reports
        for stream_name, stream_report in reports.items():
            padding["reader_count"] += 1
            padding["total_unused_bits"] += stream_report["tail_padding_bits"]
            if not stream_report["tail_padding_all_zero"]:
                padding["nonzero_count"] += 1
                if len(padding["examples"]) < 32:
                    padding["examples"].append(
                        {
                            "call": call["call"],
                            "tile": call["tile"],
                            "local_precinct": call["local"],
                            "file_precinct": call["file_precinct"],
                            "lb": lb,
                            "stream": stream_name,
                            "report": stream_report,
                        }
                    )

    detail = {
        "gcli_by_sb": gcli_by_sb,
        "gtli_by_sb": gtli_by_sb,
        "lb_reports": lb_reports,
        "raw_by_lb": raw_by_lb,
        "mode_by_sb": mode_by_sb,
        "tail": tail_report,
    }
    return buf_a, buf_b, detail


def run_truncated_substream_check(
    first_precinct: bytes,
    first_detail: dict[str, Any],
    layout: list[dict[str, int]],
) -> dict[str, Any]:
    lbs, _, _, _, _ = parse_lb_payloads(first_precinct, layout)
    payload = lbs[0]
    original = first_detail["lb_reports"]["0"]["data"]
    required_bytes = (original["consumed_bits"] + 7) // 8
    truncated = payload["data"][: max(0, required_bytes - 1)]
    reader = FastStrictBitReader(truncated, "intentional real LB0 magnitude truncation")
    try:
        for sb in LB_BANDS[0]:
            decode_magnitudes(
                first_detail["gcli_by_sb"][sb],
                first_detail["gtli_by_sb"][sb],
                reader,
            )
    except ProbeError as exc:
        if "truncated bitstream" not in str(exc):
            raise ProbeError(f"truncated-substream check failed for an unexpected reason: {exc}") from exc
        return {
            "passed": True,
            "source": "real first-precinct LB0 magnitude substream",
            "original_bytes": len(payload["data"]),
            "original_consumed_bits": original["consumed_bits"],
            "truncated_bytes": len(truncated),
            "error": str(exc),
            "zero_fill_or_padding_used": False,
        }
    raise ProbeError("intentional truncated-substream check unexpectedly decoded without error")


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_outputs(
    python_output: Path,
    cpp_output: Path,
    calls: list[dict[str, int]],
    stripe_ints: int,
) -> dict[str, Any]:
    record_bytes = stripe_ints * 2 * 4
    expected_bytes = record_bytes * len(calls)
    if python_output.stat().st_size != expected_bytes:
        raise ProbeError(
            f"Python scatter size {python_output.stat().st_size} != expected {expected_bytes}"
        )
    if cpp_output.stat().st_size != expected_bytes:
        raise ProbeError(
            f"C++ scatter size {cpp_output.stat().st_size} != expected {expected_bytes}"
        )

    total_int_differences = 0
    differing_calls = 0
    first_differences: list[dict[str, Any]] = []
    with python_output.open("rb") as py_stream, cpp_output.open("rb") as cpp_stream:
        for call in calls:
            py_record = py_stream.read(record_bytes)
            cpp_record = cpp_stream.read(record_bytes)
            if len(py_record) != record_bytes or len(cpp_record) != record_bytes:
                raise ProbeError(f"short scatter record at call {call['call']}")
            if py_record == cpp_record:
                continue
            differing_calls += 1
            call_diff_count = 0
            for index, (py_value,) in enumerate(struct.iter_unpack("<i", py_record)):
                cpp_value = struct.unpack_from("<i", cpp_record, index * 4)[0]
                if py_value != cpp_value:
                    call_diff_count += 1
                    if len(first_differences) < 24:
                        buffer_name = "A" if index < stripe_ints else "B"
                        buffer_index = index if index < stripe_ints else index - stripe_ints
                        first_differences.append(
                            {
                                "call": call["call"],
                                "tile": call["tile"],
                                "local_precinct": call["local"],
                                "file_precinct": call["file_precinct"],
                                "buffer": buffer_name,
                                "int_index": buffer_index,
                                "python": py_value,
                                "cpp": cpp_value,
                            }
                        )
            total_int_differences += call_diff_count
    return {
        "record_bytes": record_bytes,
        "calls_compared": len(calls),
        "total_int32_values_compared": len(calls) * stripe_ints * 2,
        "differing_call_count": differing_calls,
        "differing_int32_count": total_int_differences,
        "first_differences": first_differences,
        "all_buffers_zero_difference": total_int_differences == 0,
    }


def ensure_fresh(paths: list[Path]) -> None:
    resolved = [path.resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ProbeError("output paths must be distinct")
    for path in paths:
        if path.exists():
            raise ProbeError(f"refusing to overwrite existing output: {path}")


_OWNED_OUTPUTS: list[Path] = []
_KEEP_OUTPUTS = False


def open_exclusive(path: Path, mode: str):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    _OWNED_OUTPUTS.append(path)
    return os.fdopen(fd, mode)


def write_json_new(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open_exclusive(path, "w") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def _execute(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample", type=Path)
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--cpp-output", type=Path, required=True)
    parser.add_argument("--cpp-meta", type=Path, required=True)
    parser.add_argument("--python-output", type=Path, required=True)
    parser.add_argument("--python-meta", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)

    output_paths = [args.python_output, args.python_meta, args.report]
    ensure_fresh(output_paths)
    for path in (args.sample, args.tables, args.cpp_output, args.cpp_meta):
        if not path.is_file():
            raise ProbeError(f"required input does not exist: {path}")

    prediction, scale, prediction_address, dequant_address = load_entropy_tables(args.tables)
    strip, strip_info, packet_headers, stream_info, slice_boundaries = read_sample_stream(args.sample)
    header = stream_info["parsed"]
    pih = header["PIH"]
    if (pih["width"], pih["height"]) != (EXPECTED_RAW_WIDTH, EXPECTED_RAW_HEIGHT):
        raise ProbeError(f"sample dimensions changed: {pih['width']}x{pih['height']}")
    if len(packet_headers) != 1010:
        raise ProbeError(f"expected all 1010 file precincts; found {len(packet_headers)}")

    half_width = pih["width"] // 2
    py_layout, stripe_ints, pass_a_stride = compute_scatter_layout(half_width)
    geometry_layout, ng_by_sb = compute_geometry(half_width)
    wgt = header["WGT"]["bands"]
    if len(wgt) < 25:
        raise ProbeError(f"WGT contains only {len(wgt)} entries")

    cpp = json.loads(args.cpp_meta.read_text(encoding="utf-8"))
    cpp_layout = cpp.get("layout")
    if cpp.get("source_commit") != PUBLIC_SOURCE_COMMIT:
        raise ProbeError("C++ metadata is not from the pinned public source commit")
    if cpp.get("record_order") != "A then B; each is stripe_ints little-endian int32":
        raise ProbeError("C++ scatter record order or encoding metadata is unexpected")
    if cpp.get("stripe_ints") != stripe_ints or cpp.get("buffer_ints") != stripe_ints:
        raise ProbeError(
            f"Python stripe geometry {stripe_ints} disagrees with C++ metadata "
            f"{cpp.get('stripe_ints')}"
        )
    if cpp_layout != [{k: s[k] for k in ("sb", "ng", "x24", "buffer_idx")} for s in py_layout]:
        raise ProbeError("Python independently computed scatter geometry differs from C++ metadata")

    n_tiles = (pih["height"] + 63) // 64
    calls: list[dict[str, int]] = []
    distinct_indices: set[int] = set()
    tile_summaries = []
    for tile in range(n_tiles):
        base = tile * 16
        tile_calls = 0
        for local in range(18):
            file_index = base + local
            if file_index >= len(packet_headers):
                break
            packet = packet_headers[file_index]
            calls.append(
                {
                    "call": len(calls),
                    "tile": tile,
                    "local": local,
                    "file_precinct": file_index,
                    "offset": packet["offset"],
                    "size": packet["total_size"],
                    "Bp": packet["Bp"],
                    "Br": packet["Br"],
                    "reset": local == 16,
                    "stripe_ints": stripe_ints,
                }
            )
            distinct_indices.add(file_index)
            tile_calls += 1
        tile_summaries.append(
            {
                "tile": tile,
                "file_base": base,
                "call_count": tile_calls,
                "first_file_precinct": base if tile_calls else None,
                "last_file_precinct": base + tile_calls - 1 if tile_calls else None,
                "fresh_predecessor_state": True,
                "reset_local_index": 16 if tile_calls > 16 else None,
            }
        )
    if distinct_indices != set(range(len(packet_headers))):
        raise ProbeError("tile call schedule did not cover every file precinct")
    cpp_call_map = cpp.get("call_map", [])
    if len(cpp_call_map) != len(calls):
        raise ProbeError(f"C++ call count {len(cpp_call_map)} != Python {len(calls)}")
    for expected, reference in zip(calls, cpp_call_map):
        for key in ("call", "tile", "local", "file_precinct", "Bp", "Br", "reset"):
            if expected[key] != reference.get(key):
                raise ProbeError(
                    f"C++ call map differs at call {expected['call']} field {key}: "
                    f"Python={expected[key]} C++={reference.get(key)}"
                )

    args.python_output.parent.mkdir(parents=True, exist_ok=True)
    mode_counts: dict[str, Counter[str]] = {str(sb): Counter() for sb in range(26)}
    raw_counts: Counter[str] = Counter()
    padding = {"reader_count": 0, "total_unused_bits": 0, "nonzero_count": 0, "examples": []}
    first_detail: dict[str, Any] | None = None
    per_call_hashes: list[dict[str, Any]] = []
    with open_exclusive(args.python_output, "wb") as output:
        for call in calls:
            packet_header = packet_headers[call["file_precinct"]]
            start = packet_header["offset"]
            precinct = strip[start : start + packet_header["total_size"]]
            if len(precinct) != packet_header["total_size"]:
                raise ProbeError(f"short precinct slice for file precinct {call['file_precinct']}")

            if call["local"] == 16:
                previous_by_sb = zero_state(ng_by_sb)
                previous_sb23 = [0] * ng_by_sb[23]
            elif call["local"] == 0:
                # A new state is created at every tile boundary. This is
                # intentionally distinct from a reset at local index 16.
                previous_by_sb = zero_state(ng_by_sb)
                previous_sb23 = [0] * ng_by_sb[23]
            if call["tile"] == 0 and call["local"] == 0:
                previous_by_sb = zero_state(ng_by_sb)
                previous_sb23 = [0] * ng_by_sb[23]

            buf_a, buf_b, detail = decode_call(
                precinct,
                call,
                geometry_layout,
                py_layout,
                ng_by_sb,
                prediction,
                scale,
                wgt,
                previous_by_sb,
                previous_sb23,
                padding,
            )
            # Only sb 23 survives as the cross-band predecessor into the next
            # precinct; ordinary bands preserve their own decoded GCLI arrays.
            previous_sb23 = detail["gcli_by_sb"][23]
            for sb_text, mode in detail["mode_by_sb"].items():
                mode_counts[sb_text][mode] += 1
            for lb, raw in detail["raw_by_lb"].items():
                raw_counts[f"lb{lb}:{'raw' if raw else 'coded'}"] += 1
            if first_detail is None:
                first_detail = detail
            output.write(buf_a)
            output.write(buf_b)
            if call["call"] < 3 or call["local"] in (15, 16, 17):
                per_call_hashes.append(
                    {
                        "call": call["call"],
                        "tile": call["tile"],
                        "local": call["local"],
                        "file_precinct": call["file_precinct"],
                        "raw_lbs": [int(lb) for lb, raw in detail["raw_by_lb"].items() if raw],
                        "buffer_sha256": {
                            "A": hashlib.sha256(buf_a).hexdigest(),
                            "B": hashlib.sha256(buf_b).hexdigest(),
                        },
                    }
                )
    if first_detail is None:
        raise ProbeError("no precinct calls were decoded")

    truncation = run_truncated_substream_check(
        strip[
            packet_headers[0]["offset"] : packet_headers[0]["end_exclusive"]
        ],
        first_detail,
        geometry_layout,
    )
    comparison = compare_outputs(args.python_output, args.cpp_output, calls, stripe_ints)
    python_meta = {
        "decoder": "independent_python_continuous_entropy_model",
        "byte_order": "little_endian",
        "record_order": "A then B; each is stripe_ints little-endian int32",
        "output": str(args.python_output),
        "calls": len(calls),
        "file_precinct_count": len(packet_headers),
        "distinct_file_precincts": len(distinct_indices),
        "tile_count": n_tiles,
        "stripe_ints": stripe_ints,
        "buffer_ints": stripe_ints,
        "pass_a_stride": pass_a_stride,
        "layout": py_layout,
        "call_map": [
            {key: item[key] for key in ("call", "tile", "local", "file_precinct", "Bp", "Br", "reset")}
            for item in calls
        ],
    }
    write_json_new(args.python_meta, python_meta)

    reset_events = [
        {
            "tile": call["tile"],
            "local_precinct_index": call["local"],
            "file_precinct_index": call["file_precinct"],
            "meaning": "same tile's predecessor state reset immediately before this decode call",
        }
        for call in calls
        if call["reset"]
    ]
    report = {
        "probe": "ticoraw_continuous_precinct_entropy_v1",
        "status": "passed"
        if comparison["all_buffers_zero_difference"]
        and truncation["passed"]
        and padding["nonzero_count"] == 0
        else "mismatch",
        "scope": {
            "decoded": "all GCLI, magnitude, signed, dequantized coefficients and full A/B scatter buffers for every public image-orchestrated precinct call",
            "not_decoded": ["full CFA reconstruction", "display/rendered pixels", "Adobe Camera Raw execution"],
            "independent_python_model": True,
        },
        "sample": {
            "name": args.sample.name,
            "size_bytes": args.sample.stat().st_size,
            "sha256": sha256_file(args.sample),
            "strip_index": strip_info["index"],
            "strip_byte_count": strip_info["byte_count"],
            "strip_sha256": hashlib.sha256(strip).hexdigest(),
        },
        "entropy_tables": {
            "name": args.tables.name,
            "sha256": sha256_file(args.tables),
            "prediction_address": prediction_address,
            "prediction_bytes": len(prediction),
            "dequant_address": dequant_address,
            "dequant_scale_entries": len(scale),
            "provenance": "ACR-extracted bytes read directly from the supplied table JSON",
        },
        "public_source_assistance": {
            "commit": PUBLIC_SOURCE_COMMIT,
            "path": PUBLIC_SOURCE_PATH,
            "used_for": [
                "image-level tile count and 16-stride/18-precinct tile scheduling",
                "fresh predecessor state per tile",
                "decode_precinct per-call state and reset invocation",
                "sub-band predecessor routing and entropy/dequantization reference",
                "scatter layout and complete stripe buffer dimensions",
            ],
            "source_observation": {
                "n_tiles": "ceil(image_height/64)",
                "tile_file_base": "tile_index*16",
                "max_calls_per_tile": 18,
                "should_reset_gcli": "the pinned source implementation returns true exactly when precinct_index == 16, regardless of Bp/Br; called from decode_precinct before decoding that local tile call",
                "reset_source": "/private/tmp/imprint-tico-4abfcd2/src/decoders/nikon_he/nikon_he_predecessor.h",
                "tile_state_source": "/private/tmp/imprint-tico-4abfcd2/src/decoders/nikon_he/nikon_he_decode.cpp",
                "decode_source": "/private/tmp/imprint-tico-4abfcd2/src/decoders/nikon_he/nikon_he_precinct_decode.cpp",
            },
            "independent_reverse_engineering_claim": False,
            "adobe_execution": False,
        },
        "schedule": {
            "image_width": pih["width"],
            "image_height": pih["height"],
            "tile_count": n_tiles,
            "file_precinct_count": len(packet_headers),
            "tile_decode_call_count": len(calls),
            "unique_file_precincts_covered": len(distinct_indices),
            "distinct_coverage_complete": len(distinct_indices) == len(packet_headers),
            "overlap_calls": len(calls) - len(packet_headers),
            "tiles": tile_summaries,
            "reset_events": reset_events,
            "reset_event_count": len(reset_events),
            "boundary_examples": {
                "tile_0_local_15": calls[15],
                "tile_0_local_16_reset": calls[16],
                "tile_0_local_17": calls[17],
                "tile_1_local_0_fresh_state_overlap": calls[18],
                "tile_1_local_1_fresh_state_overlap": calls[19],
            },
        },
        "scatter": {
            "geometry_source": "computed in Python from width and subband decomposition, then field-by-field checked against C++ metadata",
            "stripe_ints": stripe_ints,
            "buffer_ints_per_a_or_b": stripe_ints,
            "record_order": "A then B, each a complete stripe_ints int32 array",
            "byte_order": "little_endian",
            "python_output": str(args.python_output),
            "python_output_bytes": args.python_output.stat().st_size,
            "python_output_sha256": hash_file(args.python_output),
            "cpp_output": str(args.cpp_output),
            "cpp_output_bytes": args.cpp_output.stat().st_size,
            "cpp_output_sha256": hash_file(args.cpp_output),
            "python_metadata": str(args.python_meta),
            "cpp_metadata": str(args.cpp_meta),
            "cpp_declared_calls": cpp.get("calls"),
            "cpp_declared_buffer_ints": cpp.get("buffer_ints"),
            "buffer_comparison": comparison,
            "selected_call_hashes": per_call_hashes,
        },
        "entropy_modes": {
            "mode_count_by_subband": {sb: dict(counter) for sb, counter in mode_counts.items()},
            "raw_or_coded_lb_call_count": dict(raw_counts),
            "mode_interpretation": "0x71 follows zero-prediction; all other values use the ACR prediction table with the public-source predecessor routing",
        },
        "padding": {
            **padding,
            "all_unused_bits_zero": padding["nonzero_count"] == 0,
            "packet_tail_after_all_lb_payloads_all_zero": True,
        },
        "truncated_substream_check": truncation,
        "limitations": [
            "The Python model is informed by the pinned public LibRaw implementation and directly supplied ACR-extracted entropy tables; agreement with that public implementation does not establish Adobe Camera Raw runtime equivalence.",
            "No Adobe code was executed; no claim of Adobe Camera Raw pixel or image equivalence is made.",
            "Outputs are wavelet sub-band coefficient scatter buffers for a later reconstruction stage, not CFA pixels or a rendered image.",
            "The public source's mode 0x70 handling, GTLI/WGT mapping, and dequant shift remain public-source-assisted model choices without direct ACR runtime verification.",
        ],
    }
    write_json_new(args.report, report)
    global _KEEP_OUTPUTS
    _KEEP_OUTPUTS = True
    print(
        json.dumps(
            {
                "status": report["status"],
                "calls": len(calls),
                "distinct_file_precincts": len(distinct_indices),
                "stripe_ints": stripe_ints,
                "differing_calls": comparison["differing_call_count"],
                "differing_int32": comparison["differing_int32_count"],
                "padding_anomalies": padding["nonzero_count"],
                "truncated_substream_hard_error": truncation["passed"],
                "report": str(args.report),
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["status"] == "passed" else 1


def main(argv: list[str] | None = None) -> int:
    global _KEEP_OUTPUTS
    _OWNED_OUTPUTS.clear()
    _KEEP_OUTPUTS = False
    try:
        return _execute(argv)
    finally:
        if not _KEEP_OUTPUTS:
            for path in reversed(_OWNED_OUTPUTS):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ProbeError, OSError, ValueError, KeyError, TypeError, struct.error) as exc:
        print(f"probe_ticoraw_continuous: {exc}", file=sys.stderr)
        raise SystemExit(2)
