#!/usr/bin/env python3
"""Experimental Python model for the first TicoRAW precinct's coefficients.

This bounded research probe decodes the first precinct only. It uses the
ACR-extracted entropy tables supplied by the caller and the public LibRaw
Nikon HE implementation as a decoding-scheme aid. It does not execute Adobe
code, decode later precincts, or reconstruct/display CFA pixels.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import struct
import sys
from pathlib import Path
from typing import Any

from probe_ticoraw_header import (
    EXPECTED_RAW_HEIGHT,
    EXPECTED_RAW_WIDTH,
    ClassicTiff,
    ProbeError,
    find_raw_ifd,
    read_strip_bounds,
    sha256_file,
)
from probe_ticoraw_packets import parse_header, parse_precinct_stream


LB_BANDS: tuple[tuple[int, ...], ...] = (
    tuple(range(0, 6)),
    tuple(range(6, 12)),
    (12,),
    tuple(range(13, 19)),
    (19, 20),
    (21, 22),
    (23,),
    (24, 25),
)
STAGES = ("gcli", "magnitudes", "signed", "dequantized")
PUBLIC_SOURCE_COMMIT = "4abfcd2ab82f2211b1f7e7c3cd83974c87e0d812"
PUBLIC_SOURCE_PATH = "/private/tmp/imprint-tico-4abfcd2/src/decoders/nikon_he"


class MSBBitReader:
    """MSB-first reader with a hard byte-buffer boundary and no zero padding."""

    def __init__(self, data: bytes, label: str):
        self.data = data
        self.label = label
        self.bit_position = 0
        self.total_bits = len(data) * 8

    def read_bits(self, width: int) -> int:
        if width < 0:
            raise ProbeError(f"{self.label}: negative read width {width}")
        remaining = self.total_bits - self.bit_position
        if width > remaining:
            raise ProbeError(
                f"{self.label}: truncated bitstream at bit {self.bit_position}; "
                f"requested {width}, have {remaining}"
            )
        value = 0
        for _ in range(width):
            byte = self.data[self.bit_position >> 3]
            shift = 7 - (self.bit_position & 7)
            value = (value << 1) | ((byte >> shift) & 1)
            self.bit_position += 1
        return value

    def read_unary_ones(self, max_value: int = 31) -> int:
        value = 0
        while self.read_bits(1):
            value += 1
            if value > max_value:
                raise ProbeError(
                    f"{self.label}: unary run {value} exceeds table index limit {max_value}"
                )
        return value

    def report(self) -> dict[str, Any]:
        remaining = self.total_bits - self.bit_position
        tail_value = 0
        for bit_position in range(self.bit_position, self.total_bits):
            byte = self.data[bit_position >> 3]
            shift = 7 - (bit_position & 7)
            tail_value = (tail_value << 1) | ((byte >> shift) & 1)
        tail_hex_width = (remaining + 3) // 4
        tail_hex = f"{tail_value:0{tail_hex_width}x}" if tail_hex_width else ""
        return {
            "consumed_bits": self.bit_position,
            "total_bits": self.total_bits,
            "tail_padding_bits": remaining,
            "tail_padding_hex_msb_first": tail_hex,
            "tail_padding_all_zero": tail_value == 0,
        }


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def compute_geometry(half_width: int) -> tuple[list[dict[str, int]], list[int]]:
    low = [half_width]
    high = [0]
    for _ in range(5):
        high.append(low[-1] // 2)
        low.append(low[-1] - high[-1])

    ng_5level = [
        ceil_div(low[5], 4),
        ceil_div(high[5], 4),
        ceil_div(high[4], 4),
        ceil_div(high[3], 4),
        ceil_div(high[2], 4),
        ceil_div(high[1], 4),
    ]
    ng_1level = [ceil_div(low[1], 4), ceil_div(high[1], 4)]
    ng_by_sb = [0] * 26
    for sb in LB_BANDS[0]:
        ng_by_sb[sb] = ng_5level[sb]
    for sb in LB_BANDS[1]:
        ng_by_sb[sb] = ng_5level[sb - 6]
    ng_by_sb[12] = half_width // 4
    for sb in LB_BANDS[3]:
        ng_by_sb[sb] = ng_5level[sb - 13]
    for sb in LB_BANDS[4]:
        ng_by_sb[sb] = ng_1level[sb - 19]
    for sb in LB_BANDS[5]:
        ng_by_sb[sb] = ng_1level[sb - 21]
    ng_by_sb[23] = half_width // 4
    for sb in LB_BANDS[7]:
        ng_by_sb[sb] = ng_1level[sb - 24]

    layout = []
    for lb, bands in enumerate(LB_BANDS):
        sig_bits = sum(ceil_div(ng_by_sb[sb], 8) for sb in bands)
        layout.append(
            {
                "lb": lb,
                "sig_bytes": ceil_div(sig_bits, 8),
                "first_sb": bands[0],
                "last_sb": bands[-1],
            }
        )
    return layout, ng_by_sb


def load_entropy_tables(path: Path) -> tuple[bytes, list[int], str, str]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        prediction_entry = document["prediction"]
        dequant_entry = document["dequant"]
        prediction = bytes.fromhex(prediction_entry["data"])
        dequant_bytes = bytes.fromhex(dequant_entry["data"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ProbeError(f"cannot parse entropy table JSON {path}: {exc}") from exc

    if len(prediction) != 8192 or prediction_entry.get("size") != 8192:
        raise ProbeError(
            f"prediction table must contain 8192 bytes; found {len(prediction)}"
        )
    if len(dequant_bytes) != 64 or dequant_entry.get("size") != 64:
        raise ProbeError(f"dequant table must contain 64 bytes; found {len(dequant_bytes)}")
    scale = list(struct.unpack("<16i", dequant_bytes))
    if any(value < 0 for value in scale):
        raise ProbeError("dequant table contains a negative scale value")
    return (
        prediction,
        scale,
        str(prediction_entry.get("address", "")),
        str(dequant_entry.get("address", "")),
    )


def load_reference(path: Path) -> dict[str, Any]:
    try:
        reference = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProbeError(f"cannot parse C++ reference JSON {path}: {exc}") from exc
    if (reference.get("width"), reference.get("height")) != (
        EXPECTED_RAW_WIDTH,
        EXPECTED_RAW_HEIGHT,
    ):
        raise ProbeError("C++ reference dimensions do not match the expected LCR_3520 sample")
    if len(reference.get("lbs", [])) != 8 or len(reference.get("bands", [])) != 26:
        raise ProbeError("C++ reference must contain eight LB and 26 sub-band records")
    by_sb = {band.get("sb"): band for band in reference["bands"]}
    if set(by_sb) != set(range(26)):
        raise ProbeError("C++ reference sub-band records must cover sb 0 through 25 exactly")
    for sb, band in by_sb.items():
        for stage in STAGES:
            if not isinstance(band.get(stage), list):
                raise ProbeError(f"C++ reference sb {sb} is missing its {stage} array")
    reference["_bands_by_sb"] = by_sb
    return reference


def sb_to_wgt_index(sb: int) -> int:
    if sb < 23:
        return sb
    if sb == 23:
        return 12
    return sb - 1


def compute_gtli(bp: int, br: int, sb: int, wgt: list[dict[str, int]]) -> int:
    index = sb_to_wgt_index(sb)
    if index >= len(wgt):
        raise ProbeError(f"WGT table has no entry for sub-band {sb} (index {index})")
    gain = int(wgt[index]["gain"])
    priority = int(wgt[index]["priority"])
    return max(0, min(15, bp - gain - (1 if priority < br else 0)))


def parse_dpb(data: bytes) -> list[int]:
    if len(data) != 7:
        raise ProbeError(f"Dpb header must be seven bytes; found {len(data)}")
    values: list[int] = []
    for byte in data:
        values.extend((byte >> shift) & 0x03 for shift in (6, 4, 2, 0))
    return values


def decode_prediction(
    table: bytes, gtli: int, previous: int, unary: int, sb: int, group: int
) -> int:
    if not (0 <= gtli <= 15 and 0 <= previous <= 15 and 0 <= unary <= 31):
        raise ProbeError(
            f"sb {sb} group {group}: prediction tuple outside ACR table bounds "
            f"(gtli={gtli}, previous={previous}, unary={unary})"
        )
    index = (gtli << 9) | (previous << 5) | unary
    value = table[index]
    if value == 0xFF:
        raise ProbeError(
            f"sb {sb} group {group}: ACR prediction table marks index {index} unreachable"
        )
    if value > 30:
        raise ProbeError(f"sb {sb} group {group}: predicted GCLI {value} exceeds model limit 30")
    return value


def decode_gcli_band(
    sb: int,
    ng: int,
    gtli: int,
    mode: int,
    raw: bool,
    sig_reader: MSBBitReader,
    gcli_reader: MSBBitReader,
    prediction: bytes,
    previous_values: list[int],
) -> list[int]:
    if raw:
        return [gcli_reader.read_bits(4) for _ in range(ng)]

    values: list[int] = []
    for base in range(0, ng, 8):
        block_size = min(8, ng - base)
        sig = sig_reader.read_bits(1)
        if sig:
            if mode == 0x71:
                values.extend([gtli] * block_size)
            else:
                values.extend(max(previous_values[base + i], gtli) for i in range(block_size))
            continue

        for group in range(base, base + block_size):
            unary = gcli_reader.read_unary_ones()
            if mode == 0x71:
                value = gtli + unary
                if value > 30:
                    raise ProbeError(f"sb {sb} group {group}: zero-predicted GCLI {value} exceeds 30")
            else:
                value = decode_prediction(
                    prediction, gtli, previous_values[group], unary, sb, group
                )
            values.append(value)
    return values


def decode_magnitudes(
    gcli: list[int], gtli: int, data_reader: MSBBitReader
) -> list[int]:
    values = [0] * (len(gcli) * 4)
    for group, gcli_value in enumerate(gcli):
        bitplanes = gcli_value - gtli
        if bitplanes <= 0:
            continue
        coefficients = [0, 0, 0, 0]
        for _ in range(bitplanes):
            nibble = data_reader.read_bits(4)
            coefficients[0] = (coefficients[0] << 1) | ((nibble >> 3) & 1)
            coefficients[1] = (coefficients[1] << 1) | ((nibble >> 2) & 1)
            coefficients[2] = (coefficients[2] << 1) | ((nibble >> 1) & 1)
            coefficients[3] = (coefficients[3] << 1) | (nibble & 1)
        start = group * 4
        for channel, magnitude in enumerate(coefficients):
            values[start + channel] = magnitude << gtli
    return values


def apply_signs(magnitudes: list[int], sign_reader: MSBBitReader) -> list[int]:
    signed: list[int] = []
    for magnitude in magnitudes:
        if magnitude and sign_reader.read_bits(1):
            signed.append(-magnitude)
        else:
            signed.append(magnitude)
    return signed


def dequantize(signed: list[int], gcli: list[int], gtli: int, scale: list[int]) -> list[int]:
    values = [0] * len(signed)
    for group, gcli_value in enumerate(gcli):
        planes = gcli_value - gtli
        if planes <= 0 or planes > 15:
            continue
        factor = scale[planes - 1]
        for channel in range(4):
            index = group * 4 + channel
            coefficient = signed[index]
            magnitude = abs(coefficient) >> gtli
            result = (magnitude * factor) >> (16 - gtli)
            result <<= 4
            values[index] = -result if coefficient < 0 else result
    return values


def parse_lb_payloads(
    precinct: bytes, layout: list[dict[str, int]]
) -> tuple[list[dict[str, Any]], int, int, list[int], dict[str, Any]]:
    if len(precinct) < 12:
        raise ProbeError(f"first precinct has only {len(precinct)} bytes; expected at least 12")
    bp, br = precinct[3], precinct[4]
    dpb = parse_dpb(precinct[5:12])
    cursor = 12
    lbs: list[dict[str, Any]] = []
    for lb, bands in enumerate(LB_BANDS):
        header_offset = cursor
        if cursor + 7 > len(precinct):
            raise ProbeError(f"LB {lb} mini-header at byte {cursor} is truncated")
        packed = int.from_bytes(precinct[cursor:cursor + 7], "big")
        raw = (packed >> 55) & 1
        data_bytes = (packed >> 35) & 0xFFFFF
        gcli_bytes = (packed >> 15) & 0xFFFFF
        sign_bytes = packed & 0x7FFF
        sig_bytes = 0 if raw else layout[lb]["sig_bytes"]
        cursor += 7
        sig_offset = cursor
        stream_sizes = (sig_bytes, gcli_bytes, data_bytes, sign_bytes)
        if any(size < 0 for size in stream_sizes):
            raise ProbeError(f"LB {lb} contains a negative stream length")
        payload_size = sum(stream_sizes)
        if cursor + payload_size > len(precinct):
            raise ProbeError(
                f"LB {lb} payload ends at {cursor + payload_size}, beyond precinct size {len(precinct)}"
            )
        sig_start = cursor
        gcli_start = sig_start + sig_bytes
        data_start = gcli_start + gcli_bytes
        sign_start = data_start + data_bytes
        lbs.append(
            {
                "lb": lb,
                "raw": raw,
                "header_offset": header_offset,
                "sig_offset": sig_offset,
                "sig_bytes": sig_bytes,
                "gcli_bytes": gcli_bytes,
                "data_bytes": data_bytes,
                "sign_bytes": sign_bytes,
                "bands": bands,
                "sig": precinct[sig_start:gcli_start],
                "gcli": precinct[gcli_start:data_start],
                "data": precinct[data_start:sign_start],
                "sign": precinct[sign_start:sign_start + sign_bytes],
            }
        )
        cursor += payload_size
    tail = precinct[cursor:]
    if any(tail):
        raise ProbeError(
            f"first precinct has {len(tail)} nonzero byte(s) after the eight LB payloads"
        )
    tail_report = {
        "offset": cursor,
        "byte_count": len(tail),
        "all_zero": True,
        "sha256": hashlib.sha256(tail).hexdigest(),
    }
    return lbs, bp, br, dpb, tail_report


def digest_values(stage: str, values: list[int]) -> str:
    if stage == "gcli":
        payload = bytes(values)
    else:
        payload = struct.pack(f"<{len(values)}i", *values)
    return hashlib.sha256(payload).hexdigest()


def compare_stage(model: list[int], reference: list[int]) -> dict[str, Any]:
    if len(model) != len(reference):
        return {
            "model_count": len(model),
            "reference_count": len(reference),
            "difference_count": abs(len(model) - len(reference)) + sum(
                a != b for a, b in zip(model, reference)
            ),
            "first_differences": [],
        }
    differences = []
    count = 0
    for index, (actual, expected) in enumerate(zip(model, reference)):
        if actual != expected:
            count += 1
            if len(differences) < 8:
                differences.append({"index": index, "model": actual, "reference": expected})
    return {
        "model_count": len(model),
        "reference_count": len(reference),
        "difference_count": count,
        "first_differences": differences,
    }


def compare_lb(model: dict[str, int], reference: dict[str, Any]) -> dict[str, Any]:
    keys = ("raw", "sig_bytes", "gcli_bytes", "data_bytes", "sign_bytes", "sig_offset")
    return {
        key: {"model": model[key], "reference": reference.get(key), "match": model[key] == reference.get(key)}
        for key in keys
    }


def decode_first_precinct(
    sample_path: Path,
    tables_path: Path,
    reference_path: Path,
    arrays_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, list[int]]]]:
    prediction, scale, prediction_address, dequant_address = load_entropy_tables(tables_path)
    reference = load_reference(reference_path)

    tiff = ClassicTiff(sample_path)
    try:
        directories = tiff.walk_ifds()
        selected = find_raw_ifd(directories)
        strips = read_strip_bounds(tiff, selected["metadata"])
        if len(strips) != 1:
            raise ProbeError(f"expected one RAW strip for the bounded probe; found {len(strips)}")
        strip_info = strips[0]
        strip = tiff.read_at(strip_info["offset"], strip_info["byte_count"], "complete RAW strip")
        stream_offset, segments, header, slice_boundaries = parse_header(strip)
        precinct_headers, termination = parse_precinct_stream(strip, stream_offset, slice_boundaries)
        if not precinct_headers:
            raise ProbeError("packet parser found no first precinct")
        first_header = precinct_headers[0]
        precinct_start = first_header["offset"]
        precinct_end = first_header["end_exclusive"]
        precinct = strip[precinct_start:precinct_end]
        if len(precinct) != first_header["total_size"]:
            raise ProbeError("packet boundary length does not match first precinct byte slice")
    finally:
        tiff.close()

    layout, ng_by_sb = compute_geometry(EXPECTED_RAW_WIDTH // 2)
    lbs, bp, br, dpb, precinct_tail = parse_lb_payloads(precinct, layout)
    if (bp, br) != (reference.get("Bp"), reference.get("Br")):
        raise ProbeError(
            f"first precinct Bp/Br={bp}/{br} differ from C++ reference "
            f"{reference.get('Bp')}/{reference.get('Br')}"
        )

    wgt = header["WGT"]["bands"]
    if len(wgt) < 25:
        raise ProbeError(f"WGT contains {len(wgt)} gain/priority entries; at least 25 are required")

    reference_lbs = {item["lb"]: item for item in reference["lbs"]}
    lb_summaries: list[dict[str, Any]] = []
    band_records: dict[int, dict[str, Any]] = {}
    band_arrays: dict[str, dict[str, list[int]]] = {}
    bit_readers: dict[str, dict[str, dict[str, Any]]] = {}
    dpb_modes: dict[int, int] = {}
    for lb, payload in enumerate(lbs):
        ref_lb = reference_lbs.get(lb)
        if ref_lb is None:
            raise ProbeError(f"C++ reference is missing LB {lb}")
        lb_model = {key: payload[key] for key in ("lb", "raw", "sig_bytes", "gcli_bytes", "data_bytes", "sign_bytes", "sig_offset")}
        lb_comparison = compare_lb(lb_model, ref_lb)
        lb_summaries.append({**lb_model, "reference_comparison": lb_comparison})

        sig_reader = MSBBitReader(payload["sig"], f"LB {lb} significance")
        gcli_reader = MSBBitReader(payload["gcli"], f"LB {lb} GCLI")
        data_reader = MSBBitReader(payload["data"], f"LB {lb} magnitude")
        sign_reader = MSBBitReader(payload["sign"], f"LB {lb} sign")
        for sb in payload["bands"]:
            wgt_index = sb_to_wgt_index(sb)
            gtli = compute_gtli(bp, br, sb, wgt)
            dpb_index = sb if sb < 16 else 16 + (sb - 16)
            mode = 0x70 | dpb[dpb_index]
            dpb_modes[sb] = mode
            prev = [0] * ng_by_sb[sb]
            if sb == 23:
                prev = band_records[12]["gcli_values"]
            gcli = decode_gcli_band(
                sb,
                ng_by_sb[sb],
                gtli,
                mode,
                bool(payload["raw"]),
                sig_reader,
                gcli_reader,
                prediction,
                prev,
            )
            magnitudes = decode_magnitudes(gcli, gtli, data_reader)
            signed = apply_signs(magnitudes, sign_reader)
            dequantized = dequantize(signed, gcli, gtli, scale)
            arrays = {
                "gcli": gcli,
                "magnitudes": magnitudes,
                "signed": signed,
                "dequantized": dequantized,
            }
            band_arrays[str(sb)] = arrays
            band_records[sb] = {
                "sb": sb,
                "lb": lb,
                "ng": ng_by_sb[sb],
                "wgt_index": wgt_index,
                "gtli": gtli,
                "mode": mode,
                "gcli_values": gcli,
            }
        bit_readers[str(lb)] = {
            "sig": sig_reader.report(),
            "gcli": gcli_reader.report(),
            "data": data_reader.report(),
            "sign": sign_reader.report(),
        }

    per_band: list[dict[str, Any]] = []
    stage_totals = {stage: {"value_count": 0, "difference_count": 0} for stage in STAGES}
    for sb in range(26):
        model = band_arrays[str(sb)]
        ref = reference["_bands_by_sb"][sb]
        config_match = {
            "lb": band_records[sb]["lb"] == ref.get("lb"),
            "ng": band_records[sb]["ng"] == ref.get("ng"),
            "gtli": band_records[sb]["gtli"] == ref.get("gtli"),
            "mode": band_records[sb]["mode"] == ref.get("mode"),
        }
        stage_comparison: dict[str, Any] = {}
        for stage in STAGES:
            comparison = compare_stage(model[stage], ref[stage])
            comparison["match"] = comparison["difference_count"] == 0
            stage_comparison[stage] = comparison
            stage_totals[stage]["value_count"] += comparison["model_count"]
            stage_totals[stage]["difference_count"] += comparison["difference_count"]
        per_band.append(
            {
                **{key: band_records[sb][key] for key in ("sb", "lb", "ng", "wgt_index", "gtli", "mode")},
                "configuration_matches_reference": config_match,
                "stages": {
                    stage: {
                        "sha256": digest_values(stage, model[stage]),
                        "count": len(model[stage]),
                        "nonzero_count": sum(value != 0 for value in model[stage]),
                        "first_8": model[stage][:8],
                        "comparison": stage_comparison[stage],
                    }
                    for stage in STAGES
                },
            }
        )

    total_coefficients = sum(ng * 4 for ng in ng_by_sb)
    result = {
        "probe": "nikon_ticoraw_first_precinct_coefficients_v1",
        "status": "experimental_first_precinct_model",
        "scope": {
            "decoded": "GCLI, magnitude, signed and dequantized coefficient stages for the first precinct only",
            "not_decoded": ["later-precinct predecessor state", "full CFA reconstruction", "display/rendered pixels"],
            "does_not_execute_adobe_code": True,
        },
        "sample": {
            "name": sample_path.name,
            "size_bytes": sample_path.stat().st_size,
            "sha256": sha256_file(sample_path),
            "strip_index": strip_info["index"],
            "strip_sha256": hashlib.sha256(strip).hexdigest(),
        },
        "reference": {
            "path_name": reference_path.name,
            "source_commit": reference.get("source_commit"),
            "orchestration_parity": reference.get("orchestration_parity"),
            "meaning": "fixed C++ reference output; it is not Adobe Camera Raw execution",
        },
        "entropy_tables": {
            "path_name": tables_path.name,
            "sha256": sha256_file(tables_path),
            "prediction_address": prediction_address,
            "dequant_address": dequant_address,
            "prediction_bytes": len(prediction),
            "dequant_entries_signed_little_endian_i32": len(scale),
            "provenance": "ACR-extracted tables provided by caller; bytes used directly by this Python model",
        },
        "public_source_assistance": {
            "commit": PUBLIC_SOURCE_COMMIT,
            "path": PUBLIC_SOURCE_PATH,
            "used_for": "public packet layout and entropy/dequantization scheme cross-check",
            "independent_reverse_engineering_claim": False,
        },
        "parameter_provenance": {
            "gtli_wgt_index_and_priority_formula": "implemented from the supplied probe model and cross-checked against public LibRaw; not independently traced from an Adobe Camera Raw execution",
            "dequant_shift_4": "taken from public LibRaw Nikon HE source; not independently verified against an Adobe Camera Raw sample execution",
            "prediction_and_scale_bytes": "read directly from the caller-supplied ACR-extracted table JSON",
        },
        "precinct": {
            "index": 0,
            "strip_offset": precinct_start,
            "total_size": len(precinct),
            "Bp": bp,
            "Br": br,
            "Dpb_values_28_two_bit_msb_first": dpb,
            "mode_by_subband": {str(sb): dpb_modes[sb] for sb in range(26)},
            "mode_0x70_note": "sb 25 uses prediction in this model because the public decoder predicts for every mode except 0x71; ACR-specific 0x70 mapping remains unverified",
            "zero_tail_after_lb_payloads": precinct_tail,
            "geometry": {
                "image_width": EXPECTED_RAW_WIDTH,
                "half_width": EXPECTED_RAW_WIDTH // 2,
                "subband_group_counts": ng_by_sb,
                "coefficient_count": total_coefficients,
            },
        },
        "lbs": lb_summaries,
        "readers": bit_readers,
        "comparison": {
            "stages": {
                stage: {
                    **stage_totals[stage],
                    "match": stage_totals[stage]["difference_count"] == 0,
                }
                for stage in STAGES
            },
            "all_band_configuration_fields_match": all(
                all(band["configuration_matches_reference"].values()) for band in per_band
            ),
            "all_lb_lengths_and_offsets_match": all(
                all(field["match"] for field in item["reference_comparison"].values())
                for item in lb_summaries
            ),
        },
        "bands": per_band,
        "limitations": [
            "Only the first precinct is decoded; no later precinct predecessor state is implemented.",
            "GCLI and dequantization conventions are modeled with the supplied ACR-extracted tables and assisted by the cited public LibRaw source.",
            "The sb 25 mode 0x70 prediction interpretation is a public-source modeling choice, not verified Adobe Camera Raw behavior.",
            "The GTLI/WGT mapping and public implementation's W4 shift of 4 are not independently validated by an Adobe Camera Raw sample execution.",
            "Coefficient arrays are intermediate wavelet sub-band values and must not be treated as displayable CFA pixels.",
        ],
    }
    if arrays_path is not None:
        result["arrays_output"] = {
            "path_name": arrays_path.name,
            "format": "NPZ; one array for each (sub-band, stage) pair",
        }
    return result, band_arrays


def ensure_new_output_paths(
    sample: Path,
    tables: Path,
    reference: Path,
    output: Path | None,
    arrays: Path | None,
) -> None:
    sources = {path.resolve() for path in (sample, tables, reference)}
    destinations = [path for path in (output, arrays) if path is not None]
    resolved = [path.resolve() for path in destinations]
    if len(set(resolved)) != len(resolved):
        raise ProbeError("JSON output and NPZ arrays output must use different paths")
    if any(path in sources for path in resolved):
        raise ProbeError("output path must not refer to the sample, entropy tables, or reference JSON")
    for path in destinations:
        if path.exists():
            raise ProbeError(f"output already exists; refusing to overwrite: {path}")


def write_new(path: Path, payload: bytes) -> None:
    try:
        with path.open("xb") as destination:
            destination.write(payload)
    except FileExistsError as exc:
        raise ProbeError(f"output already exists; refusing to overwrite: {path}") from exc
    except OSError as exc:
        raise ProbeError(f"cannot write output {path}: {exc}") from exc


def encode_npz(arrays: dict[str, dict[str, list[int]]]) -> bytes:
    try:
        import numpy as np
    except ImportError as exc:
        raise ProbeError("--arrays requires NumPy in the selected Python environment") from exc
    output = io.BytesIO()
    npz_arrays: dict[str, Any] = {}
    for sb_text, stages in arrays.items():
        sb = int(sb_text)
        for stage, values in stages.items():
            dtype = np.uint8 if stage == "gcli" else np.int32
            npz_arrays[f"sb{sb:02d}_{stage}"] = np.asarray(values, dtype=dtype)
    np.savez_compressed(output, **npz_arrays)
    return output.getvalue()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample", type=Path, help="LCR_3520 Nikon NEF sample")
    parser.add_argument("--tables", type=Path, required=True, help="ACR-extracted entropy table JSON")
    parser.add_argument("--reference", type=Path, required=True, help="C++ first-precinct reference JSON")
    parser.add_argument("--output", type=Path, help="write comparison JSON to a new path; default stdout")
    parser.add_argument("--arrays", type=Path, help="optionally write decoded arrays to a new NPZ path")
    args = parser.parse_args(argv)

    try:
        ensure_new_output_paths(args.sample, args.tables, args.reference, args.output, args.arrays)
        result, arrays = decode_first_precinct(args.sample, args.tables, args.reference, args.arrays)
        comparison = result["comparison"]
        comparison["reference_orchestration_parity"] = (
            result["reference"]["orchestration_parity"] is True
        )
        comparison_passed = (
            all(stage["difference_count"] == 0 for stage in comparison["stages"].values())
            and comparison["all_band_configuration_fields_match"]
            and comparison["all_lb_lengths_and_offsets_match"]
            and comparison["reference_orchestration_parity"]
        )
        comparison["passed"] = comparison_passed
        serialized = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
        if args.arrays:
            write_new(args.arrays, encode_npz(arrays))
        if args.output:
            write_new(args.output, serialized.encode("utf-8"))
        else:
            sys.stdout.write(serialized)
    except (ProbeError, OSError, ValueError, KeyError, TypeError, struct.error) as exc:
        print(f"probe_ticoraw_first_precinct: {exc}", file=sys.stderr)
        return 2
    return 0 if comparison_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
