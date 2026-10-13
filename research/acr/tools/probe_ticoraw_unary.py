#!/usr/bin/env python3
"""Synthetic-only probe for an MSB-first unary bit reader.

This exercises the bounded bit-reading hypothesis recorded for FUN_013e5828.
It does not read photographs, Adobe files, or Ghidra project data and is not a
TicoRAW entropy decoder.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Iterable


MAX_STREAM_BYTES = 4096
MAX_ALLOWED_RUN = 4096
RESEARCH_MAX_RUN = 63
RANDOM_SEED = 0xAC20261012
RANDOM_SEQUENCE_COUNT = 128


class ProbeError(ValueError):
    """Invalid synthetic probe input or output request."""


class TruncatedBitstream(ProbeError):
    """The requested codeword ended before its zero terminator."""


class UnaryRunTooLong(ProbeError):
    """A unary run exceeded this research probe's configured safety limit."""


class MSBBitReader:
    """Read a bounded bit prefix, supplementing the reservoir in BE 32-bit words."""

    def __init__(self, data: bytes, bit_length: int | None = None):
        if not isinstance(data, bytes):
            raise ProbeError("data must be bytes")
        if len(data) > MAX_STREAM_BYTES:
            raise ProbeError(f"stream exceeds the {MAX_STREAM_BYTES}-byte safety limit")
        if bit_length is None:
            bit_length = len(data) * 8
        if not isinstance(bit_length, int) or bit_length < 0:
            raise ProbeError("bit_length must be a non-negative integer")
        if bit_length > len(data) * 8:
            raise ProbeError("bit_length exceeds the available bytes")

        self._data = data
        self._byte_offset = 0
        self._reservoir = 0
        self._bits_in_reservoir = 0
        self._remaining_bits = bit_length
        self.position = 0

    def _refill(self) -> None:
        if self._remaining_bits <= 0:
            raise TruncatedBitstream("bitstream ended before a unary zero terminator")

        valid_bits = min(32, self._remaining_bits)
        byte_count = (valid_bits + 7) // 8
        chunk = self._data[self._byte_offset:self._byte_offset + byte_count]
        if len(chunk) != byte_count:
            raise TruncatedBitstream("bitstream ended while supplementing the reservoir")

        # Load at most one big-endian 32-bit word and left-align short final
        # supplements so the next consumed bit is always the most-significant bit.
        word = int.from_bytes(chunk, byteorder="big") << (32 - 8 * len(chunk))
        # Keep only the valid prefix, right-aligned for the countdown reader.
        self._reservoir = word >> (32 - valid_bits)
        self._bits_in_reservoir = valid_bits
        self._byte_offset += byte_count

    def read_bit(self) -> int:
        if self._bits_in_reservoir == 0:
            self._refill()
        bit = (self._reservoir >> (self._bits_in_reservoir - 1)) & 1
        self._bits_in_reservoir -= 1
        self._remaining_bits -= 1
        self.position += 1
        return bit


def read_unary_ones(reader: MSBBitReader, max_run: int = RESEARCH_MAX_RUN) -> int:
    """Return the number of consecutive MSB-first ones before the first zero."""
    if not isinstance(max_run, int) or max_run < 0 or max_run > MAX_ALLOWED_RUN:
        raise ProbeError(f"max_run must be between 0 and {MAX_ALLOWED_RUN}")

    run = 0
    while True:
        bit = reader.read_bit()
        if bit == 0:
            return run
        run += 1
        if run > max_run:
            raise UnaryRunTooLong(
                f"unary run exceeded research max_run={max_run}"
            )


def encode_unary(values: Iterable[int]) -> str:
    bits = []
    for value in values:
        if not isinstance(value, int) or value < 0 or value > MAX_ALLOWED_RUN:
            raise ProbeError(f"unary value must be between 0 and {MAX_ALLOWED_RUN}")
        bits.append("1" * value + "0")
    return "".join(bits)


def pack_msb_bits(bits: str) -> tuple[bytes, int]:
    if any(bit not in "01" for bit in bits):
        raise ProbeError("bit string may contain only '0' and '1'")
    bit_length = len(bits)
    padded = bits + "0" * ((-bit_length) % 8)
    data = bytes(
        int(padded[offset:offset + 8], 2)
        for offset in range(0, len(padded), 8)
    )
    return data, bit_length


def decode_count(data: bytes, bit_length: int, value_count: int) -> tuple[list[int], int]:
    reader = MSBBitReader(data, bit_length)
    values = [read_unary_ones(reader) for _ in range(value_count)]
    return values, reader.position


def run_probe() -> dict[str, object]:
    fixed_vectors = [("0", 0), ("10", 1), ("110", 2), ("1110", 3)]
    fixed_results = []
    for bits, expected in fixed_vectors:
        data, bit_length = pack_msb_bits(bits)
        decoded, consumed = decode_count(data, bit_length, 1)
        if decoded != [expected] or consumed != expected + 1:
            raise AssertionError(f"fixed vector failed: {bits}")
        fixed_results.append({"bits": bits, "expected": expected, "decoded": decoded[0], "consumed_bits": consumed, "passed": True})

    non_byte_values = [2, 1, 0, 4, 3]
    non_byte_bits = encode_unary(non_byte_values)
    non_byte_data, non_byte_length = pack_msb_bits(non_byte_bits)
    non_byte_decoded, non_byte_consumed = decode_count(
        non_byte_data, non_byte_length, len(non_byte_values)
    )
    non_byte_expected_consumed = sum(value + 1 for value in non_byte_values)
    if (
        non_byte_length % 8 == 0
        or non_byte_decoded != non_byte_values
        or non_byte_consumed != non_byte_expected_consumed
    ):
        raise AssertionError("non-byte-boundary sequence failed")

    word_crossing_cases = [
        {"name": "terminator_after_32_ones", "values": [32]},
        {"name": "codeword_starts_at_bit_31", "values": [0] * 31 + [4]},
    ]
    word_crossing_results = []
    for case in word_crossing_cases:
        values = case["values"]
        assert isinstance(values, list)
        bits = encode_unary(values)
        data, bit_length = pack_msb_bits(bits)
        decoded, consumed = decode_count(data, bit_length, len(values))
        expected_consumed = sum(value + 1 for value in values)
        if decoded != values or consumed != expected_consumed:
            raise AssertionError(f"32-bit boundary case failed: {case['name']}")
        word_crossing_results.append(
            {
                "name": case["name"],
                "bit_length": bit_length,
                "decoded_values": len(decoded),
                "consumed_bits": consumed,
                "expected_consumed_bits": expected_consumed,
                "passed": True,
            }
        )

    rng = random.Random(RANDOM_SEED)
    random_values = [rng.randrange(64) for _ in range(RANDOM_SEQUENCE_COUNT)]
    random_bits = encode_unary(random_values)
    random_data, random_bit_length = pack_msb_bits(random_bits)
    random_decoded, random_consumed = decode_count(
        random_data, random_bit_length, len(random_values)
    )
    random_expected_consumed = sum(value + 1 for value in random_values)
    if (
        random_decoded != random_values
        or random_consumed != random_expected_consumed
        or random_consumed != random_bit_length
    ):
        raise AssertionError("deterministic random sequence failed")

    truncated_data, truncated_length = pack_msb_bits("111")
    try:
        read_unary_ones(MSBBitReader(truncated_data, truncated_length))
    except TruncatedBitstream:
        truncated_passed = True
    else:
        raise AssertionError("unterminated unary codeword was not rejected")

    over_limit_bits = "1" * (RESEARCH_MAX_RUN + 1) + "0"
    over_limit_data, over_limit_length = pack_msb_bits(over_limit_bits)
    try:
        read_unary_ones(
            MSBBitReader(over_limit_data, over_limit_length),
            max_run=RESEARCH_MAX_RUN,
        )
    except UnaryRunTooLong:
        max_run_passed = True
    else:
        raise AssertionError("unary run above research max_run was not rejected")

    return {
        "probe": "ticoraw_msb_unary_synthetic_v1",
        "status": "synthetic_only",
        "source_hypothesis": "MSB-first unary interpretation supplied for FUN_013e5828; verified here only against synthetic vectors.",
        "codebook": {
            "bit_order": "MSB-first",
            "rule": "count consecutive 1 bits through the first 0; the value is that count and the codeword length is value + 1",
            "fixed_vectors": [
                {"bits": bits, "value": value}
                for bits, value in fixed_vectors
            ],
        },
        "reader": {
            "reservoir_supplement": "up to four bytes are assembled as a big-endian word; only the valid MSB-first prefix is exposed, including a partial final chunk",
            "max_stream_bytes": MAX_STREAM_BYTES,
            "max_run": RESEARCH_MAX_RUN,
            "max_run_scope": "synthetic research safety limit only; no claim that Adobe/ACR uses this limit",
        },
        "cases": {
            "fixed_vectors": {
                "count": len(fixed_results),
                "passed": all(item["passed"] for item in fixed_results),
                "results": fixed_results,
            },
            "non_byte_boundary": {
                "count": 1,
                "values": len(non_byte_values),
                "bit_length": non_byte_length,
                "consumed_bits": non_byte_consumed,
                "expected_consumed_bits": non_byte_expected_consumed,
                "passed": True,
            },
            "32bit_boundary": {
                "count": len(word_crossing_results),
                "passed": all(item["passed"] for item in word_crossing_results),
                "results": word_crossing_results,
            },
            "deterministic_random": {
                "seed": RANDOM_SEED,
                "sequence_count": RANDOM_SEQUENCE_COUNT,
                "value_range": [0, 63],
                "bit_length": random_bit_length,
                "consumed_bits": random_consumed,
                "expected_consumed_bits": random_expected_consumed,
                "passed": True,
            },
            "truncated_missing_terminator": {
                "count": 1,
                "rejected": truncated_passed,
                "exception": "TruncatedBitstream",
                "passed": truncated_passed,
            },
            "max_run_guard": {
                "count": 1,
                "configured_max_run": RESEARCH_MAX_RUN,
                "rejected_run_length": RESEARCH_MAX_RUN + 1,
                "exception": "UnaryRunTooLong",
                "passed": max_run_passed,
            },
        },
        "case_count": len(fixed_results) + 1 + len(word_crossing_results) + 1 + 1 + 1,
        "limitations": [
            "Synthetic bitstrings only; no camera sample entropy decoding was run.",
            "The MSB-first unary behavior is the supplied static interpretation of FUN_013e5828, not an independently validated full codec behavior.",
            "This is not Adobe pseudocode translation, a full decoder, or evidence of successful real CFA decoding.",
            "The 63-bit max_run is a probe safety setting and is not attributed to Adobe/ACR.",
            "Big-endian 32-bit supplementation is exercised by synthetic boundary cases only.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="write the synthetic JSON summary to a new file (existing files are never overwritten)",
    )
    args = parser.parse_args(argv)

    summary = json.dumps(run_probe(), indent=2, sort_keys=False) + "\n"
    if args.output is not None:
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as destination:
                destination.write(summary)
        except FileExistsError:
            print(f"error: refusing to overwrite existing output: {args.output}", file=sys.stderr)
            return 2
        except OSError as exc:
            print(f"error: cannot write output {args.output}: {exc}", file=sys.stderr)
            return 2
    sys.stdout.write(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
