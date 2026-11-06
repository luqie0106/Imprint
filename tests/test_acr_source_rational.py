"""Focused tests for Source's finite-rational scale helper and C bridge."""

from __future__ import annotations

import ctypes as C
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "native-renderer"
BRIDGE_SOURCE = NATIVE / "tools" / "acr_source_rational_bridge.cpp"
CORE_SOURCE = NATIVE / "src" / "acr_source_rational.cpp"
CORE_HEADER = NATIVE / "src" / "acr_source_rational.hpp"

OK = 0
NON_FINITE_FACTOR = 1
NUMERIC_OVERFLOW = 2
BRIDGE_INVALID_OUTPUT = 3
UINT32_MAX = 0xFFFFFFFF
OUTPUT_SENTINEL = (0xA5A5A5A5, 0x5A5A5A5A)


class Pair(C.Structure):
    _fields_ = [
        ("numerator", C.c_uint32),
        ("denominator", C.c_uint32),
    ]


class Bridge:
    def __init__(self, path: Path):
        self.library = C.CDLL(str(path))
        self.function = self.library.acr_source_rational_bridge
        self.function.argtypes = [
            C.c_uint32,
            C.c_uint32,
            C.c_double,
            C.POINTER(C.c_uint32),
        ]
        self.function.restype = C.c_int

    def call(self, numerator: int, denominator: int, factor: float,
             initial=OUTPUT_SENTINEL):
        output = Pair(*initial)
        output_pointer = C.cast(C.byref(output), C.POINTER(C.c_uint32))
        status = self.function(numerator, denominator, factor, output_pointer)
        return status, (output.numerator, output.denominator), output


def _compiler() -> str:
    compiler = shutil.which("clang++") or shutil.which("c++")
    assert compiler, "clang++ or c++ is required for the focused bridge"
    return compiler


def _strict_fp_flags() -> list[str]:
    return ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
            "-fno-fast-math", "-ffp-contract=off"]


@pytest.fixture(scope="module")
def bridge(tmp_path_factory):
    assert BRIDGE_SOURCE.is_file()
    assert CORE_SOURCE.is_file()
    assert CORE_HEADER.is_file()

    if sys.platform == "darwin":
        suffix, shared_flags = ".dylib", ["-dynamiclib"]
    elif sys.platform == "win32":
        suffix, shared_flags = ".dll", ["-shared"]
    else:
        suffix, shared_flags = ".so", ["-shared", "-fPIC"]

    output = tmp_path_factory.mktemp("acr_source_rational") / f"libsource_rational{suffix}"
    command = [
        _compiler(),
        *_strict_fp_flags(),
        *shared_flags,
        "-I",
        str(NATIVE / "src"),
        str(BRIDGE_SOURCE),
        str(CORE_SOURCE),
        "-o",
        str(output),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return Bridge(output)


# Constants transcribed from the saved instruction-reference boundary probes;
# tests never load research or temporary reference artifacts at runtime.
REFERENCE_GOLDENS = [
    ((14, 21), 3.0, 0, (14, 7)),
    ((21, 35), 1.0, 0, (21, 35)),
    ((1, 1), 32767.75, 0, (131071, 4)),
    ((1, 1), 32768.25, 0, (32768, 1)),
    ((1, 1), 32768.5, 0, (32769, 1)),
    ((1, 1), 1.2, 0, (19661, 16384)),
    ((10, 3), 1.2, 0, (4, 1)),
    ((0, 1), 1e-30, 1, None),
    ((1, UINT32_MAX), 0.5, 0, (0, 1)),
    ((UINT32_MAX, UINT32_MAX), 2.0, 0, (2, 1)),
    ((UINT32_MAX, 1), 2.0, 1, None),
]


@pytest.mark.parametrize("input_pair,factor,reference_status,expected_pair",
                         REFERENCE_GOLDENS)
def test_instruction_reference_goldens(bridge, input_pair, factor,
                                       reference_status, expected_pair):
    status, output, _ = bridge.call(*input_pair, factor)
    expected_status = OK if reference_status == 0 else NUMERIC_OVERFLOW
    assert status == expected_status
    if reference_status == 0:
        assert output == expected_pair
    else:
        # The original helper's error edge does not commit a result pair.
        assert output == OUTPUT_SENTINEL


def test_exact_fast_paths_and_limited_pre_reduction(bridge):
    assert bridge.call(1, 2, 0.5)[:2] == (OK, (1, 4))
    assert bridge.call(2, 5, 0.5)[:2] == (OK, (1, 5))
    assert bridge.call(1, 6, 3.0)[:2] == (OK, (1, 2))
    # The helper cancels only the observed 2/3/5 factors; it is not a GCD.
    assert bridge.call(21, 35, 1.0)[:2] == (OK, (21, 35))


def test_integer_product_overflow_uses_fractional_fallback(bridge):
    # num*2 overflows uint32, but (num/7)*2 fits. The raw falls back to the
    # fixed-denominator floating route instead of taking its error edge.
    assert bridge.call(UINT32_MAX, 7, 2.0)[:2] == (OK, (1227133513, 1))


@pytest.mark.parametrize("factor", [0.0, -0.0, -0.25, -3.0])
def test_nonpositive_factor_resets_even_zero_denominator(bridge, factor):
    assert bridge.call(UINT32_MAX, 0, factor)[:2] == (OK, (0, 1))


def test_positive_factor_preserves_zero_denominator_pair(bridge):
    pair = (UINT32_MAX, 0)
    assert bridge.call(*pair, 1.2)[:2] == (OK, pair)
    # The denominator-zero return precedes even the factor-to-u32 overflow guard.
    assert bridge.call(*pair, 1e300)[:2] == (OK, pair)


@pytest.mark.parametrize("factor", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_factor_is_out_of_domain_and_leaves_output_unchanged(bridge,
                                                                       factor):
    status, output, _ = bridge.call(14, 21, factor)
    assert status == NON_FINITE_FACTOR
    assert output == OUTPUT_SENTINEL


def test_numeric_overflow_leaves_output_unchanged(bridge):
    cases = [
        (0, 1, 1e-30),  # Must not early-return just because numerator is zero.
        (UINT32_MAX, 1, 2.0),
    ]
    for numerator, denominator, factor in cases:
        status, output, _ = bridge.call(numerator, denominator, factor)
        assert status == NUMERIC_OVERFLOW
        assert output == OUTPUT_SENTINEL


def test_null_and_misaligned_outputs_are_rejected_without_writes(bridge):
    assert bridge.function(1, 1, 2.0, None) == BRIDGE_INVALID_OUTPUT

    storage = (C.c_ubyte * 16)(*([0xA5] * 16))
    before = bytes(storage)
    misaligned = C.cast(C.addressof(storage) + 1, C.POINTER(C.c_uint32))
    assert bridge.function(1, 1, 2.0, misaligned) == BRIDGE_INVALID_OUTPUT
    assert bytes(storage) == before


def test_core_supports_input_output_alias_and_atomic_failure(tmp_path):
    harness = tmp_path / "source_rational_core_check.cpp"
    harness.write_text(
        r'''#include "acr_source_rational.hpp"
#include <cstdint>

int main() {
    using imprint::acr::SourceRational;
    using imprint::acr::SourceRationalStatus;
    SourceRational in_place{14U, 21U};
    if (imprint::acr::scale_source_rational(in_place, 3.0, in_place) !=
            SourceRationalStatus::ok ||
        in_place.numerator != 14U || in_place.denominator != 7U) return 1;

    SourceRational output{0xA5A5A5A5U, 0x5A5A5A5AU};
    const SourceRational overflow{0xFFFFFFFFU, 1U};
    if (imprint::acr::scale_source_rational(overflow, 2.0, output) !=
            SourceRationalStatus::numeric_overflow ||
        output.numerator != 0xA5A5A5A5U ||
        output.denominator != 0x5A5A5A5AU) return 2;

    SourceRational zero{0U, 1U};
    if (imprint::acr::scale_source_rational(zero, 1e-30, zero) !=
            SourceRationalStatus::numeric_overflow ||
        zero.numerator != 0U || zero.denominator != 1U) return 3;
    return 0;
}
''',
        encoding="utf-8",
    )
    executable = tmp_path / "source_rational_core_check"
    subprocess.run(
        [
            _compiler(), *_strict_fp_flags(), "-I", str(NATIVE / "src"),
            str(harness), str(CORE_SOURCE), "-o", str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(executable)], check=True, capture_output=True, text=True)
