"""Independent oracle for the ordinary sampler logical source rectangle.

The model follows the frozen ``ordinary-reverse-source-rect`` findings and
bounded ARM64 instruction windows.  It deliberately enumerates all four
corners of each half-open rectangle instead of reducing the affine extrema to
two endpoints.  It models coordinates only; it does not model pixel reads,
halo allocation, or rendered values.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Mapping


ORACLE_VERSION = "ordinary-source-rect-four-corner-v1"

STATUS_OK = 0
STATUS_INVALID_RECTANGLE = 1
STATUS_INVALID_AXIS = 2
STATUS_COORDINATE_OVERFLOW = 3

KIND_RESAMPLE = 0
KIND_IDENTITY = 1
KIND_INTEGER_TRANSLATION = 2

INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1
INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1
MASK32 = (1 << 32) - 1
MASK64 = (1 << 64) - 1


@dataclass(frozen=True)
class Rect:
    top: int
    left: int
    height: int
    width: int

    @property
    def bottom(self) -> int:
        return self.top + self.height

    @property
    def right(self) -> int:
        return self.left + self.width


def _signed(value: int, bits: int) -> int:
    mask = (1 << bits) - 1
    value &= mask
    sign = 1 << (bits - 1)
    return value - (1 << bits) if value & sign else value


def _valid_rect(rect: Rect) -> bool:
    if not (INT32_MIN <= rect.top <= INT32_MAX and
            INT32_MIN <= rect.left <= INT32_MAX):
        return False
    if not (1 <= rect.height <= INT32_MAX and 1 <= rect.width <= INT32_MAX):
        return False
    return (INT32_MIN <= rect.bottom <= INT32_MAX and
            INT32_MIN <= rect.right <= INT32_MAX)


def _mapped_high32(axis: Mapping[str, int], coordinate: int) -> int:
    # The prepared ordinary axis has one along-axis increment and no cross
    # term.  Keep signed 64-bit wrap explicit before the signed clamp.
    fixed = (axis["origin"] + axis["step"] * coordinate) & MASK64
    fixed = _signed(fixed, 64)
    fixed = max(axis["lower"], min(fixed, axis["upper"]))
    # Python's arithmetic right shift is floor division for negative values,
    # matching extraction of the signed high word from Q32.32.
    return fixed >> 32


def _mapped_lane(rect: Rect, axis: Mapping[str, int], *, vertical: bool
                 ) -> tuple[int, int] | None:
    kind = axis["kind"]
    start = rect.top if vertical else rect.left
    end = rect.bottom if vertical else rect.right

    if kind == KIND_IDENTITY:
        return start, end
    if kind == KIND_INTEGER_TRANSLATION:
        mapped_start = start - axis["integer_offset"]
        mapped_end = end - axis["integer_offset"]
        if not (INT32_MIN <= mapped_start <= INT32_MAX and
                INT32_MIN <= mapped_end <= INT32_MAX):
            return None
        return mapped_start, mapped_end

    # Enumerate the four corners as represented by the reverse-rectangle
    # methods.  This remains a four-corner model even though ordinary plans
    # store only the along-axis step and therefore have no skew term.
    other_start = rect.left if vertical else rect.top
    other_end = rect.right if vertical else rect.bottom
    along_end = end - 1
    other_end -= 1
    along_coords = (start, along_end)
    other_coords = (other_start, other_end)
    corner_results = []
    for along, _other in product(along_coords, other_coords):
        corner_results.append(_mapped_high32(axis, along))

    low = min(corner_results)
    high = max(corner_results)
    # The recovered methods add the full coefficient tap count with W32
    # semantics after selecting the inclusive high coordinate.
    mapped_end = _signed((high + axis["tap_count"]) & MASK32, 32)
    return low, mapped_end


def _replace_lane(rect: Rect, lane: tuple[int, int] | None, *, vertical: bool
                  ) -> tuple[int, Rect | None]:
    if lane is None:
        return STATUS_COORDINATE_OVERFLOW, None
    start, end = lane
    extent = end - start
    if not (1 <= extent <= INT32_MAX):
        return STATUS_INVALID_RECTANGLE, None
    if vertical:
        return STATUS_OK, Rect(start, rect.left, extent, rect.width)
    return STATUS_OK, Rect(rect.top, start, rect.height, extent)


def _valid_axis(axis: Mapping[str, int]) -> bool:
    kind = axis["kind"]
    if kind not in (KIND_RESAMPLE, KIND_IDENTITY, KIND_INTEGER_TRANSLATION):
        return False
    if kind != KIND_RESAMPLE:
        return True
    return (0 < axis["step"] <= INT64_MAX and
            INT64_MIN <= axis["lower"] <= axis["upper"] <= INT64_MAX and
            0 < axis["tap_count"] <= INT32_MAX and axis["tap_count"] % 2 == 0)


def prepare_source_request(
    destination: Rect,
    vertical: Mapping[str, int],
    horizontal: Mapping[str, int],
) -> tuple[int, Rect | None, Rect | None]:
    """Return status, source rectangle, and intermediate rectangle.

    Reverse propagation applies horizontal first and vertical second.  A
    failed request has no rectangles, which lets the caller verify that the
    C++ API leaves its output object byte-for-byte untouched.
    """
    if not _valid_rect(destination):
        return STATUS_INVALID_RECTANGLE, None, None
    if not _valid_axis(horizontal):
        return STATUS_INVALID_AXIS, None, None

    h_status, intermediate = _replace_lane(
        destination,
        _mapped_lane(destination, horizontal, vertical=False),
        vertical=False,
    )
    if h_status != STATUS_OK:
        return h_status, None, None
    assert intermediate is not None

    if not _valid_axis(vertical):
        return STATUS_INVALID_AXIS, None, None
    v_status, source = _replace_lane(
        intermediate,
        _mapped_lane(intermediate, vertical, vertical=True),
        vertical=True,
    )
    if v_status != STATUS_OK:
        return v_status, None, None
    assert source is not None
    return STATUS_OK, source, intermediate

