"""Independent scalar composition oracle for PGTM RGB image processing.

This module composes the existing independently checked references for PGTM
payload decoding, cache generation, map preparation, coordinate generation,
gain lookup, and pivot-zero RGB application.  It contains no product imports
and does not infer map selection or image geometry from metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

import profile_gain_apply_oracle as apply_oracle
import profile_gain_cache_oracle as cache_oracle
import profile_gain_coordinates_oracle as coordinates_oracle
import profile_gain_map_oracle as map_oracle
import profile_gain_oracle as gain_oracle


ORACLE_VERSION = "acr-pgtm-rgb-image-composition-2026-10-27-v2"
SCALAR_PIVOT = 0.0
SCALAR_MODE = apply_oracle.SCALAR_MODE
NO_CLIP = 0


@dataclass(frozen=True)
class PreparedImageReference:
    """Independent intermediate state used by the image oracle."""

    prepared_map: map_oracle.PreparedMap
    coordinates: tuple[np.ndarray, np.ndarray]
    cache_exponent: np.float32 | None
    negative_remap: bool | None


def prepare_reference(
    payload: bytes | bytearray | memoryview,
    tag: int,
    byteorder: int,
    amount: int,
    lookup_flag: int,
    predicate: int,
    rects: Sequence[int],
    matrix: Sequence[float],
    scalar: float,
) -> PreparedImageReference:
    """Compose the independently implemented preparation references.

    ``rects`` is exactly ``map, reference-zero, state, tile`` with TLBR lanes.
    Only the first nine row-major matrix cells are consumed by the recovered
    coordinate constructor.  The caller supplies every selection and rect.
    """
    if len(rects) != 16 or len(matrix) != 9:
        raise ValueError("expected four TLBR rectangles and nine matrix values")
    map_rect = tuple(int(value) for value in rects[0:4])
    zero_rect = tuple(int(value) for value in rects[4:8])
    state_rect = tuple(int(value) for value in rects[8:12])
    tile_rect = tuple(int(value) for value in rects[12:16])

    parsed = map_oracle.parse_payload(payload, tag, byteorder)
    prepared_map = map_oracle.prepare_map(parsed, map_rect, scalar)
    if amount == 100 and (predicate & 1) == 0:
        # The recovered route installs the parsed source pair directly here;
        # lookup_flag does not participate in this dispatch.
        exponent = None
        negative_remap = None
    else:
        cached_values, exponent, negative_remap = cache_oracle.expected_cache_values(
            parsed.table, amount, lookup_flag, predicate
        )
        prepared_map = replace(
            prepared_map, table=np.asarray(cached_values, dtype=np.float32)
        )

    coordinate_status, coefficients = coordinates_oracle.reference_prepare(
        zero_rect, state_rect, matrix
    )
    if coordinate_status != 0 or coefficients is None:
        raise ValueError(f"coordinate reference preparation failed with status {coordinate_status}")
    height = tile_rect[2] - tile_rect[0]
    width = tile_rect[3] - tile_rect[1]
    if width <= 0 or height <= 0:
        raise ValueError("tile rectangle must have positive dimensions")
    v_plane, u_plane = coordinates_oracle.reference_planes(
        coefficients, tile_rect, row_stride=width
    )
    return PreparedImageReference(
        prepared_map=prepared_map,
        coordinates=(v_plane, u_plane),
        cache_exponent=exponent,
        negative_remap=negative_remap,
    )


def reference_image(
    payload: bytes | bytearray | memoryview,
    tag: int,
    byteorder: int,
    amount: int,
    lookup_flag: int,
    predicate: int,
    rects: Sequence[int],
    matrix: Sequence[float],
    scalar: float,
    source: np.ndarray | Sequence[float],
    source_stride: int,
    destination_stride: int,
    *,
    destination_count: int | None = None,
    sentinel: float = -731.625,
) -> tuple[np.ndarray, PreparedImageReference]:
    """Return row-padded HWC output for a finite, valid float32 source image.

    Strides and counts are measured in float elements.  The per-pixel order is
    profile-gain table lookup followed by the independently transcribed
    pivot-zero, unclipped scalar channel transform.
    """
    if len(rects) != 16:
        raise ValueError("expected exactly sixteen rectangle lanes")
    tile_top, tile_left, tile_bottom, tile_right = (int(x) for x in rects[12:16])
    height = tile_bottom - tile_top
    width = tile_right - tile_left
    active_row = width * 3
    if width <= 0 or height <= 0:
        raise ValueError("tile rectangle must have positive dimensions")
    if source_stride < active_row or destination_stride < active_row:
        raise ValueError("row strides must cover active HWC RGB pixels")

    source_values = np.asarray(source, dtype=np.float32).reshape(-1)
    source_required = (height - 1) * source_stride + active_row
    output_required = (height - 1) * destination_stride + active_row
    if source_values.size < source_required:
        raise ValueError("source buffer is shorter than the active image span")
    output_size = output_required if destination_count is None else int(destination_count)
    if output_size < output_required:
        raise ValueError("destination buffer is shorter than the active image span")

    prepared = prepare_reference(
        payload,
        tag,
        byteorder,
        amount,
        lookup_flag,
        predicate,
        rects,
        matrix,
        scalar,
    )
    v_plane, u_plane = prepared.coordinates
    gain_map = prepared.prepared_map
    output = np.full(output_size, np.float32(sentinel), dtype=np.float32)

    for row in range(height):
        for column in range(width):
            pixel = row * source_stride + 3 * column
            coordinate = row * width + column
            gain = gain_oracle.reference_sample(
                gain_map.fields[:16],
                gain_map.geometry[:5],
                gain_map.table,
                source_values[pixel],
                source_values[pixel + 1],
                source_values[pixel + 2],
                v_plane[coordinate],
                u_plane[coordinate],
            )
            target = row * destination_stride + 3 * column
            for channel in range(3):
                output[target + channel] = apply_oracle.reference_value(
                    source_values[pixel + channel],
                    gain,
                    SCALAR_PIVOT,
                    SCALAR_MODE,
                    NO_CLIP,
                )

    return output, prepared

