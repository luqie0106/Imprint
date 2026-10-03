"""Fit display-space RGB curves from paired images.

This module contains only numerical helpers. It does not read images or create
XMP files; :func:`apply_display_curves` is an in-memory curve approximation for
validation and is not a measurement of Adobe Camera Raw's rendering.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

import numpy as np


Curve: TypeAlias = tuple[tuple[int, int], ...]
RGB_Curves: TypeAlias = tuple[Curve, Curve, Curve]

_BIN_COUNT = 64
_GRID_X = np.concatenate(
    (np.arange(0, 256, 4, dtype=np.float64), np.array([255.0], dtype=np.float64))
)
_MAX_CURVE_POINTS = 16
_CURVE_MAX_ERROR = 0.75
_IDENTITY_CURVE: Curve = ((0, 0), (255, 255))


def _validate_display_rgb(image: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype != np.float32:
        raise TypeError(f"{name} must have dtype float32")
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"{name} must have shape HxWx3")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError(f"{name} values must be in [0, 1]")
    return array


def _weighted_isotonic(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return the weighted nondecreasing least-squares fit of ``values``."""
    fitted = np.empty_like(values, dtype=np.float64)
    # Each block stores [start, stop, total_weight, weighted_sum].
    blocks: list[list[float]] = []
    for index, (value, weight) in enumerate(zip(values, weights, strict=True)):
        blocks.append([float(index), float(index + 1), float(weight), float(weight * value)])
        while len(blocks) > 1:
            left, right = blocks[-2], blocks[-1]
            if left[3] / left[2] <= right[3] / right[2]:
                break
            blocks[-2:] = [[left[0], right[1], left[2] + right[2], left[3] + right[3]]]

    for start, stop, weight, weighted_sum in blocks:
        fitted[int(start) : int(stop)] = weighted_sum / weight
    return fitted


def _simplify_curve(x_values: np.ndarray, y_values: np.ndarray) -> Curve:
    """Reduce a monotonic sampled curve while bounding interpolation error."""
    if x_values.ndim != 1 or y_values.ndim != 1 or x_values.size != y_values.size:
        raise ValueError("curve samples must be one-dimensional arrays of equal length")
    if x_values.size < 2:
        raise ValueError("curve samples must contain at least two points")

    selected = {0, x_values.size - 1}
    while len(selected) < _MAX_CURVE_POINTS:
        ordered_indices = sorted(selected)
        largest_error = 0.0
        largest_error_index: int | None = None
        for left, right in zip(ordered_indices, ordered_indices[1:]):
            interior_indices = np.arange(left + 1, right)
            if interior_indices.size == 0:
                continue
            predicted = np.interp(
                x_values[interior_indices],
                (x_values[left], x_values[right]),
                (y_values[left], y_values[right]),
            )
            errors = np.abs(y_values[interior_indices] - predicted)
            local_position = int(np.argmax(errors))
            local_error = float(errors[local_position])
            if local_error > largest_error:
                largest_error = local_error
                largest_error_index = int(interior_indices[local_position])

        if largest_error_index is None or largest_error <= _CURVE_MAX_ERROR:
            break
        selected.add(largest_error_index)

    return tuple(
        (int(x_values[index]), int(y_values[index]))
        for index in sorted(selected)
    )


def _fit_one_channel_samples(
    source: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    source_values = source.astype(np.float64, copy=False).ravel() * 255.0
    target_values = target.astype(np.float64, copy=False).ravel() * 255.0
    bin_indices = np.minimum(
        (source_values * (_BIN_COUNT / 256.0)).astype(np.int64), _BIN_COUNT - 1
    )

    counts = np.bincount(bin_indices, minlength=_BIN_COUNT).astype(np.float64)
    occupied = np.flatnonzero(counts)
    if occupied.size == 0:
        return _GRID_X.copy(), _GRID_X.copy()

    source_sums = np.bincount(bin_indices, weights=source_values, minlength=_BIN_COUNT)
    target_sums = np.bincount(bin_indices, weights=target_values, minlength=_BIN_COUNT)
    observed_x = source_sums[occupied] / counts[occupied]
    observed_y = target_sums[occupied] / counts[occupied]

    # A narrow or sparsely populated input histogram cannot constrain a full
    # tone curve. Reduce its influence so unsupported regions remain near the
    # identity prior.
    value_span = float(np.ptp(source_values)) / 255.0
    occupancy = min(float(occupied.size) / 20.0, 1.0)
    spread = min(value_span / 0.45, 1.0)
    coverage = occupancy * spread
    if coverage <= 0.0:
        return _GRID_X.copy(), _GRID_X.copy()

    median_count = max(float(np.median(counts[occupied])), 1.0)
    relative_support = np.sqrt(counts[occupied] / median_count)
    local_support = (counts[occupied] / (counts[occupied] + 8.0)) * np.clip(
        relative_support, 0.35, 1.5
    )
    observed_weights = 12.0 * coverage * local_support

    # Identity knots act as a conservative prior, especially where the input
    # histogram has no observations. Observed bins gain weight with both local
    # sample support and coverage of the image's tonal range.
    all_x = np.concatenate((_GRID_X, observed_x))
    all_y = np.concatenate((_GRID_X, observed_y))
    all_weights = np.concatenate((np.ones_like(_GRID_X), observed_weights))
    order = np.argsort(all_x, kind="stable")
    ordered_x = all_x[order]
    ordered_y = all_y[order]
    ordered_weights = all_weights[order]
    monotonic_y = _weighted_isotonic(ordered_y, ordered_weights)

    knot_y = np.interp(_GRID_X, ordered_x, monotonic_y)
    knot_y = np.clip(np.maximum.accumulate(knot_y), 0.0, 255.0)
    knot_y[0] = 0.0
    knot_y[-1] = 255.0
    integer_y = np.maximum.accumulate(np.rint(knot_y).astype(np.int16))
    integer_y[0] = 0
    integer_y[-1] = 255
    return _GRID_X.copy(), integer_y


def _fit_one_channel(source: np.ndarray, target: np.ndarray) -> Curve:
    x_values, y_values = _fit_one_channel_samples(source, target)
    return _simplify_curve(x_values, y_values)


def fit_dehaze_curves(source: np.ndarray, target: np.ndarray) -> RGB_Curves:
    """Fit three monotonic display-sRGB curves from paired float32 images.

    ``source`` and ``target`` must be finite HxWx3 float32 arrays in [0, 1].
    The result contains one (input, output) curve for each RGB channel, with
    integer coordinates in [0, 255] and fixed black/white endpoints.
    """
    source_array = _validate_display_rgb(source, "source")
    target_array = _validate_display_rgb(target, "target")
    if source_array.shape != target_array.shape:
        raise ValueError("source and target must have the same shape")
    if np.array_equal(source_array, target_array):
        return (_IDENTITY_CURVE, _IDENTITY_CURVE, _IDENTITY_CURVE)

    return tuple(
        _fit_one_channel(source_array[..., channel], target_array[..., channel])
        for channel in range(3)
    )  # type: ignore[return-value]


def _validate_curves(
    curves: Sequence[Sequence[tuple[int, int]]],
) -> tuple[Curve, Curve, Curve]:
    if isinstance(curves, (str, bytes)) or len(curves) != 3:
        raise ValueError("curves must contain exactly three RGB curves")

    validated: list[Curve] = []
    for channel_index, curve in enumerate(curves):
        points: list[tuple[int, int]] = []
        try:
            raw_points = tuple(curve)
        except TypeError as exc:
            raise ValueError(f"curve {channel_index} must be a sequence of points") from exc
        if len(raw_points) < 2:
            raise ValueError(f"curve {channel_index} must contain at least two points")
        for point in raw_points:
            if not isinstance(point, Sequence) or len(point) != 2:
                raise ValueError(f"curve {channel_index} points must be (x, y) pairs")
            x, y = point
            if (
                isinstance(x, bool)
                or isinstance(y, bool)
                or not isinstance(x, (int, np.integer))
                or not isinstance(y, (int, np.integer))
            ):
                raise TypeError("curve coordinates must be integers")
            x, y = int(x), int(y)
            if not (0 <= x <= 255 and 0 <= y <= 255):
                raise ValueError("curve coordinates must be in [0, 255]")
            points.append((x, y))
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        if xs[0] != 0 or xs[-1] != 255:
            raise ValueError("curves must include input endpoints 0 and 255")
        if ys[0] != 0 or ys[-1] != 255:
            raise ValueError("curves must have fixed output endpoints 0 and 255")
        if any(right <= left for left, right in zip(xs, xs[1:])):
            raise ValueError("curve input coordinates must be strictly increasing")
        if any(right < left for left, right in zip(ys, ys[1:])):
            raise ValueError("curve output coordinates must be nondecreasing")
        validated.append(tuple(points))
    return (validated[0], validated[1], validated[2])


def apply_display_curves(
    image: np.ndarray,
    curves: Sequence[Sequence[tuple[int, int]]],
) -> np.ndarray:
    """Apply the RGB curves with linear interpolation in display space.

    This helper approximates a curve lookup on pixel values for numerical
    validation. It does not reproduce Adobe Camera Raw's XMP interpretation.
    """
    image_array = _validate_display_rgb(image, "image")
    validated_curves = _validate_curves(curves)
    result = np.empty_like(image_array)
    for channel, curve in enumerate(validated_curves):
        x_values = np.fromiter((point[0] for point in curve), dtype=np.float64)
        y_values = np.fromiter((point[1] for point in curve), dtype=np.float64)
        result[..., channel] = (
            np.interp(image_array[..., channel].astype(np.float64) * 255.0, x_values, y_values)
            / 255.0
        ).astype(np.float32)
    return result


__all__ = ["Curve", "RGB_Curves", "apply_display_curves", "fit_dehaze_curves"]
