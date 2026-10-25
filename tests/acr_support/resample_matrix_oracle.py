"""Independent binary64 oracle for the ordinary ACR affine matrix factory.

The formulas and operation order are transcribed from the retained offline
ARM64 evidence.  This module does not import or inspect the native source.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import math
from enum import IntEnum
from typing import Sequence


ORACLE_VERSION = "acr-ordinary-resample-matrix-fma64-2026-10-23-v1"
DETERMINANT_THRESHOLD = 1.0e-10


class ResampleMatrixStatus(IntEnum):
    OK = 0
    NON_FINITE_INPUT = 1
    SINGULAR_MATRIX = 2
    NON_FINITE_RESULT = 3


def _load_fma():
    candidates = []
    math_library = ctypes.util.find_library("m")
    if math_library:
        candidates.append(math_library)
    candidates.append(None)
    for name in candidates:
        try:
            library = ctypes.CDLL(name)
            function = library.fma
            function.argtypes = [ctypes.c_double, ctypes.c_double,
                                 ctypes.c_double]
            function.restype = ctypes.c_double
            return function
        except (AttributeError, OSError):
            continue
    raise RuntimeError("the platform C math library does not expose fma")


_fma = _load_fma()


def _finite(values: Sequence[float]) -> bool:
    return all(math.isfinite(value) for value in values)


def _negprod(left: float, right: float) -> float:
    # The recovered FNMUL rounds the product before changing its sign.
    return -(left * right)


def _cofactor(positive_left: float, positive_right: float,
              negative_left: float, negative_right: float
              ) -> tuple[ResampleMatrixStatus, float | None]:
    negative_product = _negprod(negative_left, negative_right)
    if not math.isfinite(negative_product):
        return ResampleMatrixStatus.NON_FINITE_RESULT, None
    result = _fma(positive_left, positive_right, negative_product)
    if not math.isfinite(result):
        return ResampleMatrixStatus.NON_FINITE_RESULT, None
    return ResampleMatrixStatus.OK, result


def _invert3(matrix: tuple[float, ...]) -> tuple[ResampleMatrixStatus,
                                                  tuple[float, ...] | None]:
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = matrix

    status, c00 = _cofactor(m11, m22, m12, m21)
    if status != ResampleMatrixStatus.OK:
        return status, None
    status, c10 = _cofactor(m12, m20, m10, m22)
    if status != ResampleMatrixStatus.OK:
        return status, None
    status, c20 = _cofactor(m10, m21, m20, m11)
    if status != ResampleMatrixStatus.OK:
        return status, None
    assert c00 is not None and c10 is not None and c20 is not None

    # Preserve the separate multiply and left-to-right fused determinant sum.
    accumulator = m01 * c10
    if not math.isfinite(accumulator):
        return ResampleMatrixStatus.NON_FINITE_RESULT, None
    accumulator = _fma(m00, c00, accumulator)
    if not math.isfinite(accumulator):
        return ResampleMatrixStatus.NON_FINITE_RESULT, None
    determinant = _fma(m02, c20, accumulator)
    if not math.isfinite(determinant):
        return ResampleMatrixStatus.NON_FINITE_RESULT, None
    if abs(determinant) < DETERMINANT_THRESHOLD:
        return ResampleMatrixStatus.SINGULAR_MATRIX, None

    cofactors = []
    for positive_left, positive_right, negative_left, negative_right in (
            (m00, m11, m01, m10),
            (m20, m01, m00, m21),
            (m10, m02, m00, m12),
            (m00, m22, m02, m20),
            (m01, m12, m02, m11),
            (m02, m21, m01, m22)):
        status, value = _cofactor(positive_left, positive_right,
                                  negative_left, negative_right)
        if status != ResampleMatrixStatus.OK:
            return status, None
        assert value is not None
        cofactors.append(value)
    c22, c21, c12, c11, c02, c01 = cofactors

    # Each cell is divided independently by the signed determinant.
    adjugate = (c00, c01, c02, c10, c11, c12, c20, c21, c22)
    inverse = tuple(value / determinant for value in adjugate)
    if not _finite(inverse):
        return ResampleMatrixStatus.NON_FINITE_RESULT, None
    return ResampleMatrixStatus.OK, inverse


def _multiply3(left: tuple[float, ...],
               right: tuple[float, ...]) -> tuple[ResampleMatrixStatus,
                                                   tuple[float, ...] | None]:
    product = []
    for row in range(3):
        for column in range(3):
            accumulator = 0.0
            for inner in range(3):
                accumulator = _fma(left[row * 3 + inner],
                                   right[inner * 3 + column], accumulator)
                if not math.isfinite(accumulator):
                    return ResampleMatrixStatus.NON_FINITE_RESULT, None
            product.append(accumulator)
    return ResampleMatrixStatus.OK, tuple(product)


def decompose_ordinary_resample_matrix(
        values: Sequence[float]) -> tuple[ResampleMatrixStatus,
                                          tuple[float, ...] | None]:
    """Return (status, six factory tuples) for a row-major 3x3 matrix.

    Success tuple order is vertical translation/scale/cross followed by
    horizontal translation/scale/cross.  On failure the tuple is ``None``;
    callers that mirror the C++ output parameter should leave it untouched.
    """
    try:
        if len(values) != 9:
            raise ValueError("expected exactly nine matrix cells")
        matrix = tuple(float(value) for value in values)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("matrix must contain nine numeric values") from error

    if not _finite(matrix):
        return ResampleMatrixStatus.NON_FINITE_INPUT, None

    status, inverse = _invert3(matrix)
    if status != ResampleMatrixStatus.OK:
        return status, None
    assert inverse is not None

    # H has the recovered form [1 0 0; N10 N11 N12; 0 0 1].
    horizontal = (1.0, 0.0, 0.0,
                  inverse[3], inverse[4], inverse[5],
                  0.0, 0.0, 1.0)
    status, inverse_horizontal = _invert3(horizontal)
    if status != ResampleMatrixStatus.OK:
        return status, None
    assert inverse_horizontal is not None

    # The call order in the recovered factory is P = N * inverse(H).
    status, product = _multiply3(inverse, inverse_horizontal)
    if status != ResampleMatrixStatus.OK:
        return status, None
    assert product is not None

    result = (product[2], product[0], product[1],
              horizontal[5], horizontal[4], horizontal[3])
    if not _finite(result):
        return ResampleMatrixStatus.NON_FINITE_RESULT, None
    return ResampleMatrixStatus.OK, result
