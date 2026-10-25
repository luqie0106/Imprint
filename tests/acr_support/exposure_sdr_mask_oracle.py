"""Independent binary32 oracles for Camera Raw exposure baselines 013729ac/01372428.

The masked and no-mask entry points cover finite-intermediate prepared SDR
subsets. Their instruction order and constants are transcribed from the saved
ARM64 listings; neither reference imports or calls the C++ implementation.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import math

import numpy as np


REFERENCE_ID = "acr-013729ac-sdr-mask-no-context-f32-v1"
CONTEXT_REFERENCE_ID = "acr-013729ac-sdr-mask-five-context-f32-v1"
BASELINE_ENTRY = 0x013729AC
MASK_KERNEL_RANGE = (0x01372B54, 0x01372BC0)
SORT_RANGE = (0x01372BD4, 0x01372C78)
RATIONAL_RANGE = (0x01372C7C, 0x01372D14)
RATIONAL_CLIP_HELPER = 0x00DC6288
MAIN_CURVE_RANGE = (0x01372E44, 0x01372ECC)
SQUARE_ROOT_RANGE = (0x01372ED8, 0x01372F3C)
BASELINE_BINARY_SHA256 = "09978d95dfe76b18495fea9550b051a89f4598ab0fd4734d69d22d330496f40a"

CONTROL_GAIN = 0
CONTROL_MAIN_LUT_SCALE = 1
CONTROL_RATIONAL_SCALE = 2
CONTROL_SQRT_AMOUNT = 3
CONTROL_RATIONAL_DENOMINATOR = 4
CONTROL_EPSILON = 5
CONTROL_COUNT = 6

MASK_COEFFICIENT_BITS = 0xBEADF655
MASK_LIMIT = np.float32(16.0)
MASK_EXPONENT_SCALE = np.float32(4.0)
ONE = np.float32(1.0)
ZERO = np.float32(0.0)

_F = np.float32


class NonfiniteIntermediateError(ArithmeticError):
    """An operation in the explicitly supported finite subset overflowed."""


class LutBoundsError(IndexError):
    """A main-curve coordinate lacks its required pair of samples."""


class _LutContext:
    """Validated descriptor fields for one active optional LUT."""

    __slots__ = ("values", "max_index", "scale")

    def __init__(self, values: np.ndarray, max_index: int, scale: np.float32):
        self.values = values
        self.max_index = max_index
        self.scale = scale


def _load_float_math():
    candidates = []
    name = ctypes.util.find_library("m")
    if name:
        candidates.append(name)
    candidates.append(None)
    for candidate in candidates:
        try:
            library = ctypes.CDLL(candidate) if candidate is not None else ctypes.CDLL(None)
            fmaf = library.fmaf
            sqrtf = library.sqrtf
        except (AttributeError, OSError):
            continue
        fmaf.argtypes = (ctypes.c_float, ctypes.c_float, ctypes.c_float)
        fmaf.restype = ctypes.c_float
        sqrtf.argtypes = (ctypes.c_float,)
        sqrtf.restype = ctypes.c_float
        return fmaf, sqrtf
    raise RuntimeError("platform C math library must expose fmaf and sqrtf")


_fmaf, _sqrtf = _load_float_math()
MASK_COEFFICIENT = np.asarray([MASK_COEFFICIENT_BITS], dtype=np.uint32).view(np.float32)[0]


def f32(value: float) -> np.float32:
    """Round one value to IEEE-754 binary32."""
    with np.errstate(over="ignore", invalid="ignore"):
        return _F(value)


def _finite(value: float, operation: str) -> np.float32:
    result = f32(value)
    if not np.isfinite(result):
        raise NonfiniteIntermediateError(f"nonfinite binary32 result after {operation}")
    return result


def add32(a: float, b: float, operation: str = "FADD") -> np.float32:
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        return _finite(_F(_F(a) + _F(b)), operation)


def sub32(a: float, b: float, operation: str = "FSUB") -> np.float32:
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        return _finite(_F(_F(a) - _F(b)), operation)


def mul32(a: float, b: float, operation: str = "FMUL") -> np.float32:
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        return _finite(_F(_F(a) * _F(b)), operation)


def div32(a: float, b: float, operation: str = "FDIV") -> np.float32:
    with np.errstate(divide="ignore", over="ignore", invalid="ignore", under="ignore"):
        return _finite(_F(_F(a) / _F(b)), operation)


def fma32(a: float, b: float, c: float, operation: str = "FMADD") -> np.float32:
    return _finite(_fmaf(float(_F(a)), float(_F(b)), float(_F(c))), operation)


def sqrt32(value: float, operation: str = "FSQRT") -> np.float32:
    return _finite(_sqrtf(float(_F(value))), operation)


def _mask_gain(gain: np.float32, sample: np.float32) -> np.float32:
    # FMINNM(mask, +16), then FCSEL(-16, clamped, MI).
    mask = MASK_LIMIT if sample > MASK_LIMIT else sample
    lower = _F(-MASK_LIMIT)
    if mask < lower:
        mask = lower
    if mask == ZERO:
        return ONE

    scaled = mul32(mask, MASK_EXPONENT_SCALE, "mask FMUL by four")
    # FRINTM and FCVTMS both round toward minus infinity.  The float input is
    # exactly representable as a Python float, so floor reproduces that mode.
    exponent = math.floor(float(scaled))
    fraction = sub32(scaled, _F(exponent), "mask FSUB from FRINTM")
    exponent_bits = (0x3F800000 + (exponent << 23)) & 0xFFFFFFFF
    power = np.asarray([exponent_bits], dtype=np.uint32).view(np.float32)[0]
    complement = sub32(ONE, fraction, "mask polynomial complement")
    curve = fma32(complement, MASK_COEFFICIENT, ONE, "mask polynomial FMA 1")
    curve = fma32(curve, fraction, ONE, "mask polynomial FMA 2")
    return mul32(curve, power, "mask approximation times exponent bits")


def _sort_indices(red: np.float32, green: np.float32, blue: np.float32) -> tuple[int, int, int]:
    """Return low/middle/high indices using the raw comparison decision tree."""
    if red >= green:
        if red >= blue:
            high = 0
            if green < blue:
                return 1, 2, high
            return 2, 1, high
        return 1, 0, 2
    if green >= blue:
        high = 1
        if red < blue:
            return 0, 2, high
        return 2, 0, high
    return 0, 1, 2


def _rational(value: np.float32, scale: np.float32,
              denominator: np.float32) -> np.float32:
    product = mul32(value, scale, "rational x*A")
    first_denominator = add32(product, ONE, "rational u+1")
    first_ratio = div32(product, first_denominator, "rational u/(u+1)")
    second_ratio = div32(product, denominator, "rational u/Q")
    second_denominator = add32(second_ratio, ONE, "rational u/Q+1")
    combined = mul32(first_ratio, second_denominator, "rational product")
    return div32(combined, scale, "rational /A")


def _relative_ratio(values: list[np.float32], low: int, middle: int, high: int,
                    epsilon: np.float32) -> np.float32:
    width = sub32(values[high], values[low], "channel range")
    denominator = width if width > epsilon else epsilon
    numerator = sub32(values[middle], values[low], "middle minus low")
    return div32(numerator, denominator, "middle relative position")


def _upper_clip_preserving_middle(values: list[np.float32], low: int,
                                  middle: int, high: int,
                                  epsilon: np.float32) -> None:
    # Helper 00dc6288 computes the ratio before clipping either endpoint.
    ratio = _relative_ratio(values, low, middle, high, epsilon)
    values[low] = values[low] if values[low] < ONE else ONE
    values[high] = values[high] if values[high] < ONE else ONE
    width = sub32(values[high], values[low], "clipped channel range")
    values[middle] = fma32(ratio, width, values[low], "clip middle reconstruction")


def _main_curve(value: np.float32, scale: np.float32,
                table: np.ndarray) -> np.float32:
    coordinate = mul32(value, scale, "main LUT coordinate")
    number = float(coordinate)
    if number >= 2147483648.0:
        index = 2147483647
    elif number <= -2147483648.0:
        index = -2147483648
    else:
        # FCVTZS is toward zero, unlike the mask's FCVTMS floor.
        index = math.trunc(number)
    if index < 0 or index + 1 >= table.size:
        raise LutBoundsError(f"main LUT needs indices {index} and {index + 1}")
    left = _finite(table[index], "main LUT left sample")
    right = _finite(table[index + 1], "main LUT guard sample")
    fraction = sub32(coordinate, _F(index), "main LUT fractional coordinate")
    delta = sub32(right, left, "main LUT adjacent difference")
    return fma32(fraction, delta, left, "main LUT interpolation")


def _prepare_context(descriptor: object, name: str) -> _LutContext | None:
    """Validate an enabled descriptor without inspecting table sample values.

    A descriptor is ``(values, inclusive_max_index, scale)``. Sample finiteness
    and the adjacent guard are checked only for indices actually read, so NaNs
    in an unused tail remain permitted as in the prepared native contract.
    """
    if descriptor is None:
        return None
    if not isinstance(descriptor, (tuple, list)) or len(descriptor) != 3:
        raise ValueError(f"{name} must be (values, max_index, scale) or None")
    raw_values, raw_max_index, raw_scale = descriptor
    values = np.asarray(raw_values)
    if values.dtype != np.float32 or values.ndim != 1:
        raise ValueError(f"{name} values must be a one-dimensional float32 array")
    if isinstance(raw_max_index, (bool, np.bool_)):
        raise ValueError(f"{name} max_index must be an unsigned 32-bit integer")
    try:
        max_index = int(raw_max_index)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} max_index must be an unsigned 32-bit integer") from exc
    if max_index != raw_max_index or not 0 <= max_index <= 0x7FFFFFFF:
        raise ValueError(f"{name} max_index must fit the ARM signed index range")
    try:
        scale = _F(raw_scale)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} scale must be finite binary32") from exc
    if not np.isfinite(scale) or scale < ZERO:
        raise ValueError(f"{name} scale must be finite and nonnegative binary32")
    return _LutContext(values, max_index, scale)


def _context_index(coordinate: np.float32, context: _LutContext,
                   name: str) -> tuple[int, np.float32]:
    """Apply FCVTZS signed saturation, index checks, and guard-count checks."""
    number = float(coordinate)
    if number >= 2147483648.0:
        index = 2147483647
    elif number <= -2147483648.0:
        index = -2147483648
    else:
        index = math.trunc(number)
    if index < 0 or index > context.max_index:
        raise LutBoundsError(
            f"{name} index {index} outside [0, {context.max_index}]"
        )
    if context.values.size < index + 2:
        raise LutBoundsError(
            f"{name} needs indices {index} and {index + 1}; "
            f"count is {context.values.size}"
        )
    # SCVTF converts the signed index back to binary32 before subtraction.
    fraction = sub32(coordinate, _F(index), f"{name} fractional coordinate")
    return index, fraction


def _context_samples(context: _LutContext, index: int, fraction: np.float32,
                     name: str) -> np.float32:
    first = _finite(context.values[index], f"{name} first sample")
    next_sample = _finite(context.values[index + 1], f"{name} guard sample")
    one_minus_fraction = sub32(ONE, fraction, f"{name} complement")
    fraction_times_next = mul32(fraction, next_sample,
                                f"{name} fraction times guard sample")
    return fma32(first, one_minus_fraction, fraction_times_next,
                 f"{name} interpolation")


def _context_channel(value: np.float32, context: _LutContext,
                     name: str) -> np.float32:
    coordinate = mul32(context.scale, value, f"{name} coordinate")
    index, fraction = _context_index(coordinate, context, name)
    return _context_samples(context, index, fraction, name)


def _context_luma(values: list[np.float32], context: _LutContext,
                  epsilon: np.float32, name: str) -> None:
    # Preserve the helper's actual FADD order: R+G, then G+t, then B+t.
    total = add32(values[0], values[1], f"{name} luma R+G")
    total = add32(values[1], total, f"{name} luma G+t")
    total = add32(values[2], total, f"{name} luma B+t")
    luma = mul32(total, _F(0.25), f"{name} luma quarter")
    coordinate = mul32(context.scale, luma, f"{name} coordinate")
    index, fraction = _context_index(coordinate, context, name)
    mapped_luma = _context_samples(context, index, fraction, name)
    denominator = luma if luma > epsilon else epsilon
    gain = div32(mapped_luma, denominator, f"{name} luma gain")
    for channel in range(3):
        values[channel] = mul32(values[channel], gain,
                                f"{name} RGB gain channel {channel}")


def _pixel(rgb: np.ndarray, mask: np.float32, controls: np.ndarray,
           main_lut: np.ndarray | None,
           pre_lut: _LutContext | None = None,
           channel_luts: tuple[_LutContext, _LutContext, _LutContext] | None = None,
           post_lut: _LutContext | None = None) -> np.ndarray:
    gain = _mask_gain(controls[CONTROL_GAIN], mask)
    gain = mul32(gain, controls[CONTROL_GAIN], "masked global gain")
    values = [mul32(sample, gain, "RGB multiplied by masked gain") for sample in rgb]
    low, middle, high = _sort_indices(*values)

    rational_scale = controls[CONTROL_RATIONAL_SCALE]
    denominator = controls[CONTROL_RATIONAL_DENOMINATOR]
    epsilon = controls[CONTROL_EPSILON]
    if denominator < ONE:
        if values[high] == values[low]:
            mapped = _rational(values[low], rational_scale, denominator)
            values[low] = mapped
            values[middle] = mapped
            values[high] = mapped
        else:
            ratio = _relative_ratio(values, low, middle, high, epsilon)
            mapped_low = _rational(values[low], rational_scale, denominator)
            mapped_high = _rational(values[high], rational_scale, denominator)
            values[low] = mapped_low
            values[high] = mapped_high
            width = sub32(mapped_high, mapped_low, "rational transformed range")
            values[middle] = fma32(ratio, width, mapped_low,
                                   "rational middle reconstruction")

    _upper_clip_preserving_middle(values, low, middle, high, epsilon)

    if rational_scale > ONE:
        if values[high] == values[low]:
            ratio = ZERO
        else:
            ratio = _relative_ratio(values, low, middle, high, epsilon)
        table_scale = controls[CONTROL_MAIN_LUT_SCALE]
        # The raw kernel has no explicit bounds check. This supported subset
        # reports a guarded-table failure instead of reading outside caller data.
        if main_lut is None:
            raise LutBoundsError("main LUT is required when A > 1")
        mapped_low = _main_curve(values[low], table_scale, main_lut)
        mapped_high = _main_curve(values[high], table_scale, main_lut)
        values[low] = mapped_low
        values[high] = mapped_high
        width = sub32(mapped_high, mapped_low, "main curve transformed range")
        values[middle] = fma32(ratio, width, mapped_low,
                               "main curve middle reconstruction")

    sqrt_amount = controls[CONTROL_SQRT_AMOUNT]
    if sqrt_amount > ONE:
        if values[high] == values[low]:
            ratio = ZERO
        else:
            ratio = _relative_ratio(values, low, middle, high, epsilon)
        amount_minus_one = sub32(sqrt_amount, ONE, "sqrt amount minus one")

        def transform(value: np.float32) -> np.float32:
            square = mul32(value, value, "square/sqrt x*x")
            numerator = mul32(square, sqrt_amount, "square/sqrt x*x*B")
            denominator_value = fma32(amount_minus_one, square, ONE,
                                      "square/sqrt denominator FMA")
            radicand = div32(numerator, denominator_value,
                             "square/sqrt rational radicand")
            return sqrt32(radicand, "square/sqrt square root")

        mapped_low = transform(values[low])
        mapped_high = transform(values[high])
        values[low] = mapped_low
        values[high] = mapped_high
        width = sub32(mapped_high, mapped_low, "square/sqrt transformed range")
        values[middle] = fma32(ratio, width, mapped_low,
                               "square/sqrt middle reconstruction")

    # The raw kernel invokes the optional contexts after rational clipping,
    # main curve, and square/sqrt, in this exact physical-RGB order.
    if pre_lut is not None:
        _context_luma(values, pre_lut, controls[CONTROL_EPSILON], "pre-luma LUT")
    if channel_luts is not None:
        for channel, context in enumerate(channel_luts):
            values[channel] = _context_channel(
                values[channel], context, f"RGB channel {channel} LUT"
            )
    if post_lut is not None:
        _context_luma(values, post_lut, controls[CONTROL_EPSILON], "post-luma LUT")
        for channel in range(3):
            # FMINNM(x, 1): this is the only final upper clamp in the sequence.
            values[channel] = values[channel] if values[channel] < ONE else ONE

    return np.asarray(values, dtype=np.float32)


def apply(rgb: object, mask: object, controls: object,
          *, main_lut: object | None = None,
          pre_lut: object | None = None,
          channel_luts: object = (None, None, None),
          post_lut: object | None = None) -> np.ndarray:
    """Evaluate finite HxWx3 RGB using the prepared 013729ac SDR subset.

    Controls are ``[G, S, A, B, Q, E]``.  All ordinary operations are rounded
    independently to binary32; only recovered fused operations use ``fmaf``.
    Optional LUT descriptors are ``(values, inclusive_max_index, scale)``.
    The RGB LUT group runs only when all three descriptors are supplied; a
    partial group is ignored as a whole, including its metadata. A failure
    raises before returning output, matching the bridge's all-or-nothing write
    contract for this supported finite-intermediate set.
    """
    source = np.asarray(rgb)
    if source.dtype != np.float32 or source.ndim != 3 or source.shape[2] != 3:
        raise ValueError("RGB must be an HxWx3 float32 array")
    height, width = source.shape[:2]
    if height <= 0 or width <= 0 or height * width > (1 << 26):
        raise ValueError("RGB must contain between one pixel and 2**26 pixels")
    if not np.isfinite(source).all() or np.any(source < ZERO):
        raise ValueError("RGB samples must be finite and nonnegative")

    masks = np.asarray(mask)
    if masks.dtype != np.float32 or masks.shape != (height, width):
        raise ValueError("mask must be a matching HxW float32 array")
    if not np.isfinite(masks).all():
        raise ValueError("mask samples must be finite")

    raw_controls = np.asarray(controls)
    if raw_controls.shape != (CONTROL_COUNT,) or not np.isfinite(raw_controls).all():
        raise ValueError("controls must contain six finite values")
    with np.errstate(over="ignore", invalid="ignore"):
        parameters = np.asarray(raw_controls, dtype=np.float32)
    if not np.isfinite(parameters).all():
        raise ValueError("controls must remain finite after binary32 conversion")
    if (parameters[CONTROL_GAIN] < ZERO or parameters[CONTROL_MAIN_LUT_SCALE] < ZERO
            or parameters[CONTROL_RATIONAL_SCALE] <= ZERO
            or parameters[CONTROL_RATIONAL_DENOMINATOR] <= ZERO
            or parameters[CONTROL_EPSILON] <= ZERO):
        raise ValueError("G/S must be nonnegative and A/Q/E must be positive")

    table = None
    if parameters[CONTROL_RATIONAL_SCALE] > ONE:
        if main_lut is None:
            raise LutBoundsError("main LUT is required when A > 1")
        table = np.asarray(main_lut)
        if table.dtype != np.float32 or table.ndim != 1:
            raise ValueError("main LUT must be a one-dimensional float32 array")
    # A<=1 bypasses and ignores main_lut, matching the native adapter.

    prepared_pre = _prepare_context(pre_lut, "pre-luma LUT")
    prepared_post = _prepare_context(post_lut, "post-luma LUT")
    if not isinstance(channel_luts, (tuple, list)) or len(channel_luts) != 3:
        raise ValueError("channel_luts must contain exactly three descriptors")
    # Match the raw all-three-presence predicate. Do not inspect any descriptor
    # fields when the group is incomplete.
    if all(item is not None for item in channel_luts):
        prepared_channels = tuple(
            _prepare_context(item, f"RGB channel {index} LUT")
            for index, item in enumerate(channel_luts)
        )
        if any(item is None for item in prepared_channels):
            raise ValueError("active RGB channel LUT descriptors must be present")
        channel_contexts = prepared_channels  # type: ignore[assignment]
    else:
        channel_contexts = None

    result = np.empty(source.shape, dtype=np.float32)
    for y in range(height):
        for x in range(width):
            result[y, x] = _pixel(source[y, x], masks[y, x], parameters, table,
                                  prepared_pre, channel_contexts, prepared_post)
    return result


NOMASK_REFERENCE_ID = "acr-01372428-sdr-no-mask-five-context-f32-v1"
NOMASK_BASELINE_ENTRY = 0x01372428
NOMASK_CLIP_BRANCH_RANGE = (0x013726E0, 0x013726E4)
NOMASK_CLIP_HELPER = 0x00DC6288
NOMASK_BINARY_SHA256 = BASELINE_BINARY_SHA256


def _pixel_nomask(
    rgb: np.ndarray,
    controls: np.ndarray,
    main_lut: np.ndarray | None,
    pre_lut: _LutContext | None = None,
    channel_luts: tuple[_LutContext, _LutContext, _LutContext] | None = None,
    post_lut: _LutContext | None = None,
) -> np.ndarray:
    """Run the direct-G kernel, retaining its conditional upper clip."""
    gain = controls[CONTROL_GAIN]
    values = [mul32(sample, gain, "no-mask RGB multiplied by global gain")
              for sample in rgb]
    low, middle, high = _sort_indices(*values)

    rational_scale = controls[CONTROL_RATIONAL_SCALE]
    denominator = controls[CONTROL_RATIONAL_DENOMINATOR]
    epsilon = controls[CONTROL_EPSILON]
    if denominator < ONE:
        if values[high] == values[low]:
            mapped = _rational(values[low], rational_scale, denominator)
            values[low] = mapped
            values[middle] = mapped
            values[high] = mapped
        else:
            ratio = _relative_ratio(values, low, middle, high, epsilon)
            mapped_low = _rational(values[low], rational_scale, denominator)
            mapped_high = _rational(values[high], rational_scale, denominator)
            values[low] = mapped_low
            values[high] = mapped_high
            width = sub32(mapped_high, mapped_low,
                          "no-mask rational transformed range")
            values[middle] = fma32(ratio, width, mapped_low,
                                   "no-mask rational middle reconstruction")

    # Baseline 01372428 reaches 00dc6288 only for high>1. The masked baseline
    # above intentionally retains its unconditional invocation and rounding.
    if values[high] > ONE:
        _upper_clip_preserving_middle(values, low, middle, high, epsilon)

    if rational_scale > ONE:
        if values[high] == values[low]:
            ratio = ZERO
        else:
            ratio = _relative_ratio(values, low, middle, high, epsilon)
        table_scale = controls[CONTROL_MAIN_LUT_SCALE]
        if main_lut is None:
            raise LutBoundsError("main LUT is required when A > 1")
        mapped_low = _main_curve(values[low], table_scale, main_lut)
        mapped_high = _main_curve(values[high], table_scale, main_lut)
        values[low] = mapped_low
        values[high] = mapped_high
        width = sub32(mapped_high, mapped_low, "no-mask main curve range")
        values[middle] = fma32(ratio, width, mapped_low,
                               "no-mask main curve middle reconstruction")

    sqrt_amount = controls[CONTROL_SQRT_AMOUNT]
    if sqrt_amount > ONE:
        if values[high] == values[low]:
            ratio = ZERO
        else:
            ratio = _relative_ratio(values, low, middle, high, epsilon)
        amount_minus_one = sub32(sqrt_amount, ONE,
                                 "no-mask sqrt amount minus one")

        def transform(value: np.float32) -> np.float32:
            square = mul32(value, value, "no-mask square/sqrt x*x")
            numerator = mul32(square, sqrt_amount,
                              "no-mask square/sqrt x*x*B")
            denominator_value = fma32(amount_minus_one, square, ONE,
                                      "no-mask square/sqrt denominator FMA")
            radicand = div32(numerator, denominator_value,
                             "no-mask square/sqrt rational radicand")
            return sqrt32(radicand, "no-mask square/sqrt square root")

        mapped_low = transform(values[low])
        mapped_high = transform(values[high])
        values[low] = mapped_low
        values[high] = mapped_high
        width = sub32(mapped_high, mapped_low, "no-mask square/sqrt range")
        values[middle] = fma32(ratio, width, mapped_low,
                               "no-mask square/sqrt middle reconstruction")

    if pre_lut is not None:
        _context_luma(values, pre_lut, controls[CONTROL_EPSILON],
                      "no-mask pre-luma LUT")
    if channel_luts is not None:
        for channel, context in enumerate(channel_luts):
            values[channel] = _context_channel(
                values[channel], context, f"no-mask RGB channel {channel} LUT"
            )
    if post_lut is not None:
        _context_luma(values, post_lut, controls[CONTROL_EPSILON],
                      "no-mask post-luma LUT")
        for channel in range(3):
            values[channel] = values[channel] if values[channel] < ONE else ONE

    return np.asarray(values, dtype=np.float32)


def apply_nomask(rgb: object, controls: object,
                 *, main_lut: object | None = None,
                 pre_lut: object | None = None,
                 channel_luts: object = (None, None, None),
                 post_lut: object | None = None) -> np.ndarray:
    """Evaluate the prepared 01372428 SDR no-mask subset.

    This is the direct RGB*G baseline. It does not synthesize a zero mask and
    only calls the shared upper-clip/median helper when the post-rational high
    endpoint exceeds one. Optional contexts follow the same five-slot order
    and activation rules as :func:`apply`.
    """
    source = np.asarray(rgb)
    if source.dtype != np.float32 or source.ndim != 3 or source.shape[2] != 3:
        raise ValueError("RGB must be an HxWx3 float32 array")
    height, width = source.shape[:2]
    if height <= 0 or width <= 0 or height * width > (1 << 26):
        raise ValueError("RGB must contain between one pixel and 2**26 pixels")
    if not np.isfinite(source).all() or np.any(source < ZERO):
        raise ValueError("RGB samples must be finite and nonnegative")

    raw_controls = np.asarray(controls)
    if raw_controls.shape != (CONTROL_COUNT,) or not np.isfinite(raw_controls).all():
        raise ValueError("controls must contain six finite values")
    with np.errstate(over="ignore", invalid="ignore"):
        parameters = np.asarray(raw_controls, dtype=np.float32)
    if not np.isfinite(parameters).all():
        raise ValueError("controls must remain finite after binary32 conversion")
    if (parameters[CONTROL_GAIN] < ZERO or parameters[CONTROL_MAIN_LUT_SCALE] < ZERO
            or parameters[CONTROL_RATIONAL_SCALE] <= ZERO
            or parameters[CONTROL_RATIONAL_DENOMINATOR] <= ZERO
            or parameters[CONTROL_EPSILON] <= ZERO):
        raise ValueError("G/S must be nonnegative and A/Q/E must be positive")

    table = None
    if parameters[CONTROL_RATIONAL_SCALE] > ONE:
        if main_lut is None:
            raise LutBoundsError("main LUT is required when A > 1")
        table = np.asarray(main_lut)
        if table.dtype != np.float32 or table.ndim != 1:
            raise ValueError("main LUT must be a one-dimensional float32 array")

    prepared_pre = _prepare_context(pre_lut, "pre-luma LUT")
    prepared_post = _prepare_context(post_lut, "post-luma LUT")
    if not isinstance(channel_luts, (tuple, list)) or len(channel_luts) != 3:
        raise ValueError("channel_luts must contain exactly three descriptors")
    if all(item is not None for item in channel_luts):
        prepared_channels = tuple(
            _prepare_context(item, f"RGB channel {index} LUT")
            for index, item in enumerate(channel_luts)
        )
        if any(item is None for item in prepared_channels):
            raise ValueError("active RGB channel LUT descriptors must be present")
        channel_contexts = prepared_channels  # type: ignore[assignment]
    else:
        channel_contexts = None

    result = np.empty(source.shape, dtype=np.float32)
    for y in range(height):
        for x in range(width):
            result[y, x] = _pixel_nomask(
                source[y, x], parameters, table, prepared_pre,
                channel_contexts, prepared_post,
            )
    return result
