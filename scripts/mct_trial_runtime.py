"""Small, isolated runtime helpers for the DehazeFormer-MCT experiment.

This module is intentionally separate from Imprint's production dehaze path.
"""

from __future__ import annotations

import importlib.util
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


_CURVE_SIZE = 256
_INPUT_CHANNELS = 3
_OUTPUT_CHANNELS = 3
_CURVE_NODES = 8


@dataclass(frozen=True)
class MappingStats:
    strip_count: int
    pixel_count: int
    strip_rows: int
    elapsed_seconds: float


@dataclass(frozen=True)
class MCTRunStats:
    width: int
    height: int
    device: str
    strength: float
    input_dtype: str
    strip_count: int
    inference_seconds: float
    mapping_seconds: float


def load_model(vendor_path: str | Path, weights_path: str | Path, device: str = "cpu") -> torch.nn.Module:
    """Load the standalone upstream MCT implementation and a strict checkpoint."""
    source = Path(vendor_path)
    spec = importlib.util.spec_from_file_location("imprint_mct_vendor", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load MCT source from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "MCT"):
        raise ImportError(f"MCT class not found in {source}")

    model = module.MCT()
    checkpoint = torch.load(Path(weights_path), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError("MCT checkpoint must contain a 'state_dict' mapping")
    state_dict = checkpoint["state_dict"]
    if not isinstance(state_dict, dict):
        raise ValueError("MCT checkpoint 'state_dict' must be a mapping")
    model.load_state_dict(state_dict, strict=True)
    model.to(torch.device(device))
    model.eval()
    return model


def _validate_mapping_inputs(
    model_input: torch.Tensor,
    params: torch.Tensor,
    strip_rows: int,
) -> tuple[int, int, int]:
    if not isinstance(model_input, torch.Tensor) or model_input.ndim != 4:
        raise ValueError("model_input must have shape [B, 3, H, W]")
    if not isinstance(params, torch.Tensor) or params.ndim != 4:
        raise ValueError("params must have shape [B, 72, 256, 256]")
    batch, channels, height, width = model_input.shape
    if batch <= 0 or channels != _INPUT_CHANNELS or height <= 0 or width <= 0:
        raise ValueError("model_input must be non-empty RGB BCHW")
    if params.shape != (batch, 72, _CURVE_SIZE, _CURVE_SIZE):
        raise ValueError("params must have shape [B, 72, 256, 256]")
    if model_input.dtype != torch.float32 or params.dtype != torch.float32:
        raise ValueError("model_input and params must use float32")
    if not isinstance(strip_rows, int) or isinstance(strip_rows, bool) or strip_rows <= 0:
        raise ValueError("strip_rows must be a positive integer")
    if model_input.device.type != "cpu":
        raise ValueError("model_input must stay on CPU so strip mapping bounds device memory")
    if not torch.isfinite(model_input).all() or not torch.isfinite(params).all():
        raise ValueError("model_input and params must contain only finite values")
    return batch, height, width


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def map_curve_strips(
    model_input: torch.Tensor,
    params: torch.Tensor,
    device: str | torch.device = "cpu",
    strip_rows: int = 32,
) -> tuple[torch.Tensor, MappingStats]:
    """Apply the upstream 3D curve mapping as memory-bounded 2D strip maps.

    ``model_input`` is normalized sRGB in ``[-1, 1]`` and remains on CPU. The
    returned tensor is also on CPU; only the current strip and 72 sampled curve
    planes are resident on ``device`` at once. Coordinates use the full image's
    global ``[-1, 1]`` extent for every strip.
    """
    batch, height, width = _validate_mapping_inputs(model_input, params, strip_rows)
    target = torch.device(device)
    device_params = params.to(target)
    output = torch.empty((batch, _OUTPUT_CHANNELS, height, width), dtype=torch.float32, device="cpu")
    global_x = torch.linspace(-1.0, 1.0, width, dtype=torch.float32, device=target)
    global_y = torch.linspace(-1.0, 1.0, height, dtype=torch.float32, device=target)

    _synchronize(target)
    started = time.perf_counter()
    strip_count = 0
    with torch.inference_mode():
        for top in range(0, height, strip_rows):
            bottom = min(top + strip_rows, height)
            strip_height = bottom - top
            strip = model_input[:, :, top:bottom, :].to(target)
            coord_x = global_x.view(1, 1, width).expand(batch, strip_height, width)
            coord_y = global_y[top:bottom].view(1, strip_height, 1).expand(batch, strip_height, width)
            grid = torch.stack((coord_x, coord_y), dim=-1)

            # Sampling the 72 spatial planes first, then interpolating between
            # adjacent intensity nodes, is mathematically the same trilinear
            # interpolation performed by upstream's 5D grid_sample.
            sampled = F.grid_sample(
                device_params,
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            # Upstream first chunks the flat parameter channels into the
            # three input-image-channel groups, then chunks each group into
            # the three output channels. In the flattened 72-channel tensor,
            # the output channel is therefore the outer 24-channel axis.
            curves = sampled.reshape(
                batch,
                _OUTPUT_CHANNELS,
                _INPUT_CHANNELS,
                _CURVE_NODES,
                strip_height,
                width,
            ).permute(0, 2, 1, 3, 4, 5)

            node_position = ((strip + 1.0) * ((_CURVE_NODES - 1) / 2.0)).clamp(0.0, _CURVE_NODES - 1)
            low_index = node_position.floor().to(torch.long)
            high_index = (low_index + 1).clamp(max=_CURVE_NODES - 1)
            fraction = node_position - low_index.to(node_position.dtype)
            low = torch.gather(
                curves,
                dim=3,
                index=low_index[:, :, None, None, :, :].expand(
                    batch, _INPUT_CHANNELS, _OUTPUT_CHANNELS, 1, strip_height, width
                ),
            ).squeeze(3)
            high = torch.gather(
                curves,
                dim=3,
                index=high_index[:, :, None, None, :, :].expand(
                    batch, _INPUT_CHANNELS, _OUTPUT_CHANNELS, 1, strip_height, width
                ),
            ).squeeze(3)
            contribution = low + (high - low) * fraction[:, :, None, :, :]
            mapped = contribution.sum(dim=1)
            output[:, :, top:bottom, :] = mapped.to(device="cpu")
            strip_count += 1

    _synchronize(target)
    return output, MappingStats(
        strip_count=strip_count,
        pixel_count=height * width * batch,
        strip_rows=strip_rows,
        elapsed_seconds=time.perf_counter() - started,
    )


def _linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    return np.where(
        linear <= np.float32(0.0031308),
        np.float32(12.92) * linear,
        np.float32(1.055) * np.power(linear, np.float32(1.0 / 2.4)) - np.float32(0.055),
    ).astype(np.float32, copy=False)


def _srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    return np.where(
        srgb <= np.float32(0.04045),
        srgb / np.float32(12.92),
        np.power((srgb + np.float32(0.055)) / np.float32(1.055), np.float32(2.4)),
    ).astype(np.float32, copy=False)


def apply_mct(
    image: np.ndarray,
    model: torch.nn.Module,
    device: str | torch.device = "cpu",
    strip_rows: int = 32,
    strength: float = 1.0,
) -> tuple[np.ndarray, MCTRunStats]:
    """Run MCT on HWC linear RGB data and return a same-size, same-dtype image."""
    if not isinstance(image, np.ndarray):
        raise ValueError("image must be a NumPy array")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape [H, W, 3]")
    height, width, _ = image.shape
    if height <= 0 or width <= 0:
        raise ValueError("image must be non-empty")
    if image.dtype not in (np.dtype(np.uint16), np.dtype(np.float32)):
        raise ValueError("image dtype must be uint16 or float32")
    if not np.isfinite(float(strength)) or not 0.0 <= float(strength) <= 1.0:
        raise ValueError("strength must be finite and in [0, 1]")
    strength_value = float(strength)
    if not isinstance(strip_rows, int) or isinstance(strip_rows, bool) or strip_rows <= 0:
        raise ValueError("strip_rows must be a positive integer")
    if not np.isfinite(image).all():
        raise ValueError("image must contain only finite values")
    if image.dtype == np.float32 and (np.any(image < 0.0) or np.any(image > 1.0)):
        raise ValueError("float32 image values must be normalized linear RGB in [0, 1]")
    if not hasattr(model, "basenet"):
        raise ValueError("model must expose a basenet curve predictor")

    original = np.array(image, dtype=np.float32, copy=True, order="C")
    if image.dtype == np.uint16:
        original /= np.float32(65535.0)
    if strength_value == 0.0:
        stats = MCTRunStats(
            width=width,
            height=height,
            device=str(torch.device(device)),
            strength=0.0,
            input_dtype=str(image.dtype),
            strip_count=0,
            inference_seconds=0.0,
            mapping_seconds=0.0,
        )
        return image.copy(), asdict(stats)

    source_srgb = _linear_to_srgb(original)
    normalized = np.ascontiguousarray(source_srgb * np.float32(2.0) - np.float32(1.0))
    model_input = torch.from_numpy(normalized.transpose(2, 0, 1)[None, ...].copy())
    reduced = F.interpolate(model_input, size=(_CURVE_SIZE, _CURVE_SIZE), mode="area")
    target = torch.device(device)

    _synchronize(target)
    inference_started = time.perf_counter()
    with torch.inference_mode():
        params = model.basenet(reduced.to(target))
    _synchronize(target)
    inference_seconds = time.perf_counter() - inference_started
    if not isinstance(params, torch.Tensor) or params.shape != (1, 72, _CURVE_SIZE, _CURVE_SIZE):
        raise ValueError("basenet must return float32 [1, 72, 256, 256] curve parameters")
    if params.dtype != torch.float32:
        raise ValueError("basenet curve parameters must use float32")
    if not torch.isfinite(params).all():
        raise ValueError("basenet produced non-finite curve parameters")

    mapped, mapping_stats = map_curve_strips(model_input, params, device=target, strip_rows=strip_rows)
    mapped_srgb = ((mapped[0].permute(1, 2, 0).numpy() + np.float32(1.0)) * np.float32(0.5))
    mapped_srgb = np.clip(mapped_srgb, 0.0, 1.0)
    enhanced_linear = _srgb_to_linear(mapped_srgb)
    result_linear = original + np.float32(strength_value) * (enhanced_linear - original)
    result_linear = np.clip(result_linear, 0.0, 1.0)

    if image.dtype == np.uint16:
        result = np.rint(result_linear * np.float32(65535.0)).clip(0, 65535).astype(np.uint16)
    else:
        result = result_linear.astype(np.float32, copy=False)
    stats = MCTRunStats(
        width=width,
        height=height,
        device=str(target),
        strength=strength_value,
        input_dtype=str(image.dtype),
        strip_count=mapping_stats.strip_count,
        inference_seconds=inference_seconds,
        mapping_seconds=mapping_stats.elapsed_seconds,
    )
    return result, asdict(stats)
