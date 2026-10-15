from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.mct_trial_runtime import apply_mct, load_model, map_curve_strips


def _upstream_mapping(x: torch.Tensor, param: torch.Tensor) -> torch.Tensor:
    """Reference the upstream MCT.mapping implementation using 5D sampling."""
    curve = torch.stack(torch.chunk(param, 3, dim=1), dim=1)
    curve_list = list(torch.chunk(curve, 3, dim=2))
    x_list = list(torch.chunk(x.detach(), 3, dim=1))
    batch, _, height, width = x.size()
    coord_h, coord_w = torch.meshgrid(
        [torch.linspace(-1, 1, height), torch.linspace(-1, 1, width)], indexing="ij"
    )
    coord_h = coord_h.unsqueeze(0).unsqueeze(1).repeat(batch, 1, 1, 1)
    coord_w = coord_w.unsqueeze(0).unsqueeze(1).repeat(batch, 1, 1, 1)
    grids = [torch.stack([coord_w, coord_h, channel], dim=4) for channel in x_list]
    return sum(
        F.grid_sample(curve_part, grid, "bilinear", "border", True)
        for curve_part, grid in zip(curve_list, grids)
    ).squeeze(2)


@pytest.mark.parametrize(
    ("height", "width", "strip_rows"),
    [
        (1, 17, 4),
        (17, 1, 5),
        (9, 13, 32),
        (35, 19, 32),
        (67, 23, 11),
    ],
)
def test_strip_mapping_matches_upstream_5d_grid_sample(height, width, strip_rows):
    generator = torch.Generator().manual_seed(714 + height + width)
    x = torch.rand((1, 3, height, width), generator=generator) * 2.0 - 1.0
    x[:, :, 0, 0] = -1.0
    x[:, :, -1, -1] = 1.0
    params = torch.randn((1, 72, 256, 256), generator=generator)

    actual, stats = map_curve_strips(x, params, device="cpu", strip_rows=strip_rows)
    expected = _upstream_mapping(x, params)

    assert actual.shape == x.shape
    assert actual.device.type == "cpu"
    assert torch.max(torch.abs(actual - expected)).item() <= 1e-5
    assert stats.strip_count == math.ceil(height / strip_rows)
    assert stats.pixel_count == height * width


def _identity_curve_params() -> torch.Tensor:
    params = torch.zeros((1, 72, 256, 256), dtype=torch.float32)
    nodes = torch.linspace(-1.0, 1.0, 8)
    for input_channel in range(3):
        for output_channel in range(3):
            for node_index in range(8):
                channel_index = (output_channel * 3 + input_channel) * 8 + node_index
                if input_channel == output_channel:
                    params[0, channel_index].fill_(nodes[node_index])
    return params


class _FixedCurvePredictor(nn.Module):
    def __init__(self, params: torch.Tensor):
        super().__init__()
        self.register_buffer("params", params)
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return self.params.to(device=x.device).expand(x.shape[0], -1, -1, -1)


class _FakeMCT(nn.Module):
    def __init__(self, params: torch.Tensor):
        super().__init__()
        self.basenet = _FixedCurvePredictor(params)


def test_identity_curve_preserves_uint16_linear_rgb_and_input():
    image = np.arange(7 * 11 * 3, dtype=np.uint16).reshape(7, 11, 3) * np.uint16(281)
    before = image.copy()
    result, stats = apply_mct(image, _FakeMCT(_identity_curve_params()), strip_rows=3)

    assert result.dtype == np.uint16
    assert result.shape == image.shape
    assert np.max(np.abs(result.astype(np.int32) - before.astype(np.int32))) <= 1
    np.testing.assert_array_equal(image, before)
    assert stats["strip_count"] == math.ceil(image.shape[0] / 3)
    assert stats["input_dtype"] == "uint16"


def test_zero_strength_returns_exact_copy_without_running_model():
    image = np.linspace(0.0, 1.0, 5 * 7 * 3, dtype=np.float32).reshape(5, 7, 3)
    before = image.copy()
    model = _FakeMCT(_identity_curve_params())

    result, stats = apply_mct(image, model, strength=0.0)

    np.testing.assert_array_equal(result, before)
    np.testing.assert_array_equal(image, before)
    assert result is not image
    assert result.dtype == np.float32
    assert model.basenet.calls == 0
    assert stats["strip_count"] == 0


def test_float32_output_range_dtype_and_input_immutability():
    image = np.random.default_rng(918).random((5, 9, 3), dtype=np.float32)
    before = image.copy()
    result, _ = apply_mct(image, _FakeMCT(_identity_curve_params()), strip_rows=2)

    assert result.dtype == np.float32
    assert result.shape == image.shape
    assert np.isfinite(result).all()
    assert np.all((result >= 0.0) & (result <= 1.0))
    np.testing.assert_array_equal(image, before)
    np.testing.assert_allclose(result, before, atol=1e-6, rtol=0.0)


@pytest.mark.parametrize(
    "image",
    [
        np.empty((0, 2, 3), dtype=np.float32),
        np.empty((2, 0, 3), dtype=np.uint16),
        np.zeros((2, 3, 4), dtype=np.float32),
        np.zeros((2, 3), dtype=np.float32),
        np.zeros((2, 3, 3), dtype=np.float64),
        np.full((2, 3, 3), np.nan, dtype=np.float32),
        np.full((2, 3, 3), np.inf, dtype=np.float32),
        np.full((2, 3, 3), -0.1, dtype=np.float32),
        np.full((2, 3, 3), 1.1, dtype=np.float32),
    ],
)
def test_apply_rejects_invalid_image_inputs(image):
    with pytest.raises(ValueError):
        apply_mct(image, _FakeMCT(_identity_curve_params()))


def test_load_model_uses_strict_state_dict_and_eval(tmp_path):
    vendor = tmp_path / "tiny_vendor.py"
    vendor.write_text(
        "import torch\n"
        "import torch.nn as nn\n"
        "class MCT(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.basenet = nn.Conv2d(3, 72, 1)\n",
        encoding="utf-8",
    )
    weights = tmp_path / "weights.pt"
    source_model = nn.Conv2d(3, 72, 1)
    torch.save({"state_dict": {f"basenet.{k}": v for k, v in source_model.state_dict().items()}}, weights)

    model = load_model(vendor, weights, device="cpu")

    assert not model.training
    assert all(not child.training for child in model.modules())
    assert all(parameter.requires_grad for parameter in model.parameters())
    for actual, expected in zip(model.basenet.state_dict().values(), source_model.state_dict().values()):
        torch.testing.assert_close(actual, expected)

    torch.save({"state_dict": {"unexpected": torch.tensor(1)}}, weights)
    with pytest.raises(RuntimeError):
        load_model(vendor, weights, device="cpu")
