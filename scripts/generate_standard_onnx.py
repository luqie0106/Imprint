#!/usr/bin/env python3
"""
generate_standard_onnx.py

从 LAION 官方仓库下载预训练通用美学线性权重 (sa_0_4_vit_b_32_linear.pth)，
并与 CLIP ViT-B/32 视觉底座熔铸生成端到端 ONNX 模型：models/standard_aesthetic_model.onnx
"""

import sys
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_MODELS = _ROOT / "models"
_MODELS.mkdir(parents=True, exist_ok=True)

LAION_WEIGHTS_URL = "https://github.com/LAION-AI/aesthetic-predictor/raw/main/sa_0_4_vit_b_32_linear.pth"
WEIGHTS_PATH = _MODELS / "sa_0_4_vit_b_32_linear.pth"
OUTPUT_ONNX_PATH = _MODELS / "standard_aesthetic_model.onnx"


def download_weights():
    if WEIGHTS_PATH.exists() and WEIGHTS_PATH.stat().st_size > 1000:
        print(f"✅ 已检测到本地权重: {WEIGHTS_PATH}")
        return
    print(f"⬇️ 正在从官方仓库下载通用美学线性权重...\n{LAION_WEIGHTS_URL}")
    req = urllib.request.Request(LAION_WEIGHTS_URL, headers={"User-Agent": "PhotoSort-Setup/1.0"})
    with urllib.request.urlopen(req) as resp, open(WEIGHTS_PATH, "wb") as f:
        f.write(resp.read())
    print(f"✅ 下载完成: {WEIGHTS_PATH} ({WEIGHTS_PATH.stat().st_size} bytes)")


def fuse_and_export_onnx():
    import torch
    import torch.nn as nn
    from transformers import CLIPModel

    class StandardAestheticONNX(nn.Module):
        def __init__(self, clip_model: CLIPModel, linear_weight: torch.Tensor, linear_bias: torch.Tensor):
            super().__init__()
            self.clip = clip_model
            self.linear = nn.Linear(512, 1)
            self.linear.weight = nn.Parameter(linear_weight)
            self.linear.bias = nn.Parameter(linear_bias)

        def forward(self, pixel_values):
            vision_outputs = self.clip.vision_model(pixel_values=pixel_values, return_dict=False)
            pooled_output = vision_outputs[1]
            features = self.clip.visual_projection(pooled_output)
            features = features / features.norm(dim=-1, keepdim=True)
            raw_score = self.linear(features)
            norm_score = torch.clamp((raw_score - 1.0) / 9.0, 0.0, 1.0)
            return norm_score.squeeze(-1)

    clip_source = str(_MODELS / "clip-vit-base-patch32")
    if not (Path(clip_source) / "config.json").exists():
        clip_source = "openai/clip-vit-base-patch32"

    print(f"📦 正在加载 CLIP 视觉底座: {clip_source} ...")
    clip_model = CLIPModel.from_pretrained(clip_source, return_dict=False).eval()
    state = torch.load(str(WEIGHTS_PATH), map_location="cpu")

    model = StandardAestheticONNX(clip_model, state["weight"], state["bias"]).eval()
    dummy_input = torch.randn(1, 3, 224, 224)

    print(f"⚡ 正在导出标准通用 ONNX 模型至: {OUTPUT_ONNX_PATH} ...")
    torch.onnx.export(
        model,
        (dummy_input,),
        str(OUTPUT_ONNX_PATH),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["pixel_values"],
        output_names=["aesthetic_score"],
        dynamic_axes={"pixel_values": {0: "batch_size"}, "aesthetic_score": {0: "batch_size"}},
        dynamo=False,
    )
    print(f"🎉 导出成功！文件大小: {OUTPUT_ONNX_PATH.stat().st_size / 1024 / 1024:.2f} MB")


def main():
    download_weights()
    fuse_and_export_onnx()


if __name__ == "__main__":
    main()
