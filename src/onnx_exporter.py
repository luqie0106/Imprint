"""
onnx_exporter.py — 将 CLIP 视觉编码器与个人 AestheticMLP 熔铸为单个 ONNX 模型
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:
    import torch
    import torch.nn as nn
    from transformers import CLIPModel
    TORCH_EXPORT_AVAILABLE = True
except ImportError:
    TORCH_EXPORT_AVAILABLE = False


# AestheticMLP 结构（与 trainer / filter 保持严格一致）
if TORCH_EXPORT_AVAILABLE:
    class _AestheticMLP(nn.Module):
        def __init__(self, input_dim: int = 512):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 2),
            )

        def forward(self, x):
            return self.net(x)

    class _CombinedAestheticModel(nn.Module):
        def __init__(self, clip_model: CLIPModel, mlp: _AestheticMLP):
            super().__init__()
            self.clip = clip_model
            self.mlp = mlp

        def forward(self, pixel_values):
            # 1. CLIP 提取视觉特征
            vision_outputs = self.clip.vision_model(pixel_values=pixel_values, return_dict=False)
            pooled_output = vision_outputs[1]  # pooler_output
            features = self.clip.visual_projection(pooled_output)

            # 2. L2 归一化
            features = features / features.norm(dim=-1, keepdim=True)
            # 3. 过 MLP 分类头
            logits = self.mlp(features)
            # 4. Softmax 输出 Like 概率 (索引 1)
            probs = torch.softmax(logits, dim=1)
            return probs[:, 1]


def export_to_onnx(
    project_root: Path | None = None,
    mlp_path: Path | None = None,
    onnx_path: Path | None = None,
    clip_source: str | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> Path:
    """
    熔铸 CLIP + AestheticMLP 并导出为单一 ONNX 模型。
    返回导出的 ONNX 文件路径。
    """
    if not TORCH_EXPORT_AVAILABLE:
        raise RuntimeError("导出 ONNX 需要 PyTorch 和 transformers 依赖，当前环境中未安装。")

    root = project_root or PROJECT_ROOT
    if mlp_path is None:
        mlp_path = root / "aesthetic_mlp.pth"
    if onnx_path is None:
        onnx_path = root / "photo_sort_model.onnx"

    if not mlp_path.exists():
        raise FileNotFoundError(f"找不到 MLP 权重文件: {mlp_path}，请先训练模型。")

    # 确定 CLIP 来源（优先本地 models/clip-vit-base-patch32）
    if clip_source is None:
        local_clip_dir = root / "models" / "clip-vit-base-patch32"
        if local_clip_dir.exists() and (local_clip_dir / "config.json").exists():
            clip_source = str(local_clip_dir)
        else:
            clip_source = "openai/clip-vit-base-patch32"

    def _notify(msg: str):
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg)

    _notify(f"正在加载 CLIP 视觉模型: {clip_source} ...")
    clip_model = CLIPModel.from_pretrained(clip_source, return_dict=False)
    clip_model.eval()

    _notify(f"正在加载 AestheticMLP 权重: {mlp_path.name} ...")
    mlp = _AestheticMLP(input_dim=512)
    mlp.load_state_dict(torch.load(str(mlp_path), map_location="cpu", weights_only=True))
    mlp.eval()

    _notify("正在构建端到端联合模型...")
    combined = _CombinedAestheticModel(clip_model, mlp)
    combined.eval()

    dummy_input = torch.randn(1, 3, 224, 224)

    _notify(f"正在导出 ONNX 模型至: {onnx_path.name} ...")
    torch.onnx.export(
        combined,
        (dummy_input,),
        str(onnx_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["pixel_values"],
        output_names=["like_prob"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "like_prob": {0: "batch_size"},
        },
    )

    _notify(f"✅ ONNX 熔铸成功！模型已保存至: {onnx_path}")
    return onnx_path
