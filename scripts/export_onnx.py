#!/usr/bin/env python3
"""
export_onnx.py

合并 CLIP 视觉模型和我们自己训练的 AestheticMLP，
并将其导出为一个统一的 ONNX 模型。
"""

import sys
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    from transformers import CLIPModel
except ImportError:
    print("Error: 请在带有 torch 和 transformers 的环境中运行本脚本。")
    sys.exit(1)

# 与 trainer_gui.py 中相同的 MLP 结构
class AestheticMLP(nn.Module):
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

# 合并后的端到端模型
class CombinedAestheticModel(nn.Module):
    def __init__(self, clip_model: CLIPModel, mlp: AestheticMLP):
        super().__init__()
        self.clip = clip_model
        self.mlp = mlp

    def forward(self, pixel_values):
        # 1. 过 CLIP 提取特征：手动走底层模型，强制 return_dict=False 确保返回 tuple，这对于 ONNX tracing 最为安全
        vision_outputs = self.clip.vision_model(pixel_values=pixel_values, return_dict=False)
        pooled_output = vision_outputs[1]  # tuple 的第二个元素是 pooler_output
        features = self.clip.visual_projection(pooled_output)
        
        # 2. L2 归一化
        features = features / features.norm(dim=-1, keepdim=True)
        # 3. 过 MLP
        logits = self.mlp(features)
        # 4. Softmax 输出 Like 的概率 (类别 1)
        probs = torch.softmax(logits, dim=1)
        return probs[:, 1]  # 只返回 Like 概率

def main():
    root = Path(__file__).resolve().parent.parent
    mlp_path = root / "aesthetic_mlp.pth"
    onnx_path = root / "photo_sort_model.onnx"

    if not mlp_path.exists():
        print(f"Error: 找不到 {mlp_path}，请先运行 trainer_gui.py 训练。")
        sys.exit(1)

    print("加载 CLIP 视觉编码器...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", return_dict=False)
    clip_model.eval()

    print("加载 AestheticMLP...")
    mlp = AestheticMLP(input_dim=512)
    mlp.load_state_dict(torch.load(str(mlp_path), map_location="cpu", weights_only=True))
    mlp.eval()

    print("构建联合模型...")
    combined = CombinedAestheticModel(clip_model, mlp)
    combined.eval()

    # CLIP 的输入要求：(B, C, H, W) = (1, 3, 224, 224)
    dummy_input = torch.randn(1, 3, 224, 224)

    print(f"正在导出至 ONNX: {onnx_path}")
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
            "like_prob": {0: "batch_size"}
        }
    )

    print("✅ ONNX 导出成功！")

if __name__ == "__main__":
    main()
