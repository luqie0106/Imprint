"""
onnx_exporter.py — 将 CLIP 视觉编码器与个人 AestheticMLP 熔铸为单个 ONNX 模型
支持纯 ONNX 拓扑图直接注入（零 PyTorch 依赖）以及 PyTorch 导出兼容模式。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:
    import onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    from transformers import CLIPModel
    TORCH_EXPORT_AVAILABLE = True
except ImportError:
    TORCH_EXPORT_AVAILABLE = False


def fuse_mlp_weights_to_onnx(
    base_onnx_path: Path | str,
    out_onnx_path: Path | str,
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: np.ndarray,
) -> Path:
    """
    使用纯 ONNX Protobuf 库将训练好的 MLP 权重直接熔铸进基础 CLIP ONNX 模型中。
    无需 PyTorch 依赖，生成端到端硬件加速的 custom_aesthetic_model.onnx。
    """
    if not ONNX_AVAILABLE:
        raise RuntimeError("缺少 onnx 依赖库，请确保已安装 onnx")

    base_model = onnx.load(str(base_onnx_path))
    out_onnx_path = Path(out_onnx_path)
    out_onnx_path.parent.mkdir(parents=True, exist_ok=True)

    # 归一化特征输出节点名 (CLIP 视觉投影后 L2 归一化输出)
    feat_output_name = "/Div_output_0"

    # 过滤掉原有标准线性头或旧 MLP 节点
    old_head_node_names = {
        "/linear/Gemm", "/Constant", "/Sub", "/Constant_1", "/Div_1",
        "/Constant_2", "/Constant_3", "/Clip", "/Constant_4", "/Squeeze",
        "/mlp/net.0/Gemm", "/mlp/net.1/Relu", "/mlp/net.3/Gemm", "/mlp/Softmax", "/mlp/Gather",
    }
    new_nodes = [
        n for n in base_model.graph.node
        if n.name not in old_head_node_names and n.output != ["aesthetic_score"] and n.output != ["like_prob"]
    ]

    hidden_dim, in_dim = W1.shape
    out_dim, _ = W2.shape

    init_W1 = onnx.helper.make_tensor("mlp.net.0.weight", onnx.TensorProto.FLOAT, [hidden_dim, in_dim], W1.astype(np.float32).flatten().tolist())
    init_b1 = onnx.helper.make_tensor("mlp.net.0.bias", onnx.TensorProto.FLOAT, [hidden_dim], b1.astype(np.float32).flatten().tolist())
    init_W2 = onnx.helper.make_tensor("mlp.net.3.weight", onnx.TensorProto.FLOAT, [out_dim, hidden_dim], W2.astype(np.float32).flatten().tolist())
    init_b2 = onnx.helper.make_tensor("mlp.net.3.bias", onnx.TensorProto.FLOAT, [out_dim], b2.astype(np.float32).flatten().tolist())
    init_gather_idx = onnx.helper.make_tensor("mlp_gather_idx_1", onnx.TensorProto.INT64, [], [1])

    new_inits = [
        init for init in base_model.graph.initializer
        if not init.name.startswith(("linear.", "mlp.")) and init.name != "mlp_gather_idx_1"
    ]
    new_inits.extend([init_W1, init_b1, init_W2, init_b2, init_gather_idx])

    # 组装 2 层 MLP 分类头 + Softmax + 抽取 Like 概率 (index 1)
    node_gemm1 = onnx.helper.make_node(
        "Gemm", [feat_output_name, "mlp.net.0.weight", "mlp.net.0.bias"],
        ["/mlp/net.0/Gemm_output_0"], name="/mlp/net.0/Gemm", transB=1
    )
    node_relu = onnx.helper.make_node(
        "Relu", ["/mlp/net.0/Gemm_output_0"],
        ["/mlp/net.1/Relu_output_0"], name="/mlp/net.1/Relu"
    )
    node_gemm2 = onnx.helper.make_node(
        "Gemm", ["/mlp/net.1/Relu_output_0", "mlp.net.3.weight", "mlp.net.3.bias"],
        ["/mlp/net.3/Gemm_output_0"], name="/mlp/net.3/Gemm", transB=1
    )
    node_softmax = onnx.helper.make_node(
        "Softmax", ["/mlp/net.3/Gemm_output_0"],
        ["/mlp/Softmax_output_0"], name="/mlp/Softmax", axis=-1
    )
    node_gather = onnx.helper.make_node(
        "Gather", ["/mlp/Softmax_output_0", "mlp_gather_idx_1"],
        ["like_prob"], name="/mlp/Gather", axis=1
    )

    new_nodes.extend([node_gemm1, node_relu, node_gemm2, node_softmax, node_gather])
    new_outputs = [onnx.helper.make_tensor_value_info("like_prob", onnx.TensorProto.FLOAT, [None])]

    new_graph = onnx.helper.make_graph(
        new_nodes,
        base_model.graph.name,
        base_model.graph.input,
        new_outputs,
        new_inits,
        value_info=base_model.graph.value_info,
    )
    new_model = onnx.helper.make_model(new_graph, opset_imports=base_model.opset_import)
    onnx.save(new_model, str(out_onnx_path))
    return out_onnx_path


# AestheticMLP 结构（兼容旧 PyTorch 模式）
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
            pooled_output = vision_outputs[1]
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
    """
    root = project_root or PROJECT_ROOT
    if mlp_path is None:
        if (root / "models" / "aesthetic_mlp.pth").exists():
            mlp_path = root / "models" / "aesthetic_mlp.pth"
        else:
            mlp_path = root / "aesthetic_mlp.pth"
    if onnx_path is None:
        onnx_path = root / "models" / "custom_aesthetic_model.onnx"

    if not mlp_path.exists():
        raise FileNotFoundError(f"找不到 MLP 权重文件: {mlp_path}，请先训练模型。")

    def _notify(msg: str):
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg)

    # 如果有标准 ONNX 底座，优先走纯 ONNX 极速熔铸
    std_onnx = root / "models" / "standard_aesthetic_model.onnx"
    if std_onnx.exists() and ONNX_AVAILABLE:
        try:
            _notify(f"正在通过 ONNX 图注入熔铸权重: {mlp_path.name} ...")
            # 尝试加载权重字典
            if TORCH_EXPORT_AVAILABLE:
                state = torch.load(str(mlp_path), map_location="cpu", weights_only=True)
                w1 = state["net.0.weight"].detach().cpu().numpy()
                b1 = state["net.0.bias"].detach().cpu().numpy()
                w2 = state["net.3.weight"].detach().cpu().numpy()
                b2 = state["net.3.bias"].detach().cpu().numpy()
                fuse_mlp_weights_to_onnx(std_onnx, onnx_path, w1, b1, w2, b2)
                _notify(f"✅ ONNX 模型极速导出成功: {onnx_path.name}")
                return onnx_path
        except Exception as e:
            _notify(f"纯 ONNX 注入失败 ({e})，尝试 PyTorch 导出回退...")

    if not TORCH_EXPORT_AVAILABLE:
        raise RuntimeError("导出 ONNX 需要 PyTorch 和 transformers 依赖，当前环境中未安装。")

    # 确定 CLIP 来源
    if clip_source is None:
        local_clip_dir = root / "models" / "clip-vit-base-patch32"
        if local_clip_dir.exists() and (local_clip_dir / "config.json").exists():
            clip_source = str(local_clip_dir)
        else:
            clip_source = "openai/clip-vit-base-patch32"

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
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
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
        dynamo=False,
    )
    legacy_path = root / "photo_sort_model.onnx"
    try:
        import shutil
        shutil.copy2(str(onnx_path), str(legacy_path))
    except Exception:
        pass

    _notify(f"✅ ONNX 模型导出成功: {onnx_path}")
    return onnx_path
