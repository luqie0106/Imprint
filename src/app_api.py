"""
app_api.py — Photo Sort FastAPI Sidecar 后端服务
为 Tauri 2.0 前端提供 HTTP + SSE 接口支持。
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import AsyncGenerator, Literal, Optional

# 确保标准输出为 UTF-8 编码，防止 Windows GBK 环境下 Emoji 引发 UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# 确保 src 目录在 sys.path 中
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from burst_filter import BurstFilter
from model_manager import (
    PROJECT_ROOT,
    MODELS_DIR,
    check_all_models,
    download_clip_l14_model,
    download_clip_model,
    get_active_model_mode,
    set_active_model_mode,
    get_resolved_mlp_path,
    get_resolved_mlp_l14_path,
    get_resolved_standard_onnx_path,
    get_resolved_standard_l14_onnx_path,
)
from onnx_exporter import fuse_mlp_weights_to_onnx, export_to_onnx, TORCH_EXPORT_AVAILABLE

import io
import onnx
import onnxruntime as ort
import numpy as np
from PIL import Image
import rawpy

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pass

try:
    import pillow_jxl
except Exception:
    pass

app = FastAPI(title="Imprint API", version="2.0.5")

# 配置 CORS 中间件，允许 Tauri 桌面端以及本地开发环境请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_free_port() -> int:
    """获取一个随机可用的本地 TCP 端口"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ══════════════════════════════════════════════════════════════════════════════
# 数据模型定义
# ══════════════════════════════════════════════════════════════════════════════


class BurstRequest(BaseModel):
    input_dir: str
    gap_seconds: float = 1.5
    max_hamming_distance: int = 12
    review_subdir: str = "审查_连拍淘汰"
    keep_count: int = 1
    max_workers: int = 4
    use_gpu: bool = False


class DownloadModelRequest(BaseModel):
    model: Literal["clip_b32", "clip_l14"]
    use_mirror: bool = True


class SetModeRequest(BaseModel):
    mode: Literal["standard", "standard_l14", "custom", "custom_l14"]


class TrainerRequest(BaseModel):
    photos_dir: str
    model_type: Literal["standard", "l14", "b32", "standard_l14", "custom_l14"] = "standard"
    epochs: int = 15
    lr: float = 1e-3


# ══════════════════════════════════════════════════════════════════════════════
# 接口一：POST /api/burst/run (SSE 连拍筛选)
# ══════════════════════════════════════════════════════════════════════════════


@app.post("/api/burst/run")
async def run_burst(req: BurstRequest):
    """
    执行连拍照片筛选优选，通过 SSE 流式返回实时进度、处理结果或异常信息。
    """
    queue: asyncio.Queue[dict] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def on_progress(msg: str):
        loop.call_soon_threadsafe(queue.put_nowait, {"type": "progress", "msg": msg})

    async def run_in_thread():
        try:
            target_path = Path(req.input_dir)
            if not target_path.exists() or not target_path.is_dir():
                await queue.put({"type": "error", "msg": f"目标目录不存在: {req.input_dir}"})
                return

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                flt = BurstFilter(
                    gap_seconds=req.gap_seconds,
                    max_hamming_distance=req.max_hamming_distance,
                    review_subdir=req.review_subdir,
                    keep_count=req.keep_count,
                    max_workers=req.max_workers,
                    use_gpu=req.use_gpu,
                    progress_callback=on_progress,
                )
                result = await loop.run_in_executor(pool, lambda: flt.run(target_path))

                # 转换 BurstFilterResult 字段
                done_payload = {
                    "type": "done",
                    "total": getattr(result, "total", 0),
                    "burst_groups": getattr(result, "burst_groups", 0),
                    "moved": getattr(result, "moved", 0),
                    "skipped_single": getattr(result, "skipped_single", 0),
                    "errors": getattr(result, "errors", []),
                    "review_dir": str(result.review_dir) if getattr(result, "review_dir", None) else "",
                }
                await queue.put(done_payload)
        except Exception as exc:
            await queue.put({"type": "error", "msg": f"连拍筛选执行异常: {str(exc)}"})

    asyncio.create_task(run_in_thread())

    async def event_stream() -> AsyncGenerator[str, None]:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=10.0)
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                if event.get("type") in ("done", "error"):
                    break
            except asyncio.TimeoutError:
                # SSE 注释行：不触发前端 onmessage，仅用于保持 TCP 连接活跃
                yield ": keep-alive\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ══════════════════════════════════════════════════════════════════════════════
# 接口二：GET /api/models/status (模型状态检测)
# ══════════════════════════════════════════════════════════════════════════════


def _get_gpu_info() -> tuple[bool, str]:
    try:
        providers = ort.get_available_providers()
        if "CoreMLExecutionProvider" in providers:
            return True, "Apple Silicon Metal / 神经引擎 (CoreML)"
        if "DmlExecutionProvider" in providers:
            return True, "DirectX 12 显卡硬件加速 (DirectML)"
        if "CUDAExecutionProvider" in providers:
            return True, "NVIDIA 显卡硬件加速 (CUDA)"
        if "ROCMExecutionProvider" in providers:
            return True, "AMD 显卡硬件加速 (ROCm)"
    except Exception:
        pass
    return False, "CPU 多核心并行计算"


@app.get("/api/models/status")
async def get_models_status():
    """获取所有模型就绪状态、GPU 加速检测及当前激活模式"""
    gpu_available, gpu_name = _get_gpu_info()
    try:
        active_mode = get_active_model_mode()
        status = check_all_models()
        return {
            "mode": active_mode,
            "gpu_available": gpu_available,
            "gpu_name": gpu_name,
            "clip_b32_ready": status.clip_ready,
            "clip_l14_ready": status.clip_l14_ready,
            "standard_onnx_ready": status.standard_onnx_ready,
            "standard_l14_onnx_ready": status.standard_l14_onnx_ready,
            "custom_onnx_ready": status.custom_onnx_ready,
            "custom_l14_onnx_ready": status.custom_l14_onnx_ready,
            "mlp_ready": status.mlp_ready,
            "mlp_path": status.mlp_path,
            "mlp_l14_ready": status.mlp_l14_ready,
            "mlp_l14_path": status.mlp_l14_path,
        }
    except Exception as exc:
        return JSONResponse(
            status_code=200,
            content={
                "mode": "standard",
                "gpu_available": gpu_available,
                "gpu_name": gpu_name,
                "clip_b32_ready": False,
                "clip_l14_ready": False,
                "standard_onnx_ready": False,
                "standard_l14_onnx_ready": False,
                "custom_onnx_ready": False,
                "custom_l14_onnx_ready": False,
                "mlp_ready": False,
                "mlp_path": "",
                "mlp_l14_ready": False,
                "mlp_l14_path": "",
                "error": str(exc),
            },
        )


# ══════════════════════════════════════════════════════════════════════════════
# 接口三：POST /api/models/download (SSE 模型下载)
# ══════════════════════════════════════════════════════════════════════════════


@app.post("/api/models/download")
async def download_model(req: DownloadModelRequest):
    """
    下载 CLIP 基础模型，通过 SSE 返回实时下载进度。
    """
    queue: asyncio.Queue[dict] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def on_progress(msg: str, pct: float):
        loop.call_soon_threadsafe(
            queue.put_nowait,
            {"type": "progress", "msg": msg, "pct": round(pct, 4)},
        )

    async def run_in_thread():
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                if req.model == "clip_b32":
                    success = await loop.run_in_executor(
                        pool, lambda: download_clip_model(use_mirror=req.use_mirror, progress_callback=on_progress)
                    )
                else:
                    success = await loop.run_in_executor(
                        pool, lambda: download_clip_l14_model(use_mirror=req.use_mirror, progress_callback=on_progress)
                    )

                if success:
                    await queue.put({"type": "done", "success": True, "msg": "模型下载完成！"})
                else:
                    await queue.put({"type": "error", "msg": "模型下载失败或已被取消。"})
        except Exception as exc:
            await queue.put({"type": "error", "msg": f"模型下载出现异常: {str(exc)}"})

    asyncio.create_task(run_in_thread())

    async def event_stream() -> AsyncGenerator[str, None]:
        while True:
            event = await queue.get()
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            if event.get("type") in ("done", "error"):
                break

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ══════════════════════════════════════════════════════════════════════════════
# 接口四：POST /api/models/set-mode (设置当前模型模式)
# ══════════════════════════════════════════════════════════════════════════════


@app.post("/api/models/set-mode")
async def set_model_mode(req: SetModeRequest):
    """切换并持久化当前激活的美学评分模型模式"""
    try:
        set_active_model_mode(req.mode)
        return {"ok": True, "mode": req.mode}
    except Exception as exc:
        return JSONResponse(status_code=400, content={"ok": False, "error": str(exc)})


# ══════════════════════════════════════════════════════════════════════════════
# 接口：POST /api/models/fuse-onnx (SSE 个人 PTH 权重熔铸为 ONNX 模型)
# ══════════════════════════════════════════════════════════════════════════════


class FuseOnnxRequest(BaseModel):
    model_type: Literal["b32", "l14"] = "b32"


@app.post("/api/models/fuse-onnx")
async def fuse_custom_onnx(req: FuseOnnxRequest):
    """
    将已训练的 aesthetic_mlp.pth 熔铸为 ONNX 模型，通过 SSE 返回实时进度。
    需要当前 Python 环境已安装 torch 和 transformers。
    """
    queue: asyncio.Queue[dict] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    if not TORCH_EXPORT_AVAILABLE:
        async def err_stream():
            yield f"data: {json.dumps({'type': 'error', 'msg': '熔铸 ONNX 需要 PyTorch 与 transformers，当前运行环境未安装。请在源码开发环境下运行此功能。'}, ensure_ascii=False)}\n\n"
        return StreamingResponse(err_stream(), media_type="text/event-stream")

    def on_progress(msg: str):
        loop.call_soon_threadsafe(
            queue.put_nowait,
            {"type": "progress", "msg": msg},
        )

    async def run_in_thread():
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                if req.model_type == "b32":
                    mlp_p = get_resolved_mlp_path()
                    if not mlp_p:
                        await queue.put({"type": "error", "msg": "未找到 aesthetic_mlp.pth，请先完成偏好训练。"})
                        return
                    result = await loop.run_in_executor(
                        pool,
                        lambda: export_to_onnx(
                            project_root=PROJECT_ROOT,
                            mlp_path=mlp_p,
                            progress_callback=on_progress,
                        ),
                    )
                else:
                    # L14 custom: 调用与 b32 相同的 export_to_onnx，但指向 L14 权重和模型路径
                    from pathlib import Path
                    mlp_p = get_resolved_mlp_l14_path()
                    if not mlp_p:
                        await queue.put({"type": "error", "msg": "未找到 aesthetic_mlp_l14.pth，请先完成 L14 偏好训练。"})
                        return
                    onnx_out = PROJECT_ROOT / "models" / "custom_aesthetic_l14_model.onnx"
                    result = await loop.run_in_executor(
                        pool,
                        lambda: export_to_onnx(
                            project_root=PROJECT_ROOT,
                            mlp_path=mlp_p,
                            onnx_path=onnx_out,
                            clip_source=str(PROJECT_ROOT / "models" / "clip-vit-large-patch14"),
                            progress_callback=on_progress,
                        ),
                    )
                await queue.put({
                    "type": "done",
                    "msg": f"✅ ONNX 熔铸完成：{result.name}",
                    "onnx_path": str(result),
                })
        except Exception as exc:
            await queue.put({"type": "error", "msg": f"熔铸失败: {str(exc)}"})

    asyncio.create_task(run_in_thread())

    async def event_stream() -> AsyncGenerator[str, None]:
        while True:
            event = await queue.get()
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            if event.get("type") in ("done", "error"):
                break

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


# ══════════════════════════════════════════════════════════════════════════════
# 接口五：POST /api/trainer/run (SSE 偏好模型微调与 ONNX 熔铸)
# ══════════════════════════════════════════════════════════════════════════════

RAW_SUFFIXES = {
    ".nef", ".nrw", ".arw", ".srf", ".sr2", ".cr2", ".cr3", ".crw",
    ".rw2", ".raw", ".dng", ".raf", ".orf", ".ori", ".pef", ".ptx",
    ".3fr", ".fff", ".iiq", ".srw", ".x3f", ".mrw", ".gpr", ".erf", ".mef", ".mos",
}
STANDARD_SUFFIXES = {
    ".jpg", ".jpeg", ".jpe", ".jxl", ".hif", ".heif", ".heic", ".png", ".webp", ".tiff", ".tif", ".bmp"
}
ALL_PHOTO_SUFFIXES = RAW_SUFFIXES | STANDARD_SUFFIXES


def _preprocess_photo_for_clip(path: Path) -> np.ndarray | None:
    img = None
    suffix = path.suffix.lower()
    if suffix in RAW_SUFFIXES:
        try:
            with rawpy.imread(str(path)) as raw:
                thumb = raw.extract_thumb()
            img = Image.open(io.BytesIO(thumb.data)).convert("RGB")
        except Exception:
            try:
                with rawpy.imread(str(path)) as raw:
                    arr = raw.postprocess(half_size=True, use_camera_wb=True, output_bps=8)
                img = Image.fromarray(arr).convert("RGB")
            except Exception:
                pass
    if img is None:
        try:
            with Image.open(path) as im:
                img = im.convert("RGB")
        except Exception:
            return None

    # CLIP 图像预处理: 等比缩放短边至 224 并中心裁剪
    w, h = img.size
    scale = 224.0 / min(w, h)
    new_w, new_h = max(224, int(round(w * scale))), max(224, int(round(h * scale)))
    img_resized = img.resize((new_w, new_h), Image.Resampling.BICUBIC)

    left = (new_w - 224) // 2
    top = (new_h - 224) // 2
    img_cropped = img_resized.crop((left, top, left + 224, top + 224))

    # 标准 CLIP 归一化
    arr = np.array(img_cropped, dtype=np.float32) / 255.0
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    arr = (arr - mean) / std
    return arr.transpose(2, 0, 1)


@app.post("/api/trainer/run")
async def run_trainer(req: TrainerRequest):
    """
    启动纯原生 (ONNX + NumPy) 偏好微调训练与 ONNX 熔铸进程，零外部 Python / PyTorch 依赖。
    """
    queue: asyncio.Queue[dict] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    data_dir = Path(req.photos_dir)
    if not data_dir.exists() or not data_dir.is_dir():
        async def err_stream():
            yield f"data: {json.dumps({'type': 'error', 'msg': f'训练样本目录不存在: {req.photos_dir}'}, ensure_ascii=False)}\n\n"
        return StreamingResponse(err_stream(), media_type="text/event-stream")

    def _notify(msg: str, pct: float | None = None):
        loop.call_soon_threadsafe(
            queue.put_nowait,
            {"type": "progress", "msg": msg, "pct": pct},
        )

    async def run_training_worker():
        try:
            _notify("🚀 启动审美偏好微调与 ONNX 熔铸引擎 (纯原生极速引擎)...")

            # 1. 扫描标注样本
            samples: list[tuple[Path, int]] = []
            for label, dname in ((1, "like"), (0, "dislike")):
                sub_dir = data_dir / dname
                if sub_dir.exists() and sub_dir.is_dir():
                    for p in sub_dir.iterdir():
                        if p.is_file() and p.suffix.lower() in ALL_PHOTO_SUFFIXES:
                            samples.append((p, label))

            _notify(f"✅ 成功扫描到 {len(samples)} 张标注样本照片 (like/dislike)")
            if len(samples) == 0:
                raise ValueError("数据集为空，请确保在所选目录下包含 like/ 与 dislike/ 子文件夹并放置标注图片")

            like_count = sum(1 for _, l in samples if l == 1)
            dislike_count = len(samples) - like_count
            _notify(f"📊 样本分布: 喜欢 (Like) {like_count} 张 | 不喜欢 (Dislike) {dislike_count} 张")

            # 2. 确定底座模型
            backbone_key = "l14" if req.model_type in ("l14", "standard_l14", "custom_l14") else "b32"
            dim = 768 if backbone_key == "l14" else 512
            backbone_desc = "CLIP ViT-L/14 (专业高精 · 768维)" if backbone_key == "l14" else "CLIP ViT-B/32 (标准极速 · 512维)"
            out_onnx_name = "custom_aesthetic_l14_model.onnx" if backbone_key == "l14" else "custom_aesthetic_model.onnx"

            if backbone_key == "l14":
                base_onnx_path = get_resolved_standard_l14_onnx_path()
            else:
                base_onnx_path = get_resolved_standard_onnx_path()

            if not base_onnx_path or not base_onnx_path.exists():
                raise FileNotFoundError(f"未找到基础视觉底座模型 ({backbone_desc})，请先在“模型管理”中下载或就绪底座。")

            _notify(f"📦 正在加载视觉底座: {backbone_desc} ...")

            def _extract_and_train():
                base_model = onnx.load(str(base_onnx_path))
                output_names = [o.name for o in base_model.graph.output]
                if "/Div_output_0" not in output_names:
                    div_out = onnx.helper.make_tensor_value_info("/Div_output_0", onnx.TensorProto.FLOAT, [None, dim])
                    base_model.graph.output.append(div_out)

                model_bytes = base_model.SerializeToString()
                providers = []
                available = ort.get_available_providers()
                for p in ["CoreMLExecutionProvider", "DmlExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]:
                    if p in available:
                        providers.append(p)
                session = ort.InferenceSession(model_bytes, providers=providers)
                active_provider = session.get_providers()[0]
                _notify(f"⚡ 特征提取硬件加速后端: {active_provider}")

                # 4. 批量预处理并提取特征向量
                _notify("🎯 正在提取全量样本特征向量...")
                all_feats: list[np.ndarray] = []
                all_labels: list[int] = []
                batch_imgs: list[np.ndarray] = []
                batch_labels: list[int] = []

                batch_size = 16
                for idx, (img_path, label) in enumerate(samples):
                    arr = _preprocess_photo_for_clip(img_path)
                    if arr is not None:
                        batch_imgs.append(arr)
                        batch_labels.append(label)

                    if len(batch_imgs) >= batch_size or (idx == len(samples) - 1 and batch_imgs):
                        batch_tensor = np.stack(batch_imgs, axis=0).astype(np.float32)
                        feats = session.run(["/Div_output_0"], {"pixel_values": batch_tensor})[0]
                        all_feats.append(feats)
                        all_labels.extend(batch_labels)
                        batch_imgs.clear()
                        batch_labels.clear()

                        pct = round((idx + 1) / len(samples) * 0.5, 4)
                        _notify(f"  特征提取进度: [{idx + 1}/{len(samples)}]", pct=pct)

                if not all_feats:
                    raise RuntimeError("无法成功解析样本中的任何图片，请检查图片格式是否损坏")

                X = np.concatenate(all_feats, axis=0)  # (N, dim)
                y = np.array(all_labels, dtype=np.int64)  # (N,)
                N = len(y)
                _notify(f"✅ 特征提取完成！共计 {N} 个样本特征向量 (维度: {dim})")

                # 5. 纯 NumPy Adam 训练 2 层 MLP 分类头
                _notify(f"🔥 启动神经网络微调训练 (共 {req.epochs} 轮)...")
                hidden = 256
                np.random.seed(42)
                W1 = (np.random.randn(hidden, dim).astype(np.float32) * np.sqrt(2.0 / dim))
                b1 = np.zeros(hidden, dtype=np.float32)
                W2 = (np.random.randn(2, hidden).astype(np.float32) * np.sqrt(2.0 / hidden))
                b2 = np.zeros(2, dtype=np.float32)

                mW1, vW1 = np.zeros_like(W1), np.zeros_like(W1)
                mb1, vb1 = np.zeros_like(b1), np.zeros_like(b1)
                mW2, vW2 = np.zeros_like(W2), np.zeros_like(W2)
                mb2, vb2 = np.zeros_like(b2), np.zeros_like(b2)

                lr = float(req.lr) if req.lr > 0 else 0.001
                beta1, beta2, eps = 0.9, 0.999, 1e-8
                epochs = max(1, req.epochs)
                train_batch_sz = min(16, N)
                t = 0

                for ep in range(epochs):
                    perm = np.random.permutation(N)
                    total_loss, correct = 0.0, 0
                    for i in range(0, N, train_batch_sz):
                        t += 1
                        b_idx = perm[i : i + train_batch_sz]
                        xb, yb = X[b_idx], y[b_idx]
                        B = len(yb)

                        # 前向传播
                        z1 = np.dot(xb, W1.T) + b1
                        a1 = np.maximum(0, z1)
                        z2 = np.dot(a1, W2.T) + b2
                        # Softmax
                        exp_z2 = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
                        probs = exp_z2 / (np.sum(exp_z2, axis=1, keepdims=True) + 1e-12)

                        loss = -np.mean(np.log(probs[np.arange(B), yb] + 1e-12))
                        total_loss += loss * B
                        correct += int(np.sum(np.argmax(probs, axis=1) == yb))

                        # 反向传播求梯度
                        dz2 = probs.copy()
                        dz2[np.arange(B), yb] -= 1.0
                        dz2 /= B

                        dW2 = np.dot(dz2.T, a1)
                        db2 = np.sum(dz2, axis=0)

                        da1 = np.dot(dz2, W2)
                        dz1 = da1 * (z1 > 0)

                        dW1 = np.dot(dz1.T, xb)
                        db1 = np.sum(dz1, axis=0)

                        # Adam 优化器参数更新
                        for p_arr, g_arr, m_arr, v_arr in [
                            (W1, dW1, mW1, vW1),
                            (b1, db1, mb1, vb1),
                            (W2, dW2, mW2, vW2),
                            (b2, db2, mb2, vb2),
                        ]:
                            m_arr[:] = beta1 * m_arr + (1 - beta1) * g_arr
                            v_arr[:] = beta2 * v_arr + (1 - beta2) * (g_arr ** 2)
                            m_hat = m_arr / (1.0 - beta1 ** t)
                            v_hat = v_arr / (1.0 - beta2 ** t)
                            p_arr -= lr * m_hat / (np.sqrt(v_hat) + eps)

                    ep_loss = total_loss / max(1, N)
                    ep_acc = correct / max(1, N)
                    train_pct = 0.5 + round((ep + 1) / epochs * 0.45, 4)
                    _notify(
                        f"  Epoch [{ep+1:02d}/{epochs:02d}] Loss: {ep_loss:.4f} 准确率: {ep_acc*100:.1f}%",
                        pct=train_pct,
                    )

                # 6. 熔铸为单个 ONNX 模型
                _notify(f"⚡ 正在将专属微调权重直接注入视觉底座生成 ONNX ({out_onnx_name})...")
                MODELS_DIR.mkdir(parents=True, exist_ok=True)
                out_onnx_path = MODELS_DIR / out_onnx_name
                fuse_mlp_weights_to_onnx(base_onnx_path, out_onnx_path, W1, b1, W2, b2)

                if backbone_key == "b32":
                    legacy_path = PROJECT_ROOT / "photo_sort_model.onnx"
                    try:
                        import shutil
                        shutil.copy2(str(out_onnx_path), str(legacy_path))
                    except Exception:
                        pass

                _notify(f"🎉 ONNX 模型熔铸完毕！文件已就绪: models/{out_onnx_name} ({out_onnx_path.stat().st_size/(1024*1024):.1f} MB)")
                return out_onnx_path

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                await loop.run_in_executor(pool, _extract_and_train)

            await queue.put({
                "type": "done",
                "success": True,
                "msg": "🎉 专属审美偏好模型微调与 ONNX 熔铸成功！模型已保存在 models/ 目录下，可即刻在“模型管理”中选用！",
            })
        except Exception as exc:
            await queue.put({"type": "error", "msg": f"训练过程发生异常: {str(exc)}"})

    asyncio.create_task(run_training_worker())

    async def event_stream() -> AsyncGenerator[str, None]:
        while True:
            event = await queue.get()
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            if event.get("type") in ("done", "error"):
                break

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ══════════════════════════════════════════════════════════════════════════════
# 主启动入口
# ══════════════════════════════════════════════════════════════════════════════


def main():
    # 确定监听端口：优先环境变量 APP_PORT，否则随机分配可用端口
    port_env = os.environ.get("APP_PORT")
    if port_env and port_env.isdigit() and int(port_env) > 0:
        port = int(port_env)
    else:
        port = get_free_port()

    # 向 stdout 打印单行 JSON 端口信息，供 Tauri Rust 进程读取
    port_json = json.dumps({"port": port})
    print(port_json, flush=True)
    sys.stdout.flush()

    # 启动 uvicorn
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
