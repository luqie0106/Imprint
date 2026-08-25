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
    check_all_models,
    download_clip_l14_model,
    download_clip_model,
    get_active_model_mode,
    set_active_model_mode,
    get_resolved_mlp_path,
    get_resolved_mlp_l14_path,
)
from onnx_exporter import export_to_onnx, TORCH_EXPORT_AVAILABLE

app = FastAPI(title="PhotoSort API", version="2.0.1")

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
# 接口二：GET /api/models/status (模型状态检测)
# ══════════════════════════════════════════════════════════════════════════════


@app.get("/api/models/status")
async def get_models_status():
    """获取所有模型就绪状态及当前激活模式"""
    try:
        active_mode = get_active_model_mode()
        status = check_all_models()
        return {
            "mode": active_mode,
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


@app.post("/api/trainer/run")
async def run_trainer(req: TrainerRequest):
    """
    启动偏好微调训练与 ONNX 熔铸进程，通过 SSE 返回实时日志和 Epoch 进度。
    """
    queue: asyncio.Queue[dict] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    data_dir = Path(req.photos_dir)
    if not data_dir.exists() or not data_dir.is_dir():
        async def err_stream():
            yield f"data: {json.dumps({'type': 'error', 'msg': f'训练样本目录不存在: {req.photos_dir}'}, ensure_ascii=False)}\n\n"
        return StreamingResponse(err_stream(), media_type="text/event-stream")

    backbone_key = "l14" if req.model_type in ("l14", "standard_l14", "custom_l14") else "b32"
    repo_id = "openai/clip-vit-base-patch32" if backbone_key == "b32" else "openai/clip-vit-large-patch14"
    dim = 512 if backbone_key == "b32" else 768
    pth_name = "aesthetic_mlp.pth" if backbone_key == "b32" else "aesthetic_mlp_l14.pth"
    onnx_name = "custom_aesthetic_model.onnx" if backbone_key == "b32" else "custom_aesthetic_l14_model.onnx"
    backbone_desc = "CLIP ViT-B/32 (标准极速 · 512维)" if backbone_key == "b32" else "CLIP ViT-L/14 (专业高精 · 768维)"

    training_script = f"""
import sys
from pathlib import Path
data_dir = Path(r"{data_dir}")
epochs = {req.epochs}
lr = {req.lr}
backbone_repo = "{repo_id}"
feat_dim = {dim}
pth_filename = "{pth_name}"
onnx_filename = "{onnx_name}"
project_root = Path(r"{PROJECT_ROOT}")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from transformers import CLIPProcessor, CLIPModel
    from PIL import Image
    import rawpy, io
except ImportError as err:
    print(f"❌ 运行环境缺少依赖: {{err}}\\n请确保所选 Python 环境已安装 torch, transformers, rawpy, Pillow", flush=True)
    sys.exit(1)

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if sys.platform == "darwin" and torch.backends.mps.is_available() else "cpu"))
if device.type == 'cuda':
    dev_title = f"NVIDIA CUDA 显卡硬件加速 ({{torch.cuda.get_device_name(0)}})"
elif device.type == 'mps':
    dev_title = "Apple Silicon Metal 显卡硬件加速 (MPS)"
else:
    dev_title = "CPU 多核心并行计算"

print(f"⚡ 深度学习加速设备: {{dev_title}}", flush=True)
print(f"📦 正在加载视觉主干底座: {backbone_desc} ...", flush=True)

models_local = project_root / "models"
local_clip_dir = models_local / ("clip-vit-base-patch32" if feat_dim == 512 else "clip-vit-large-patch14")
clip_src = str(local_clip_dir) if local_clip_dir.exists() and (local_clip_dir / "config.json").exists() else backbone_repo

clip_model = CLIPModel.from_pretrained(clip_src).to(device).eval()
for p in clip_model.parameters(): p.requires_grad_(False)
clip_processor = CLIPProcessor.from_pretrained(clip_src)

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pass

try:
    import pillow_jxl
except Exception:
    pass

RAW_SUFFIXES = {{
    ".nef", ".nrw", ".arw", ".srf", ".sr2", ".cr2", ".cr3", ".crw",
    ".rw2", ".raw", ".dng", ".raf", ".orf", ".ori", ".pef", ".ptx",
    ".3fr", ".fff", ".iiq", ".srw", ".x3f", ".mrw", ".gpr", ".erf", ".mef", ".mos",
}}
STANDARD_SUFFIXES = {{
    ".jpg", ".jpeg", ".jpe", ".jxl", ".hif", ".heif", ".heic", ".png", ".webp", ".tiff", ".tif", ".bmp"
}}
ALL_PHOTO_SUFFIXES = RAW_SUFFIXES | STANDARD_SUFFIXES

class RawDataset:
    def __init__(self, root):
        self.samples = []
        for l, dname in ((1, "like"), (0, "dislike")):
            d = root / dname
            if d.exists():
                for p in d.iterdir():
                    if p.is_file() and p.suffix.lower() in ALL_PHOTO_SUFFIXES:
                        self.samples.append((p, l))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        p, l = self.samples[idx]
        img = None
        if p.suffix.lower() in RAW_SUFFIXES:
            try:
                with rawpy.imread(str(p)) as raw: thumb = raw.extract_thumb()
                img = Image.open(io.BytesIO(thumb.data)).convert("RGB")
            except Exception:
                try:
                    with rawpy.imread(str(p)) as raw: arr = raw.postprocess(half_size=True, use_camera_wb=True, output_bps=8)
                    img = Image.fromarray(arr).convert("RGB")
                except Exception:
                    pass
        if img is None:
            try:
                with Image.open(p) as im:
                    img = im.convert("RGB")
            except Exception:
                img = Image.new("RGB", (224, 224), (0, 0, 0))
        inp = clip_processor(images=img, return_tensors="pt", padding=True)
        return {{k: v.squeeze(0) for k, v in inp.items()}}, l

def collate_fn(batch):
    inputs_list, labels = zip(*batch)
    return {{k: torch.stack([d[k] for d in inputs_list]) for k in inputs_list[0]}}, torch.tensor(labels, dtype=torch.long)

dataset = RawDataset(data_dir)
print(f"✅ 成功扫描到 {{len(dataset)}} 张标注样本照片 (like/dislike)", flush=True)
if len(dataset) == 0:
    print("❌ 数据集为空，请确保在所选目录下包含 like/ 与 dislike/ 子文件夹并放置标注图片", flush=True)
    sys.exit(1)

batch_sz = 16 if device.type in ('cuda', 'mps') else 8
dataloader = DataLoader(dataset, batch_size=batch_sz, shuffle=True, collate_fn=collate_fn)
mlp = nn.Sequential(nn.Linear(feat_dim, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 2)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mlp.parameters(), lr=lr)

print("🎯 正在执行特征提取与微调训练...", flush=True)
for ep in range(epochs):
    mlp.train()
    running_loss, correct, total = 0.0, 0, 0
    for b_in, labels in dataloader:
        b_in = {{k: v.to(device) for k, v in b_in.items()}}
        labels = labels.to(device)
        with torch.no_grad():
            vout = clip_model.vision_model(pixel_values=b_in['pixel_values'], return_dict=False)
            feat = clip_model.visual_projection(vout[1])
            feat = feat / feat.norm(dim=-1, keepdim=True)
        optimizer.zero_grad()
        out = mlp(feat)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, pred = torch.max(out.data, 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()
    print(f"  Epoch [{{ep+1:02d}}/{{epochs:02d}}] Loss: {{running_loss/max(1, len(dataloader)):.4f}} Acc: {{100*correct/max(1, total):.1f}}%", flush=True)

save_dir = project_root / "models"
save_dir.mkdir(parents=True, exist_ok=True)
save_path = save_dir / pth_filename
torch.save(mlp.state_dict(), save_path)
print(f"💾 专属模型头权重已保存至: models/{{save_path.name}}", flush=True)

print(f"⚡ 正在将专属模型头与视觉底座熔铸为 ONNX 硬件加速模型 ({{onnx_filename}})...", flush=True)
class Combined(nn.Module):
    def __init__(self, clip, mlp):
        super().__init__()
        self.clip = clip
        self.mlp = mlp
    def forward(self, pixel_values):
        vout = self.clip.vision_model(pixel_values=pixel_values, return_dict=False)
        feat = self.clip.visual_projection(vout[1])
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return torch.softmax(self.mlp(feat), dim=1)[:, 1]
comb = Combined(clip_model, mlp).eval()
onnx_path = save_dir / onnx_filename
torch.onnx.export(
    comb,
    (torch.randn(1, 3, 224, 224).to(device),),
    str(onnx_path),
    opset_version=14,
    input_names=["pixel_values"],
    output_names=["like_prob"],
    dynamic_axes={{"pixel_values": {{0: "batch_size"}}, "like_prob": {{0: "batch_size"}}}},
    dynamo=False,
)
if feat_dim == 512:
    legacy_onnx = project_root / "photo_sort_model.onnx"
    try:
        import shutil
        shutil.copy2(str(onnx_path), str(legacy_onnx))
    except Exception:
        pass
print(f"🎉 ONNX 模型熔铸完毕！文件已就绪: models/{{onnx_filename}} ({{onnx_path.stat().st_size/(1024*1024):.1f}} MB)", flush=True)
"""

    async def run_training_subprocess():
        try:
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {"type": "progress", "msg": "🚀 启动审美偏好微调训练进程..."},
            )

            def _proc_worker():
                proc = subprocess.Popen(
                    [sys.executable, "-c", training_script],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                for line in iter(proc.stdout.readline, ""):
                    l = line.strip()
                    if l:
                        pct = None
                        if "Epoch [" in l:
                            try:
                                cur_ep = int(l.split("[")[1].split("/")[0])
                                pct = round(cur_ep / max(1, req.epochs), 4)
                            except Exception:
                                pass
                        loop.call_soon_threadsafe(
                            queue.put_nowait,
                            {"type": "progress", "msg": l, "pct": pct},
                        )
                proc.stdout.close()
                ret = proc.wait()
                return ret

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                ret = await loop.run_in_executor(pool, _proc_worker)

            if ret == 0:
                await queue.put({
                    "type": "done",
                    "success": True,
                    "msg": "🎉 专属审美偏好模型微调与 ONNX 熔铸成功！模型已保存在 models/ 目录下，可即刻选用！",
                })
            else:
                await queue.put({
                    "type": "error",
                    "msg": f"训练进程异常退出 (退出码: {ret})，请检查日志信息。",
                })
        except Exception as exc:
            await queue.put({"type": "error", "msg": f"训练过程发生异常: {str(exc)}"})

    asyncio.create_task(run_training_subprocess())

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
