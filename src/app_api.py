"""
app_api.py — Photo Sort FastAPI Sidecar 后端服务
为 Tauri 2.0 前端提供 HTTP + SSE 接口支持。
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import copy
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from types import MappingProxyType
from typing import Any, AsyncGenerator, Literal, Mapping, Optional

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
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from burst_filter import BurstFilter, RawEvaluator
from model_manager import (
    PROJECT_ROOT,
    MODELS_DIR,
    check_all_models,
    download_clip_l14_model,
    download_clip_model,
    download_face_landmarker_model,
    get_active_model_mode,
    set_active_model_mode,
    get_resolved_mlp_path,
    get_resolved_mlp_l14_path,
    get_resolved_standard_onnx_path,
    get_resolved_standard_l14_onnx_path,
)
from onnx_exporter import fuse_mlp_weights_to_onnx, export_to_onnx, TORCH_EXPORT_AVAILABLE
from dehaze import DehazeParams, apply_dehaze, get_gpu_status
from dng_writer import write_enhanced_dng, write_linear_dng
from image_io import (
    OUTPUT_DIR_NAME, SUPPORTED_SUFFIXES,
    EnhancedDNGColorError, camera_profile_names, enhanced_dng_source_data,
    matching_embedded_profile_dng, read_image,
    scan_photo_directory, to_uint16,
)
from ricoh_filter import (
    apply_basic_preview_effect,
    RicohBatchLimitError,
    apply_ricoh_preset,
    list_ricoh_presets,
)
from lens_correction import (
    LensCorrectionError,
    LensCorrectionNotAppliedError,
    LensMatchError,
    LensfunUnavailableError,
    apply_lens_correction,
)

import io
import cv2
import onnx
import onnxruntime as ort
import numpy as np
from PIL import Image
import rawpy
from ricoh_filter import (
    apply_ricoh_preset_to_session,
    apply_ricoh_preview_effect,
    read_photo_settings,
    write_dehaze_session_settings,
    write_dehaze_settings,
    write_ricoh_preset,
    validate_basic_params,
)

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pass

try:
    import pillow_jxl
except Exception:
    pass

app = FastAPI(title="Imprint API", version="2.1.1")

# 最近几次筛选结果的缩略图访问表。只保存不可猜测的临时 ID 与本地路径映射，
# 不把任意文件路径暴露为公开查询参数。
_PREVIEW_SESSIONS: dict[str, dict[str, dict[str, Any]]] = {}
_PREVIEW_CACHE: dict[tuple[str, str, str], bytes] = {}
_PREVIEW_SESSION_LOCK = threading.Lock()
_MAX_PREVIEW_SESSIONS = 3
_MAX_PREVIEW_GROUPS = 40

# 去朦胧使用完全独立的会话、预览缓存和后台任务；会话中保存真实路径，HTTP
# 接口只暴露随机 ID，避免把任意本地路径做成可读取的 GET 参数。
_ENHANCE_SESSIONS: dict[str, dict[str, Any]] = {}
_ENHANCE_PREVIEW_CACHE: dict[tuple, bytes] = {}
_ENHANCE_THUMBNAIL_CACHE: dict[tuple[str, str], bytes] = {}
_ENHANCE_JOBS: dict[str, dict[str, Any]] = {}
_RICOH_PREVIEW_CACHE: dict[tuple, bytes] = {}
_RICOH_JOBS: dict[str, dict[str, Any]] = {}
_ENHANCE_LOCK = threading.RLock()
_MAX_ENHANCE_SESSIONS = 8
_MAX_ENHANCE_THUMBNAILS = 256

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
    defect_subdir: str = "审查_明显废片"
    keep_count: int = 1
    max_workers: int = 4
    use_gpu: bool = False
    include_previews: bool = False
    weight_mode: Literal["adaptive", "custom"] = "adaptive"
    custom_weights: dict[str, float] | None = None
    all_blurry_action: Literal["keep", "review", "reject"] = "keep"
    eye_detection: bool = False


class BurstDecisionRequest(BaseModel):
    kept: bool


class DownloadModelRequest(BaseModel):
    model: Literal["clip_b32", "clip_l14", "face_landmarker"]
    use_mirror: bool = True


class SetModeRequest(BaseModel):
    mode: Literal["standard", "standard_l14", "custom", "custom_l14"]


class TrainerRequest(BaseModel):
    photos_dir: str
    model_type: Literal["standard", "l14", "b32", "standard_l14", "custom_l14"] = "standard"
    epochs: int = 15
    lr: float = 1e-3


class EnhanceParamsRequest(BaseModel):
    strength: float = Field(default=0.45, ge=0.0, le=1.0)
    naturalness: float = Field(default=0.70, ge=0.0, le=1.0)
    fog_retention: float = Field(default=0.55, ge=0.0, le=1.0)
    local_contrast: float = Field(default=0.25, ge=0.0, le=1.0)
    color_recovery: float = Field(default=0.35, ge=0.0, le=1.0)
    color_protection: float = Field(default=0.80, ge=0.0, le=1.0)
    highlight_protection: float = Field(default=0.75, ge=0.0, le=1.0)
    shadow_protection: float = Field(default=0.75, ge=0.0, le=1.0)
    brightness_protection: float = Field(default=0.70, ge=0.0, le=1.0)

    def to_params(self) -> DehazeParams:
        return DehazeParams(**self.model_dump())


class BasicParamsRequest(BaseModel):
    exposure: float = Field(default=0, ge=-5, le=5)
    contrast: float = Field(default=0, ge=-100, le=100)
    highlights: float = Field(default=0, ge=-100, le=100)
    shadows: float = Field(default=0, ge=-100, le=100)
    whites: float = Field(default=0, ge=-100, le=100)
    blacks: float = Field(default=0, ge=-100, le=100)
    vibrance: float = Field(default=0, ge=-100, le=100)
    saturation: float = Field(default=0, ge=-100, le=100)

    def values(self) -> dict[str, float]:
        return validate_basic_params(self.model_dump())


class EnhanceSessionRequest(BaseModel):
    paths: list[str] = Field(default_factory=list)
    input_dir: str = ""


class EnhancePreviewRequest(BaseModel):
    session_id: str
    photo_id: str
    params: EnhanceParamsRequest = Field(default_factory=EnhanceParamsRequest)
    basic_params: BasicParamsRequest = Field(default_factory=BasicParamsRequest)
    max_edge: int = Field(default=1800, ge=320, le=3000)
    mode: Literal["original", "dehazed"] = "dehazed"
    use_gpu: bool = False


class EnhanceRevealRequest(BaseModel):
    session_id: str
    photo_id: str


class EnhanceRunRequest(BaseModel):
    session_id: str
    output_dir: str = ""
    params: EnhanceParamsRequest = Field(default_factory=EnhanceParamsRequest)
    params_by_photo: dict[str, EnhanceParamsRequest] = Field(default_factory=dict)
    basic_params_by_photo: dict[str, BasicParamsRequest] = Field(default_factory=dict)
    use_gpu: bool = False


class RicohApplyRequest(BaseModel):
    paths: list[str] = Field(min_length=1, max_length=32)
    preset_id: str = Field(min_length=1, max_length=80)


class EnhanceXmpRequest(BaseModel):
    session_id: str
    params_by_photo: dict[str, EnhanceParamsRequest] = Field(default_factory=dict)
    basic_params_by_photo: dict[str, BasicParamsRequest] = Field(default_factory=dict)


class RicohSessionRequest(BaseModel):
    session_id: str
    preset_id: str = Field(min_length=1, max_length=80)
    basic_params_by_photo: dict[str, BasicParamsRequest] = Field(default_factory=dict)


class RicohPreviewRequest(BaseModel):
    session_id: str
    photo_id: str
    preset_id: str = Field(min_length=1, max_length=80)
    basic_params: BasicParamsRequest = Field(default_factory=BasicParamsRequest)
    max_edge: int = Field(default=1800, ge=320, le=3000)


class RicohRunRequest(BaseModel):
    session_id: str
    preset_id: str = Field(min_length=1, max_length=80)
    output_dir: str = ""
    basic_params_by_photo: dict[str, BasicParamsRequest] = Field(default_factory=dict)


def _register_preview_groups(groups: list[dict]) -> tuple[str, list[dict]]:
    session_id = uuid.uuid4().hex
    path_map: dict[str, dict[str, Any]] = {}
    serialized: list[dict] = []

    # 优先展示低置信度组，其余按原始顺序补齐。
    ordered = sorted(
        groups,
        key=lambda group: (not bool(group.get("needs_review")), int(group.get("index", 0))),
    )[:_MAX_PREVIEW_GROUPS]

    for group_position, group in enumerate(ordered):
        shots = []
        for shot_position, shot in enumerate(group.get("shots", [])):
            photo_id = f"g{group_position}-p{shot_position}-{uuid.uuid4().hex[:8]}"
            current_paths = [Path(str(path)) for path in shot.get("paths", [])]
            if not current_paths and shot.get("path"):
                current_paths = [Path(str(shot["path"]))]
            original_paths = [Path(str(path)) for path in shot.get("original_paths", [])]
            if not original_paths:
                original_paths = list(current_paths)
            reject_category = "defect" if shot.get("category") == "defect" else "review"
            reject_dir_value = (
                group.get("defect_dir") if reject_category == "defect"
                else group.get("review_dir")
            )
            path_map[photo_id] = {
                "current_paths": current_paths,
                "original_paths": original_paths,
                "review_dir": Path(str(reject_dir_value or "")),
                "reject_category": reject_category,
                "category": str(shot.get("category", "keep" if shot.get("kept") else reject_category)),
                "kept": bool(shot.get("kept")),
            }
            shots.append({
                key: value for key, value in shot.items()
                if key not in {"path", "paths", "original_paths"}
            } | {"photo_id": photo_id})
        serialized.append({
            key: value for key, value in group.items()
            if key not in {"shots", "review_dir"}
        } | {"shots": shots})

    with _PREVIEW_SESSION_LOCK:
        _PREVIEW_SESSIONS[session_id] = path_map
        while len(_PREVIEW_SESSIONS) > _MAX_PREVIEW_SESSIONS:
            expired = next(iter(_PREVIEW_SESSIONS))
            _PREVIEW_SESSIONS.pop(expired, None)
            for cache_key in [key for key in _PREVIEW_CACHE if key[0] == expired]:
                _PREVIEW_CACHE.pop(cache_key, None)

    return session_id, serialized


def _review_destination(review_dir: Path, original_path: Path, current_path: Path) -> Path:
    """为审查目录生成不覆盖现有文件的目标路径。"""
    candidate = review_dir / original_path.name
    if candidate == current_path or not candidate.exists():
        return candidate
    index = 1
    while True:
        candidate = review_dir / f"{original_path.stem}_dup{index}{original_path.suffix}"
        if candidate == current_path or not candidate.exists():
            return candidate
        index += 1


def _apply_preview_decision(record: dict[str, Any], kept: bool) -> int:
    """在原目录与审查目录之间移动照片及伴生文件，失败时尽量回滚。"""
    if bool(record.get("kept")) == kept:
        return 0

    current_paths = [Path(path) for path in record.get("current_paths", [])]
    original_paths = [Path(path) for path in record.get("original_paths", [])]
    review_dir_value = record.get("review_dir")
    review_dir = Path(review_dir_value) if review_dir_value else Path()
    if not current_paths or len(current_paths) != len(original_paths):
        raise ValueError("照片伴生文件记录不完整")
    if not kept and not review_dir_value:
        raise ValueError("审查目录无效")

    targets: list[Path] = []
    for current_path, original_path in zip(current_paths, original_paths):
        if not current_path.exists():
            raise FileNotFoundError(f"照片不存在: {current_path.name}")
        if kept:
            target = original_path
            if target != current_path and target.exists():
                raise FileExistsError(f"原目录已存在同名文件: {target.name}")
        else:
            target = _review_destination(review_dir, original_path, current_path)
        targets.append(target)

    moved_pairs: list[tuple[Path, Path]] = []
    try:
        for current_path, target in zip(current_paths, targets):
            if current_path == target:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(current_path), str(target))
            moved_pairs.append((current_path, target))
    except Exception:
        for original_location, moved_location in reversed(moved_pairs):
            try:
                if moved_location.exists() and not original_location.exists():
                    shutil.move(str(moved_location), str(original_location))
            except Exception:
                pass
        raise

    record["current_paths"] = targets
    record["kept"] = kept
    record["category"] = "keep" if kept else record.get("reject_category", "review")
    return len(moved_pairs)


# ══════════════════════════════════════════════════════════════════════════════
# 接口一：POST /api/burst/run (SSE 连拍筛选)
# ══════════════════════════════════════════════════════════════════════════════


@app.get("/api/ricoh/presets")
def get_ricoh_presets():
    """List the reviewed GR2 and GR3 Camera Raw sidecar presets."""
    return {"presets": list_ricoh_presets()}


@app.post("/api/ricoh/apply")
def apply_ricoh_preset_request(req: RicohApplyRequest):
    """Merge Camera Raw XMP settings beside selected photos."""
    try:
        return apply_ricoh_preset(req.paths, req.preset_id)
    except KeyError:
        return JSONResponse(status_code=400, content={"error": "未知的理光预设"})
    except RicohBatchLimitError:
        return JSONResponse(status_code=413, content={"error": "一次最多处理 500 张照片"})


@app.post("/api/ricoh/apply-session")
def apply_ricoh_preset_session(req: RicohSessionRequest):
    with _ENHANCE_LOCK:
        session = _ENHANCE_SESSIONS.get(req.session_id)
        records = list(session.get("files", {}).items()) if session else []
    if session is None:
        return JSONResponse(status_code=404, content={"error": "照片会话已失效"})
    try:
        return apply_ricoh_preset_to_session(
            records, req.preset_id,
            {key: value.values() for key, value in req.basic_params_by_photo.items()},
        )
    except KeyError:
        return JSONResponse(status_code=400, content={"error": "未知的理光预设"})
    except RicohBatchLimitError:
        return JSONResponse(status_code=413, content={"error": "一次最多处理 500 张照片"})


@app.post("/api/ricoh/preview")
def create_ricoh_preview(req: RicohPreviewRequest):
    with _ENHANCE_LOCK:
        path = _ENHANCE_SESSIONS.get(req.session_id, {}).get("files", {}).get(req.photo_id)
    if path is None or not Path(path).is_file():
        return JSONResponse(status_code=404, content={"error": "照片预览会话已失效"})
    basic = req.basic_params.values()
    cache_key = (req.session_id, req.photo_id, req.preset_id, req.max_edge,
                 tuple(basic[key] for key in sorted(basic)))
    with _ENHANCE_LOCK:
        cached = _RICOH_PREVIEW_CACHE.get(cache_key)
    if cached is not None:
        return Response(
            content=cached, media_type="image/jpeg",
            headers={"Cache-Control": "private, max-age=3600", "X-Preview-Approximation": "true"},
        )
    try:
        image, metadata = read_image(path, preview=True, max_edge=req.max_edge)
        display_image = _display_rgb8(image, linear=getattr(metadata, "color_space", "") == "Linear sRGB")
        effected = apply_ricoh_preview_effect(display_image, req.preset_id, basic)
        payload = _encode_preview(effected)
        with _ENHANCE_LOCK:
            if len(_RICOH_PREVIEW_CACHE) >= 128:
                _RICOH_PREVIEW_CACHE.pop(next(iter(_RICOH_PREVIEW_CACHE)), None)
            _RICOH_PREVIEW_CACHE[cache_key] = payload
        return Response(
            content=payload,
            media_type="image/jpeg",
            headers={
                "Cache-Control": "private, max-age=3600",
                "X-Image-Width": str(metadata.width),
                "X-Image-Height": str(metadata.height),
                "X-Preview-Approximation": "true",
            },
        )
    except KeyError:
        return JSONResponse(status_code=400, content={"error": "未知的理光预设"})
    except Exception as exc:
        status_code = 422 if _is_enhance_unsupported_error(exc) else 500
        return JSONResponse(status_code=status_code, content={"error": _enhance_error_message(exc)})


@app.post("/api/burst/run")
async def run_burst(req: BurstRequest):
    """
    执行连拍照片筛选优选，通过 SSE 流式返回实时进度、处理结果或异常信息。
    """
    queue: asyncio.Queue[dict] = asyncio.Queue()
    loop = asyncio.get_running_loop()
    last_progress = ""
    progress_lock = threading.Lock()

    def on_progress(msg: str):
        nonlocal last_progress
        with progress_lock:
            if msg == last_progress:
                return
            last_progress = msg
        loop.call_soon_threadsafe(queue.put_nowait, {"type": "progress", "msg": msg})

    async def run_in_thread():
        try:
            clean_dir = str(req.input_dir).strip().strip('\'"')
            target_path = Path(clean_dir)
            if not target_path.exists() or not target_path.is_dir():
                await queue.put({"type": "error", "msg": f"目标目录不存在: {clean_dir}"})
                return

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                flt = BurstFilter(
                    gap_seconds=req.gap_seconds,
                    max_hamming_distance=req.max_hamming_distance,
                    review_subdir=req.review_subdir,
                    defect_subdir=req.defect_subdir,
                    keep_count=req.keep_count,
                    max_workers=req.max_workers,
                    use_gpu=req.use_gpu,
                    progress_callback=on_progress,
                    weight_mode=req.weight_mode,
                    custom_weights=req.custom_weights,
                    all_blurry_action=req.all_blurry_action,
                    eye_detection=req.eye_detection,
                )
                result = await loop.run_in_executor(pool, lambda: flt.run(target_path))

                if req.include_previews:
                    preview_session, preview_groups = _register_preview_groups(
                        getattr(result, "groups", [])
                    )
                else:
                    preview_session, preview_groups = "", []

                # 转换 BurstFilterResult 字段
                done_payload = {
                    "type": "done",
                    "total": getattr(result, "total", 0),
                    "burst_groups": getattr(result, "burst_groups", 0),
                    "moved": getattr(result, "moved", 0),
                    "defect_moved": getattr(result, "defect_moved", 0),
                    "skipped_single": getattr(result, "skipped_single", 0),
                    "errors": getattr(result, "errors", []),
                    "review_dir": str(result.review_dir) if getattr(result, "review_dir", None) else "",
                    "defect_dir": str(result.defect_dir) if getattr(result, "defect_dir", None) else "",
                    "preview_session": preview_session,
                    "groups": preview_groups,
                    "groups_shown": len(preview_groups),
                }
                await queue.put(done_payload)
        except Exception as exc:
            await queue.put({"type": "error", "msg": f"连拍筛选执行异常: {str(exc)}"})

    asyncio.create_task(run_in_thread())

    async def event_stream() -> AsyncGenerator[str, None]:
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=10.0)
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                    if event.get("type") in ("done", "error"):
                        break
                except asyncio.TimeoutError:
                    # SSE 注释行：不触发前端 onmessage，仅用于保持 TCP 连接活跃
                    yield ": keep-alive\n\n"
        except (asyncio.CancelledError, GeneratorExit, Exception):
            pass

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/burst/decision/{session_id}/{photo_id}")
def set_burst_decision(
    session_id: str,
    photo_id: str,
    decision: BurstDecisionRequest,
):
    """人工复核时立即改变一张照片的去留，并同步移动其 RAW/JPG 伴生文件。"""
    with _PREVIEW_SESSION_LOCK:
        record = _PREVIEW_SESSIONS.get(session_id, {}).get(photo_id)
        if record is None:
            return JSONResponse(status_code=404, content={"error": "复核会话已失效"})
        try:
            moved_files = _apply_preview_decision(record, decision.kept)
        except FileExistsError as exc:
            return JSONResponse(status_code=409, content={"error": str(exc)})
        except FileNotFoundError as exc:
            return JSONResponse(status_code=404, content={"error": str(exc)})
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": f"移动照片失败: {exc}"})

    return {
        "ok": True,
        "kept": decision.kept,
        "category": record.get("category", "keep" if decision.kept else "review"),
        "moved_files": moved_files,
        "message": "已恢复到原目录" if decision.kept else "已移入审查目录",
    }


@app.get("/api/burst/preview/{session_id}/{photo_id}")
def get_burst_preview(
    session_id: str,
    photo_id: str,
    kind: Literal["full", "focus", "review"] = "full",
):
    """按不可猜测的会话 ID 返回筛选结果缩略图或清晰度检查裁切。"""
    record = _PREVIEW_SESSIONS.get(session_id, {}).get(photo_id)
    current_paths = record.get("current_paths", []) if record else []
    path = Path(current_paths[0]) if current_paths else None
    if path is None or not path.exists() or not path.is_file():
        return JSONResponse(status_code=404, content={"error": "预览已失效或照片不存在"})

    cache_key = (session_id, photo_id, kind)
    cached = _PREVIEW_CACHE.get(cache_key)
    if cached is not None:
        return Response(
            content=cached,
            media_type="image/jpeg",
            headers={"Cache-Control": "private, max-age=3600"},
        )

    try:
        evaluator = RawEvaluator()
        image_rgb = evaluator.extract_preview(path)

        if kind == "focus":
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            faces = evaluator._face_regions(gray)
            height, width = image_rgb.shape[:2]
            if faces:
                x, y, face_width, face_height = max(faces, key=lambda face: face[2] * face[3])
                center_x = x + face_width // 2
                center_y = y + face_height // 2
                side = int(max(face_width, face_height) * 1.8)
            else:
                center_x, center_y = width // 2, height // 2
                side = int(min(width, height) * 0.46)
            side = max(32, min(side, width, height))
            left = max(0, min(width - side, center_x - side // 2))
            top = max(0, min(height - side, center_y - side // 2))
            image_rgb = image_rgb[top:top + side, left:left + side]

        image = Image.fromarray(image_rgb).convert("RGB")
        if kind == "review":
            target_size = (2048, 2048)
        elif kind == "focus":
            target_size = (260, 260)
        else:
            target_size = (360, 220)
        image.thumbnail(target_size, Image.Resampling.LANCZOS)
        output = io.BytesIO()
        image.save(
            output,
            format="JPEG",
            quality=90 if kind == "review" else 84,
            optimize=True,
        )
        payload = output.getvalue()
        _PREVIEW_CACHE[cache_key] = payload
        return Response(
            content=payload,
            media_type="image/jpeg",
            headers={"Cache-Control": "private, max-age=3600"},
        )
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": f"生成预览失败: {exc}"})


# ══════════════════════════════════════════════════════════════════════════════
# 独立功能：去朦胧会话、参数化预览与可取消批处理
# ══════════════════════════════════════════════════════════════════════════════


def _enhance_public_job(job: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy({key: value for key, value in job.items() if key not in {"_cancel", "_thread"}})


def _exception_text(exc: BaseException) -> str:
    """Return exception text only for internal classification, never for responses."""
    try:
        return str(exc).casefold()
    except Exception:
        return ""


def _is_enhance_unsupported_error(exc: BaseException) -> bool:
    """Recognize RAW/container errors that should be reported as HTTP 422."""
    if isinstance(exc, rawpy.LibRawFileUnsupportedError):
        return True
    return "unsupported file format or not raw file" in _exception_text(exc)


def _enhance_error_message(exc: BaseException, *, output_dir: bool = False) -> str:
    """Map enhancement failures to stable, path-free messages for the API."""
    text = _exception_text(exc)
    if _is_enhance_unsupported_error(exc):
        return (
            "当前 RAW 或压缩方式暂不受支持。若为尼康 HE/HE★，建议改用无损压缩 RAW，"
            "或先用 Nikon NX Studio 转换为 TIFF 后再导入。"
        )
    if isinstance(exc, LensfunUnavailableError):
        return "镜头校正组件不可用，已停止导出，避免生成未校正的 DNG。"
    if isinstance(exc, LensMatchError):
        return "找不到可靠的相机或镜头校正配置，未生成未校正的 DNG。"
    if isinstance(exc, LensCorrectionNotAppliedError):
        return "镜头配置没有可用的畸变或横向色差数据，未生成未校正的 DNG。"
    if isinstance(exc, LensCorrectionError):
        return "镜头像素校正失败，未生成未校正的 DNG。"
    if isinstance(exc, EnhancedDNGColorError):
        return "无法可靠转换相机色彩数据，未生成可能颜色错误的 DNG。"
    if isinstance(exc, (MemoryError, rawpy.LibRawUnsufficientMemoryError, rawpy.LibRawMemPoolOverflowError)) or any(
        marker in text for marker in ("out of memory", "insufficient memory", "cannot allocate memory")
    ):
        return "处理照片时内存不足，请关闭其他应用后重试。"
    if isinstance(exc, FileNotFoundError) or any(
        marker in text for marker in ("no such file", "file not found", "does not exist", "不存在的照片")
    ):
        return "照片文件不存在或已被移动，请重新选择后重试。"
    if isinstance(exc, PermissionError):
        return (
            "输出目录不可用，请检查目录权限和磁盘空间。"
            if output_dir
            else "没有权限读取照片或写入输出目录，请检查文件和目录权限。"
        )
    if output_dir and isinstance(exc, (OSError, ValueError)):
        return "输出目录不可用，请检查目录权限和磁盘空间。"
    if isinstance(exc, (rawpy.LibRawDataError, rawpy.LibRawFatalError)) or any(
        marker in text
        for marker in (
            "corrupt",
            "corrupted",
            "truncated",
            "unexpected end",
            "invalid image",
            "failed to decode",
            "decode error",
        )
    ):
        return "照片数据可能已损坏或不完整，请重新复制文件后重试。"
    return "照片解码/处理失败，请确认文件完整且格式受支持。"


_LINEAR_LEVELS = np.arange(65536, dtype=np.float32) / 65535.0
_LINEAR_TO_SRGB8 = np.clip(
    np.rint(
        np.where(
            _LINEAR_LEVELS <= 0.0031308,
            _LINEAR_LEVELS * 12.92,
            1.055 * np.power(_LINEAR_LEVELS, 1.0 / 2.4) - 0.055,
        ) * 255.0
    ), 0, 255,
).astype(np.uint8)
del _LINEAR_LEVELS


def _display_rgb8(image_rgb: np.ndarray, *, linear: bool = False) -> np.ndarray:
    if image_rgb.dtype == np.uint16:
        if linear:
            return _LINEAR_TO_SRGB8[image_rgb]
        return np.clip(np.rint(image_rgb.astype(np.float32) / 257.0), 0, 255).astype(np.uint8)
    return image_rgb


def _encode_preview(image_rgb: np.ndarray, *, linear: bool = False) -> bytes:
    image_rgb = _display_rgb8(image_rgb, linear=linear)
    stream = io.BytesIO()
    Image.fromarray(image_rgb, "RGB").save(stream, "JPEG", quality=91, optimize=True)
    return stream.getvalue()


def _srgb16_to_linear16(image_rgb: np.ndarray) -> np.ndarray:
    encoded = image_rgb.astype(np.float32) / 65535.0
    linear = np.where(encoded <= 0.04045, encoded / 12.92, ((encoded + 0.055) / 1.055) ** 2.4)
    return np.clip(np.rint(linear * 65535.0), 0, 65535).astype(np.uint16)


def _linear16_to_srgb16(image_rgb: np.ndarray) -> np.ndarray:
    linear = image_rgb.astype(np.float32) / 65535.0
    encoded = np.where(linear <= 0.0031308, linear * 12.92, 1.055 * np.power(linear, 1.0 / 2.4) - 0.055)
    return np.clip(np.rint(encoded * 65535.0), 0, 65535).astype(np.uint16)


@app.post("/api/enhance/reveal")
def reveal_enhance_original(req: EnhanceRevealRequest):
    with _ENHANCE_LOCK:
        path = _ENHANCE_SESSIONS.get(req.session_id, {}).get("files", {}).get(req.photo_id)
    if path is None:
        return JSONResponse(status_code=404, content={"error": "去朦胧预览会话已失效"})
    try:
        current_path = Path(path).resolve(strict=True)
    except (FileNotFoundError, OSError, RuntimeError):
        return JSONResponse(status_code=404, content={"error": "当前原图不存在"})
    if not current_path.is_file():
        return JSONResponse(status_code=404, content={"error": "当前原图不存在"})
    return {"path": str(current_path)}


@app.post("/api/enhance/session")
def create_enhance_session(req: EnhanceSessionRequest):
    try:
        if req.input_dir.strip():
            source_dir = Path(req.input_dir.strip().strip('\"\'')).expanduser().resolve()
            sources = scan_photo_directory(source_dir)
            default_output = (source_dir / OUTPUT_DIR_NAME).resolve()
        else:
            sources = []
            seen: set[Path] = set()
            for raw_path in req.paths:
                path = Path(raw_path.strip().strip('\"\'')).expanduser().resolve()
                if path in seen:
                    continue
                if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
                    continue
                if OUTPUT_DIR_NAME in path.parts:
                    continue
                seen.add(path)
                sources.append(path)
            default_output = (sources[0].parent / OUTPUT_DIR_NAME).resolve() if sources else Path()
        if not sources:
            return JSONResponse(status_code=400, content={"error": "没有找到可处理的照片"})
        if len(sources) > 5000:
            return JSONResponse(status_code=400, content={"error": "单次最多处理 5000 张照片"})

        session_id = uuid.uuid4().hex
        records: dict[str, Path] = {}
        files: list[dict[str, Any]] = []
        for index, path in enumerate(sources):
            photo_id = f"p{index}-{uuid.uuid4().hex[:10]}"
            records[photo_id] = path
            settings = read_photo_settings(path)
            files.append({
                "photo_id": photo_id,
                "name": path.name,
                "extension": path.suffix.lower(),
                "dehaze_params": settings["dehaze_params"],
                "ricoh_preset_id": settings["ricoh_preset_id"],
                "basic_params": settings["basic_params"],
            })
        with _ENHANCE_LOCK:
            _ENHANCE_SESSIONS[session_id] = {
                "files": records,
                "created": time.time(),
            }
            while len(_ENHANCE_SESSIONS) > _MAX_ENHANCE_SESSIONS:
                expired = next(iter(_ENHANCE_SESSIONS))
                _ENHANCE_SESSIONS.pop(expired, None)
                for key in [key for key in _ENHANCE_PREVIEW_CACHE if key[0] == expired]:
                    _ENHANCE_PREVIEW_CACHE.pop(key, None)
                for key in [key for key in _ENHANCE_THUMBNAIL_CACHE if key[0] == expired]:
                    _ENHANCE_THUMBNAIL_CACHE.pop(key, None)
                for key in [key for key in _RICOH_PREVIEW_CACHE if key[0] == expired]:
                    _RICOH_PREVIEW_CACHE.pop(key, None)
        return {
            "session_id": session_id,
            "count": len(files),
            "files": files,
            "default_output_dir": str(default_output),
            "ricoh_default_output_dir": str(default_output / "理光预设输出"),
            "output_format": "Source bit-depth Linear/Demosaiced DNG",
        }
    except Exception as exc:
        return JSONResponse(status_code=400, content={"error": _enhance_error_message(exc)})


@app.post("/api/enhance/xmp")
def save_enhance_session_xmp(req: EnhanceXmpRequest):
    with _ENHANCE_LOCK:
        session = _ENHANCE_SESSIONS.get(req.session_id)
        records = dict(session.get("files", {})) if session else {}
    if session is None:
        return JSONResponse(status_code=404, content={"error": "去朦胧会话已失效"})
    scoped = [
        (photo_id, records[photo_id], params.to_params().__dict__)
        for photo_id, params in req.params_by_photo.items()
        if photo_id in records
    ]
    try:
        return write_dehaze_session_settings(
            scoped, {key: value.values() for key, value in req.basic_params_by_photo.items()},
        )
    except RicohBatchLimitError:
        return JSONResponse(status_code=413, content={"error": "一次最多处理 5000 张照片"})


@app.get("/api/enhance/gpu-status")
def get_enhance_gpu_status():
    """Return accelerator availability for the dehaze preview/export path."""
    try:
        return get_gpu_status()
    except Exception:
        # Device probing is best-effort.  Keep the response stable and avoid
        # exposing driver paths or exception details if an optional runtime is
        # partially installed or unavailable.
        return {
            "available": False,
            "backends": [],
            "recommended": None,
            "label": "未检测到可用 GPU",
        }


@app.post("/api/enhance/preview")
def create_enhance_preview(req: EnhancePreviewRequest):
    with _ENHANCE_LOCK:
        path = _ENHANCE_SESSIONS.get(req.session_id, {}).get("files", {}).get(req.photo_id)
    if path is None or not Path(path).is_file():
        return JSONResponse(status_code=404, content={"error": "去朦胧预览会话已失效"})
    params = req.params.to_params()
    token = params.cache_token()
    basic = req.basic_params.values()
    cache_key = (req.session_id, req.photo_id, token, req.max_edge, req.mode, bool(req.use_gpu),
                 tuple(basic[key] for key in sorted(basic)))
    with _ENHANCE_LOCK:
        cached = _ENHANCE_PREVIEW_CACHE.get(cache_key)
    if cached is not None:
        return Response(content=cached, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})
    try:
        image, metadata = read_image(path, preview=True, max_edge=req.max_edge)
        if req.mode == "dehazed":
            image = apply_dehaze(image, params, backend="auto" if req.use_gpu else "cpu")
            display = _display_rgb8(image, linear=getattr(metadata, "color_space", "") == "Linear sRGB")
            image = apply_basic_preview_effect(display, basic)
            payload = _encode_preview(image)
        else:
            payload = _encode_preview(image, linear=getattr(metadata, "color_space", "") == "Linear sRGB")
        with _ENHANCE_LOCK:
            if len(_ENHANCE_PREVIEW_CACHE) >= 128:
                _ENHANCE_PREVIEW_CACHE.pop(next(iter(_ENHANCE_PREVIEW_CACHE)), None)
            _ENHANCE_PREVIEW_CACHE[cache_key] = payload
        return Response(
            content=payload,
            media_type="image/jpeg",
            headers={
                "Cache-Control": "private, max-age=3600",
                "X-Image-Width": str(metadata.width),
                "X-Image-Height": str(metadata.height),
            },
        )
    except Exception as exc:
        status_code = 422 if _is_enhance_unsupported_error(exc) else 500
        return JSONResponse(status_code=status_code, content={"error": _enhance_error_message(exc)})


@app.get("/api/enhance/thumbnail/{session_id}/{photo_id}")
def get_enhance_thumbnail(session_id: str, photo_id: str):
    """Return a small, cached JPEG for a photo in an active enhance session."""
    cache_key = (session_id, photo_id)
    with _ENHANCE_LOCK:
        path = _ENHANCE_SESSIONS.get(session_id, {}).get("files", {}).get(photo_id)
        cached = _ENHANCE_THUMBNAIL_CACHE.get(cache_key)
    if path is None:
        return JSONResponse(status_code=404, content={"error": "去朦胧缩略图会话已失效"})
    if not Path(path).is_file():
        return JSONResponse(status_code=404, content={"error": "当前照片不存在"})
    if cached is not None:
        return Response(
            content=cached,
            media_type="image/jpeg",
            headers={"Cache-Control": "private, max-age=3600"},
        )

    try:
        image, metadata = read_image(path, preview=True, max_edge=360)
        payload = _encode_preview(image, linear=getattr(metadata, "color_space", "") == "Linear sRGB")
        with _ENHANCE_LOCK:
            current_path = _ENHANCE_SESSIONS.get(session_id, {}).get("files", {}).get(photo_id)
            if current_path is None or Path(current_path) != Path(path):
                return JSONResponse(status_code=404, content={"error": "去朦胧缩略图会话已失效"})
            if len(_ENHANCE_THUMBNAIL_CACHE) >= _MAX_ENHANCE_THUMBNAILS:
                _ENHANCE_THUMBNAIL_CACHE.pop(next(iter(_ENHANCE_THUMBNAIL_CACHE)), None)
            _ENHANCE_THUMBNAIL_CACHE[cache_key] = payload
        return Response(
            content=payload,
            media_type="image/jpeg",
            headers={"Cache-Control": "private, max-age=3600"},
        )
    except FileNotFoundError:
        return JSONResponse(status_code=404, content={"error": "当前照片不存在"})
    except Exception:
        return JSONResponse(status_code=500, content={"error": "生成缩略图失败"})


def _snapshot_enhance_params(
    req: EnhanceRunRequest,
    photo_ids: set[str] | None = None,
) -> tuple[DehazeParams, Mapping[str, DehazeParams]]:
    """Freeze request parameters before a background job starts.

    Only photo IDs from the active session are retained.  ``DehazeParams`` is
    frozen itself, and the mapping proxy prevents a caller from changing the
    selection while the worker is processing the batch.
    """
    default_params = req.params.to_params()
    scoped_params = {
        photo_id: photo_params.to_params()
        for photo_id, photo_params in req.params_by_photo.items()
        if photo_ids is None or photo_id in photo_ids
    }
    return default_params, MappingProxyType(scoped_params)


def _select_enhance_params(
    photo_id: str,
    params_by_photo: Mapping[str, DehazeParams],
    default_params: DehazeParams,
) -> DehazeParams:
    """Return a per-photo snapshot, falling back to the batch default."""
    return params_by_photo.get(photo_id, default_params)


def _run_enhance_job(
    job_id: str,
    session_id: str,
    output_dir: Path,
    default_params: DehazeParams,
    params_by_photo: Mapping[str, DehazeParams],
    backend: str = "cpu",
    basic_params_by_photo: Mapping[str, dict[str, float]] | None = None,
) -> None:
    with _ENHANCE_LOCK:
        job = _ENHANCE_JOBS[job_id]
        records = list(_ENHANCE_SESSIONS.get(session_id, {}).get("files", {}).items())
        job["status"] = "running"
    for photo_id, path in records:
        with _ENHANCE_LOCK:
            cancel_event: threading.Event = job["_cancel"]
            if cancel_event.is_set():
                job["status"] = "cancelled"
                break
            item = next(item for item in job["files"] if item["photo_id"] == photo_id)
            item["status"] = "processing"
            job["current_file"] = Path(path).name
        try:
            image, metadata = read_image(path, preview=False)
            photo_params = _select_enhance_params(photo_id, params_by_photo, default_params)
            enhanced = apply_dehaze(to_uint16(image), photo_params, backend=backend)
            corrected, correction = apply_lens_correction(
                enhanced,
                metadata,
                # RAW exports must never silently succeed without a real
                # Lensfun geometry/TCA pass. Standard RGB inputs remain
                # exportable as required by the existing product behavior.
                require_correction=getattr(metadata, "source_kind", "") == "raw",
            )
            basic = (basic_params_by_photo or {}).get(photo_id)
            if basic and any(basic.values()):
                display = (_linear16_to_srgb16(corrected)
                           if getattr(metadata, "color_space", "") == "Linear sRGB" else corrected)
                adjusted = apply_basic_preview_effect(display, basic)
                corrected = (_srgb16_to_linear16(adjusted)
                             if getattr(metadata, "color_space", "") == "Linear sRGB" else adjusted)
            del enhanced
            if cancel_event.is_set():
                with _ENHANCE_LOCK:
                    item["status"] = "cancelled"
                    job["status"] = "cancelled"
                break
            output_metadata = dict(metadata.exif)
            operations = []
            if correction.distortion_applied:
                operations.append("distortion")
            if correction.tca_applied:
                operations.append("tca")
            if correction.vignetting_applied:
                operations.append("vignetting")
            if correction.applied:
                output_metadata.update(
                    {
                        "LensCorrectionApplied": True,
                        "LensCorrectionEngine": "Lensfun/lensfunpy",
                        "LensCorrectionCamera": correction.camera_name or "",
                        "LensCorrectionLens": correction.lens_name or "",
                        "LensCorrectionOperations": ",".join(operations),
                    }
                )
            is_raw = getattr(metadata, "source_kind", "") == "raw"
            if is_raw:
                output_metadata.update(camera_profile_names(path))
                profile_reference = matching_embedded_profile_dng(path, output_metadata)
                if profile_reference is not None:
                    output_metadata["DNGProfileReferencePath"] = profile_reference
                camera_rgb, mosaic, cfa_pattern, profile, orientation = enhanced_dng_source_data(
                    corrected, image, path, output_metadata,
                )
                output_metadata.update(profile)
                del image
                # Build a high-resolution display preview from the same
                # linear pixels as the exported enhanced layer.  Resize in
                # linear light before applying the sRGB transfer curve.
                preview_edge = 4096
                if max(corrected.shape[:2]) > preview_edge:
                    scale = preview_edge / max(corrected.shape[:2])
                    preview_linear = cv2.resize(
                        corrected,
                        (max(1, round(corrected.shape[1] * scale)),
                         max(1, round(corrected.shape[0] * scale))),
                        interpolation=cv2.INTER_AREA,
                    )
                else:
                    preview_linear = corrected
                preview_rgb = _display_rgb8(preview_linear, linear=True)
                if orientation == 8:
                    preview_rgb = np.rot90(preview_rgb, -1)
                elif orientation == 6:
                    preview_rgb = np.rot90(preview_rgb, 1)
                elif orientation == 3:
                    preview_rgb = np.rot90(preview_rgb, 2)
                preview_rgb = np.ascontiguousarray(preview_rgb)
                output_path = write_enhanced_dng(
                    camera_rgb, mosaic, cfa_pattern, path, output_dir,
                    output_metadata, orientation=orientation, preview_rgb16=preview_rgb,
                )
            else:
                del image
                output_path = write_linear_dng(
                    corrected, path, output_dir, output_metadata,
                    bits_per_sample=metadata.bit_depth,
                )
            xmp_status = "failed"
            try:
                write_dehaze_settings(path, photo_params.__dict__, basic)
                xmp_status = "written"
            except Exception:
                # DNG export is the primary job result. A sidecar failure is
                # reported separately and never rolls back a valid DNG.
                pass
            with _ENHANCE_LOCK:
                item.update(
                    {
                        "status": "success",
                        "output": str(output_path),
                        "xmp_status": xmp_status,
                        "lens_correction": {
                            "applied": correction.applied,
                            "camera": correction.camera_name,
                            "lens": correction.lens_name,
                            "distortion": correction.distortion_applied,
                            "tca": correction.tca_applied,
                            "vignetting": correction.vignetting_applied,
                        },
                    }
                )
                job["success"] += 1
        except Exception as exc:
            with _ENHANCE_LOCK:
                item.update({"status": "failed", "error": _enhance_error_message(exc)})
                job["failed"] += 1
        finally:
            with _ENHANCE_LOCK:
                job["processed"] += 1
                job["progress"] = round(job["processed"] / max(1, job["total"]), 4)
    with _ENHANCE_LOCK:
        if job["status"] != "cancelled":
            job["status"] = "completed"
        job["current_file"] = ""


@app.post("/api/enhance/run")
def run_enhance(req: EnhanceRunRequest):
    with _ENHANCE_LOCK:
        session = _ENHANCE_SESSIONS.get(req.session_id)
    if session is None:
        return JSONResponse(status_code=404, content={"error": "去朦胧会话已失效"})
    records = list(session["files"].items())
    default_params, params_by_photo = _snapshot_enhance_params(
        req,
        {photo_id for photo_id, _ in records},
    )
    first_path = Path(records[0][1])
    output_dir = Path(req.output_dir.strip().strip('\"\'')) if req.output_dir.strip() else first_path.parent / OUTPUT_DIR_NAME
    try:
        output_dir = output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        return JSONResponse(status_code=400, content={"error": _enhance_error_message(exc, output_dir=True)})
    job_id = uuid.uuid4().hex
    job: dict[str, Any] = {
        "job_id": job_id, "status": "queued", "total": len(records), "processed": 0,
        "success": 0, "failed": 0, "progress": 0.0, "current_file": "",
        "output_dir": str(output_dir),
        "files": [{"photo_id": photo_id, "name": Path(path).name, "status": "waiting"} for photo_id, path in records],
        "_cancel": threading.Event(),
    }
    thread = threading.Thread(
        target=_run_enhance_job,
        args=(job_id, req.session_id, output_dir, default_params, params_by_photo,
              "auto" if req.use_gpu else "cpu",
              {key: value.values() for key, value in req.basic_params_by_photo.items()}),
        name=f"enhance-{job_id[:8]}", daemon=True,
    )
    job["_thread"] = thread
    with _ENHANCE_LOCK:
        _ENHANCE_JOBS[job_id] = job
    thread.start()
    return _enhance_public_job(job)


def _ricoh_public_job(job: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy({key: value for key, value in job.items() if key not in {"_cancel", "_thread"}})


def _run_ricoh_job(job_id: str, session_id: str, preset_id: str, output_dir: Path,
                   basic_params_by_photo: Mapping[str, dict[str, float]] | None = None) -> None:
    with _ENHANCE_LOCK:
        job = _RICOH_JOBS[job_id]
        records = list(_ENHANCE_SESSIONS.get(session_id, {}).get("files", {}).items())
        job["status"] = "running"
    for photo_id, path in records:
        with _ENHANCE_LOCK:
            cancel_event: threading.Event = job["_cancel"]
            if cancel_event.is_set():
                job["status"] = "cancelled"
                break
            item = next(item for item in job["files"] if item["photo_id"] == photo_id)
            item["status"] = "processing"
            job["current_file"] = Path(path).name
        try:
            image, metadata = read_image(path, preview=False)
            source16 = to_uint16(image)
            # Camera Raw controls operate on a display-referred rendering.
            # Convert back to linear RGB before writing the Linear DNG.
            display16 = (
                _linear16_to_srgb16(source16)
                if getattr(metadata, "color_space", "") == "Linear sRGB"
                else source16
            )
            basic = (basic_params_by_photo or {}).get(photo_id)
            effected_display16 = apply_ricoh_preview_effect(display16, preset_id, basic)
            effected_linear16 = _srgb16_to_linear16(effected_display16)
            output_path = write_linear_dng(
                effected_linear16, path, output_dir, dict(metadata.exif), bits_per_sample=16,
                name_suffix="_ricoh",
            )
            # The exported DNG already contains rendered preset pixels. Keep
            # Camera Raw adjustments on the source only, or Adobe would apply
            # the same look a second time when opening the DNG.
            xmp_status: dict[str, str] = {"source": "failed", "output": "rendered"}
            try:
                write_ricoh_preset(path, preset_id, basic)
                xmp_status["source"] = "written"
            except Exception:
                xmp_status["source"] = "failed"
            with _ENHANCE_LOCK:
                item.update({
                    "status": "success",
                    "output": str(output_path),
                    "xmp_status": xmp_status,
                })
                job["success"] += 1
        except Exception as exc:
            with _ENHANCE_LOCK:
                item.update({"status": "failed", "error": _enhance_error_message(exc)})
                job["failed"] += 1
        finally:
            with _ENHANCE_LOCK:
                job["processed"] += 1
                job["progress"] = round(job["processed"] / max(1, job["total"]), 4)
    with _ENHANCE_LOCK:
        if job["status"] != "cancelled":
            job["status"] = "completed"
        job["current_file"] = ""


@app.post("/api/ricoh/run")
def run_ricoh(req: RicohRunRequest):
    if req.preset_id not in {preset["id"] for preset in list_ricoh_presets()}:
        return JSONResponse(status_code=400, content={"error": "未知的理光预设"})
    with _ENHANCE_LOCK:
        session = _ENHANCE_SESSIONS.get(req.session_id)
    if session is None:
        return JSONResponse(status_code=404, content={"error": "照片会话已失效"})
    records = list(session["files"].items())
    if not records:
        return JSONResponse(status_code=400, content={"error": "没有可处理的照片"})
    if len(records) > 500:
        return JSONResponse(status_code=413, content={"error": "一次最多处理 500 张照片"})
    first_path = Path(records[0][1])
    output_dir = (
        Path(req.output_dir.strip().strip('"\''))
        if req.output_dir.strip()
        else first_path.parent / OUTPUT_DIR_NAME / "理光预设输出"
    )
    try:
        output_dir = output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        return JSONResponse(status_code=400, content={"error": _enhance_error_message(exc, output_dir=True)})
    job_id = uuid.uuid4().hex
    job: dict[str, Any] = {
        "job_id": job_id,
        "status": "queued",
        "preset_id": req.preset_id,
        "total": len(records),
        "processed": 0,
        "success": 0,
        "failed": 0,
        "progress": 0.0,
        "current_file": "",
        "output_dir": str(output_dir),
        "files": [{"photo_id": photo_id, "name": Path(path).name, "status": "waiting"} for photo_id, path in records],
        "_cancel": threading.Event(),
    }
    thread = threading.Thread(
        target=_run_ricoh_job,
        args=(job_id, req.session_id, req.preset_id, output_dir,
              {key: value.values() for key, value in req.basic_params_by_photo.items()}),
        name=f"ricoh-{job_id[:8]}", daemon=True,
    )
    job["_thread"] = thread
    with _ENHANCE_LOCK:
        _RICOH_JOBS[job_id] = job
    thread.start()
    return _ricoh_public_job(job)


@app.get("/api/ricoh/job/{job_id}")
def get_ricoh_job(job_id: str):
    with _ENHANCE_LOCK:
        job = _RICOH_JOBS.get(job_id)
        if job is None:
            return JSONResponse(status_code=404, content={"error": "任务不存在或已过期"})
        return _ricoh_public_job(job)


@app.post("/api/ricoh/cancel/{job_id}")
def cancel_ricoh_job(job_id: str):
    with _ENHANCE_LOCK:
        job = _RICOH_JOBS.get(job_id)
        if job is None:
            return JSONResponse(status_code=404, content={"error": "任务不存在或已过期"})
        job["_cancel"].set()
        if job["status"] == "queued":
            job["status"] = "cancelled"
        return {"ok": True, "message": "已请求停止；当前照片完成后不会再处理下一张"}


@app.get("/api/enhance/job/{job_id}")
def get_enhance_job(job_id: str):
    with _ENHANCE_LOCK:
        job = _ENHANCE_JOBS.get(job_id)
        if job is None:
            return JSONResponse(status_code=404, content={"error": "任务不存在或已过期"})
        return _enhance_public_job(job)


@app.post("/api/enhance/cancel/{job_id}")
def cancel_enhance_job(job_id: str):
    with _ENHANCE_LOCK:
        job = _ENHANCE_JOBS.get(job_id)
        if job is None:
            return JSONResponse(status_code=404, content={"error": "任务不存在或已过期"})
        job["_cancel"].set()
        if job["status"] == "queued":
            job["status"] = "cancelled"
        return {"ok": True, "message": "已请求停止；当前照片完成后不会再处理下一张"}


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
            "face_landmarker_ready": status.face_landmarker_ready,
            "face_landmarker_path": status.face_landmarker_path,
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
                "face_landmarker_ready": False,
                "face_landmarker_path": "",
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
                elif req.model == "clip_l14":
                    success = await loop.run_in_executor(
                        pool, lambda: download_clip_l14_model(use_mirror=req.use_mirror, progress_callback=on_progress)
                    )
                else:
                    success = await loop.run_in_executor(
                        pool, lambda: download_face_landmarker_model(progress_callback=on_progress)
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

    clean_photos_dir = str(req.photos_dir).strip().strip('\'"')
    data_dir = Path(clean_photos_dir)
    if not data_dir.exists() or not data_dir.is_dir():
        async def err_stream():
            yield f"data: {json.dumps({'type': 'error', 'msg': f'训练样本目录不存在: {clean_photos_dir}'}, ensure_ascii=False)}\n\n"
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
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=10.0)
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                    if event.get("type") in ("done", "error"):
                        break
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
        except (asyncio.CancelledError, GeneratorExit, Exception):
            pass

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
