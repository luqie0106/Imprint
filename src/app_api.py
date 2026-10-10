"""
app_api.py — Photo Sort FastAPI Sidecar 后端服务
为 Tauri 2.0 前端提供 HTTP + SSE 接口支持。
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import copy
import ctypes
import ctypes.util
from collections import OrderedDict
from dataclasses import asdict
from functools import lru_cache
import gc
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
from pydantic import BaseModel, ConfigDict, Field

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
from dehaze import ALGORITHM_VERSION as DEHAZE_ALGORITHM_VERSION
from dehaze import DehazeParams, resolve_nonlocal_mode
from auto_exposure import ALGORITHM_VERSION as AUTO_EXPOSURE_ALGORITHM_VERSION
from auto_exposure import apply_auto_exposure, estimate_auto_exposure
from dehaze_physical import apply_physical_dehaze, get_last_physical_backend
from native_renderer import (
    get_native_status, get_native_physical_status, native_basic, native_ricoh,
)
from native_sort import get_native_sort_status
from dng_writer import write_enhanced_dng, write_linear_dng
from camera_profile import (
    PROFILE_PREVIEW_VERSION, resolve_profile, load_camera_rgb,
    transfer_enhancement, render_profile, get_last_profile_backend,
    clear_camera_profile_gpu_cache,
)
from image_io import (
    OUTPUT_DIR_NAME, SUPPORTED_SUFFIXES,
    EnhancedDNGColorError, build_enhanced_dng_source_cache,
    camera_profile_names, enhanced_dng_source_data,
    enhanced_dng_source_identity,
    matching_embedded_profile_dng, read_image,
    read_photo_metadata, scan_photo_directory, to_uint16,
)
from ricoh_filter import (
    apply_basic_preview_effect,
    RicohBatchLimitError,
    apply_ricoh_preset,
    list_ricoh_presets,
)
from measured_response import standard_preview_to_srgb
from lens_correction import (
    LensCorrectionError,
    LensMetadataUnavailableError,
    LensCorrectionNotAppliedError,
    LensCorrectionResult,
    LensMatchError,
    LensfunUnavailableError,
    apply_lens_correction,
)
from dng_gainmap import apply_dng_gain_map
from dng_warp import apply_dng_warp_correction
from native_renderer import clear_native_warp_cache

LENS_PREVIEW_VERSION = "dng-warp-ordered-gainmap-gpu-v5"
BASIC_PREVIEW_VERSION = "camera-profile-ricoh-linear-exposure-v4"

import io
import logging
import cv2
import onnx
import onnxruntime as ort
import numpy as np
from PIL import Image, ImageCms, ImageOps
import rawpy
from ricoh_filter import (
    apply_ricoh_preset_to_session,
    apply_ricoh_preview_effect,
    read_photo_settings,
    write_dehaze_session_settings,
    write_dehaze_settings,
    write_ricoh_preset,
    write_photo_settings,
    validate_basic_params,
)

_DEHAZE_DEFAULTS = asdict(DehazeParams())

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pass

try:
    import pillow_jxl
except Exception:
    pass

app = FastAPI(title="Imprint API", version="3.0.3")

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
_ENHANCE_PREVIEW_CACHE: dict[tuple, tuple[bytes, int, int, str, str, float, str]] = {}
_MAX_ENHANCE_PREVIEW_CACHE_BYTES = 16 * 1024 * 1024
_ENHANCE_THUMBNAIL_CACHE: dict[tuple[str, str, int], bytes] = {}
_ENHANCE_JOBS: dict[str, dict[str, Any]] = {}
_RICOH_PREVIEW_CACHE: dict[tuple, bytes] = {}
_MAX_RICOH_PREVIEW_CACHE_BYTES = 16 * 1024 * 1024
_RICOH_JOBS: dict[str, dict[str, Any]] = {}
_ENHANCE_LOCK = threading.RLock()
_FULL_RESOLUTION_PREVIEW_LOCK = threading.Lock()
# One decoded source and one display RGB8 base. The source lets repeated full
# preview edits skip RAW decoding; both are cleared together on photo changes.
_FULL_RESOLUTION_DECODED_CACHE: dict[tuple, tuple[np.ndarray, Any]] = {}
_FULL_RESOLUTION_DECODED_SOURCE_IDENTITIES: dict[tuple, tuple] = {}
_MAX_FULL_RESOLUTION_DECODED_BYTES = 256 * 1024 * 1024
_FULL_RESOLUTION_BASE_CACHE: dict[tuple, tuple[np.ndarray, tuple]] = {}
_MAX_FULL_RESOLUTION_BASE_BYTES = 96 * 1024 * 1024
# The camera profile path keeps at most one full-resolution source and one
# pre-exposure transferred image. All full-resolution arrays share one hard
# budget so large sources can simply bypass retention.
_FULL_RESOLUTION_CAMERA_SOURCE_CACHE: dict[
    tuple, tuple[np.ndarray, np.ndarray, np.ndarray]
] = {}
_FULL_RESOLUTION_CAMERA_PROCESSED_CACHE: dict[
    tuple, tuple[np.ndarray, np.ndarray, tuple]
] = {}
_FULL_RESOLUTION_EXPORT_SOURCE_CACHE: dict[tuple, Any] = {}
_MAX_FULL_RESOLUTION_PREVIEW_CACHE_BYTES = 640 * 1024 * 1024
_FULL_RESOLUTION_ACTIVE_SOURCE_KEY: tuple | None = None
_FULL_RESOLUTION_ACTIVE_SOURCE_SHAPE: tuple[int, ...] | None = None
_FULL_RESOLUTION_ACTIVE_SOURCE_IDENTITY: tuple | None = None
_FULL_RESOLUTION_CACHE_IDLE_SECONDS = 60.0
_FULL_RESOLUTION_CACHE_GENERATION = 0
_FULL_RESOLUTION_CACHE_TIMER: threading.Timer | None = None
_DISPLAY_PREVIEW_LOCK = threading.RLock()
_DISPLAY_PREVIEW_CACHE: OrderedDict[tuple, np.ndarray] = OrderedDict()
_MAX_DISPLAY_PREVIEW_BYTES = 64 * 1024 * 1024
_DEHAZED_PREVIEW_CACHE: OrderedDict[tuple, np.ndarray] = OrderedDict()
_MAX_DEHAZED_PREVIEW_BYTES = 64 * 1024 * 1024
_DEHAZED_PREVIEW_STATUS_CACHE: OrderedDict[tuple, tuple] = OrderedDict()
_MAX_DEHAZED_PREVIEW_STATUS_ENTRIES = 512
_PROCESSING_PREVIEW_CACHE: OrderedDict[tuple, tuple[np.ndarray, Any]] = OrderedDict()
_MAX_PROCESSING_PREVIEW_BYTES = 96 * 1024 * 1024
_MAX_ENHANCE_SESSIONS = 8
_MAX_ENHANCE_THUMBNAILS = 256

# 配置 CORS 中间件，允许 Tauri 桌面端以及本地开发环境请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "X-Image-Width", "X-Image-Height", "X-Dehaze-Nonlocal-Status",
        "X-Dehaze-Nonlocal-Reason", "X-Auto-Exposure-EV",
        "X-Auto-Exposure-Reason", "Server-Timing", "X-Preview-Base-Cache",
        "X-Preview-Decode-Cache", "X-Preview-Camera-Cache",
    ],
)


def get_free_port() -> int:
    """获取一个随机可用的本地 TCP 端口"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _insert_bounded_jpeg_preview(
    cache: dict[tuple, Any], key: tuple, entry: Any, *,
    max_entries: int, max_bytes: int, payload_index: int | None = None,
) -> None:
    """Insert a JPEG cache entry under its owning lock with entry and byte caps."""
    def payload_size(value: Any) -> int:
        payload = value if payload_index is None else value[payload_index]
        return len(payload)

    size = payload_size(entry)
    if size > max_bytes:
        return
    cache.pop(key, None)
    retained_bytes = sum(payload_size(value) for value in cache.values())
    while cache and (len(cache) >= max_entries or retained_bytes + size > max_bytes):
        oldest_key = next(iter(cache))
        oldest = cache.pop(oldest_key)
        retained_bytes -= payload_size(oldest)
    cache[key] = entry


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
    sort_backend: Literal["python", "native"] = "python"
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
    strength: float = Field(default=_DEHAZE_DEFAULTS["strength"], ge=0.0, le=1.0)
    naturalness: float = Field(default=_DEHAZE_DEFAULTS["naturalness"], ge=0.0, le=1.0)
    fog_retention: float = Field(default=_DEHAZE_DEFAULTS["fog_retention"], ge=0.0, le=1.0)
    local_contrast: float = Field(default=_DEHAZE_DEFAULTS["local_contrast"], ge=0.0, le=1.0)
    color_recovery: float = Field(default=_DEHAZE_DEFAULTS["color_recovery"], ge=0.0, le=1.0)
    color_protection: float = Field(default=_DEHAZE_DEFAULTS["color_protection"], ge=0.0, le=1.0)
    highlight_protection: float = Field(default=_DEHAZE_DEFAULTS["highlight_protection"], ge=0.0, le=1.0)
    shadow_protection: float = Field(default=_DEHAZE_DEFAULTS["shadow_protection"], ge=0.0, le=1.0)
    brightness_protection: float = Field(default=_DEHAZE_DEFAULTS["brightness_protection"], ge=0.0, le=1.0)

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
    auto_mode: bool = False
    auto_exposure: bool = False
    nonlocal_mode: Literal["off", "conservative", "strong"] = "off"
    algorithm: Literal["physical"] = "physical"
    basic_params: BasicParamsRequest = Field(default_factory=BasicParamsRequest)
    max_edge: int = Field(default=1800, ge=320, le=3000)
    preview_level: Literal[0, 1, 2] = 0
    full_resolution: bool = False
    mode: Literal["original", "dehazed"] = "dehazed"
    color_manage_srgb: bool = False
    use_gpu: bool = False
    render_backend: Literal["auto", "native", "pytorch", "cpu"] | None = None
    basic_backend: Literal["python", "native"] | None = None
    ricoh_backend: Literal["python", "native"] = "python"
    ricoh_preset_id: str | None = Field(default=None, min_length=1, max_length=80)


class EnhanceRevealRequest(BaseModel):
    session_id: str
    photo_id: str


class EnhanceRunRequest(BaseModel):
    session_id: str
    output_dir: str = ""
    photo_ids: list[str] | None = None
    params: EnhanceParamsRequest = Field(default_factory=EnhanceParamsRequest)
    params_by_photo: dict[str, EnhanceParamsRequest] = Field(default_factory=dict)
    auto_mode: bool = False
    auto_modes_by_photo: dict[str, bool] = Field(default_factory=dict)
    auto_exposure: bool = False
    auto_exposures_by_photo: dict[str, bool] = Field(default_factory=dict)
    nonlocal_mode: Literal["off", "conservative", "strong"] = "off"
    nonlocal_modes_by_photo: dict[str, Literal["off", "conservative", "strong"]] = Field(default_factory=dict)
    algorithms_by_photo: dict[str, Literal["physical"]] = Field(default_factory=dict)
    basic_params_by_photo: dict[str, BasicParamsRequest] = Field(default_factory=dict)
    preset_ids_by_photo: dict[str, str | None] = Field(default_factory=dict)
    ricoh_backend: Literal["python", "native"] = "python"
    compression: Literal["none", "lossless_jpeg", "jpegxl"] = "lossless_jpeg"
    bit_depth: Literal["source", "16"] = "source"
    use_gpu: bool = False
    render_backend: Literal["auto", "native", "pytorch", "cpu"] | None = None
    basic_backend: Literal["python", "native"] | None = None


class RicohApplyRequest(BaseModel):
    paths: list[str] = Field(min_length=1, max_length=32)
    preset_id: str = Field(min_length=1, max_length=80)


class EnhanceXmpRequest(BaseModel):
    session_id: str
    params_by_photo: dict[str, EnhanceParamsRequest] = Field(default_factory=dict)
    auto_mode: bool = False
    auto_modes_by_photo: dict[str, bool] = Field(default_factory=dict)
    auto_exposure: bool = False
    auto_exposures_by_photo: dict[str, bool] = Field(default_factory=dict)
    nonlocal_mode: Literal["off", "conservative", "strong"] = "off"
    nonlocal_modes_by_photo: dict[str, Literal["off", "conservative", "strong"]] = Field(default_factory=dict)
    algorithms_by_photo: dict[str, Literal["physical"]] = Field(default_factory=dict)
    basic_params_by_photo: dict[str, BasicParamsRequest] = Field(default_factory=dict)
    preset_ids_by_photo: dict[str, str | None] = Field(default_factory=dict)


class RicohSessionRequest(BaseModel):
    session_id: str
    preset_id: str = Field(min_length=1, max_length=80)
    basic_params_by_photo: dict[str, BasicParamsRequest] = Field(default_factory=dict)
    preset_ids_by_photo: dict[str, str | None] = Field(default_factory=dict)


class PhotoSettingsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    photo_id: str
    dehaze_params: EnhanceParamsRequest
    auto_mode: bool = False
    auto_exposure: bool = False
    nonlocal_mode: Literal["off", "conservative", "strong"] = "off"
    dehaze_algorithm: Literal["physical"] = "physical"
    basic_params: BasicParamsRequest
    ricoh_preset_id: str | None = Field(default=None, min_length=1, max_length=80)


class RicohPreviewRequest(BaseModel):
    session_id: str
    photo_id: str
    preset_id: str = Field(min_length=1, max_length=80)
    basic_params: BasicParamsRequest = Field(default_factory=BasicParamsRequest)
    max_edge: int = Field(default=1800, ge=320, le=3000)
    ricoh_backend: Literal["python", "native"] = "python"


class RicohRunRequest(BaseModel):
    session_id: str
    preset_id: str | None = Field(default=None, min_length=1, max_length=80)
    output_dir: str = ""
    ricoh_backend: Literal["python", "native"] = "python"
    basic_backend: Literal["python", "native"] = "python"
    basic_params_by_photo: dict[str, BasicParamsRequest] = Field(default_factory=dict)
    preset_ids_by_photo: dict[str, str | None] = Field(default_factory=dict)


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
            req.preset_ids_by_photo,
        )
    except KeyError:
        return JSONResponse(status_code=400, content={"error": "未知的理光预设"})
    except RicohBatchLimitError:
        return JSONResponse(status_code=413, content={"error": "一次最多处理 500 张照片"})


@app.post("/api/photo/settings")
def save_photo_settings_snapshot(req: PhotoSettingsRequest):
    """Save one complete settings snapshot for a photo in an active session."""
    with _ENHANCE_LOCK:
        session = _ENHANCE_SESSIONS.get(req.session_id)
        path = session.get("files", {}).get(req.photo_id) if session else None
    if session is None or path is None:
        return JSONResponse(status_code=404, content={"error": "照片会话或照片已失效"})
    try:
        result = write_photo_settings(
            path,
            req.dehaze_params.to_params().__dict__,
            req.basic_params.values(),
            req.ricoh_preset_id,
            req.auto_mode,
            nonlocal_mode=req.nonlocal_mode,
            auto_exposure=req.auto_exposure,
            compatibility_curves=_build_dehaze_xmp_curves(
                req.session_id, req.photo_id, Path(path), req.dehaze_params.to_params(),
                req.auto_mode, req.nonlocal_mode, req.auto_exposure,
            ),
        )
        return {
            "photo_id": req.photo_id,
            "status": "saved",
            "sidecar_status": result["status"],
            "name": result["name"],
            "dehaze_compatibility": "approximate" if req.dehaze_params.strength > 0 else "disabled",
        }
    except KeyError:
        return JSONResponse(status_code=400, content={"error": "未知的理光预设"})
    except FileNotFoundError:
        return JSONResponse(status_code=404, content={"error": "当前照片不存在"})
    except FileExistsError:
        return JSONResponse(status_code=409, content={"error": "已存在同名 XMP，未覆盖"})
    except ValueError:
        return JSONResponse(status_code=400, content={"error": "照片设置无效"})
    except Exception:
        # Never include server-side photo paths or filesystem error details.
        return JSONResponse(status_code=500, content={"error": "保存照片设置失败"})


def _cached_display_preview(session_id: str, photo_id: str, path: Path,
                            max_edge: int, *, optical_correction: bool = False) -> np.ndarray:
    """Share decoded, color-managed previews across original/effect requests.

    A bounded cache avoids decoding the same RAW again for every slider change.
    Decoding happens outside the cache lock so different photos can load in parallel.
    """
    stat = path.stat()
    key = (session_id, photo_id, max_edge, stat.st_mtime_ns, stat.st_size,
           optical_correction, LENS_PREVIEW_VERSION)
    with _DISPLAY_PREVIEW_LOCK:
        cached = _DISPLAY_PREVIEW_CACHE.get(key)
        if cached is not None:
            _DISPLAY_PREVIEW_CACHE.move_to_end(key)
            return cached

    image, metadata = _cached_processing_preview(session_id, photo_id, path, max_edge)
    if optical_correction:
        image, _, _ = _correct_enhanced_raw(image, metadata, path, preview=True)
    display = _display_rgb8(image, linear=getattr(metadata, "color_space", "") == "Linear sRGB")
    if getattr(metadata, "source_kind", "") == "rgb":
        display = standard_preview_to_srgb(display, path)
    display.setflags(write=False)

    if display.nbytes <= _MAX_DISPLAY_PREVIEW_BYTES:
        with _DISPLAY_PREVIEW_LOCK:
            cached = _DISPLAY_PREVIEW_CACHE.get(key)
            if cached is not None:
                _DISPLAY_PREVIEW_CACHE.move_to_end(key)
                return cached
            for old_key in [old_key for old_key in _DISPLAY_PREVIEW_CACHE
                            if old_key[:2] == key[:2] and old_key != key]:
                _DISPLAY_PREVIEW_CACHE.pop(old_key, None)
            while (_DISPLAY_PREVIEW_CACHE and
                   sum(value.nbytes for value in _DISPLAY_PREVIEW_CACHE.values())
                   + display.nbytes > _MAX_DISPLAY_PREVIEW_BYTES):
                _DISPLAY_PREVIEW_CACHE.popitem(last=False)
            _DISPLAY_PREVIEW_CACHE[key] = display
    return display


def _render_mode(render_backend: str | None, use_gpu: bool = False) -> str:
    return render_backend or ("legacy_gpu" if use_gpu else "cpu")


def _camera_profile_render_backend(render_backend: str) -> str:
    """Map legacy GPU selectors onto the camera profile renderer's GPU mode."""
    if render_backend in ("auto", "native"):
        return render_backend
    if render_backend in ("legacy_gpu", "pytorch"):
        return "auto"
    return "cpu"


def _basic_mode(basic_backend: str | None, render_mode: str) -> str:
    if basic_backend is not None:
        return basic_backend
    return "native" if render_mode in ("auto", "native") else "python"


@lru_cache(maxsize=1)
def _load_lcms2() -> Any:
    """Load Pillow's bundled LittleCMS for RGB16 ICC transforms."""
    candidates: list[str] = []
    pil_dir = Path(ImageCms.__file__).resolve().parent
    for pattern in (
        ".dylibs/liblcms2*.dylib", ".libs/liblcms2*.so*", "lcms2.dll",
        "liblcms2.dll", "liblcms2*.dylib", "liblcms2*.so*",
    ):
        candidates.extend(str(candidate) for candidate in sorted(pil_dir.glob(pattern)))
    discovered = ctypes.util.find_library("lcms2")
    if discovered:
        candidates.append(discovered)
    errors = []
    for candidate in dict.fromkeys(candidates):
        try:
            library = ctypes.CDLL(candidate)
            library.cmsOpenProfileFromMem.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
            library.cmsOpenProfileFromMem.restype = ctypes.c_void_p
            library.cmsCreate_sRGBProfile.argtypes = []
            library.cmsCreate_sRGBProfile.restype = ctypes.c_void_p
            library.cmsCreateTransform.argtypes = [
                ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p,
                ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
            ]
            library.cmsCreateTransform.restype = ctypes.c_void_p
            library.cmsDoTransform.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32,
            ]
            library.cmsDoTransform.restype = None
            library.cmsDeleteTransform.argtypes = [ctypes.c_void_p]
            library.cmsDeleteTransform.restype = None
            library.cmsCloseProfile.argtypes = [ctypes.c_void_p]
            library.cmsCloseProfile.restype = ctypes.c_int
            return library
        except (OSError, AttributeError) as exc:
            errors.append(str(exc))
    detail = errors[-1] if errors else "LittleCMS library not found"
    raise RuntimeError(f"无法加载 LittleCMS 进行 16-bit ICC 色彩转换: {detail}")


def _icc_rgb16_to_srgb(image: np.ndarray, profile_bytes: bytes) -> np.ndarray:
    """Transform uint16 RGB through ICC without reducing it to 8-bit."""
    if not profile_bytes:
        return image
    lcms = _load_lcms2()
    profile_buffer = ctypes.create_string_buffer(profile_bytes)
    source_profile = lcms.cmsOpenProfileFromMem(
        ctypes.cast(profile_buffer, ctypes.c_void_p), len(profile_bytes),
    )
    if not source_profile:
        raise ValueError("无法读取照片中的 ICC 色彩配置")
    target_profile = lcms.cmsCreate_sRGBProfile()
    if not target_profile:
        lcms.cmsCloseProfile(source_profile)
        raise RuntimeError("无法创建 sRGB 色彩配置")
    # LittleCMS TYPE_RGB_16: PT_RGB=4, three channels and two bytes/sample.
    rgb16_format = (4 << 16) | (3 << 3) | 2
    transform = lcms.cmsCreateTransform(
        source_profile, rgb16_format, target_profile, rgb16_format, 0, 0,
    )
    if not transform:
        lcms.cmsCloseProfile(source_profile)
        lcms.cmsCloseProfile(target_profile)
        raise ValueError("无法将照片 ICC 色彩配置转换到 sRGB")
    source = np.ascontiguousarray(image, dtype=np.uint16)
    output = np.empty_like(source)
    try:
        pixel_count = int(source.shape[0]) * int(source.shape[1])
        lcms.cmsDoTransform(
            transform,
            ctypes.c_void_p(source.ctypes.data),
            ctypes.c_void_p(output.ctypes.data),
            pixel_count,
        )
    finally:
        lcms.cmsDeleteTransform(transform)
        lcms.cmsCloseProfile(source_profile)
        lcms.cmsCloseProfile(target_profile)
    return output


def _standard_rgb_to_srgb(image: np.ndarray, path: Path) -> np.ndarray:
    """Apply the embedded RGB profile while retaining high-bit-depth samples."""
    if image.dtype == np.uint8:
        return standard_preview_to_srgb(image, path)
    with Image.open(path) as source:
        profile_bytes = source.info.get("icc_profile")
    if not profile_bytes:
        return image
    if image.dtype == np.uint16:
        return _icc_rgb16_to_srgb(image, profile_bytes)
    if image.dtype == np.float32:
        encoded16 = np.clip(np.rint(image * 65535.0), 0, 65535).astype(np.uint16)
        return _icc_rgb16_to_srgb(encoded16, profile_bytes).astype(np.float32) / 65535.0
    raise TypeError("ICC 色彩转换仅支持 8-bit、16-bit 或归一化 float32 RGB")


def _prepare_dehaze_input(
    image: np.ndarray, metadata: Any, path: str | Path, *, color_manage_srgb: bool,
) -> np.ndarray:
    """Convert decoded pixels to contiguous linear float32 RGB for the core."""
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("去朦胧输入必须是 RGB 图像")
    source_kind = getattr(metadata, "source_kind", "")
    is_raw = source_kind == "raw"
    already_linear = is_raw or getattr(metadata, "color_space", "") == "Linear sRGB"
    working = image
    if not already_linear and color_manage_srgb:
        working = _standard_rgb_to_srgb(image, Path(path))
    if working.dtype == np.uint8:
        normalized = working.astype(np.float32) / 255.0
    elif working.dtype == np.uint16:
        normalized = working.astype(np.float32) / 65535.0
    elif working.dtype == np.float32:
        normalized = working
    else:
        raise TypeError("去朦胧输入仅支持 uint8、uint16 或 float32 RGB")
    if not np.isfinite(normalized).all() or np.any(normalized < 0) or np.any(normalized > 1):
        raise ValueError("去朦胧输入像素必须有限且位于 [0, 1]")
    if already_linear:
        return np.ascontiguousarray(normalized, dtype=np.float32)
    encoded = normalized
    linear = np.where(
        encoded <= 0.04045,
        encoded / 12.92,
        np.power((encoded + 0.055) / 1.055, 2.4),
    )
    return np.ascontiguousarray(linear, dtype=np.float32)


def _linear_float_to_uint16(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.float32 or image.ndim != 3 or image.shape[2] != 3:
        raise TypeError("线性去朦胧结果必须是 float32 RGB")
    if not np.isfinite(image).all():
        raise ValueError("线性去朦胧结果包含非有限值")
    return np.clip(np.rint(image * 65535.0), 0, 65535).astype(np.uint16)


def _render_dehaze(image: np.ndarray, params: DehazeParams, mode: str,
                   auto_mode: bool = False,
                   nonlocal_mode: str | None = None,
                   diagnostics: dict[str, Any] | None = None,
                   auto_exposure: bool = False,
                   auto_exposure_ev: float | None = None,
                   auto_exposure_reason: str | None = None) -> np.ndarray:
    """Render linear float RGB through the single physical dehaze operator."""
    # Historical GPU settings now prefer the optional float native operator.
    # CPU fallback always retains the same inverse, including older bundles.
    backend = "native" if mode == "native" else (
        "auto" if mode in ("auto", "legacy_gpu", "pytorch") else "cpu"
    )
    kwargs: dict[str, Any] = {
        "backend": backend, "spatial": auto_mode, "nonlocal_mode": nonlocal_mode,
    }
    source = image
    exposure_diagnostics: dict[str, Any] = {}
    apply_exposure = auto_exposure or auto_exposure_ev is not None
    if apply_exposure:
        source = apply_auto_exposure(
            image, enabled=True, ev_override=auto_exposure_ev,
            reason_override=auto_exposure_reason,
            diagnostics=exposure_diagnostics,
        )
    physical_diagnostics: dict[str, Any] | None = {} if diagnostics is not None else None
    if physical_diagnostics is not None:
        kwargs["diagnostics"] = physical_diagnostics
    rendered = apply_physical_dehaze(source, params, **kwargs)
    if diagnostics is not None:
        diagnostics.clear()
        if physical_diagnostics is not None:
            diagnostics.update(physical_diagnostics)
        if apply_exposure:
            diagnostics.update(exposure_diagnostics)
        else:
            diagnostics.update(auto_exposure_ev=0.0, auto_exposure_reason="off")
    return rendered


_NONLOCAL_STATUS_REASON_CODES = {
    "manual_mode", "zero_strength", "low_airlight", "uncertain_airlight",
    "clipped_highlights", "backlit_scene", "no_reliable_rays",
    "solver_nonconverged", "reliable_estimate_unavailable",
}


def _nonlocal_preview_status(
    nonlocal_mode: str | None, diagnostics: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """Describe the nonlocal result reported by the render that produced the image."""
    mode = resolve_nonlocal_mode(nonlocal_mode)
    details = diagnostics or {}
    reason = str(details.get("fallback_reason") or "")
    if reason not in _NONLOCAL_STATUS_REASON_CODES:
        reason = "reliable_estimate_unavailable" if mode != "off" else ""
    if mode == "off":
        return "off", reason
    if bool(details.get("nonlocal_active")):
        return "active", ""
    return "fallback", reason or "reliable_estimate_unavailable"


def _remember_dehazed_preview_status(key: tuple, status: tuple[str, str, float, str]) -> None:
    """Keep status metadata paired with the bounded dehazed-display image cache."""
    with _DISPLAY_PREVIEW_LOCK:
        for old_key in [
            old_key for old_key in _DEHAZED_PREVIEW_STATUS_CACHE
            if old_key[:2] == key[:2] and old_key[2:4] != key[2:4]
        ]:
            _DEHAZED_PREVIEW_STATUS_CACHE.pop(old_key, None)
        _DEHAZED_PREVIEW_STATUS_CACHE[key] = status
        _DEHAZED_PREVIEW_STATUS_CACHE.move_to_end(key)
        while len(_DEHAZED_PREVIEW_STATUS_CACHE) > _MAX_DEHAZED_PREVIEW_STATUS_ENTRIES:
            evicted_key, _ = _DEHAZED_PREVIEW_STATUS_CACHE.popitem(last=False)
            _DEHAZED_PREVIEW_CACHE.pop(evicted_key, None)


def _set_preview_status(
    target: dict[str, Any] | None, status: tuple[str, str, float, str],
) -> None:
    if target is not None:
        target.clear()
        target.update(
            status=status[0], reason=status[1],
            auto_exposure_ev=status[2], auto_exposure_reason=status[3],
            camera_profile_status=status[4] if len(status) > 4 else "legacy",
            camera_profile_backend=status[5] if len(status) > 5 else "legacy",
        )


def _render_basic(image: np.ndarray, basic: dict[str, float], mode: str) -> np.ndarray:
    if mode == "native":
        try:
            return native_basic(image, basic)
        except Exception:
            pass
    return apply_basic_preview_effect(image, basic)


# Low-resolution camera/reference pairs use a bounded multi-photo cache.
# Full-resolution profile layers use the single-photo caches above and the
# shared full-resolution memory budget.
_CAMERA_PROFILE_SOURCE_CACHE: OrderedDict[tuple, tuple[np.ndarray, np.ndarray, np.ndarray]] = OrderedDict()
_MAX_CAMERA_PROFILE_SOURCE_BYTES = 128 * 1024 * 1024


def _full_resolution_preview_cache_bytes() -> int:
    """Count retained array payloads across every full-resolution preview layer."""
    arrays: dict[int, np.ndarray] = {}

    def include(array: Any) -> None:
        if isinstance(array, np.ndarray):
            arrays[id(array)] = array

    for value in _FULL_RESOLUTION_DECODED_CACHE.values():
        include(value[0])
    for value in _FULL_RESOLUTION_BASE_CACHE.values():
        include(value[0])
    for camera, reference, neutral in _FULL_RESOLUTION_CAMERA_SOURCE_CACHE.values():
        for array in (camera, reference, neutral):
            include(array)
    for processed, neutral, _status in _FULL_RESOLUTION_CAMERA_PROCESSED_CACHE.values():
        include(processed)
        include(neutral)
    for source in _FULL_RESOLUTION_EXPORT_SOURCE_CACHE.values():
        for name in ("camera_rgb16", "reference_rgb16", "srgb_to_camera"):
            include(getattr(source, name, None))
    return sum(array.nbytes for array in arrays.values())


def _clear_full_resolution_camera_source_cache() -> None:
    """Drop the preview camera pair and the export snapshot that shares it."""
    _FULL_RESOLUTION_CAMERA_SOURCE_CACHE.clear()
    _FULL_RESOLUTION_EXPORT_SOURCE_CACHE.clear()


def _full_resolution_export_source_key(
    session_id: str, photo_id: str, source_identity: tuple,
    shape: tuple[int, ...], correction_version: str,
) -> tuple:
    return (session_id, photo_id, source_identity, tuple(shape), correction_version)


def _valid_full_resolution_export_source_cache(
    source: Any, source_identity: tuple, shape: tuple[int, ...],
    correction_version: str,
) -> bool:
    if (getattr(source, "source_identity", None) != source_identity
            or tuple(getattr(source, "source_shape", ())) != tuple(shape)
            or getattr(source, "correction_version", None) != correction_version):
        return False
    camera = getattr(source, "camera_rgb16", None)
    reference = getattr(source, "reference_rgb16", None)
    matrix = getattr(source, "srgb_to_camera", None)
    return bool(
        isinstance(camera, np.ndarray) and camera.dtype == np.uint16
        and camera.shape == tuple(shape) and camera.flags.c_contiguous
        and not camera.flags.writeable
        and isinstance(reference, np.ndarray) and reference.dtype == np.uint16
        and reference.shape == tuple(shape) and reference.flags.c_contiguous
        and not reference.flags.writeable
        and isinstance(matrix, np.ndarray) and matrix.shape == (3, 3)
        and np.isfinite(matrix).all() and not matrix.flags.writeable
    )


def _trim_full_resolution_preview_cache_budget() -> None:
    """Evict redundant full-resolution layers until their combined cap holds."""
    while _full_resolution_preview_cache_bytes() > _MAX_FULL_RESOLUTION_PREVIEW_CACHE_BYTES:
        if _FULL_RESOLUTION_BASE_CACHE:
            _FULL_RESOLUTION_BASE_CACHE.clear()
        elif _FULL_RESOLUTION_DECODED_CACHE:
            _FULL_RESOLUTION_DECODED_CACHE.clear()
            _FULL_RESOLUTION_DECODED_SOURCE_IDENTITIES.clear()
        elif _FULL_RESOLUTION_CAMERA_SOURCE_CACHE:
            # The transferred stage still renders exposure edits without these
            # source arrays. Keep it before the more expensive reusable source.
            _clear_full_resolution_camera_source_cache()
        elif _FULL_RESOLUTION_EXPORT_SOURCE_CACHE:
            _FULL_RESOLUTION_EXPORT_SOURCE_CACHE.clear()
        elif _FULL_RESOLUTION_CAMERA_PROCESSED_CACHE:
            _FULL_RESOLUTION_CAMERA_PROCESSED_CACHE.clear()
            clear_camera_profile_gpu_cache()
        else:
            break


def _profile_display_base(processed: np.ndarray, reference: np.ndarray,
                          metadata: Any, path: Path, profile: Any,
                          exposure_ev: float, *, preview: bool,
                          backend: str = "cpu",
                          full_cache_identity: tuple | None = None,
                          full_export_identity: tuple | None = None,
                          stage_cache_key: tuple | None = None,
                          stage_status: tuple | None = None,
                          camera_cache_out: dict[str, str] | None = None
                          ) -> tuple[np.ndarray, str, str]:
    """Render the native camera anchor before display quantization and EV clipping.

    On failure the old display path still applies EV exactly once. The caller
    consequently clears only the manual exposure from subsequent basic edits.
    """
    try:
        if getattr(metadata, "source_kind", "") != "raw":
            raise ValueError("camera profile requires RAW input")
        if reference.shape != processed.shape:
            reference = cv2.resize(reference, (processed.shape[1], processed.shape[0]),
                                   interpolation=cv2.INTER_AREA)
        stat = path.stat()
        key = (str(path), stat.st_mtime_ns, stat.st_size, reference.shape,
               preview, LENS_PREVIEW_VERSION)
        if preview:
            with _DISPLAY_PREVIEW_LOCK:
                cached = _CAMERA_PROFILE_SOURCE_CACHE.get(key)
                if cached is not None:
                    _CAMERA_PROFILE_SOURCE_CACHE.move_to_end(key)
            if cached is None:
                camera, neutral = load_camera_rgb(path, reference.shape, preview)
                neutral = np.asarray(neutral, dtype=np.float64)
                camera, _, _ = _correct_enhanced_raw(
                    camera, metadata, path, preview=True, highlight_reference=reference,
                )
                corrected_reference, _, _ = _correct_enhanced_raw(reference, metadata, path, preview=True)
                cached = (camera, corrected_reference, neutral)
                size = sum(value.nbytes for value in cached)
                if size <= _MAX_CAMERA_PROFILE_SOURCE_BYTES:
                    for value in cached:
                        value.setflags(write=False)
                    with _DISPLAY_PREVIEW_LOCK:
                        while (_CAMERA_PROFILE_SOURCE_CACHE and
                               sum(sum(a.nbytes for a in value) for value in _CAMERA_PROFILE_SOURCE_CACHE.values())
                               + size > _MAX_CAMERA_PROFILE_SOURCE_BYTES):
                            _CAMERA_PROFILE_SOURCE_CACHE.popitem(last=False)
                        _CAMERA_PROFILE_SOURCE_CACHE[key] = cached
            camera, corrected_reference, neutral = cached
            camera_processed = transfer_enhancement(camera, corrected_reference, processed)
        else:
            if full_cache_identity is None:
                raise ValueError("full profile cache requires a source identity")
            source_key = (*full_cache_identity, tuple(reference.shape), LENS_PREVIEW_VERSION)
            cached_source = _FULL_RESOLUTION_CAMERA_SOURCE_CACHE.get(source_key)
            if cached_source is None:
                uncorrected_camera, neutral = load_camera_rgb(path, reference.shape, False)
                neutral = np.asarray(neutral, dtype=np.float64).copy()
                camera, camera_lens_result, camera_gain_applied = _correct_enhanced_raw(
                    uncorrected_camera, metadata, path, preview=True,
                    highlight_reference=reference,
                )
                corrected_reference, reference_lens_result, reference_gain_applied = (
                    _correct_enhanced_raw(reference, metadata, path, preview=True)
                )
                cached_source = (camera, corrected_reference, neutral)
                source_cache = None
                if full_export_identity is not None:
                    source_cache = build_enhanced_dng_source_cache(
                        path, uncorrected_camera, reference, camera, corrected_reference,
                        camera_lens_result, reference_lens_result,
                        correction_version=LENS_PREVIEW_VERSION,
                        camera_gain_applied=camera_gain_applied,
                        reference_gain_applied=reference_gain_applied,
                    )
                if camera.dtype == np.uint16 and corrected_reference.dtype == np.uint16:
                    for value in cached_source:
                        value.setflags(write=False)
                    if sum(value.nbytes for value in cached_source) <= _MAX_FULL_RESOLUTION_PREVIEW_CACHE_BYTES:
                        _clear_full_resolution_camera_source_cache()
                        _FULL_RESOLUTION_CAMERA_SOURCE_CACHE[source_key] = cached_source
                        if (source_cache is not None and full_export_identity is not None
                                and _valid_full_resolution_export_source_cache(
                                    source_cache, full_export_identity,
                                    tuple(reference.shape), LENS_PREVIEW_VERSION,
                                )):
                            export_key = _full_resolution_export_source_key(
                                full_cache_identity[0], full_cache_identity[1],
                                full_export_identity, tuple(reference.shape), LENS_PREVIEW_VERSION,
                            )
                            _FULL_RESOLUTION_EXPORT_SOURCE_CACHE[export_key] = source_cache
                        _trim_full_resolution_preview_cache_budget()
            else:
                if camera_cache_out is not None:
                    camera_cache_out["source"] = "hit"
            camera, corrected_reference, neutral = cached_source

            cached_processed = (
                _FULL_RESOLUTION_CAMERA_PROCESSED_CACHE.get(stage_cache_key)
                if stage_cache_key is not None else None
            )
            if cached_processed is not None:
                camera_processed, neutral, _cached_status = cached_processed
                if camera_cache_out is not None:
                    camera_cache_out["stage"] = "hit"
            else:
                camera_processed = transfer_enhancement(camera, corrected_reference, processed)
                if (stage_cache_key is not None and camera_processed.dtype == np.uint16
                        and camera_processed.shape == processed.shape
                        and camera_processed.nbytes + neutral.nbytes <= _MAX_FULL_RESOLUTION_PREVIEW_CACHE_BYTES):
                    camera_processed.setflags(write=False)
                    neutral.setflags(write=False)
                    _FULL_RESOLUTION_CAMERA_PROCESSED_CACHE.clear()
                    _FULL_RESOLUTION_CAMERA_PROCESSED_CACHE[stage_cache_key] = (
                        camera_processed, neutral, tuple(stage_status or ()),
                    )
                    _trim_full_resolution_preview_cache_budget()
                if camera_cache_out is not None:
                    camera_cache_out["stage"] = "miss"
            if camera_cache_out is not None:
                camera_cache_out.setdefault("source", "miss")
        display = render_profile(
            camera_processed, neutral, profile, exposure_ev,
            backend=_camera_profile_render_backend(backend),
        )
        return display, "applied", get_last_profile_backend()
    except Exception as exc:
        # A bad/missing optional profile must not break slider interaction or
        # accidentally suppress exposure. Keep errors out of HTTP image bodies.
        logging.getLogger(__name__).debug("Camera profile preview fallback: %s", type(exc).__name__)
        display = _display_rgb8(processed, linear=True)
        if exposure_ev:
            display = _render_basic(display, BasicParamsRequest(exposure=exposure_ev).values(), "python")
        return display, "fallback", "legacy"


def _full_resolution_profile_stage_key(
    source_identity: tuple, shape: tuple[int, ...], params: DehazeParams,
    req: EnhancePreviewRequest, backend: str, profile: Any,
) -> tuple:
    """Identify the transferred camera RGB16 layer before manual exposure."""
    return (
        source_identity, shape, LENS_PREVIEW_VERSION,
        DEHAZE_ALGORITHM_VERSION, AUTO_EXPOSURE_ALGORITHM_VERSION,
        tuple(params.__dict__.items()), req.algorithm, req.auto_mode,
        req.auto_exposure, req.nonlocal_mode, req.color_manage_srgb,
        backend, PROFILE_PREVIEW_VERSION, BASIC_PREVIEW_VERSION,
        profile.fingerprint,
        req.ricoh_preset_id if req.mode == "dehazed" else None,
        req.ricoh_backend,
    )


def _complete_export_basic_params(
    params: Mapping[str, float] | None,
) -> dict[str, float]:
    """Return the complete Camera Raw settings packet for a DNG export."""
    return BasicParamsRequest(**dict(params or {})).values()


def _export_metadata_with_basic_params(
    source_metadata: Mapping[str, Any], basic_params: dict[str, float],
) -> dict[str, Any]:
    """Keep source EXIF and explicitly attach this export's editable settings."""
    output_metadata = dict(source_metadata)
    # These values describe an existing source-side edit and must not be copied
    # to a new DNG, where they could apply the edit a second time.
    for key in (
        "ACRBasicParams", "XMP", "XMPPacket", "XMPData",
        "DehazeCompatibilityCurves",
    ):
        output_metadata.pop(key, None)
    output_metadata["ACRBasicParams"] = dict(basic_params)
    return output_metadata


def _clear_full_resolution_preview_caches() -> None:
    """Clear full-resolution caches. The caller must hold the full-preview lock."""
    global _FULL_RESOLUTION_ACTIVE_SOURCE_KEY, _FULL_RESOLUTION_ACTIVE_SOURCE_SHAPE
    global _FULL_RESOLUTION_ACTIVE_SOURCE_IDENTITY
    _FULL_RESOLUTION_DECODED_CACHE.clear()
    _FULL_RESOLUTION_DECODED_SOURCE_IDENTITIES.clear()
    _FULL_RESOLUTION_BASE_CACHE.clear()
    _clear_full_resolution_camera_source_cache()
    _FULL_RESOLUTION_CAMERA_PROCESSED_CACHE.clear()
    clear_camera_profile_gpu_cache()
    clear_native_warp_cache()
    _FULL_RESOLUTION_ACTIVE_SOURCE_KEY = None
    _FULL_RESOLUTION_ACTIVE_SOURCE_SHAPE = None
    _FULL_RESOLUTION_ACTIVE_SOURCE_IDENTITY = None


def _release_unused_preview_memory() -> None:
    """Best-effort return of freed preview pages to the platform allocator."""
    try:
        gc.collect()
    except Exception:
        pass

    try:
        if sys.platform == "darwin":
            allocator = ctypes.CDLL(None)
            release = getattr(allocator, "malloc_zone_pressure_relief", None)
            if release is None:
                return
            release.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            release.restype = ctypes.c_size_t
            # The return value describes allocator behavior and is not a byte
            # count suitable for reporting as reclaimed process RSS.
            release(None, 0)
        elif sys.platform.startswith("linux"):
            allocator = ctypes.CDLL(None)
            trim = getattr(allocator, "malloc_trim", None)
            if trim is None:
                return
            trim.argtypes = [ctypes.c_size_t]
            trim.restype = ctypes.c_int
            trim(0)
    except Exception:
        # Allocator symbols vary by libc and OS release; cache cleanup remains
        # correct even when the platform does not expose a release hook.
        pass


def _expire_full_resolution_preview_caches(generation: int) -> None:
    """Expire idle full-resolution arrays without retaining them in the timer."""
    global _FULL_RESOLUTION_CACHE_TIMER
    with _FULL_RESOLUTION_PREVIEW_LOCK:
        if generation != _FULL_RESOLUTION_CACHE_GENERATION:
            return
        _clear_full_resolution_preview_caches()
        _FULL_RESOLUTION_CACHE_TIMER = None
        _release_unused_preview_memory()


def _touch_full_resolution_preview_cache() -> None:
    """Reset the idle timer. Called with the full-preview lock held."""
    global _FULL_RESOLUTION_CACHE_GENERATION, _FULL_RESOLUTION_CACHE_TIMER
    old_timer = _FULL_RESOLUTION_CACHE_TIMER
    if old_timer is not None:
        old_timer.cancel()
    _FULL_RESOLUTION_CACHE_GENERATION += 1
    timer = threading.Timer(
        _FULL_RESOLUTION_CACHE_IDLE_SECONDS,
        _expire_full_resolution_preview_caches,
        args=(_FULL_RESOLUTION_CACHE_GENERATION,),
    )
    timer.daemon = True
    _FULL_RESOLUTION_CACHE_TIMER = timer
    timer.start()


def _shutdown_full_resolution_preview_cache() -> None:
    """Cancel the idle callback and release full-resolution arrays on shutdown."""
    global _FULL_RESOLUTION_CACHE_GENERATION, _FULL_RESOLUTION_CACHE_TIMER
    with _FULL_RESOLUTION_PREVIEW_LOCK:
        _FULL_RESOLUTION_CACHE_GENERATION += 1
        timer = _FULL_RESOLUTION_CACHE_TIMER
        _FULL_RESOLUTION_CACHE_TIMER = None
        if timer is not None:
            timer.cancel()
        _clear_full_resolution_preview_caches()
        _release_unused_preview_memory()
    with _DISPLAY_PREVIEW_LOCK:
        _CAMERA_PROFILE_SOURCE_CACHE.clear()


app.router.add_event_handler("shutdown", _shutdown_full_resolution_preview_cache)


def _full_resolution_decoded_image(
    req: EnhancePreviewRequest, path: Path,
) -> tuple[np.ndarray, Any, bool]:
    """Return one immutable full-resolution decode, called under its cache lock."""
    global _FULL_RESOLUTION_ACTIVE_SOURCE_KEY, _FULL_RESOLUTION_ACTIVE_SOURCE_SHAPE
    global _FULL_RESOLUTION_ACTIVE_SOURCE_IDENTITY
    stat = path.stat()
    key = (req.session_id, req.photo_id, str(path), stat.st_mtime_ns, stat.st_size)
    strong_identity = enhanced_dng_source_identity(path)
    if (_FULL_RESOLUTION_ACTIVE_SOURCE_KEY != key
            or _FULL_RESOLUTION_ACTIVE_SOURCE_IDENTITY != strong_identity):
        # All retained full-size layers belong to one active photo/source.
        _clear_full_resolution_preview_caches()
        _FULL_RESOLUTION_ACTIVE_SOURCE_KEY = key
        _FULL_RESOLUTION_ACTIVE_SOURCE_IDENTITY = strong_identity
    cached = _FULL_RESOLUTION_DECODED_CACHE.get(key)
    if cached is not None:
        if _FULL_RESOLUTION_DECODED_SOURCE_IDENTITIES.get(key) != strong_identity:
            # Preserve the historical decoded-cache key shape while validating
            # its content against the stronger file identity separately.
            _clear_full_resolution_preview_caches()
            _FULL_RESOLUTION_ACTIVE_SOURCE_KEY = key
            _FULL_RESOLUTION_ACTIVE_SOURCE_IDENTITY = strong_identity
            cached = None
    if cached is not None:
        _FULL_RESOLUTION_ACTIVE_SOURCE_SHAPE = tuple(cached[0].shape)
        _touch_full_resolution_preview_cache()
        return cached[0], cached[1], True

    image, metadata = read_image(path, preview=False)
    if enhanced_dng_source_identity(path) != strong_identity:
        raise RuntimeError("source changed while decoding full-resolution preview")
    image.setflags(write=False)
    _FULL_RESOLUTION_ACTIVE_SOURCE_SHAPE = tuple(image.shape)
    if image.nbytes <= _MAX_FULL_RESOLUTION_DECODED_BYTES:
        _FULL_RESOLUTION_DECODED_CACHE[key] = (image, metadata)
        _FULL_RESOLUTION_DECODED_SOURCE_IDENTITIES[key] = strong_identity
        _trim_full_resolution_preview_cache_budget()
    _touch_full_resolution_preview_cache()
    return image, metadata, False


def _capture_full_resolution_export_sources(
    session_id: str, photo_id: str, path: Path,
) -> tuple[np.ndarray | None, Any, Any, bool, tuple]:
    """Take strong references to a matching decode and camera snapshot under lock."""
    global _FULL_RESOLUTION_ACTIVE_SOURCE_KEY, _FULL_RESOLUTION_ACTIVE_SOURCE_IDENTITY
    global _FULL_RESOLUTION_ACTIVE_SOURCE_SHAPE
    stat = path.stat()
    decoded_key = (session_id, photo_id, str(path), stat.st_mtime_ns, stat.st_size)
    source_identity = enhanced_dng_source_identity(path)
    with _FULL_RESOLUTION_PREVIEW_LOCK:
        active_key = _FULL_RESOLUTION_ACTIVE_SOURCE_KEY
        if active_key is None:
            return None, None, None, False, source_identity
        if active_key[:2] != decoded_key[:2]:
            # Batch export of another photo must not evict the editor's active
            # full-resolution preview layers.
            return None, None, None, False, source_identity
        if _FULL_RESOLUTION_ACTIVE_SOURCE_IDENTITY != source_identity:
            _clear_full_resolution_preview_caches()
            _FULL_RESOLUTION_ACTIVE_SOURCE_KEY = decoded_key
            _FULL_RESOLUTION_ACTIVE_SOURCE_IDENTITY = source_identity
        elif active_key != decoded_key:
            # The same session/photo now resolves to a different path or legacy
            # stat key, so its previous decoded and camera layers are stale.
            _clear_full_resolution_preview_caches()
            _FULL_RESOLUTION_ACTIVE_SOURCE_KEY = decoded_key
            _FULL_RESOLUTION_ACTIVE_SOURCE_IDENTITY = source_identity

        decoded = _FULL_RESOLUTION_DECODED_CACHE.get(decoded_key)
        if (decoded is not None
                and _FULL_RESOLUTION_DECODED_SOURCE_IDENTITIES.get(decoded_key) != source_identity):
            _clear_full_resolution_preview_caches()
            _FULL_RESOLUTION_ACTIVE_SOURCE_KEY = decoded_key
            _FULL_RESOLUTION_ACTIVE_SOURCE_IDENTITY = source_identity
            decoded = None

        snapshot = None
        snapshot_shape = None
        for cache_key, candidate in _FULL_RESOLUTION_EXPORT_SOURCE_CACHE.items():
            if (len(cache_key) == 5 and cache_key[0] == session_id
                    and cache_key[1] == photo_id and cache_key[2] == source_identity
                    and cache_key[4] == LENS_PREVIEW_VERSION
                    and tuple(getattr(candidate, "source_shape", ())) == tuple(cache_key[3])
                    and _valid_full_resolution_export_source_cache(
                        candidate, source_identity, tuple(cache_key[3]), LENS_PREVIEW_VERSION,
                    )):
                snapshot = candidate
                snapshot_shape = tuple(cache_key[3])
                break

        if decoded is not None:
            decoded_shape = tuple(decoded[0].shape)
            _FULL_RESOLUTION_ACTIVE_SOURCE_SHAPE = decoded_shape
            if snapshot is not None and snapshot_shape != decoded_shape:
                snapshot = None
        elif snapshot is not None:
            _FULL_RESOLUTION_ACTIVE_SOURCE_SHAPE = snapshot_shape

        _touch_full_resolution_preview_cache()
        return (
            decoded[0] if decoded is not None else None,
            decoded[1] if decoded is not None else None,
            snapshot,
            decoded is not None,
            source_identity,
        )


def _invalidate_full_resolution_export_source(
    session_id: str, photo_id: str, source_identity: tuple,
) -> None:
    """Clear stale preview layers only when they still belong to this source."""
    with _FULL_RESOLUTION_PREVIEW_LOCK:
        active_key = _FULL_RESOLUTION_ACTIVE_SOURCE_KEY
        if (active_key is not None and active_key[:2] == (session_id, photo_id)
                and _FULL_RESOLUTION_ACTIVE_SOURCE_IDENTITY == source_identity):
            _clear_full_resolution_preview_caches()


def _full_resolution_display_base(req: EnhancePreviewRequest, path: Path,
                                  params: DehazeParams, backend: str,
                                  profile: Any = None, exposure_ev: float = 0.0
                                  ) -> tuple[np.ndarray, tuple, bool, str]:
    """Called under the full-preview lock; cache a single immutable RGB8 base."""
    global _FULL_RESOLUTION_ACTIVE_SOURCE_IDENTITY
    stat = path.stat()
    source_identity = (req.session_id, req.photo_id, str(path), stat.st_mtime_ns, stat.st_size)
    full_export_identity = enhanced_dng_source_identity(path)
    if (_FULL_RESOLUTION_ACTIVE_SOURCE_KEY == source_identity
            and _FULL_RESOLUTION_ACTIVE_SOURCE_IDENTITY != full_export_identity):
        _clear_full_resolution_preview_caches()
    key = (*source_identity,
           DEHAZE_ALGORITHM_VERSION, AUTO_EXPOSURE_ALGORITHM_VERSION,
           LENS_PREVIEW_VERSION, tuple(params.__dict__.items()), req.algorithm, req.auto_mode,
           req.auto_exposure, req.nonlocal_mode, req.color_manage_srgb, backend,
           PROFILE_PREVIEW_VERSION, profile.fingerprint if profile is not None else None,
           exposure_ev if profile is not None else 0.0,
           req.ricoh_preset_id if req.mode == "dehazed" else None,
           req.ricoh_backend, BASIC_PREVIEW_VERSION)
    cached = _FULL_RESOLUTION_BASE_CACHE.get(key)
    if cached is not None:
        decoded_key = key[:5]
        decode_cache = "hit" if decoded_key in _FULL_RESOLUTION_DECODED_CACHE else "not-needed"
        _touch_full_resolution_preview_cache()
        return cached[0], cached[1], True, decode_cache

    def remember_base(display: np.ndarray, status: tuple) -> None:
        display.setflags(write=False)
        if display.nbytes <= _MAX_FULL_RESOLUTION_BASE_BYTES:
            _FULL_RESOLUTION_BASE_CACHE[key] = (display, status)
            _trim_full_resolution_preview_cache_budget()

    stage_key: tuple | None = None
    active_shape = (
        _FULL_RESOLUTION_ACTIVE_SOURCE_SHAPE
        if _FULL_RESOLUTION_ACTIVE_SOURCE_KEY == source_identity else None
    )
    if profile is not None and active_shape is not None:
        stage_key = _full_resolution_profile_stage_key(
            source_identity, active_shape, params, req, backend, profile,
        )
        cached_stage = _FULL_RESOLUTION_CAMERA_PROCESSED_CACHE.get(stage_key)
        if cached_stage is not None:
            camera_processed, neutral, cached_status = cached_stage
            try:
                display = render_profile(
                    camera_processed, neutral, profile, exposure_ev,
                    backend=_camera_profile_render_backend(backend),
                )
            except Exception as exc:
                logging.getLogger(__name__).debug(
                    "Cached camera profile render fallback: %s", type(exc).__name__,
                )
            else:
                preview_status = (
                    *cached_status, "applied", get_last_profile_backend(), "stage-hit",
                )
                remember_base(display, preview_status)
                _touch_full_resolution_preview_cache()
                return display, preview_status, False, "not-needed"
        elif _FULL_RESOLUTION_CAMERA_PROCESSED_CACHE:
            _FULL_RESOLUTION_CAMERA_PROCESSED_CACHE.clear()
            clear_camera_profile_gpu_cache()
    elif profile is None:
        _clear_full_resolution_camera_source_cache()
        _FULL_RESOLUTION_CAMERA_PROCESSED_CACHE.clear()
        clear_camera_profile_gpu_cache()

    # Keep the same decoded source for parameter edits, but release its previous
    # rendered base before allocating another full-size working set.
    _FULL_RESOLUTION_BASE_CACHE.clear()
    image, metadata, decode_hit = _full_resolution_decoded_image(req, path)
    full_export_identity = _FULL_RESOLUTION_ACTIVE_SOURCE_IDENTITY
    reference_shape = tuple(image.shape)
    if profile is not None:
        stage_key = _full_resolution_profile_stage_key(
            source_identity, reference_shape, params, req, backend, profile,
        )
        if any(old_key != stage_key for old_key in _FULL_RESOLUTION_CAMERA_PROCESSED_CACHE):
            _FULL_RESOLUTION_CAMERA_PROCESSED_CACHE.clear()
            clear_camera_profile_gpu_cache()
    linear_input = _prepare_dehaze_input(image, metadata, path,
                                        color_manage_srgb=req.color_manage_srgb)
    # A float32 linear decode can be returned directly by the preparation helper.
    # Give renderers a writable working array while retaining an immutable source.
    if not linear_input.flags.writeable or np.shares_memory(linear_input, image):
        linear_input = np.array(linear_input, dtype=np.float32, order="C", copy=True)
    diagnostics: dict[str, Any] = {}
    enhanced = _render_dehaze(
        linear_input, params, backend, auto_mode=req.auto_mode,
        nonlocal_mode=req.nonlocal_mode, diagnostics=diagnostics,
        **({"auto_exposure": True} if req.auto_exposure else {}))
    del linear_input
    status, reason = _nonlocal_preview_status(req.nonlocal_mode, diagnostics)
    preview_status = (status, reason, float(diagnostics.get("auto_exposure_ev", 0.0)),
                      str(diagnostics.get("auto_exposure_reason", "off")))
    enhanced16 = _linear_float_to_uint16(enhanced)
    del enhanced
    enhanced16, _, _ = _correct_enhanced_raw(enhanced16, metadata, path, preview=True)
    if req.mode == "dehazed" and req.ricoh_preset_id is not None:
        enhanced16 = _bake_ricoh_linear(
            enhanced16, req.ricoh_preset_id, req.ricoh_backend,
        )
    if profile is not None:
        camera_cache: dict[str, str] = {}
        display, profile_status, profile_backend = _profile_display_base(
            enhanced16, image, metadata, path, profile, exposure_ev, preview=False,
            backend=backend, full_cache_identity=source_identity,
            full_export_identity=full_export_identity,
            stage_cache_key=stage_key, stage_status=preview_status,
            camera_cache_out=camera_cache)
        preview_status += (profile_status, profile_backend,
                           camera_cache.get("stage", "fallback"))
    else:
        display = _display_rgb8(enhanced16, linear=True)
    del enhanced16
    remember_base(display, preview_status)
    return display, preview_status, False, "hit" if decode_hit else "miss"


def _build_dehaze_xmp_curves(session_id: str, photo_id: str, path: Path,
                             params: DehazeParams, auto_mode: bool,
                             nonlocal_mode: str | None = None,
                             auto_exposure: bool = False):
    """Fit the real physical operator's display RGB change before creative edits.

    A bounded preview supplies the sample. These global curves cannot represent
    spatial transmission or Adobe's different camera profile and processing order.
    A decoding/fitting failure aborts the XMP write, preserving the old packet.
    """
    if params.strength <= 0 and not auto_exposure:
        return None
    from dehaze_xmp import fit_dehaze_curves
    image, metadata = _cached_processing_preview(session_id, photo_id, path, 1024)
    source = _prepare_dehaze_input(image, metadata, path, color_manage_srgb=True)
    target = _render_dehaze(
        source, params, "cpu", auto_mode=auto_mode,
        nonlocal_mode=nonlocal_mode,
        **({"auto_exposure": True} if auto_exposure else {}),
    )
    def display(rgb):
        return np.where(rgb <= .0031308, rgb * 12.92,
                        1.055 * np.power(np.maximum(rgb, 0), 1 / 2.4) - .055).astype(np.float32)
    return fit_dehaze_curves(np.clip(display(source), 0, 1), np.clip(display(target), 0, 1))


def _render_ricoh(image: np.ndarray, preset_id: str, basic: dict[str, float] | None,
                  mode: str, *, use_measured_color: bool = False) -> np.ndarray:
    if mode == "native":
        try:
            return native_ricoh(image, preset_id, basic, use_measured_color=use_measured_color)
        except KeyError:
            raise
        except Exception:
            pass
    if use_measured_color:
        return apply_ricoh_preview_effect(image, preset_id, basic,
                                          use_measured_color=True)
    return apply_ricoh_preview_effect(image, preset_id, basic)


def _bake_ricoh_linear(
    linear16: np.ndarray, preset_id: str, backend: str,
) -> np.ndarray:
    """Apply the measured Ricoh appearance to 16-bit linear RGB samples."""
    display16 = _linear16_to_srgb16(linear16)
    effected16 = _render_ricoh(
        display16, preset_id, BasicParamsRequest().values(), backend,
        use_measured_color=True,
    )
    return _srgb16_to_linear16(effected16)


def _cached_processing_preview(
    session_id: str, photo_id: str, path: Path, max_edge: int,
) -> tuple[np.ndarray, Any]:
    """Keep one decoded preview for rapid parameter changes on fallback paths."""
    stat = path.stat()
    key = (session_id, photo_id, stat.st_mtime_ns, stat.st_size, max_edge)
    with _DISPLAY_PREVIEW_LOCK:
        cached = _PROCESSING_PREVIEW_CACHE.get(key)
        if cached is not None:
            _PROCESSING_PREVIEW_CACHE.move_to_end(key)
            return cached

    image, metadata = read_image(path, preview=True, max_edge=max_edge)
    image.setflags(write=False)
    result = (image, metadata)
    if image.nbytes <= _MAX_PROCESSING_PREVIEW_BYTES:
        with _DISPLAY_PREVIEW_LOCK:
            cached = _PROCESSING_PREVIEW_CACHE.get(key)
            if cached is not None:
                _PROCESSING_PREVIEW_CACHE.move_to_end(key)
                return cached
            for old_key in [old_key for old_key in _PROCESSING_PREVIEW_CACHE
                            if old_key[:2] == key[:2] and old_key != key]:
                _PROCESSING_PREVIEW_CACHE.pop(old_key, None)
            while (_PROCESSING_PREVIEW_CACHE and
                   sum(value[0].nbytes for value in _PROCESSING_PREVIEW_CACHE.values())
                   + image.nbytes > _MAX_PROCESSING_PREVIEW_BYTES):
                _PROCESSING_PREVIEW_CACHE.popitem(last=False)
            _PROCESSING_PREVIEW_CACHE[key] = result
    return result


def _correct_enhanced_raw(
    image: np.ndarray, metadata: Any, path: Path, *, require_correction: bool = False,
    preview: bool = False, backend: str = "auto",
    highlight_reference: np.ndarray | None = None,
) -> tuple[np.ndarray, Any, bool]:
    """Use the same lens and DNG gain corrections for preview and export."""
    is_raw = getattr(metadata, "source_kind", "") == "raw"
    if preview and not is_raw:
        return image, LensCorrectionResult(False, None, None, False, False, False), False
    if is_raw and path.suffix.lower() == ".dng":
        diagnostics: dict[str, Any] = {}
        corrected, warp_applied, gain_applied = apply_dng_warp_correction(
            to_uint16(image), path, backend=backend, preserve_highlights=True,
            highlight_reference=highlight_reference, diagnostics=diagnostics,
            full_frame=True,
        )
        if warp_applied:
            exif = getattr(metadata, "exif", {}) or {}
            return corrected, LensCorrectionResult(
                True, str(exif.get("Model") or ""), "DNG WarpRectilinear",
                True, bool(diagnostics.get("tca_applied")), False,
                engine="DNG/WarpRectilinear", backend=diagnostics.get("backend"),
            ), gain_applied
    corrected, lens_result = apply_lens_correction(
        to_uint16(image), metadata, require_correction=is_raw and require_correction,
    )
    gain_map_applied = False
    if is_raw and path.suffix.lower() == ".dng" and not lens_result.vignetting_applied:
        corrected, gain_map_applied = apply_dng_gain_map(
            corrected, path, preserve_highlights=True,
        )
    return corrected, lens_result, gain_map_applied


_LENS_CORRECTION_FALLBACK_WARNING = "缺少可靠的镜头校正数据，已跳过镜头校正并继续导出。"


def _correct_enhanced_raw_for_export(
    image: np.ndarray, metadata: Any, path: Path, *, backend: str = "auto",
) -> tuple[np.ndarray, LensCorrectionResult, bool, str | None]:
    """Keep export strict for processing faults and degrade only missing lens data."""
    try:
        corrected, result, gain_map_applied = _correct_enhanced_raw(
            image, metadata, path, require_correction=True, backend=backend,
        )
        return corrected, result, gain_map_applied, None
    except (
        LensMatchError,
        LensfunUnavailableError,
        LensCorrectionNotAppliedError,
        LensMetadataUnavailableError,
    ):
        # Missing calibration must not prevent a DNG export. Preserve any valid
        # source DNG GainMap, while allowing malformed opcode data to fail it.
        corrected = np.ascontiguousarray(to_uint16(image).copy())
        gain_map_applied = False
        is_raw = getattr(metadata, "source_kind", "") == "raw"
        if is_raw and path.suffix.lower() == ".dng":
            corrected, gain_map_applied = apply_dng_gain_map(
                corrected, path, preserve_highlights=True,
            )
        return (
            corrected,
            LensCorrectionResult(False, None, None, False, False, False),
            gain_map_applied,
            _LENS_CORRECTION_FALLBACK_WARNING,
        )


def _cached_dehazed_display_preview(
    session_id: str,
    photo_id: str,
    path: Path,
    max_edge: int,
    params: DehazeParams,
    backend: str,
    color_manage_srgb: bool,
    preview_level: int = 0,
    auto_mode: bool = False,
    auto_exposure: bool = False,
    nonlocal_mode: str | None = None,
    status_out: dict[str, Any] | None = None,
    profile: Any = None,
    manual_exposure_ev: float = 0.0,
    ricoh_preset_id: str | None = None,
    ricoh_backend: str = "python",
) -> np.ndarray:
    """Cache the styled, profile-rendered display base before remaining basics."""
    effective_nonlocal_mode = resolve_nonlocal_mode(nonlocal_mode)
    stat = path.stat()
    dehaze_values = tuple((key, float(value)) for key, value in params.__dict__.items())
    key = (
        session_id, photo_id, stat.st_mtime_ns, stat.st_size,
        DEHAZE_ALGORITHM_VERSION, AUTO_EXPOSURE_ALGORITHM_VERSION,
        LENS_PREVIEW_VERSION, dehaze_values,
        max_edge, preview_level, backend,
        bool(auto_mode),
        bool(auto_exposure),
        effective_nonlocal_mode,
        bool(color_manage_srgb),
        PROFILE_PREVIEW_VERSION, profile.fingerprint if profile is not None else None,
        manual_exposure_ev if profile is not None else 0.0,
        ricoh_preset_id, ricoh_backend, BASIC_PREVIEW_VERSION,
    )
    with _DISPLAY_PREVIEW_LOCK:
        cached = _DEHAZED_PREVIEW_CACHE.get(key)
        if cached is not None:
            status = _DEHAZED_PREVIEW_STATUS_CACHE.get(key)
            if status is not None:
                _DEHAZED_PREVIEW_CACHE.move_to_end(key)
                _DEHAZED_PREVIEW_STATUS_CACHE.move_to_end(key)
                _set_preview_status(status_out, status)
                return cached
            _DEHAZED_PREVIEW_CACHE.pop(key, None)

    image, metadata = _cached_processing_preview(
        session_id, photo_id, path, max_edge,
    )
    linear = _prepare_dehaze_input(
        image, metadata, path, color_manage_srgb=color_manage_srgb,
    )
    exposure_ev, exposure_reason = (
        estimate_auto_exposure(linear) if auto_exposure else (0.0, "off")
    )
    if preview_level:
        scale = 2 ** preview_level
        linear = cv2.resize(
            linear,
            (max(1, linear.shape[1] // scale), max(1, linear.shape[0] // scale)),
            interpolation=cv2.INTER_AREA,
        )
        np.clip(linear, 0.0, 1.0, out=linear)
    diagnostics: dict[str, Any] = {}
    dehazed = _render_dehaze(
        np.ascontiguousarray(linear, dtype=np.float32), params, backend,
        auto_mode=auto_mode, nonlocal_mode=effective_nonlocal_mode,
        diagnostics=diagnostics,
        **({"auto_exposure": True, "auto_exposure_ev": exposure_ev,
            "auto_exposure_reason": exposure_reason} if auto_exposure else {}),
    )
    nonlocal_status, nonlocal_reason = _nonlocal_preview_status(
        effective_nonlocal_mode, diagnostics,
    )
    status = (
        nonlocal_status, nonlocal_reason,
        float(diagnostics.get("auto_exposure_ev", 0.0)),
        str(diagnostics.get("auto_exposure_reason", "off")),
    )
    _set_preview_status(status_out, status)
    dehazed16 = _linear_float_to_uint16(dehazed)
    corrected, _, _ = _correct_enhanced_raw(dehazed16, metadata, path, preview=True)
    if ricoh_preset_id is not None:
        corrected = _bake_ricoh_linear(corrected, ricoh_preset_id, ricoh_backend)
    if profile is not None:
        display, profile_status, profile_backend = _profile_display_base(
            corrected, image, metadata, path, profile, manual_exposure_ev,
            preview=True, backend=backend)
        status += (profile_status, profile_backend)
        _set_preview_status(status_out, status)
    else:
        display = _display_rgb8(corrected, linear=True)
    display.setflags(write=False)

    if display.nbytes <= _MAX_DEHAZED_PREVIEW_BYTES:
        with _DISPLAY_PREVIEW_LOCK:
            cached = _DEHAZED_PREVIEW_CACHE.get(key)
            if cached is not None:
                cached_status = _DEHAZED_PREVIEW_STATUS_CACHE.get(key)
                if cached_status is not None:
                    _DEHAZED_PREVIEW_CACHE.move_to_end(key)
                    _DEHAZED_PREVIEW_STATUS_CACHE.move_to_end(key)
                    _set_preview_status(status_out, cached_status)
                    return cached
                _DEHAZED_PREVIEW_CACHE.pop(key, None)
            # Drop stale copies if the source file changed during the session.
            for old_key in [
                old_key for old_key in _DEHAZED_PREVIEW_CACHE
                if old_key[:2] == key[:2] and old_key[2:4] != key[2:4]
            ]:
                _DEHAZED_PREVIEW_CACHE.pop(old_key, None)
                _DEHAZED_PREVIEW_STATUS_CACHE.pop(old_key, None)
            while (
                _DEHAZED_PREVIEW_CACHE
                and sum(value.nbytes for value in _DEHAZED_PREVIEW_CACHE.values())
                + display.nbytes > _MAX_DEHAZED_PREVIEW_BYTES
            ):
                evicted_key, _ = _DEHAZED_PREVIEW_CACHE.popitem(last=False)
                _DEHAZED_PREVIEW_STATUS_CACHE.pop(evicted_key, None)
            _DEHAZED_PREVIEW_CACHE[key] = display
            _remember_dehazed_preview_status(key, status)
    return display


@app.post("/api/ricoh/preview")
def create_ricoh_preview(req: RicohPreviewRequest):
    response = create_enhance_preview(EnhancePreviewRequest(
        session_id=req.session_id,
        photo_id=req.photo_id,
        params=EnhanceParamsRequest(strength=0.0),
        basic_params=req.basic_params,
        max_edge=req.max_edge,
        color_manage_srgb=True,
        ricoh_backend=req.ricoh_backend,
        ricoh_preset_id=req.preset_id,
    ))
    if response.status_code == 200 and response.media_type == "image/jpeg":
        response.headers["X-Preview-Approximation"] = "true"
        response.headers["X-Preview-Empirical-Color"] = "hsl-and-grading-response"
    return response


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
                    sort_backend=req.sort_backend,
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
    # Huffman-table optimization adds a full coding pass. Default tables keep
    # the same quantized pixels and make local interactive previews faster.
    Image.fromarray(image_rgb, "RGB").save(stream, "JPEG", quality=91, optimize=False)
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
                "dehaze_auto_mode": settings.get("dehaze_auto_mode", True),
                "dehaze_auto_exposure": settings.get("dehaze_auto_exposure", False),
                "dehaze_nonlocal_mode": settings.get("dehaze_nonlocal_mode", "off"),
                "dehaze_algorithm": "physical",
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
                with _FULL_RESOLUTION_PREVIEW_LOCK:
                    if (_FULL_RESOLUTION_ACTIVE_SOURCE_KEY is not None
                            and _FULL_RESOLUTION_ACTIVE_SOURCE_KEY[0] == expired):
                        _clear_full_resolution_preview_caches()
                with _DISPLAY_PREVIEW_LOCK:
                    for key in [key for key in _DISPLAY_PREVIEW_CACHE if key[0] == expired]:
                        _DISPLAY_PREVIEW_CACHE.pop(key, None)
                    for key in [key for key in _DEHAZED_PREVIEW_CACHE if key[0] == expired]:
                        _DEHAZED_PREVIEW_CACHE.pop(key, None)
                    for key in [key for key in _DEHAZED_PREVIEW_STATUS_CACHE if key[0] == expired]:
                        _DEHAZED_PREVIEW_STATUS_CACHE.pop(key, None)
                    for key in [key for key in _PROCESSING_PREVIEW_CACHE if key[0] == expired]:
                        _PROCESSING_PREVIEW_CACHE.pop(key, None)
        return {
            "session_id": session_id,
            "count": len(files),
            "files": files,
            "dehaze_defaults": _DEHAZE_DEFAULTS.copy(),
            "default_output_dir": str(default_output),
            "ricoh_default_output_dir": str(default_output / "理光预设输出"),
            "output_format": "Source bit-depth Linear/Demosaiced DNG",
        }
    except Exception as exc:
        return JSONResponse(status_code=400, content={"error": _enhance_error_message(exc)})


@app.get("/api/enhance/metadata/{session_id}/{photo_id}")
def get_enhance_photo_metadata(session_id: str, photo_id: str):
    """Return compact capture metadata for a photo scoped to an active session."""
    with _ENHANCE_LOCK:
        session = _ENHANCE_SESSIONS.get(session_id)
        path = session.get("files", {}).get(photo_id) if session else None
    if path is None:
        return JSONResponse(status_code=404, content={"error": "照片会话或照片已失效"})

    try:
        current_path = Path(path).resolve(strict=True)
    except (FileNotFoundError, OSError, RuntimeError):
        return JSONResponse(status_code=404, content={"error": "当前照片不存在"})
    if not current_path.is_file():
        return JSONResponse(status_code=404, content={"error": "当前照片不存在"})

    try:
        metadata = read_photo_metadata(current_path)
    except FileNotFoundError:
        return JSONResponse(status_code=404, content={"error": "当前照片不存在"})
    except Exception:
        # Do not include local paths or parser details in this public response.
        return JSONResponse(status_code=422, content={"error": "读取照片信息失败"})

    # The session may be evicted while metadata is being read. Re-check that
    # the same opaque IDs still authorize this response before returning it.
    with _ENHANCE_LOCK:
        current_session = _ENHANCE_SESSIONS.get(session_id)
        current_record = current_session.get("files", {}).get(photo_id) if current_session else None
    if current_record is None or Path(current_record) != path:
        return JSONResponse(status_code=404, content={"error": "照片会话或照片已失效"})
    return metadata


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
    scoped_nonlocal_modes = MappingProxyType({
        photo_id: mode for photo_id, mode in req.nonlocal_modes_by_photo.items()
        if photo_id in records
    })
    scoped_auto_exposures = MappingProxyType({
        photo_id: bool(enabled)
        for photo_id, enabled in req.auto_exposures_by_photo.items()
        if photo_id in records
    })
    default_nonlocal_mode = req.nonlocal_mode
    try:
        return write_dehaze_session_settings(
            scoped,
            {key: value.values() for key, value in req.basic_params_by_photo.items()},
            req.preset_ids_by_photo,
            req.auto_modes_by_photo,
            req.auto_mode,
            curve_builder=lambda photo_id, path, params, mode: _build_dehaze_xmp_curves(
                req.session_id, photo_id, path, DehazeParams(**params), mode,
                scoped_nonlocal_modes.get(photo_id, default_nonlocal_mode),
                scoped_auto_exposures.get(photo_id, req.auto_exposure),
            ),
            nonlocal_mode=default_nonlocal_mode,
            nonlocal_modes_by_photo=scoped_nonlocal_modes,
            auto_exposure=req.auto_exposure,
            auto_exposures_by_photo=dict(scoped_auto_exposures),
        )
    except KeyError:
        return JSONResponse(status_code=400, content={"error": "未知的理光预设"})
    except RicohBatchLimitError:
        return JSONResponse(status_code=413, content={"error": "一次最多处理 5000 张照片"})


@app.get("/api/enhance/gpu-status")
def get_enhance_gpu_status():
    """Return only acceleration supported by the physical float renderer."""
    try:
        status = get_native_physical_status()
        if status.get("gpu_available") and str(status.get("backend", "")).lower() == "metal":
            return {
                "available": True,
                "backends": ["metal"],
                "recommended": "metal",
                "label": "Metal GPU 可用",
            }
    except Exception:
        # Device probing is best-effort.  Keep the response stable and avoid
        # exposing driver paths or exception details if an optional runtime is
        # partially installed or unavailable.
        pass
    return {
        "available": False,
        "backends": [],
        "recommended": None,
        "label": "未检测到可用 GPU",
    }


@app.get("/api/enhance/render-status")
def get_enhance_render_status():
    native = get_native_status()
    spatial = get_native_physical_status()
    spatial["last_used"] = get_last_physical_backend()
    return {"native": native, "spatial": spatial,
            "sort": get_native_sort_status(), "pytorch": {
        "available": False, "backends": [],
    }}


@app.post("/api/enhance/preview")
def create_enhance_preview(req: EnhancePreviewRequest):
    with _ENHANCE_LOCK:
        path = _ENHANCE_SESSIONS.get(req.session_id, {}).get("files", {}).get(req.photo_id)
    if path is None or not Path(path).is_file():
        return JSONResponse(status_code=404, content={"error": "去朦胧预览会话已失效"})
    if req.full_resolution and req.preview_level != 0:
        return JSONResponse(
            status_code=400,
            content={"error": "全分辨率预览要求 preview_level 为 0"},
        )
    params = req.params.to_params()
    dehaze_values = tuple((key, float(value)) for key, value in params.__dict__.items())
    basic = req.basic_params.values()
    stat = Path(path).stat()
    effective_preset_id = req.ricoh_preset_id if req.mode == "dehazed" else None
    render_mode = _render_mode(req.render_backend, req.use_gpu)
    basic_mode = _basic_mode(req.basic_backend, render_mode)
    profile = resolve_profile(Path(path), {})
    adjustment_basic = dict(basic)
    if profile is not None:
        adjustment_basic["exposure"] = 0.0
    cache_key = (req.session_id, req.photo_id, stat.st_mtime_ns, stat.st_size,
                 DEHAZE_ALGORITHM_VERSION, AUTO_EXPOSURE_ALGORITHM_VERSION,
                 LENS_PREVIEW_VERSION, BASIC_PREVIEW_VERSION,
                 dehaze_values, req.max_edge,
                 req.preview_level, bool(req.full_resolution), req.mode,
                 req.algorithm, bool(req.auto_mode), bool(req.auto_exposure), req.nonlocal_mode,
                 effective_preset_id, render_mode, basic_mode, req.ricoh_backend,
                 bool(req.color_manage_srgb), PROFILE_PREVIEW_VERSION,
                 profile.fingerprint if profile is not None else None,
                 tuple(basic[key] for key in sorted(basic)))

    if effective_preset_id is not None:
        valid_preset_ids = {preset["id"] for preset in list_ricoh_presets()}
        if effective_preset_id not in valid_preset_ids:
            return JSONResponse(status_code=400, content={"error": "未知的理光预设"})
    if not req.full_resolution:
        with _ENHANCE_LOCK:
            cached = _ENHANCE_PREVIEW_CACHE.get(cache_key)
        if cached is not None:
            (payload, width, height, nonlocal_status, nonlocal_reason,
             auto_exposure_ev, auto_exposure_reason, camera_profile_status, camera_profile_backend) = cached
            return Response(
                content=payload,
                media_type="image/jpeg",
                headers={
                    "Cache-Control": "private, max-age=3600",
                    "X-Image-Width": str(width),
                    "X-Image-Height": str(height),
                    "X-Dehaze-Nonlocal-Status": nonlocal_status,
                    "X-Dehaze-Nonlocal-Reason": nonlocal_reason,
                    "X-Auto-Exposure-EV": format(auto_exposure_ev, ".8g"),
                    "X-Auto-Exposure-Reason": auto_exposure_reason,
                    "X-Camera-Profile-Status": camera_profile_status,
                    "X-Camera-Profile-Backend": camera_profile_backend,
                },
            )
    try:
        nonlocal_status = "off"
        nonlocal_reason = ""
        auto_exposure_ev = 0.0
        auto_exposure_reason = "off"
        preview_headers: dict[str, str] = {}
        camera_profile_status = "legacy"
        camera_profile_backend = "legacy"
        camera_cache_status = "not-needed"
        if req.full_resolution:
            started = time.perf_counter()
            waiting = started
            with _FULL_RESOLUTION_PREVIEW_LOCK:
                lock_wait = time.perf_counter() - waiting
                base_started = time.perf_counter()
                if req.mode == "original":
                    image, metadata, decode_hit = _full_resolution_decoded_image(req, Path(path))
                    full_export_identity = _FULL_RESOLUTION_ACTIVE_SOURCE_IDENTITY
                    reference = image
                    image, _, _ = _correct_enhanced_raw(image, metadata, Path(path), preview=True)
                    if profile is not None:
                        source_identity = (
                            req.session_id, req.photo_id, str(Path(path)),
                            stat.st_mtime_ns, stat.st_size,
                        )
                        camera_cache: dict[str, str] = {}
                        original_stage_key = (
                            source_identity, tuple(reference.shape), LENS_PREVIEW_VERSION,
                            "original", render_mode, PROFILE_PREVIEW_VERSION,
                            profile.fingerprint,
                        )
                        display, camera_profile_status, camera_profile_backend = _profile_display_base(
                            image, reference, metadata, Path(path), profile, 0.0,
                            preview=False, backend=render_mode,
                            full_cache_identity=source_identity,
                            full_export_identity=full_export_identity,
                            stage_cache_key=original_stage_key,
                            camera_cache_out=camera_cache,
                        )
                        camera_cache_status = camera_cache.get("stage", "fallback")
                    else:
                        display = _display_rgb8(image, linear=getattr(metadata, "color_space", "") == "Linear sRGB")
                    if req.color_manage_srgb and getattr(metadata, "source_kind", "") == "rgb":
                        display = standard_preview_to_srgb(display, path)
                    base_cache = "bypass"
                    decode_cache = "hit" if decode_hit else "miss"
                else:
                    display, preview_status, hit, decode_hit = _full_resolution_display_base(
                        req, Path(path), params, render_mode,
                        **({"profile": profile, "exposure_ev": basic["exposure"]} if profile is not None else {}),
                    )
                    nonlocal_status, nonlocal_reason, auto_exposure_ev, auto_exposure_reason = preview_status[:4]
                    camera_profile_status = preview_status[4] if len(preview_status) > 4 else "legacy"
                    camera_profile_backend = preview_status[5] if len(preview_status) > 5 else "legacy"
                    camera_cache_status = (
                        "base-hit" if hit else
                        (preview_status[6] if len(preview_status) > 6 else "not-needed")
                    )
                    base_cache = "hit" if hit else "miss"
                    decode_cache = decode_hit
                width, height = int(display.shape[1]), int(display.shape[0])
                base_time = time.perf_counter() - base_started
                adjustments_started = time.perf_counter()
                if req.mode == "original":
                    effected = display
                else:
                    effected = _render_basic(display, adjustment_basic, basic_mode)
                adjustments_time = time.perf_counter() - adjustments_started
                encoding_started = time.perf_counter()
                payload = _encode_preview(effected)
                encoding_time = time.perf_counter() - encoding_started
                total_time = time.perf_counter() - started
                preview_headers = {
                    "X-Preview-Base-Cache": base_cache,
                    "X-Preview-Decode-Cache": decode_cache,
                    "X-Preview-Camera-Cache": camera_cache_status,
                    "Server-Timing": ", ".join(f"{name};dur={duration * 1000:.2f}" for name, duration in (
                        ("wait", lock_wait), ("base", base_time), ("adjustments", adjustments_time),
                        ("encode", encoding_time), ("total", total_time))),
                }
                # Cold decodes may take longer than the idle window. Start the
                # complete idle interval after rendering/encoding finishes.
                _touch_full_resolution_preview_cache()
        elif req.mode == "original" and profile is not None:
            original_status: dict[str, Any] = {}
            display = _cached_dehazed_display_preview(
                req.session_id, req.photo_id, Path(path), req.max_edge,
                DehazeParams(strength=0.0), "cpu", req.color_manage_srgb,
                profile=profile, status_out=original_status,
            )
            camera_profile_status = str(original_status.get("camera_profile_status", "legacy"))
            camera_profile_backend = str(original_status.get("camera_profile_backend", "legacy"))
            payload = _encode_preview(display)
            width, height = display.shape[1], display.shape[0]
        elif req.mode == "original" and req.color_manage_srgb:
            display = _cached_display_preview(
                req.session_id, req.photo_id, Path(path), req.max_edge,
                optical_correction=True,
            )
            payload = _encode_preview(display)
            width, height = display.shape[1], display.shape[0]
        elif req.mode == "dehazed":
            status_out: dict[str, Any] = {}
            display = _cached_dehazed_display_preview(
                req.session_id, req.photo_id, Path(path), req.max_edge, params,
                backend=render_mode,
                color_manage_srgb=req.color_manage_srgb,
                preview_level=req.preview_level,
                auto_mode=req.auto_mode,
                auto_exposure=req.auto_exposure,
                nonlocal_mode=req.nonlocal_mode,
                status_out=status_out,
                **({"profile": profile, "manual_exposure_ev": basic["exposure"]} if profile is not None else {}),
                ricoh_preset_id=effective_preset_id,
                ricoh_backend=req.ricoh_backend,
            )
            nonlocal_status = status_out.get("status", "off")
            nonlocal_reason = status_out.get("reason", "")
            auto_exposure_ev = float(status_out.get("auto_exposure_ev", 0.0))
            auto_exposure_reason = str(status_out.get("auto_exposure_reason", "off"))
            camera_profile_status = str(status_out.get("camera_profile_status", "legacy"))
            camera_profile_backend = str(status_out.get("camera_profile_backend", "legacy"))
            effected = _render_basic(display, adjustment_basic, basic_mode)
            payload = _encode_preview(effected)
            width, height = display.shape[1], display.shape[0]
        else:
            image, metadata = read_image(path, preview=True, max_edge=req.max_edge)
            if req.mode == "original":
                image, _, _ = _correct_enhanced_raw(image, metadata, Path(path), preview=True)
            width, height = metadata.width, metadata.height
            payload = _encode_preview(image, linear=getattr(metadata, "color_space", "") == "Linear sRGB")
        if not req.full_resolution:
            with _ENHANCE_LOCK:
                _insert_bounded_jpeg_preview(
                    _ENHANCE_PREVIEW_CACHE, cache_key,
                    (payload, width, height, nonlocal_status, nonlocal_reason,
                     auto_exposure_ev, auto_exposure_reason, camera_profile_status, camera_profile_backend),
                    max_entries=128, max_bytes=_MAX_ENHANCE_PREVIEW_CACHE_BYTES,
                    payload_index=0,
                )
        return Response(
            content=payload,
            media_type="image/jpeg",
            headers={
                "Cache-Control": "private, no-store" if req.full_resolution else "private, max-age=3600",
                **preview_headers,
                "X-Image-Width": str(width),
                "X-Image-Height": str(height),
                "X-Dehaze-Nonlocal-Status": nonlocal_status,
                "X-Dehaze-Nonlocal-Reason": nonlocal_reason,
                "X-Auto-Exposure-EV": format(auto_exposure_ev, ".8g"),
                "X-Auto-Exposure-Reason": auto_exposure_reason,
                "X-Camera-Profile-Status": camera_profile_status,
                "X-Camera-Profile-Backend": camera_profile_backend,
            },
        )
    except KeyError:
        return JSONResponse(status_code=400, content={"error": "未知的理光预设"})
    except Exception as exc:
        status_code = 422 if _is_enhance_unsupported_error(exc) else 500
        return JSONResponse(status_code=status_code, content={"error": _enhance_error_message(exc)})


def _orient_enhance_thumbnail(thumbnail: Image.Image, path: Path) -> Image.Image:
    """Match the list thumbnail to the camera's displayed orientation."""
    if path.suffix.lower() in RAW_SUFFIXES:
        # LibRaw rotates postprocessed previews, but extract_thumb returns the
        # embedded JPEG's pixels unchanged. Some cameras already rotate that
        # JPEG, so only rotate when its aspect still matches the sensor.
        try:
            with rawpy.imread(str(path)) as raw:
                flip = int(raw.sizes.flip)
                sensor_landscape = raw.sizes.width > raw.sizes.height
        except Exception:
            return thumbnail
        if flip in (5, 6) and (thumbnail.width > thumbnail.height) == sensor_landscape:
            return thumbnail.transpose(
                Image.Transpose.ROTATE_90 if flip == 5 else Image.Transpose.ROTATE_270
            )
        return thumbnail

    # RawEvaluator converts standard files to RGB without applying EXIF.
    try:
        with Image.open(path) as source:
            orientation = int(source.getexif().get(274, 1))
    except Exception:
        return thumbnail
    transpose = {
        2: Image.Transpose.FLIP_LEFT_RIGHT,
        3: Image.Transpose.ROTATE_180,
        4: Image.Transpose.FLIP_TOP_BOTTOM,
        5: Image.Transpose.TRANSPOSE,
        6: Image.Transpose.ROTATE_270,
        7: Image.Transpose.TRANSVERSE,
        8: Image.Transpose.ROTATE_90,
    }.get(orientation)
    return thumbnail.transpose(transpose) if transpose is not None else thumbnail


@app.get("/api/enhance/thumbnail/{session_id}/{photo_id}")
def get_enhance_thumbnail(
    session_id: str,
    photo_id: str,
    max_edge: int = 360,
):
    """Return a small, cached JPEG for a photo in an active enhance session."""
    if not 1 <= max_edge <= 1800:
        return JSONResponse(status_code=422, content={"error": "缩略图尺寸无效"})
    cache_key = (session_id, photo_id, max_edge)
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
        # The list needs a small visual cue, so use the camera's embedded RAW
        # preview when available instead of postprocessing every full RAW.
        # RawEvaluator retains its half-size decode fallback for RAW files
        # without an embedded preview.
        if Path(path).suffix.lower() in RAW_SUFFIXES:
            image_rgb = RawEvaluator().extract_preview(Path(path))
            thumbnail = _orient_enhance_thumbnail(
                Image.fromarray(image_rgb, "RGB"), Path(path),
            )
        else:
            # Let Pillow decode a JPEG near the requested size instead of
            # expanding the entire photo into a NumPy array first.
            with Image.open(path) as source:
                if source.format == "JPEG":
                    source.draft("RGB", (max_edge, max_edge))
                thumbnail = ImageOps.exif_transpose(source).convert("RGB")
        thumbnail.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
        output = io.BytesIO()
        thumbnail.save(output, format="JPEG", quality=84, optimize=True)
        payload = output.getvalue()
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


def _snapshot_enhance_modes(
    req: EnhanceRunRequest,
    photo_ids: set[str] | None = None,
) -> tuple[bool, Mapping[str, bool]]:
    """Freeze the independent per-photo automatic/manual mode selection."""
    scoped_modes = {
        photo_id: bool(auto_mode)
        for photo_id, auto_mode in req.auto_modes_by_photo.items()
        if photo_ids is None or photo_id in photo_ids
    }
    return bool(req.auto_mode), MappingProxyType(scoped_modes)


def _snapshot_enhance_nonlocal_modes(
    req: EnhanceRunRequest,
    photo_ids: set[str] | None = None,
) -> tuple[str, Mapping[str, str]]:
    """Freeze the experiment mode per active photo before a batch starts."""
    scoped_modes = {
        photo_id: nonlocal_mode
        for photo_id, nonlocal_mode in req.nonlocal_modes_by_photo.items()
        if photo_ids is None or photo_id in photo_ids
    }
    return req.nonlocal_mode, MappingProxyType(scoped_modes)


def _snapshot_enhance_auto_exposures(
    req: EnhanceRunRequest,
    photo_ids: set[str] | None = None,
) -> tuple[bool, Mapping[str, bool]]:
    """Freeze the independent per-photo auto-exposure option for the job."""
    scoped = {
        photo_id: bool(enabled)
        for photo_id, enabled in req.auto_exposures_by_photo.items()
        if photo_ids is None or photo_id in photo_ids
    }
    return bool(req.auto_exposure), MappingProxyType(scoped)


def _snapshot_enhance_presets(
    req: EnhanceRunRequest,
    photo_ids: set[str],
) -> Mapping[str, str | None]:
    """Freeze only selected per-photo Ricoh presets for the export job."""
    return MappingProxyType({
        photo_id: preset_id
        for photo_id, preset_id in req.preset_ids_by_photo.items()
        if photo_id in photo_ids
    })


def _parse_bits_per_sample(value: Any) -> int | None:
    """Read a usable integer sample depth from common EXIF representations."""
    if isinstance(value, (tuple, list, np.ndarray)):
        values = list(np.asarray(value, dtype=object).reshape(-1))
    elif isinstance(value, str):
        values = value.strip().strip("[]()").replace(",", " ").replace(";", " ").split()
    else:
        values = [value]

    parsed: list[int] = []
    for item in values:
        if isinstance(item, str) and "/" in item:
            item = item.split("/", 1)[0]
        try:
            number = float(item)
        except (TypeError, ValueError, OverflowError):
            continue
        if np.isfinite(number) and number.is_integer() and 1 <= number <= 16:
            parsed.append(int(number))
    return max(parsed) if parsed else None


def _infer_bits_from_white_level(value: Any) -> int | None:
    """Estimate sensor depth from a white-level threshold, not from RGB dtype.

    TIFF/DNG white level is commonly the exclusive upper threshold (for
    example 4096 for a 12-bit sensor). Using ``bit_length()`` directly would
    turn that value into 13 bits and round it up to 14. Subtracting one first
    handles both 4095 and 4096 as 12-bit, and 16383/16384 as 14-bit.
    """
    if isinstance(value, (tuple, list, np.ndarray)):
        values = list(np.asarray(value, dtype=object).reshape(-1))
    elif isinstance(value, str):
        values = value.strip().strip("[]()").replace(",", " ").replace(";", " ").split()
    else:
        values = [value]

    levels: list[int] = []
    for item in values:
        if isinstance(item, str) and "/" in item:
            numerator, _, denominator = item.partition("/")
            try:
                divisor = float(denominator)
                item = float(numerator) / divisor if divisor else None
            except (TypeError, ValueError, OverflowError):
                continue
        try:
            number = float(item)
        except (TypeError, ValueError, OverflowError):
            continue
        if np.isfinite(number) and number.is_integer() and 1 <= number <= 65536:
            levels.append(int(number))
    if not levels:
        return None

    detected = max(1, (max(levels) - 1).bit_length())
    return next((depth for depth in (8, 10, 12, 14, 16) if detected <= depth), 16)


def _resolve_export_bit_depth(metadata: Any, requested: str) -> tuple[int, int | None, str]:
    """Resolve RGB output and source RAW depths, preferring the actual EXIF tag."""
    if getattr(metadata, "source_kind", "") != "raw":
        # Standard raster inputs are always exported as 16-bit Linear DNG.
        return 16, None, "rgb_16"

    exif = getattr(metadata, "exif", {}) or {}
    # Container EXIF can describe the 8-bit embedded JPEG rather than the
    # sensor (DJI DNG stores that JPEG in IFD0). LibRaw's sensor metadata is
    # authoritative for CFA depth; never use the preview's BitsPerSample.
    raw_bits = _parse_bits_per_sample(getattr(metadata, "bit_depth", None))
    source = "sensor_white_level" if raw_bits is not None else ""
    if raw_bits is None:
        raw_bits = _infer_bits_from_white_level(exif.get("DNGWhiteLevel"))
        if raw_bits is not None:
            source = "dng_white_level"
    if raw_bits is None:
        raw_bits = _parse_bits_per_sample(exif.get("BitsPerSample"))
        if raw_bits is not None:
            source = "exif"
    if raw_bits is None:
        raw_bits = 16
        source = "fallback_16"

    target_bits = raw_bits if requested == "source" else 16
    return target_bits, raw_bits, source


def _resolve_output_bits_per_sample(target_bits: int, compression: str | None) -> int:
    """Honor the writer's JPEG XL 16-bit constraint without changing CFA depth."""
    return 16 if compression == "jpegxl" and target_bits < 16 else target_bits


def _select_enhance_params(
    photo_id: str,
    params_by_photo: Mapping[str, DehazeParams],
    default_params: DehazeParams,
) -> DehazeParams:
    """Return a per-photo snapshot, falling back to the batch default."""
    return params_by_photo.get(photo_id, default_params)


def _select_enhance_mode(
    photo_id: str,
    auto_modes_by_photo: Mapping[str, bool],
    default_auto_mode: bool,
) -> bool:
    return auto_modes_by_photo.get(photo_id, default_auto_mode)


def _select_enhance_nonlocal_mode(
    photo_id: str,
    nonlocal_modes_by_photo: Mapping[str, str],
    default_nonlocal_mode: str,
) -> str:
    return nonlocal_modes_by_photo.get(photo_id, default_nonlocal_mode)


def _select_enhance_auto_exposure(
    photo_id: str,
    auto_exposures_by_photo: Mapping[str, bool],
    default_auto_exposure: bool,
) -> bool:
    return auto_exposures_by_photo.get(photo_id, default_auto_exposure)


def _run_enhance_job(
    job_id: str,
    session_id: str,
    output_dir: Path,
    default_params: DehazeParams,
    params_by_photo: Mapping[str, DehazeParams],
    backend: str = "cpu",
    basic_params_by_photo: Mapping[str, dict[str, float]] | None = None,
    basic_backend: str = "python",
    default_auto_mode: bool = False,
    auto_modes_by_photo: Mapping[str, bool] | None = None,
    default_nonlocal_mode: str = "off",
    nonlocal_modes_by_photo: Mapping[str, str] | None = None,
    default_auto_exposure: bool = False,
    auto_exposures_by_photo: Mapping[str, bool] | None = None,
    preset_ids_by_photo: Mapping[str, str | None] | None = None,
    ricoh_backend: str = "python",
    compression: str | None = None,
    requested_bit_depth: str = "source",
) -> None:
    with _ENHANCE_LOCK:
        job = _ENHANCE_JOBS[job_id]
        selected_photo_ids = {item["photo_id"] for item in job.get("files", [])}
        records = [
            (photo_id, path)
            for photo_id, path in _ENHANCE_SESSIONS.get(session_id, {}).get("files", {}).items()
            if photo_id in selected_photo_ids
        ]
        job["status"] = "running"
    for photo_id, path in records:
        with _ENHANCE_LOCK:
            cancel_event: threading.Event = job["_cancel"]
            if cancel_event.is_set():
                job["status"] = "cancelled"
                break
            item = next(item for item in job["files"] if item["photo_id"] == photo_id)
            item["status"] = "processing"
            item["warnings"] = []
            item["source_cache"] = {"decoded": "miss", "camera": "not-needed"}
            job["current_file"] = Path(path).name
        try:
            image, metadata, export_source_cache, decode_hit, source_identity = (
                _capture_full_resolution_export_sources(
                    session_id, photo_id, Path(path),
                )
            )
            current_identity = enhanced_dng_source_identity(path)
            if current_identity != source_identity:
                # A file may be replaced just after the locked snapshot handoff.
                # Drop all retained layers and restart from a fresh full decode.
                _invalidate_full_resolution_export_source(
                    session_id, photo_id, source_identity,
                )
                source_identity = current_identity
                image = None
                metadata = None
                export_source_cache = None
                decode_hit = False
            with _ENHANCE_LOCK:
                item["source_cache"]["decoded"] = "hit" if decode_hit else "miss"
            if image is None:
                image, metadata = read_image(path, preview=False)
                if enhanced_dng_source_identity(path) != source_identity:
                    _invalidate_full_resolution_export_source(
                        session_id, photo_id, source_identity,
                    )
                    raise RuntimeError("source changed while decoding export image")
            photo_params = _select_enhance_params(photo_id, params_by_photo, default_params)
            photo_auto_mode = _select_enhance_mode(
                photo_id, auto_modes_by_photo or {}, default_auto_mode,
            )
            photo_nonlocal_mode = _select_enhance_nonlocal_mode(
                photo_id, nonlocal_modes_by_photo or {}, default_nonlocal_mode,
            )
            photo_auto_exposure = _select_enhance_auto_exposure(
                photo_id, auto_exposures_by_photo or {}, default_auto_exposure,
            )
            linear_input = _prepare_dehaze_input(
                image, metadata, path, color_manage_srgb=True,
            )
            render_diagnostics: dict[str, Any] = {}
            enhanced = _render_dehaze(
                linear_input, photo_params, backend, auto_mode=photo_auto_mode,
                nonlocal_mode=photo_nonlocal_mode,
                auto_exposure=photo_auto_exposure, diagnostics=render_diagnostics,
            )
            photo_auto_exposure_ev = float(render_diagnostics.get("auto_exposure_ev", 0.0))
            photo_auto_exposure_reason = str(render_diagnostics.get("auto_exposure_reason", "off"))
            enhanced16 = _linear_float_to_uint16(enhanced)
            corrected, correction, gain_map_applied, lens_warning = (
                _correct_enhanced_raw_for_export(enhanced16, metadata, Path(path))
            )
            source_basic = (basic_params_by_photo or {}).get(photo_id, {})
            basic = BasicParamsRequest(**dict(source_basic or {})).values()
            # Automatic exposure precedes dehaze in the preview and is baked
            # in the same order here. Only manual basic controls stay editable.
            export_basic = _complete_export_basic_params(basic)
            photo_preset_id = (preset_ids_by_photo or {}).get(photo_id)
            if photo_preset_id is not None:
                corrected = _bake_ricoh_linear(
                    corrected, photo_preset_id, ricoh_backend,
                )
            del enhanced
            if cancel_event.is_set():
                with _ENHANCE_LOCK:
                    item["status"] = "cancelled"
                    job["status"] = "cancelled"
                break
            output_metadata = _export_metadata_with_basic_params(
                metadata.exif, export_basic,
            )
            for key in (
                "LensCorrectionApplied", "LensCorrectionEngine",
                "LensCorrectionCamera", "LensCorrectionLens",
                "LensCorrectionOperations", "LensCorrectionUnavailable",
            ):
                output_metadata.pop(key, None)
            operations = []
            if correction.distortion_applied:
                operations.append("distortion")
            if correction.tca_applied:
                operations.append("tca")
            if correction.vignetting_applied:
                operations.append("vignetting")
            if gain_map_applied:
                output_metadata["DNGGainMapApplied"] = True
                output_metadata["DNGGainMapHighlightProtection"] = True
            if correction.applied:
                output_metadata.update(
                    {
                        "LensCorrectionApplied": True,
                        "LensCorrectionEngine": getattr(correction, "engine", "Lensfun/lensfunpy"),
                        "LensCorrectionCamera": correction.camera_name or "",
                        "LensCorrectionLens": correction.lens_name or "",
                        "LensCorrectionOperations": ",".join(operations),
                    }
                )
            elif lens_warning is not None:
                output_metadata["LensCorrectionUnavailable"] = True
            is_raw = getattr(metadata, "source_kind", "") == "raw"
            bits_per_sample, raw_bits_per_sample, bit_depth_source = _resolve_export_bit_depth(
                metadata, requested_bit_depth,
            )
            # Enhanced image data is already linearized Stage 3 data. Adobe
            # interprets integer samples in full uint16 range regardless of
            # the source CFA depth. Keep precision separate from storage.
            output_bits_per_sample = (
                16 if is_raw else _resolve_output_bits_per_sample(bits_per_sample, compression)
            )
            writer_options: dict[str, Any] = (
                {} if is_raw and compression is None
                else {"bits_per_sample": bits_per_sample if is_raw else output_bits_per_sample}
            )
            # Older direct callers and test doubles can omit these options;
            # the HTTP export route always supplies the request's default.
            if compression is not None:
                writer_options["compression"] = compression
                if is_raw:
                    writer_options["raw_bits_per_sample"] = raw_bits_per_sample
            if is_raw:
                output_metadata.update(camera_profile_names(path))
                profile_reference = matching_embedded_profile_dng(path, output_metadata)
                if profile_reference is not None:
                    output_metadata["DNGProfileReferencePath"] = profile_reference
                if enhanced_dng_source_identity(path) != source_identity:
                    _invalidate_full_resolution_export_source(
                        session_id, photo_id, source_identity,
                    )
                    raise RuntimeError("source changed before camera-source export")
                cache_diagnostics: dict[str, str] = {}
                cache_compatible = bool(
                    lens_warning is None
                    and export_source_cache is not None
                    and _valid_full_resolution_export_source_cache(
                        export_source_cache, source_identity, tuple(image.shape),
                        LENS_PREVIEW_VERSION,
                    )
                )
                if cache_compatible:
                    camera_rgb, mosaic, cfa_pattern, profile, orientation = enhanced_dng_source_data(
                        corrected, image, path, output_metadata,
                        source_cache=export_source_cache,
                        correction_version=LENS_PREVIEW_VERSION,
                        cache_diagnostics=cache_diagnostics,
                    )
                else:
                    camera_rgb, mosaic, cfa_pattern, profile, orientation = enhanced_dng_source_data(
                        corrected, image, path, output_metadata,
                    )
                    cache_diagnostics["camera"] = "miss"
                with _ENHANCE_LOCK:
                    item["source_cache"]["camera"] = cache_diagnostics.get("camera", "miss")
                if enhanced_dng_source_identity(path) != source_identity:
                    _invalidate_full_resolution_export_source(
                        session_id, photo_id, source_identity,
                    )
                    raise RuntimeError("source changed during camera-source export")
                output_metadata.update(profile)
                if lens_warning is not None:
                    for key in (
                        "LensCorrectionApplied", "LensCorrectionEngine",
                        "LensCorrectionCamera", "LensCorrectionLens",
                        "LensCorrectionOperations",
                    ):
                        output_metadata.pop(key, None)
                    output_metadata["LensCorrectionUnavailable"] = True
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
                if any(export_basic.values()):
                    preview_rgb = np.ascontiguousarray(
                        _render_basic(preview_rgb.copy(), export_basic, basic_backend),
                    )
                output_path = write_enhanced_dng(
                    camera_rgb, mosaic, cfa_pattern, path, output_dir,
                    output_metadata, orientation=orientation, preview_rgb16=preview_rgb,
                    **writer_options,
                )
            else:
                del image
                output_path = write_linear_dng(
                    corrected, path, output_dir, output_metadata,
                    **writer_options,
                )
            if preset_ids_by_photo is None:
                # Keep the legacy private-worker behavior for existing callers.
                xmp_status = "failed"
                try:
                    write_dehaze_settings(
                        path, photo_params.__dict__, source_basic, auto_mode=photo_auto_mode,
                        nonlocal_mode=photo_nonlocal_mode,
                        auto_exposure=photo_auto_exposure,
                        compatibility_curves=_build_dehaze_xmp_curves(
                            session_id, photo_id, Path(path), photo_params, photo_auto_mode,
                            photo_nonlocal_mode, photo_auto_exposure,
                        ),
                    )
                    xmp_status = "written"
                except Exception:
                    # A sidecar failure never rolls back a valid DNG.
                    pass
            else:
                # The editor's debounced settings save owns XMP updates. Export
                # must not replace or clear an existing Ricoh preset on source.
                xmp_status = "unchanged"
            with _ENHANCE_LOCK:
                item.update(
                    {
                        "status": "success",
                        "output": str(output_path),
                        "xmp_status": xmp_status,
                        "bits_per_sample": output_bits_per_sample,
                        "output_bits_per_sample": output_bits_per_sample,
                        "sample_precision_bits": (
                            bits_per_sample if compression is not None else output_bits_per_sample
                        ),
                        "raw_bits_per_sample": raw_bits_per_sample,
                        "bit_depth_source": bit_depth_source,
                        "auto_exposure_ev": photo_auto_exposure_ev,
                        "auto_exposure_reason": photo_auto_exposure_reason,
                        "lens_correction": {
                            "applied": correction.applied,
                            "engine": getattr(correction, "engine", "Lensfun/lensfunpy"),
                            "backend": getattr(correction, "backend", None),
                            "camera": correction.camera_name,
                            "lens": correction.lens_name,
                            "distortion": correction.distortion_applied,
                            "tca": correction.tca_applied,
                            "vignetting": correction.vignetting_applied,
                        },
                        "warnings": [lens_warning] if lens_warning is not None else [],
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
    all_records = list(session["files"].items())
    session_photo_ids = {photo_id for photo_id, _ in all_records}
    if req.photo_ids is not None:
        if not req.photo_ids:
            return JSONResponse(status_code=400, content={"error": "请选择要处理的照片"})
        unknown_photo_ids = set(req.photo_ids) - session_photo_ids
        if unknown_photo_ids:
            return JSONResponse(status_code=400, content={"error": "所选照片不属于当前会话"})
        selected_photo_ids = set(req.photo_ids)
        records = [record for record in all_records if record[0] in selected_photo_ids]
    else:
        selected_photo_ids = session_photo_ids
        records = all_records
    if not records:
        return JSONResponse(status_code=400, content={"error": "没有可处理的照片"})
    if len(records) > 500:
        return JSONResponse(status_code=413, content={"error": "一次最多处理 500 张照片"})
    if set(req.preset_ids_by_photo) - session_photo_ids:
        return JSONResponse(status_code=400, content={"error": "理光预设对应的照片不属于当前会话"})
    valid_preset_ids = {preset["id"] for preset in list_ricoh_presets()}
    if any(
        preset_id is not None and preset_id not in valid_preset_ids
        for preset_id in req.preset_ids_by_photo.values()
    ):
        return JSONResponse(status_code=400, content={"error": "未知的理光预设"})

    default_params, params_by_photo = _snapshot_enhance_params(
        req,
        selected_photo_ids,
    )
    default_auto_mode, auto_modes_by_photo = _snapshot_enhance_modes(
        req,
        selected_photo_ids,
    )
    default_nonlocal_mode, nonlocal_modes_by_photo = _snapshot_enhance_nonlocal_modes(
        req,
        selected_photo_ids,
    )
    default_auto_exposure, auto_exposures_by_photo = _snapshot_enhance_auto_exposures(
        req,
        selected_photo_ids,
    )
    preset_ids_by_photo = _snapshot_enhance_presets(req, selected_photo_ids)
    basic_params_by_photo = MappingProxyType({
        photo_id: req.basic_params_by_photo[photo_id].values()
        for photo_id in selected_photo_ids
        if photo_id in req.basic_params_by_photo
    })
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
        "photo_ids": [photo_id for photo_id, _ in records],
        "compression": req.compression,
        "bit_depth": req.bit_depth,
        "files": [{"photo_id": photo_id, "name": Path(path).name, "status": "waiting"} for photo_id, path in records],
        "_cancel": threading.Event(),
    }
    thread = threading.Thread(
        target=_run_enhance_job,
        args=(job_id, req.session_id, output_dir, default_params, params_by_photo,
              _render_mode(req.render_backend, req.use_gpu),
              basic_params_by_photo,
              _basic_mode(req.basic_backend, _render_mode(req.render_backend, req.use_gpu)),
              default_auto_mode, auto_modes_by_photo,
              default_nonlocal_mode, nonlocal_modes_by_photo,
              default_auto_exposure, auto_exposures_by_photo,
              preset_ids_by_photo, req.ricoh_backend, req.compression, req.bit_depth),
        name=f"enhance-{job_id[:8]}", daemon=True,
    )
    job["_thread"] = thread
    with _ENHANCE_LOCK:
        _ENHANCE_JOBS[job_id] = job
    thread.start()
    return _enhance_public_job(job)


def _ricoh_public_job(job: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy({key: value for key, value in job.items() if key not in {"_cancel", "_thread"}})


def _run_ricoh_job(job_id: str, session_id: str, preset_id: str | None, output_dir: Path,
                   basic_params_by_photo: Mapping[str, dict[str, float]] | None = None,
                   preset_ids_by_photo: Mapping[str, str | None] | None = None,
                   ricoh_backend: str = "python", basic_backend: str = "python") -> None:
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
            photo_preset_id = (preset_ids_by_photo or {}).get(photo_id, preset_id)
            image, metadata = read_image(path, preview=False)
            source16 = to_uint16(image)
            source_basic = (basic_params_by_photo or {}).get(photo_id)
            basic = BasicParamsRequest(**dict(source_basic or {})).values()
            export_basic = _complete_export_basic_params(basic)
            if photo_preset_id is None:
                # No appearance controls are baked without a Ricoh preset.
                # Keep already-linear samples exact instead of round-tripping
                # them through the display transfer function.
                effected_linear16 = (
                    source16
                    if getattr(metadata, "color_space", "") == "Linear sRGB"
                    else _srgb16_to_linear16(source16)
                )
            else:
                source_linear16 = (
                    source16
                    if getattr(metadata, "color_space", "") == "Linear sRGB"
                    else _srgb16_to_linear16(source16)
                )
                effected_linear16 = _bake_ricoh_linear(
                    source_linear16, photo_preset_id, ricoh_backend,
                )
            output_path = write_linear_dng(
                effected_linear16, path, output_dir,
                _export_metadata_with_basic_params(metadata.exif, export_basic),
                bits_per_sample=16,
                name_suffix="_ricoh",
            )
            # Ricoh presets remain baked into the DNG. Basic controls are
            # embedded as editable Camera Raw settings by the DNG writer.
            xmp_status: dict[str, str] = {"source": "not_applicable", "output": "embedded"}
            if photo_preset_id is not None:
                xmp_status["source"] = "failed"
                try:
                    write_ricoh_preset(path, photo_preset_id, source_basic)
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
    valid_preset_ids = {preset["id"] for preset in list_ricoh_presets()}
    if (req.preset_id is not None and req.preset_id not in valid_preset_ids) or any(
        preset_id is not None and preset_id not in valid_preset_ids
        for preset_id in req.preset_ids_by_photo.values()
    ):
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
        "preset_ids_by_photo": dict(req.preset_ids_by_photo),
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
              {key: value.values() for key, value in req.basic_params_by_photo.items()},
              dict(req.preset_ids_by_photo), req.ricoh_backend, req.basic_backend),
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


@app.get("/api/health")
def get_health():
    """Lightweight readiness check that does not inspect model state."""
    return {"ok": True}


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
