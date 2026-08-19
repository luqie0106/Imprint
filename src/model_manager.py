"""
model_manager.py — Photo Sort 模型文件检测与本地自动下载管理
"""

from __future__ import annotations

import os
import shutil
import sys
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# 项目根目录与打包资源目录
if getattr(sys, 'frozen', False):
    PROJECT_ROOT = Path(sys.executable).parent
    BUNDLE_ROOT = Path(sys._MEIPASS)
else:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    BUNDLE_ROOT = PROJECT_ROOT

MODELS_DIR = PROJECT_ROOT / "models"
CLIP_MODEL_DIR = MODELS_DIR / "clip-vit-base-patch32"
MLP_WEIGHTS_PATH = PROJECT_ROOT / "aesthetic_mlp.pth"
ONNX_MODEL_PATH = PROJECT_ROOT / "photo_sort_model.onnx"
ONNX_BUNDLE_PATH = BUNDLE_ROOT / "photo_sort_model.onnx"

# HuggingFace 默认系统缓存路径
HF_HUB_CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub" / "models--openai--clip-vit-base-patch32"

# HuggingFace 仓库信息
CLIP_REPO_ID = "openai/clip-vit-base-patch32"
HF_OFFICIAL_URL = "https://huggingface.co"
HF_MIRROR_URL = "https://hf-mirror.com"

# 完整的文件列表（用于直接从镜像/官方下载或同步）
CLIP_ALL_FILES = [
    "config.json",
    "preprocessor_config.json",
    "model.safetensors",
    "pytorch_model.bin",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
]


@dataclass
class ModelStatus:
    clip_ready: bool
    clip_location: str  # "local", "hf_cache", "none"
    clip_path: str
    mlp_ready: bool
    mlp_path: str
    onnx_ready: bool
    onnx_path: str
    is_fully_ready: bool


def find_hf_cache_files() -> dict[str, Path]:
    """扫描系统 HuggingFace 缓存中的 CLIP 模型文件并解析符号链接。"""
    found: dict[str, Path] = {}
    snapshots_dir = HF_HUB_CACHE_DIR / "snapshots"
    if not snapshots_dir.exists() or not snapshots_dir.is_dir():
        return found

    for snap in snapshots_dir.iterdir():
        if not snap.is_dir():
            continue
        for item in snap.iterdir():
            fname = item.name
            # 解析真实指向路径（Hugging Face 快照使用软链接指向 blobs）
            real_path = item.resolve()
            if real_path.exists() and real_path.stat().st_size > 0:
                if fname not in found:
                    found[fname] = real_path

    return found


def is_clip_in_hf_cache() -> bool:
    """检查 HuggingFace 系统缓存中是否包含完整可用的 CLIP ViT-B/32 模型。"""
    files = find_hf_cache_files()
    has_config = "config.json" in files
    has_preprocessor = "preprocessor_config.json" in files
    has_weight = False

    for wf in ["model.safetensors", "pytorch_model.bin"]:
        if wf in files and files[wf].stat().st_size > 100 * 1024 * 1024:
            has_weight = True
            break

    return has_config and has_preprocessor and has_weight


def is_clip_model_downloaded() -> bool:
    """检查本地 models/clip-vit-base-patch32 目录是否存在且核心文件齐全。"""
    if not CLIP_MODEL_DIR.exists() or not CLIP_MODEL_DIR.is_dir():
        return False

    has_config = (CLIP_MODEL_DIR / "config.json").exists()
    has_preprocessor = (CLIP_MODEL_DIR / "preprocessor_config.json").exists()

    has_weights = False
    for wf in ["model.safetensors", "pytorch_model.bin"]:
        p = CLIP_MODEL_DIR / wf
        if p.exists() and p.stat().st_size > 100 * 1024 * 1024:
            has_weights = True
            break

    return has_config and has_preprocessor and has_weights


def get_clip_model_path() -> str:
    """
    返回可用于 transformers.from_pretrained(...) 的路径。
    优先级：本地 models 目录 > HuggingFace 缓存 / 远程标识。
    """
    if is_clip_model_downloaded():
        return str(CLIP_MODEL_DIR)
    return CLIP_REPO_ID


def import_from_hf_cache(progress_callback: Callable[[str, float], None] | None = None) -> bool:
    """直接将系统 Hugging Face 缓存中的 CLIP 模型文件秒级同步/复制到本地 models 目录。"""
    cached_files = find_hf_cache_files()
    if not is_clip_in_hf_cache():
        return False

    CLIP_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    total = len(cached_files)

    for idx, (fname, src_file) in enumerate(cached_files.items()):
        dest_file = CLIP_MODEL_DIR / fname
        if dest_file.exists() and dest_file.stat().st_size == src_file.stat().st_size:
            continue

        if progress_callback:
            progress_callback(f"正在从本地系统缓存复制 {fname}...", idx / total)

        try:
            # 尝试硬链接以节省磁盘空间，若跨文件系统则回退到普通复制
            try:
                if dest_file.exists():
                    dest_file.unlink()
                os.link(str(src_file), str(dest_file))
            except Exception:
                shutil.copy2(str(src_file), str(dest_file))
        except Exception as exc:
            if progress_callback:
                progress_callback(f"复制 {fname} 遇到警告: {exc}", idx / total)

    success = is_clip_model_downloaded()
    if success and progress_callback:
        progress_callback("✅ 已从系统 HuggingFace 缓存秒级导入至本地 models/ 目录！", 1.0)
    return success


def check_all_models() -> ModelStatus:
    """全面检查所有模型状态。"""
    is_local = is_clip_model_downloaded()
    is_cached = is_clip_in_hf_cache()

    if is_local:
        clip_loc = "local"
    elif is_cached:
        clip_loc = "hf_cache"
    else:
        clip_loc = "none"

    clip_ok = (clip_loc != "none")
    mlp_ok = MLP_WEIGHTS_PATH.exists() and MLP_WEIGHTS_PATH.stat().st_size > 1000
    has_local_onnx = ONNX_MODEL_PATH.exists() and ONNX_MODEL_PATH.stat().st_size > 100 * 1024 * 1024
    has_bundle_onnx = ONNX_BUNDLE_PATH.exists() and ONNX_BUNDLE_PATH.stat().st_size > 100 * 1024 * 1024
    onnx_ok = has_local_onnx or has_bundle_onnx
    actual_onnx_path = ONNX_MODEL_PATH if has_local_onnx else ONNX_BUNDLE_PATH

    return ModelStatus(
        clip_ready=clip_ok,
        clip_location=clip_loc,
        clip_path=str(CLIP_MODEL_DIR if is_local else CLIP_REPO_ID),
        mlp_ready=mlp_ok,
        mlp_path=str(MLP_WEIGHTS_PATH),
        onnx_ready=onnx_ok,
        onnx_path=str(actual_onnx_path),
        is_fully_ready=(onnx_ok or (clip_ok and mlp_ok)),
    )


def download_file_with_progress(
    url: str,
    dest_path: Path,
    progress_callback: Callable[[str, float], None] | None = None,
    file_label: str = "",
) -> None:
    """下载单个文件并实时回报进度 (0.0~1.0)。"""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(".tmp")

    headers = {"User-Agent": "PhotoSort-ModelDownloader/1.0"}
    req = urllib.request.Request(url, headers=headers)

    with urllib.request.urlopen(req, timeout=30) as response:
        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0
        chunk_size = 1024 * 512  # 512KB

        with open(temp_path, "wb") as out_file:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                if total_size > 0 and progress_callback:
                    pct = downloaded / total_size
                    mb_cur = downloaded / (1024 * 1024)
                    mb_tot = total_size / (1024 * 1024)
                    msg = f"{file_label} ({mb_cur:.1f}MB / {mb_tot:.1f}MB)"
                    progress_callback(msg, pct)

    if temp_path.exists():
        if dest_path.exists():
            dest_path.unlink()
        temp_path.rename(dest_path)


def download_clip_model(
    use_mirror: bool = True,
    progress_callback: Callable[[str, float], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> bool:
    """
    下载 CLIP 基础模型至本地 models/clip-vit-base-patch32 目录。
    1. 若系统 HF 缓存已存在，直接秒级导入，无需联网！
    2. 否则使用 huggingface_hub / 镜像源下载。
    """
    # 步骤 1：优先秒级导入已有的缓存
    if is_clip_in_hf_cache():
        if progress_callback:
            progress_callback("检测到系统缓存中已存在该模型，正在秒级同步到当前项目...", 0.5)
        success = import_from_hf_cache(progress_callback)
        if success:
            return True

    # 步骤 2：联网下载
    CLIP_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    base_host = HF_MIRROR_URL if use_mirror else HF_OFFICIAL_URL

    if progress_callback:
        progress_callback("正在准备下载 CLIP 基础视觉模型...", 0.0)

    try:
        from huggingface_hub import snapshot_download
        endpoint = base_host if use_mirror else None
        if progress_callback:
            progress_callback(f"通过 HuggingFace Hub 从 {base_host} 下载中...", 0.1)

        snapshot_download(
            repo_id=CLIP_REPO_ID,
            local_dir=str(CLIP_MODEL_DIR),
            endpoint=endpoint,
            ignore_patterns=["*.msgpack", "*.h5", "*.tflite", "*.ot"],
        )
        if is_clip_model_downloaded():
            if progress_callback:
                progress_callback("✅ CLIP 基础模型下载完成！", 1.0)
            return True
    except Exception:
        pass

    # 步骤 3：直接 HTTP 文件下载 fallback
    files_to_download = [
        "config.json",
        "preprocessor_config.json",
        "model.safetensors",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
    ]
    total_files = len(files_to_download)
    for idx, fname in enumerate(files_to_download):
        if cancel_event and cancel_event.is_set():
            if progress_callback:
                progress_callback("❌ 下载已取消", 0.0)
            return False

        target_file = CLIP_MODEL_DIR / fname
        if target_file.exists() and target_file.stat().st_size > 0:
            if fname == "model.safetensors" and target_file.stat().st_size > 100 * 1024 * 1024:
                continue
            elif fname != "model.safetensors":
                continue

        url = f"{base_host}/{CLIP_REPO_ID}/resolve/main/{fname}"
        label = f"[{idx+1}/{total_files}] 下载 {fname}"

        def _file_progress(msg: str, file_pct: float):
            overall_pct = (idx + file_pct) / total_files
            if progress_callback:
                progress_callback(msg, overall_pct)

        try:
            download_file_with_progress(url, target_file, _file_progress, label)
        except Exception as exc:
            if fname == "model.safetensors":
                bin_fname = "pytorch_model.bin"
                bin_url = f"{base_host}/{CLIP_REPO_ID}/resolve/main/{bin_fname}"
                bin_target = CLIP_MODEL_DIR / bin_fname
                try:
                    download_file_with_progress(
                        bin_url, bin_target, _file_progress, f"[{idx+1}/{total_files}] 下载 {bin_fname}"
                    )
                except Exception as bin_exc:
                    raise RuntimeError(f"下载权重失败: {exc} | {bin_exc}") from exc
            else:
                raise RuntimeError(f"下载文件 {fname} 失败: {exc}") from exc

    success = is_clip_model_downloaded()
    if success and progress_callback:
        progress_callback("✅ CLIP 基础模型下载完成！", 1.0)
    return success
