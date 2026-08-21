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
CLIP_L14_MODEL_DIR = MODELS_DIR / "clip-vit-large-patch14"

# 1. 官方标准通用美学模型 (ViT-B/32 极速版 ~335MB)
STANDARD_ONNX_PATH = MODELS_DIR / "standard_aesthetic_model.onnx"
STANDARD_ONNX_BUNDLE_PATH = BUNDLE_ROOT / "models" / "standard_aesthetic_model.onnx"
STANDARD_ONNX_ROOT_BUNDLE_PATH = BUNDLE_ROOT / "standard_aesthetic_model.onnx"

# 2. Aesthetic 3 官方专业大模型 (ViT-L/14 高精版 ~900MB)
STANDARD_L14_ONNX_PATH = MODELS_DIR / "standard_aesthetic_l14_model.onnx"
STANDARD_L14_ONNX_BUNDLE_PATH = BUNDLE_ROOT / "models" / "standard_aesthetic_l14_model.onnx"
L14_LINEAR_WEIGHTS_PATH = MODELS_DIR / "sa_0_4_vit_l_14_linear.pth"
L14_LINEAR_WEIGHTS_URL = "https://github.com/LAION-AI/aesthetic-predictor/raw/main/sa_0_4_vit_l_14_linear.pth"

# 3. 个人专属训练模型 (ViT-B/32 & ViT-L/14)
CUSTOM_ONNX_PATH = MODELS_DIR / "custom_aesthetic_model.onnx"
CUSTOM_ONNX_LEGACY_PATH = PROJECT_ROOT / "photo_sort_model.onnx"
CUSTOM_ONNX_BUNDLE_PATH = BUNDLE_ROOT / "photo_sort_model.onnx"

CUSTOM_L14_ONNX_PATH = MODELS_DIR / "custom_aesthetic_l14_model.onnx"

# 4. 个人训练 PyTorch 权重
MLP_WEIGHTS_PATH = MODELS_DIR / "aesthetic_mlp.pth"
MLP_WEIGHTS_LEGACY_PATH = PROJECT_ROOT / "aesthetic_mlp.pth"
MLP_L14_WEIGHTS_PATH = MODELS_DIR / "aesthetic_mlp_l14.pth"

# 5. 本地配置持久化路径
CONFIG_FILE_PATH = PROJECT_ROOT / "config.json"

# HuggingFace 系统缓存路径
HF_HUB_CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub" / "models--openai--clip-vit-base-patch32"
HF_HUB_CACHE_L14_DIR = Path.home() / ".cache" / "huggingface" / "hub" / "models--openai--clip-vit-large-patch14"

# HuggingFace 仓库信息
CLIP_REPO_ID = "openai/clip-vit-base-patch32"
CLIP_L14_REPO_ID = "openai/clip-vit-large-patch14"
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
    clip_l14_ready: bool
    clip_l14_location: str
    clip_l14_path: str
    standard_onnx_ready: bool
    standard_onnx_path: str
    standard_onnx_size_mb: float
    standard_l14_onnx_ready: bool
    standard_l14_onnx_path: str
    standard_l14_onnx_size_mb: float
    custom_onnx_ready: bool
    custom_onnx_path: str
    custom_onnx_size_mb: float
    custom_l14_onnx_ready: bool
    custom_l14_onnx_path: str
    custom_l14_onnx_size_mb: float
    mlp_ready: bool
    mlp_path: str
    mlp_l14_ready: bool
    mlp_l14_path: str
    active_mode: str  # "standard_b32" | "standard_l14" | "custom_b32" | "custom_l14" | "custom"
    active_onnx_path: str
    is_fully_ready: bool


def get_active_model_mode() -> str:
    """读取用户选择的激活模型模式：'standard_b32'、'standard_l14'、'custom_b32'、'custom_l14' 或 'custom'。"""
    import json
    if CONFIG_FILE_PATH.exists():
        try:
            with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                mode = data.get("aesthetic_model_mode", "standard_b32")
                if mode in ("standard", "standard_b32", "standard_l14", "custom", "custom_b32", "custom_l14"):
                    return mode
        except Exception:
            pass
    return "standard_b32"


def set_active_model_mode(mode: str) -> None:
    """持久化保存用户选择的激活模型模式。"""
    import json
    if mode not in ("standard", "standard_b32", "standard_l14", "custom", "custom_b32", "custom_l14"):
        mode = "standard_b32"
    data = {}
    if CONFIG_FILE_PATH.exists():
        try:
            with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
    data["aesthetic_model_mode"] = mode
    try:
        with open(CONFIG_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def get_resolved_standard_onnx_path() -> Path | None:
    """寻找可用的官方标准通用 (ViT-B/32) ONNX 模型路径。"""
    for p in [STANDARD_ONNX_PATH, STANDARD_ONNX_BUNDLE_PATH, STANDARD_ONNX_ROOT_BUNDLE_PATH]:
        if p.exists() and p.stat().st_size > 100 * 1024 * 1024:
            return p
    return None


def get_resolved_standard_l14_onnx_path() -> Path | None:
    """寻找可用的 Aesthetic 3 (ViT-L/14) ONNX 模型路径。"""
    for p in [STANDARD_L14_ONNX_PATH, STANDARD_L14_ONNX_BUNDLE_PATH, BUNDLE_ROOT / "standard_aesthetic_l14_model.onnx"]:
        if p.exists() and p.stat().st_size > 100 * 1024 * 1024:
            return p
    return None


def get_resolved_custom_onnx_path() -> Path | None:
    """寻找可用的个人专属训练 (ViT-B/32) ONNX 模型路径。"""
    for p in [CUSTOM_ONNX_PATH, CUSTOM_ONNX_LEGACY_PATH, CUSTOM_ONNX_BUNDLE_PATH]:
        if p.exists() and p.stat().st_size > 100 * 1024 * 1024:
            return p
    return None


def get_resolved_custom_l14_onnx_path() -> Path | None:
    """寻找可用的个人专属训练 (ViT-L/14) ONNX 模型路径。"""
    for p in [CUSTOM_L14_ONNX_PATH, BUNDLE_ROOT / "models" / "custom_aesthetic_l14_model.onnx"]:
        if p.exists() and p.stat().st_size > 100 * 1024 * 1024:
            return p
    return None


def get_resolved_mlp_path() -> Path | None:
    """寻找可用的个人训练 (ViT-B/32) PyTorch 权重文件。"""
    for p in [MLP_WEIGHTS_PATH, MLP_WEIGHTS_LEGACY_PATH, BUNDLE_ROOT / "models" / "aesthetic_mlp.pth", BUNDLE_ROOT / "aesthetic_mlp.pth"]:
        if p.exists() and p.stat().st_size > 1000:
            return p
    return None


def get_resolved_mlp_l14_path() -> Path | None:
    """寻找可用的个人训练 (ViT-L/14) PyTorch 权重文件。"""
    for p in [MLP_L14_WEIGHTS_PATH, BUNDLE_ROOT / "models" / "aesthetic_mlp_l14.pth"]:
        if p.exists() and p.stat().st_size > 1000:
            return p
    return None


def get_active_aesthetic_model_path() -> Path | None:
    """根据当前配置与文件存在情况，返回最终用于推理的 ONNX 模型路径。"""
    mode = get_active_model_mode()
    if mode in ("standard_l14",):
        p = get_resolved_standard_l14_onnx_path()
        if p:
            return p
        return get_resolved_standard_onnx_path()
    elif mode in ("custom_l14",):
        p = get_resolved_custom_l14_onnx_path()
        if p:
            return p
        p_std_l14 = get_resolved_standard_l14_onnx_path()
        if p_std_l14:
            return p_std_l14
        return get_resolved_standard_onnx_path()
    elif mode in ("custom", "custom_b32"):
        p = get_resolved_custom_onnx_path()
        if p:
            return p
        return get_resolved_standard_onnx_path()
    else:  # standard / standard_b32
        p = get_resolved_standard_onnx_path()
        if p:
            return p
        return get_resolved_standard_l14_onnx_path() or get_resolved_custom_onnx_path()




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


def find_hf_cache_l14_files() -> dict[str, Path]:
    """扫描系统 HuggingFace 缓存中的 CLIP ViT-L/14 模型文件并解析符号链接。"""
    found: dict[str, Path] = {}
    snapshots_dir = HF_HUB_CACHE_L14_DIR / "snapshots"
    if not snapshots_dir.exists() or not snapshots_dir.is_dir():
        return found

    for snap in snapshots_dir.iterdir():
        if not snap.is_dir():
            continue
        for item in snap.iterdir():
            fname = item.name
            real_path = item.resolve()
            if real_path.exists() and real_path.stat().st_size > 0:
                if fname not in found:
                    found[fname] = real_path

    return found


def is_clip_l14_in_hf_cache() -> bool:
    """检查 HuggingFace 系统缓存中是否包含完整可用的 CLIP ViT-L/14 模型。"""
    files = find_hf_cache_l14_files()
    has_config = "config.json" in files
    has_preprocessor = "preprocessor_config.json" in files
    has_weight = False

    for wf in ["model.safetensors", "pytorch_model.bin"]:
        if wf in files and files[wf].stat().st_size > 300 * 1024 * 1024:
            has_weight = True
            break

    return has_config and has_preprocessor and has_weight


def is_clip_l14_model_downloaded() -> bool:
    """检查本地 models/clip-vit-large-patch14 目录是否存在且核心文件齐全。"""
    if not CLIP_L14_MODEL_DIR.exists() or not CLIP_L14_MODEL_DIR.is_dir():
        return False

    has_config = (CLIP_L14_MODEL_DIR / "config.json").exists()
    has_preprocessor = (CLIP_L14_MODEL_DIR / "preprocessor_config.json").exists()

    has_weights = False
    for wf in ["model.safetensors", "pytorch_model.bin"]:
        p = CLIP_L14_MODEL_DIR / wf
        if p.exists() and p.stat().st_size > 300 * 1024 * 1024:
            has_weights = True
            break

    return has_config and has_preprocessor and has_weights


def get_clip_l14_model_path() -> str:
    """返回可用于 transformers.from_pretrained(...) 的 ViT-L/14 路径。"""
    if is_clip_l14_model_downloaded():
        return str(CLIP_L14_MODEL_DIR)
    return CLIP_L14_REPO_ID


def import_from_hf_cache_l14(progress_callback: Callable[[str, float], None] | None = None) -> bool:
    """直接将系统 Hugging Face 缓存中的 CLIP ViT-L/14 模型文件秒级同步/复制到本地 models 目录。"""
    cached_files = find_hf_cache_l14_files()
    if not is_clip_l14_in_hf_cache():
        return False

    CLIP_L14_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    total = len(cached_files)

    for idx, (fname, src_file) in enumerate(cached_files.items()):
        dest_file = CLIP_L14_MODEL_DIR / fname
        if dest_file.exists() and dest_file.stat().st_size == src_file.stat().st_size:
            continue

        if progress_callback:
            progress_callback(f"正在从本地系统缓存复制 {fname}...", idx / total)

        try:
            try:
                if dest_file.exists():
                    dest_file.unlink()
                os.link(str(src_file), str(dest_file))
            except Exception:
                shutil.copy2(str(src_file), str(dest_file))
        except Exception as exc:
            if progress_callback:
                progress_callback(f"复制 {fname} 遇到警告: {exc}", idx / total)

    success = is_clip_l14_model_downloaded()
    if success and progress_callback:
        progress_callback("✅ 已从系统 HuggingFace 缓存秒级导入至本地 models/clip-vit-large-patch14 目录！", 1.0)
    return success


def check_all_models() -> ModelStatus:
    """全面检查所有模型状态（ViT-B/32 与 ViT-L/14 标准模型及个人专属模型）。"""
    # 1. CLIP ViT-B/32
    is_b32_local = is_clip_model_downloaded()
    is_b32_cached = is_clip_in_hf_cache()
    if is_b32_local:
        clip_loc = "local"
    elif is_b32_cached:
        clip_loc = "hf_cache"
    else:
        clip_loc = "none"
    clip_ok = (clip_loc != "none")

    # 2. CLIP ViT-L/14 (Aesthetic 3)
    is_l14_local = is_clip_l14_model_downloaded()
    is_l14_cached = is_clip_l14_in_hf_cache()
    if is_l14_local:
        clip_l14_loc = "local"
    elif is_l14_cached:
        clip_l14_loc = "hf_cache"
    else:
        clip_l14_loc = "none"
    clip_l14_ok = (clip_l14_loc != "none")

    # 3. 官方标准通用模型 (ViT-B/32)
    std_p = get_resolved_standard_onnx_path()
    std_ok = (std_p is not None)
    std_size = (std_p.stat().st_size / (1024 * 1024)) if std_ok else 0.0

    # 4. Aesthetic 3 官方专业大模型 (ViT-L/14)
    std_l14_p = get_resolved_standard_l14_onnx_path()
    std_l14_ok = (std_l14_p is not None)
    std_l14_size = (std_l14_p.stat().st_size / (1024 * 1024)) if std_l14_ok else 0.0

    # 5. 个人专属训练模型 (ViT-B/32)
    custom_p = get_resolved_custom_onnx_path()
    custom_ok = (custom_p is not None)
    custom_size = (custom_p.stat().st_size / (1024 * 1024)) if custom_ok else 0.0

    # 6. 个人专属训练模型 (ViT-L/14)
    custom_l14_p = get_resolved_custom_l14_onnx_path()
    custom_l14_ok = (custom_l14_p is not None)
    custom_l14_size = (custom_l14_p.stat().st_size / (1024 * 1024)) if custom_l14_ok else 0.0

    # 7. 个人训练 PyTorch 权重
    mlp_p = get_resolved_mlp_path()
    mlp_ok = (mlp_p is not None)

    mlp_l14_p = get_resolved_mlp_l14_path()
    mlp_l14_ok = (mlp_l14_p is not None)

    # 8. 当前激活模式与最终生效模型
    active_mode = get_active_model_mode()
    active_onnx = get_active_aesthetic_model_path()
    active_path_str = str(active_onnx) if active_onnx else ""

    is_ready = bool(std_ok or std_l14_ok or custom_ok or custom_l14_ok or (clip_ok and mlp_ok))

    return ModelStatus(
        clip_ready=clip_ok,
        clip_location=clip_loc,
        clip_path=str(CLIP_MODEL_DIR if is_b32_local else CLIP_REPO_ID),
        clip_l14_ready=clip_l14_ok,
        clip_l14_location=clip_l14_loc,
        clip_l14_path=str(CLIP_L14_MODEL_DIR if is_l14_local else CLIP_L14_REPO_ID),
        standard_onnx_ready=std_ok,
        standard_onnx_path=str(std_p) if std_p else str(STANDARD_ONNX_PATH),
        standard_onnx_size_mb=std_size,
        standard_l14_onnx_ready=std_l14_ok,
        standard_l14_onnx_path=str(std_l14_p) if std_l14_p else str(STANDARD_L14_ONNX_PATH),
        standard_l14_onnx_size_mb=std_l14_size,
        custom_onnx_ready=custom_ok,
        custom_onnx_path=str(custom_p) if custom_p else str(CUSTOM_ONNX_PATH),
        custom_onnx_size_mb=custom_size,
        custom_l14_onnx_ready=custom_l14_ok,
        custom_l14_onnx_path=str(custom_l14_p) if custom_l14_p else str(CUSTOM_L14_ONNX_PATH),
        custom_l14_onnx_size_mb=custom_l14_size,
        mlp_ready=mlp_ok,
        mlp_path=str(mlp_p) if mlp_p else str(MLP_WEIGHTS_PATH),
        mlp_l14_ready=mlp_l14_ok,
        mlp_l14_path=str(mlp_l14_p) if mlp_l14_p else str(MLP_L14_WEIGHTS_PATH),
        active_mode=active_mode,
        active_onnx_path=active_path_str,
        is_fully_ready=is_ready,
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


def download_clip_l14_model(
    use_mirror: bool = True,
    progress_callback: Callable[[str, float], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> bool:
    """
    下载 CLIP ViT-L/14 (Aesthetic 3) 专业大模型至本地 models/clip-vit-large-patch14 目录。
    1. 若系统 HF 缓存已存在，直接秒级导入！
    2. 否则使用 huggingface_hub / 镜像源下载。
    """
    if is_clip_l14_in_hf_cache():
        if progress_callback:
            progress_callback("检测到系统缓存中已存在 ViT-L/14 模型，正在秒级同步...", 0.5)
        success = import_from_hf_cache_l14(progress_callback)
        if success:
            return True

    CLIP_L14_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    base_host = HF_MIRROR_URL if use_mirror else HF_OFFICIAL_URL

    if progress_callback:
        progress_callback("正在准备下载 CLIP ViT-L/14 (Aesthetic 3) 大模型...", 0.0)

    try:
        from huggingface_hub import snapshot_download
        endpoint = base_host if use_mirror else None
        if progress_callback:
            progress_callback(f"通过 HuggingFace Hub 从 {base_host} 下载中...", 0.1)

        snapshot_download(
            repo_id=CLIP_L14_REPO_ID,
            local_dir=str(CLIP_L14_MODEL_DIR),
            endpoint=endpoint,
            ignore_patterns=["*.msgpack", "*.h5", "*.tflite", "*.ot"],
        )
        if is_clip_l14_model_downloaded():
            if progress_callback:
                progress_callback("✅ CLIP ViT-L/14 模型下载完成！", 1.0)
            return True
    except Exception:
        pass

    # HTTP 直接下载 fallback
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

        target_file = CLIP_L14_MODEL_DIR / fname
        if target_file.exists() and target_file.stat().st_size > 0:
            if fname == "model.safetensors" and target_file.stat().st_size > 300 * 1024 * 1024:
                continue
            elif fname != "model.safetensors":
                continue

        url = f"{base_host}/{CLIP_L14_REPO_ID}/resolve/main/{fname}"
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
                bin_url = f"{base_host}/{CLIP_L14_REPO_ID}/resolve/main/{bin_fname}"
                bin_target = CLIP_L14_MODEL_DIR / bin_fname
                try:
                    download_file_with_progress(
                        bin_url, bin_target, _file_progress, f"[{idx+1}/{total_files}] 下载 {bin_fname}"
                    )
                except Exception as bin_exc:
                    raise RuntimeError(f"下载 ViT-L/14 权重失败: {exc} | {bin_exc}") from exc
            else:
                raise RuntimeError(f"下载文件 {fname} 失败: {exc}") from exc

    success = is_clip_l14_model_downloaded()
    if success and progress_callback:
        progress_callback("✅ CLIP ViT-L/14 大模型下载完成！", 1.0)
    return success


def fuse_standard_l14_onnx(
    progress_callback: Callable[[str], None] | None = None
) -> Path:
    """
    下载官方 ViT-L/14 美学线性权重并与 CLIP ViT-L/14 熔铸导出为 standard_aesthetic_l14_model.onnx
    """
    import torch
    import torch.nn as nn
    from transformers import CLIPModel

    def _notify(msg: str):
        if progress_callback:
            progress_callback(msg)

    # 1. 确保权重就位
    if not L14_LINEAR_WEIGHTS_PATH.exists() or L14_LINEAR_WEIGHTS_PATH.stat().st_size < 1000:
        _notify("正在下载官方 ViT-L/14 美学线性权重...")
        req = urllib.request.Request(L14_LINEAR_WEIGHTS_URL, headers={"User-Agent": "PhotoSort-App/1.0"})
        with urllib.request.urlopen(req) as resp, open(L14_LINEAR_WEIGHTS_PATH, "wb") as f:
            f.write(resp.read())

    # 2. 加载 CLIP ViT-L/14
    clip_source = get_clip_l14_model_path()
    _notify(f"正在加载 CLIP ViT-L/14: {clip_source} ...")
    clip_model = CLIPModel.from_pretrained(clip_source, return_dict=False).eval()
    state = torch.load(str(L14_LINEAR_WEIGHTS_PATH), map_location="cpu")

    class StandardL14AestheticONNX(nn.Module):
        def __init__(self, clip, weight, bias):
            super().__init__()
            self.clip = clip
            self.linear = nn.Linear(768, 1)
            self.linear.weight = nn.Parameter(weight)
            self.linear.bias = nn.Parameter(bias)

        def forward(self, pixel_values):
            vision_outputs = self.clip.vision_model(pixel_values=pixel_values, return_dict=False)
            pooled_output = vision_outputs[1]
            features = self.clip.visual_projection(pooled_output)
            features = features / features.norm(dim=-1, keepdim=True)
            raw_score = self.linear(features)
            norm_score = torch.clamp((raw_score - 1.0) / 9.0, 0.0, 1.0)
            return norm_score.squeeze(-1)

    model = StandardL14AestheticONNX(clip_model, state["weight"], state["bias"]).eval()
    dummy_input = torch.randn(1, 3, 224, 224)

    _notify(f"正在导出 Aesthetic 3 ONNX 模型至: {STANDARD_L14_ONNX_PATH.name} ...")
    STANDARD_L14_ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (dummy_input,),
        str(STANDARD_L14_ONNX_PATH),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["pixel_values"],
        output_names=["aesthetic_score"],
        dynamic_axes={"pixel_values": {0: "batch_size"}, "aesthetic_score": {0: "batch_size"}},
        dynamo=False,
    )
    _notify(f"✅ Aesthetic 3 ONNX 导出成功 ({STANDARD_L14_ONNX_PATH.stat().st_size / (1024*1024):.1f} MB)！")
    return STANDARD_L14_ONNX_PATH

