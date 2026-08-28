"""
burst_filter.py — RAW 连拍优选与冗余片自动移动

算法说明（BurstGrouper）：
  采用"首帧锡点比对"法。每个子组以第一张照片作为基准帧。
  后续照片满足以下两个条件才归入同一子组：
    1. 与前一张的时间间隔 <= gap_seconds
    2. 与当前子组基准帧的 dHash 汉明距离 <= max_hamming_distance
  任意一个条件不满足，则截断当前子组，以该照片开启新子组。
"""

from __future__ import annotations

import os
import shutil
import sys
import threading
import warnings

# 确保标准输出为 UTF-8 编码，防止 Windows GBK 环境下 Emoji 引发 UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np
import rawpy
from PIL import Image

# ── 注册现代高效率图像格式解码器 (HIF / HEIF / HEIC / JPEG XL) ───────────
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pass

try:
    import pillow_jxl
except Exception:
    pass

# ── 支持的照片扩展名格式集 ───────────────────────────────────────────────────
RAW_SUFFIXES = {
    # 尼康 Nikon
    ".nef", ".nrw",
    # 索尼 Sony
    ".arw", ".srf", ".sr2",
    # 佳能 Canon (CR2 / CR3 / CRW)
    ".cr2", ".cr3", ".crw",
    # 松下 / 徕卡 Panasonic / Lumix / Leica (RW2 / RAW)
    ".rw2", ".raw",
    # Adobe DNG (通用 RAW / 徕卡 / 大疆无人机 / Apple ProRAW / 理光 / 宾得)
    ".dng",
    # 富士 Fujifilm
    ".raf",
    # 奥林巴斯 / OM System Olympus
    ".orf", ".ori",
    # 宾得 / 理光 Pentax / Ricoh
    ".pef", ".ptx",
    # 哈苏 / 飞思 / 三星 / 适马 / 柯达 / 美能达 / GoPro 等
    ".3fr", ".fff", ".iiq", ".srw", ".x3f", ".mrw", ".gpr", ".erf", ".mef", ".mos",
}

STANDARD_IMAGE_SUFFIXES = {
    ".jpg", ".jpeg", ".jpe",
    ".jxl",
    ".hif", ".heif", ".heic",
    ".png", ".webp", ".tiff", ".tif", ".bmp"
}

SUPPORTED_PHOTO_SUFFIXES = RAW_SUFFIXES | STANDARD_IMAGE_SUFFIXES


# ── 人脸检测器（加载失败时降级为中心区域锐度）───────────────────────────
try:
    _FACE_CASCADE = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    _FACE_DETECTION_AVAILABLE = not _FACE_CASCADE.empty()
except Exception:
    _FACE_CASCADE = None
    _FACE_DETECTION_AVAILABLE = False

# ── EXIF 常量 ─────────────────────────────────────────────────────────────────
_EXIF_IFD_TAG = 34665
_DATETIME_ORIGINAL_TAG = 36867
_DATETIME_FMT = "%Y:%m:%d %H:%M:%S"

# ── 默认参数 ──────────────────────────────────────────────────────────────────
DEFAULT_TIME_GAP_SECONDS: float = 1.5
DEFAULT_MAX_HAMMING_DISTANCE: int = 12   # dHash 64 位中允许的最大不同位数
DEFAULT_REVIEW_SUBDIR: str = "审查_连拍淘汰"

_CENTER_CROP_RATIO: float = 0.6

# ── 动态加载 ONNX Runtime ───────────────────────────────────────────────────
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════════
# 数据类
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ScoredPhoto:
    path: Path
    score: float = 0.0


@dataclass
class PhotoShot:
    """
    单次快门拍摄的实体对象。
    若开启了 RAW+JPG / RAW+HIF / XMP 伴生存储，同一个 PhotoShot 会囊括该次快门生成的所有伴生文件，
    在连拍聚类、打分以及优选移动时均作为同一个整体处理。
    """
    primary_path: Path
    all_paths: list[Path] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.all_paths:
            self.all_paths = [self.primary_path]
        elif self.primary_path not in self.all_paths:
            self.all_paths.insert(0, self.primary_path)

    @property
    def path(self) -> Path:
        return self.primary_path

    @property
    def name(self) -> str:
        return self.primary_path.name

    @property
    def stem(self) -> str:
        return self.primary_path.stem

    def __hash__(self) -> int:
        return hash(self.primary_path)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, PhotoShot):
            return self.primary_path == other.primary_path
        if isinstance(other, Path):
            return self.primary_path == other
        return False



@dataclass
class BurstFilterResult:
    total: int = 0
    skipped_single: int = 0
    burst_groups: int = 0
    moved: int = 0
    review_dir: Path | None = None
    errors: list[str] = field(default_factory=list)



# ── 亚秒 EXIF 标签 ────────────────────────────────────────────────────────────
_SUBSEC_TIME_ORIGINAL_TAG  = 37521   # SubSecTimeOriginal
_SUBSEC_TIME_DIGITIZED_TAG = 37522   # SubSecTimeDigitized

# ══════════════════════════════════════════════════════════════════════════════
# EXIF 时间读取
# ══════════════════════════════════════════════════════════════════════════════

class RawExifReader:
    """读取拍摄时间（精确到微秒），失败时回退 mtime。适用于各种相机 RAW 格式及通用图像。"""

    def read_datetime(self, path: Path) -> datetime:
        try:
            return self._from_exif(path)
        except Exception:
            return self._from_mtime(path)

    def _parse_exif_from_img(self, img: Image.Image) -> datetime:
        exif = img.getexif()
        if not exif:
            raise ValueError("No EXIF")
        ifd = exif.get_ifd(_EXIF_IFD_TAG) if hasattr(exif, "get_ifd") else {}

        # 整秒时间
        raw = ifd.get(_DATETIME_ORIGINAL_TAG) or exif.get(_DATETIME_ORIGINAL_TAG)
        if not raw:
            raise ValueError("No DateTimeOriginal")
        dt = datetime.strptime(str(raw).strip(), _DATETIME_FMT)

        # 亚秒时间（优先 SubSecTimeOriginal，次选 SubSecTimeDigitized）
        subsec_raw = (
            ifd.get(_SUBSEC_TIME_ORIGINAL_TAG)
            or ifd.get(_SUBSEC_TIME_DIGITIZED_TAG)
            or exif.get(_SUBSEC_TIME_ORIGINAL_TAG)
            or exif.get(_SUBSEC_TIME_DIGITIZED_TAG)
        )
        microseconds = 0
        if subsec_raw is not None:
            try:
                # 字段如 "45" 表示 0.45 秒，左对齐填充到 6 位
                s = str(subsec_raw).strip()
                microseconds = int(s[:6].ljust(6, "0"))
            except (ValueError, TypeError):
                microseconds = 0

        return dt.replace(microsecond=microseconds)

    def _from_exif(self, path: Path) -> datetime:
        # 1. 优先使用 Pillow 直接读取 EXIF (适用于 DNG, CR2, NEF, ARW, JPG, PNG 等)
        try:
            with Image.open(path) as img:
                return self._parse_exif_from_img(img)
        except Exception:
            pass

        # 2. 如果 Pillow 无法直接解析该 RAW 容器格式 (如 CR3, RW2 等)，尝试从 rawpy 提取内嵌缩略图读取 EXIF
        if path.suffix.lower() in RAW_SUFFIXES:
            try:
                with rawpy.imread(str(path)) as raw:
                    thumb = raw.extract_thumb()
                if thumb.format == rawpy.ThumbFormat.JPEG:
                    with Image.open(BytesIO(thumb.data)) as img:
                        return self._parse_exif_from_img(img)
            except Exception:
                pass

        raise ValueError(f"No EXIF DateTime found in {path.name}")

    @staticmethod
    def _from_mtime(path: Path) -> datetime:
        return datetime.fromtimestamp(path.stat().st_mtime)


# ══════════════════════════════════════════════════════════════════════════════
# 预览提取 & 多维度评估
# ══════════════════════════════════════════════════════════════════════════════

class RawEvaluator:
    """
    适用于所有 RAW 格式的预览提取器 + 多维度评估器。

    evaluate(img) 返回 (sharpness: float, exposure_score: float)：
      - sharpness      : Laplacian 方差 + Tenengrad 梯度（越大越清晰），
                         优先在人脸区域计算，无人脸则用中心裁剪区。
      - exposure_score : 0.0~1.0，对高光过曝重罚、对欠曝宽容。
    """

    def extract_preview(self, path: Path) -> np.ndarray:
        suffix = path.suffix.lower()
        if suffix in RAW_SUFFIXES:
            # 1. 优先从 rawpy 提取内嵌 JPEG / BITMAP 缩略图 (耗时极低 ~5-15ms)
            try:
                return self._extract_thumb(path)
            except Exception:
                pass
            # 2. 若无内嵌缩略图或提取失败，尝试 rawpy 半采样解码 (half_decode)
            try:
                return self._half_decode(path)
            except Exception:
                pass
            # 3. 兜底尝试 Pillow 打开 (如某些特殊 DNG / TIFF 格式)
            try:
                with Image.open(path) as img:
                    return np.asarray(img.convert("RGB"))
            except Exception as exc:
                raise RuntimeError(f"无法读取 RAW 预览: {path.name}") from exc
        else:
            try:
                with Image.open(path) as img:
                    return np.asarray(img.convert("RGB"))
            except Exception as exc:
                raise RuntimeError(f"无法读取图像文件: {path.name}") from exc

    @staticmethod
    def _extract_thumb(path: Path) -> np.ndarray:
        with rawpy.imread(str(path)) as raw:
            thumb = raw.extract_thumb()
        if thumb.format == rawpy.ThumbFormat.JPEG:
            img = Image.open(BytesIO(thumb.data))
        elif thumb.format == rawpy.ThumbFormat.BITMAP:
            img = Image.fromarray(thumb.data)
        else:
            raise ValueError(f"Unknown thumb format: {thumb.format}")
        return np.asarray(img.convert("RGB"))

    @staticmethod
    def _half_decode(path: Path) -> np.ndarray:
        with rawpy.imread(str(path)) as raw:
            return raw.postprocess(half_size=True, use_camera_wb=True, output_bps=8)


    # ── 人脸优先的锐度计算 ────────────────────────────────────────────────────

    def sharpness(self, img_rgb: np.ndarray) -> float:
        """计算锐度：优先在人脸框内计算，无人脸则中心裁剪。"""
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        regions = self._face_regions(gray)
        if regions:
            scores = [self._region_sharpness(gray, x, y, w, h) for x, y, w, h in regions]
            return float(np.mean(scores))
        return self._center_sharpness(gray)

    @staticmethod
    def _face_regions(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
        """检测人脸，返回 [(x, y, w, h), ...]；无法检测则返回空列表。"""
        if not _FACE_DETECTION_AVAILABLE:
            return []
        try:
            faces = _FACE_CASCADE.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            if len(faces) == 0:
                return []
            return [tuple(f) for f in faces]
        except Exception:
            return []

    @staticmethod
    def _region_sharpness(gray: np.ndarray, x: int, y: int, w: int, h: int) -> float:
        """计算指定矩形区域的 Laplacian + Sobel 锐度。"""
        roi = gray[y:y + h, x:x + w].astype(np.float64)
        lap = cv2.Laplacian(roi, cv2.CV_64F).var()
        sx = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
        return lap + float((sx ** 2 + sy ** 2).mean())

    def _center_sharpness(self, gray: np.ndarray) -> float:
        """无人脸时：计算画面中心 60% 区域的锐度。"""
        h, w = gray.shape
        mh = int(h * (1 - _CENTER_CROP_RATIO) / 2)
        mw = int(w * (1 - _CENTER_CROP_RATIO) / 2)
        return self._region_sharpness(gray, mw, mh, w - 2 * mw, h - 2 * mh)

    # ── 曝光评分（倾向于欠曝，重惩高光溢出）──────────────────────────────────

    @staticmethod
    def exposure_score(img_rgb: np.ndarray) -> float:
        """
        返回 0.0~1.0 的曝光评分。
        公式：1.0 - pct_white * 2.0 - pct_black * 0.5，夹紧到 [0, 1]。
        """
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        total = gray.size
        pct_black = float(np.sum(gray <= 5)) / total
        pct_white = float(np.sum(gray >= 250)) / total
        raw = 1.0 - pct_white * 2.0 - pct_black * 0.5
        return float(np.clip(raw, 0.0, 1.0))

    # ── 向后兼容：旧的 score() 接口仍可用（BurstGrouper 不使用此方法）──────

    def score(self, img_rgb: np.ndarray) -> float:
        """兼容旧接口，返回绝对锐度值。"""
        return self.sharpness(img_rgb)


# ══════════════════════════════════════════════════════════════════════════════
# AI 美学评分器（CLIP 特征提取 + 轻量 MLP 分类头）
# ══════════════════════════════════════════════════════════════════════════════


class AestheticScorer:
    """
    优先使用 ONNX 引擎（极速、脱离 PyTorch）。
    如果 ONNX 模型不存在或依赖缺失，则回退到 PyTorch + Transformers（供本地训练后快速验证）。
    如果均不可用，自动降级（score 返回 1.0 全部通过）。
    """

    _CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

    def __init__(self, project_root: Path, use_gpu: bool = True):
        self.available = False
        self.engine = None  # "onnx" or "torch"
        self.model_name = "未加载"
        self._infer_lock = threading.Lock()

        # 尝试从 model_manager 解析当前激活的模型
        onnx_path = None
        mlp_path = None
        try:
            from model_manager import get_active_aesthetic_model_path, get_resolved_mlp_path, get_active_model_mode
            onnx_path = get_active_aesthetic_model_path()
            mlp_path = get_resolved_mlp_path()
            active_mode = get_active_model_mode()
        except Exception:
            active_mode = "standard"

        if onnx_path is None:
            if getattr(sys, 'frozen', False):
                bundle_root = Path(sys._MEIPASS)
                exe_root = Path(sys.executable).parent
            else:
                bundle_root = project_root
                exe_root = project_root

            for candidate in [
                exe_root / "models" / "standard_aesthetic_model.onnx",
                exe_root / "models" / "custom_aesthetic_model.onnx",
                exe_root / "photo_sort_model.onnx",
                bundle_root / "models" / "standard_aesthetic_model.onnx",
                bundle_root / "standard_aesthetic_model.onnx",
                bundle_root / "photo_sort_model.onnx",
            ]:
                if candidate.exists() and candidate.stat().st_size > 100 * 1024 * 1024:
                    onnx_path = candidate
                    break

        if mlp_path is None:
            for candidate in [
                project_root / "models" / "aesthetic_mlp.pth",
                project_root / "aesthetic_mlp.pth",
            ]:
                if candidate.exists() and candidate.stat().st_size > 1000:
                    mlp_path = candidate
                    break

        # 1. 尝试初始化 ONNX 引擎
        if ONNX_AVAILABLE and onnx_path and onnx_path.exists():
            try:
                # 动态探寻本地支持的硬件加速器
                if use_gpu:
                    available_providers = ort.get_available_providers()
                    target_providers = ['CUDAExecutionProvider', 'CoreMLExecutionProvider', 'DmlExecutionProvider', 'CPUExecutionProvider']
                    active_providers = [p for p in target_providers if p in available_providers]
                else:
                    active_providers = ['CPUExecutionProvider']
                
                self._session = ort.InferenceSession(str(onnx_path), providers=active_providers)
                self.engine = "onnx"
                if "l14" in onnx_path.name.lower():
                    self.model_name = "Aesthetic 3 官方专业大模型 (ViT-L/14)" if "standard" in onnx_path.name else "个人专属训练模型 (ViT-L/14)"
                elif "standard" in onnx_path.name:
                    self.model_name = "官方标准通用模型 (ViT-B/32)"
                else:
                    self.model_name = "个人专属训练模型 (ViT-B/32)"
                self.available = True
                return


            except Exception as exc:
                warnings.warn(f"无法加载 ONNX 模型，尝试降级: {exc}")

        # 2. 尝试回退到 PyTorch 引擎
        if mlp_path and mlp_path.exists():
            try:
                import torch
                import torch.nn as nn
                from transformers import CLIPProcessor, CLIPModel

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

                if use_gpu:
                    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    if sys.platform == "darwin" and torch.backends.mps.is_available():
                        self.device = torch.device("mps")
                else:
                    self.device = torch.device("cpu")

                local_clip_dir = project_root / "models" / "clip-vit-base-patch32"
                clip_source = str(local_clip_dir) if local_clip_dir.exists() and (local_clip_dir / "config.json").exists() else self._CLIP_MODEL_NAME

                self._clip_model = CLIPModel.from_pretrained(clip_source)
                self._clip_model.to(self.device)
                self._clip_model.eval()
                for p in self._clip_model.parameters():
                    p.requires_grad_(False)

                self._clip_processor = CLIPProcessor.from_pretrained(clip_source)

                self._mlp = _AestheticMLP(input_dim=512).to(self.device)
                state = torch.load(str(mlp_path), map_location=self.device, weights_only=True)
                self._mlp.load_state_dict(state)
                self._mlp.eval()

                self.engine = "torch"
                self.available = True
                return
            except Exception as exc:
                warnings.warn(f"无法加载 PyTorch 模型: {exc}")

    def _preprocess_onnx(self, img_rgb: np.ndarray) -> np.ndarray:
        """纯 numpy/cv2 实现 CLIP 的图像预处理"""
        h, w = img_rgb.shape[:2]
        
        if h < w:
            new_h = 224
            new_w = int(w * (224 / h))
        else:
            new_w = 224
            new_h = int(h * (224 / w))
        
        img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        start_y = (new_h - 224) // 2
        start_x = (new_w - 224) // 2
        img_cropped = img_resized[start_y:start_y+224, start_x:start_x+224]
        
        img_float = img_cropped.astype(np.float32) / 255.0
        mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
        std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
        img_norm = (img_float - mean) / std
        
        img_transposed = np.transpose(img_norm, (2, 0, 1))
        return np.expand_dims(img_transposed, axis=0)

    def score(self, img_rgb: np.ndarray) -> float:
        """返回 0.0~1.0 的美学/偏好概率，不可用时返回 1.0（全部通过）。"""
        if not self.available:
            return 1.0

        if self.engine == "onnx":
            try:
                inputs = self._preprocess_onnx(img_rgb)
                ort_inputs = {self._session.get_inputs()[0].name: inputs}
                with self._infer_lock:
                    probs = self._session.run(None, ort_inputs)[0]
                val = float(np.squeeze(probs))
                return float(np.clip(val, 0.0, 1.0))
            except Exception:
                return 1.0
        
        elif self.engine == "torch":
            try:
                pil_img = Image.fromarray(img_rgb)
                inputs = self._clip_processor(
                    images=pil_img, return_tensors="pt", padding=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with self._infer_lock:
                    with torch.no_grad():
                        vision_out = self._clip_model.vision_model(
                            pixel_values=inputs['pixel_values']
                        )
                        pooled = vision_out.pooler_output
                        features = self._clip_model.visual_projection(pooled)
                        features = features / features.norm(dim=-1, keepdim=True)
                        logits = self._mlp(features)
                        prob = torch.softmax(logits, dim=1)[0][1].item()
                return float(np.clip(prob, 0.0, 1.0))
            except Exception:
                return 1.0



# ══════════════════════════════════════════════════════════════════════════════
# 连拍分组（首帧锚点比令人）
# ══════════════════════════════════════════════════════════════════════════════

class BurstGrouper:
    """
    首帧锚点比对法（Anchor-frame）连拍分组。

    对每张照片（按拍摄时间排序后）：
      - 若与前一张时间间隔 > gap_seconds      → 截断，开新组
      - 否则计算当前帧与锚点帧的 dHash 汉明距离：
          距离 <= max_hamming_distance → 加入当前子组（锚点不变）
          距离 >  max_hamming_distance → 截断（构图/角度变化），开新子组

    dHash 优于直方图：对曝光变化不敏感，对结构/边缘更敏感。
    返回 list[list[Path]]，单元素 = 单拍，多元素 = 连拍子组。
    """

    def __init__(
        self,
        exif_reader: RawExifReader,
        preview_extractor: RawEvaluator,
        gap_seconds: float = DEFAULT_TIME_GAP_SECONDS,
        max_hamming_distance: int = DEFAULT_MAX_HAMMING_DISTANCE,
        max_workers: int | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> None:
        self._exif = exif_reader
        self._previewer = preview_extractor
        self.gap_seconds = gap_seconds
        self.max_hamming_distance = max_hamming_distance
        if max_workers is None or max_workers <= 0:
            max_workers = max(1, round((os.cpu_count() or 4) * 0.8))
        self.max_workers = max_workers
        self.progress_callback = progress_callback

    def group(self, items: Sequence[Path | PhotoShot]) -> list[list[Any]]:
        if not items:
            return []

        # 统一封装为 PhotoShot
        shots: list[PhotoShot] = [
            it if isinstance(it, PhotoShot) else PhotoShot(primary_path=it, all_paths=[it])
            for it in items
        ]

        total = len(shots)

        # 1. 多线程并发读取时间与提取 dHash，每完成 50 张回调一次进度
        def _parse_meta(shot: PhotoShot) -> tuple[PhotoShot, datetime, np.ndarray | None]:
            p = shot.primary_path
            return shot, self._exif.read_datetime(p), self._safe_dhash(p)

        import concurrent.futures
        import os
        workers = max(1, self.max_workers)

        meta_map: dict[PhotoShot, tuple[datetime, np.ndarray | None]] = {}
        _PROGRESS_BATCH = 50   # 每完成多少张通知一次
        completed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_parse_meta, shot): shot for shot in shots}
            for fut in concurrent.futures.as_completed(futures):
                shot, dt, h = fut.result()
                meta_map[shot] = (dt, h)
                completed += 1
                if self.progress_callback and (
                    completed % _PROGRESS_BATCH == 0 or completed == total
                ):
                    self.progress_callback(
                        f"正在提取图像特征哈希… ({completed}/{total})"
                    )

        timed: list[tuple[datetime, PhotoShot]] = sorted(
            ((meta_map[s][0], s) for s in shots),
            key=lambda x: x[0],
        )

        # 2. 首帧锚点比对聚类
        result: list[list[Any]] = []
        current_group: list[Any] = [timed[0][1]]
        anchor_hash: np.ndarray | None = meta_map[timed[0][1]][1]
        prev_time = timed[0][0]

        for cur_time, cur_shot in timed[1:]:
            gap = (cur_time - prev_time).total_seconds()
            prev_time = cur_time

            # 条件 1：时间超出阈值 → 直接截断
            if gap > self.gap_seconds:
                result.append(current_group)
                current_group = [cur_shot]
                anchor_hash = meta_map[cur_shot][1]
                continue

            # 条件 2：与锚点帧比对 dHash 汉明距离
            cur_hash = meta_map[cur_shot][1]
            if anchor_hash is None or cur_hash is None:
                # 预览提取失败，保守截断
                result.append(current_group)
                current_group = [cur_shot]
                anchor_hash = cur_hash
                continue

            dist = self._hamming(anchor_hash, cur_hash)
            if dist <= self.max_hamming_distance:
                # 同一场景，加入当前子组（锚点保持第一张不变）
                current_group.append(cur_shot)
            else:
                # 构图/角度变化，截断并以当前帧开启新子组
                result.append(current_group)
                current_group = [cur_shot]
                anchor_hash = cur_hash

        result.append(current_group)
        return result

    def _safe_dhash(self, path: Path) -> np.ndarray | None:
        """提取 dHash；失败返回 None，不会泄漏 numpy 数组引用。"""
        try:
            preview = self._previewer.extract_preview(path)
            h = self._dhash(preview)
            del preview
            return h
        except Exception:
            return None

    @staticmethod
    def _dhash(img_rgb: np.ndarray) -> np.ndarray:
        """
        差异哈希（dHash）：将图像缩小为 9x8 灰度图，
        比较每行相邻像素大小，生成 64 位 bool 数组。
        """
        small = cv2.resize(
            cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY),
            (9, 8),
            interpolation=cv2.INTER_AREA,
        )
        # 每行 9 个像素，比较相邻两列差值：生成 8x8=64 位
        return small[:, :-1] > small[:, 1:]   # shape (8, 8), dtype bool

    @staticmethod
    def _hamming(a: np.ndarray, b: np.ndarray) -> int:
        """两个 dHash 之间的汉明距离（不同位数）。"""
        return int(np.count_nonzero(a != b))


# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════

class BurstFilter:
    """
    连拍优选主控器。
      result = BurstFilter().run(Path("/path/to/folder"))
    """

    _RAW_SUFFIXES = {s for s in SUPPORTED_PHOTO_SUFFIXES} | {s.upper() for s in SUPPORTED_PHOTO_SUFFIXES}

    def __init__(
        self,
        gap_seconds: float = DEFAULT_TIME_GAP_SECONDS,
        max_hamming_distance: int = DEFAULT_MAX_HAMMING_DISTANCE,
        review_subdir: str = DEFAULT_REVIEW_SUBDIR,
        keep_count: int = 1,
        max_workers: int | None = None,
        use_gpu: bool = True,
        progress_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.gap_seconds = gap_seconds
        self.max_hamming_distance = max_hamming_distance
        self.review_subdir = review_subdir
        self.keep_count = max(1, keep_count)
        if max_workers is None or max_workers <= 0:
            max_workers = max(1, round((os.cpu_count() or 4) * 0.8))
        self.max_workers = max_workers
        self.use_gpu = use_gpu
        self.progress_callback = progress_callback

        self._exif_reader = RawExifReader()
        self._scorer = RawEvaluator()
        self._grouper = BurstGrouper(
            exif_reader=self._exif_reader,
            preview_extractor=self._scorer,
            gap_seconds=gap_seconds,
            max_hamming_distance=max_hamming_distance,
            max_workers=max_workers,
            progress_callback=self.progress_callback,
        )
        
        # 尝试加载美学模型（支持 ONNX 与 PyTorch 双引擎）
        if getattr(sys, 'frozen', False):
            # PyInstaller 环境
            project_root = Path(sys._MEIPASS)
        else:
            project_root = Path(__file__).resolve().parent.parent

        self._aesthetic_scorer = AestheticScorer(project_root, use_gpu=self.use_gpu)
        if self._aesthetic_scorer.available:
            engine_str = "ONNX" if self._aesthetic_scorer.engine == "onnx" else "PyTorch"
            self._notify(f"🚀 已启用 {engine_str} 美学评分模型！")
        else:
            self._notify("ℹ️ 未检测到有效的美学模型，降级为纯 OpenCV 锐度过滤。")

    def run(self, input_dir: Path) -> BurstFilterResult:
        result = BurstFilterResult()
        photo_files = self._scan_raw(input_dir)
        result.total = len(photo_files)
        if not photo_files:
            return result

        # 伴生文件聚合（同一次快门的 RAW+JPG/HIF 等视为同一个 PhotoShot）
        shots = self._pair_shots(photo_files)
        paired_count = len(photo_files) - len(shots)
        if paired_count > 0:
            self._notify(f"扫描到 {result.total} 个文件（包含 {paired_count} 组 RAW+JPG 伴生照片，已合并为 {len(shots)} 张独立快门照片），正在分析连拍组…")
        else:
            self._notify(f"扫描到 {result.total} 张照片，正在分析连拍组…")

        groups = self._grouper.group(shots)

        burst_groups = [g for g in groups if len(g) > 1]
        result.skipped_single = sum(
            sum(len(item.all_paths) if isinstance(item, PhotoShot) else 1 for item in g)
            for g in groups if len(g) == 1
        )
        result.burst_groups = len(burst_groups)

        if not burst_groups:
            self._notify("未检测到连拍组，所有文件保留原位。")
            return result

        review_dir = input_dir / self.review_subdir
        review_dir.mkdir(parents=True, exist_ok=True)
        result.review_dir = review_dir

        for idx, group in enumerate(burst_groups, 1):
            self._notify(f"处理连拍组 {idx}/{len(burst_groups)}（包含 {len(group)} 次连拍拍摄）…")
            moved, errors = self._process_group(group, review_dir)
            result.moved += moved
            result.errors.extend(errors)

        return result

    def _scan_raw(self, input_dir: Path) -> list[Path]:
        return sorted(
            p for p in input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_PHOTO_SUFFIXES
        )

    def _scan_photos(self, input_dir: Path) -> list[Path]:
        return self._scan_raw(input_dir)

    def _pair_shots(self, photo_files: Sequence[Path]) -> list[PhotoShot]:
        """
        将同一次快门拍摄的伴生文件（如同名 RAW + JPG / RAW + HIF / XMP 等）聚合成同一个 PhotoShot。
        优先选择 RAW 文件作为主文件（primary_path）用于打分与预览提取。
        """
        from collections import OrderedDict
        groups: dict[tuple[Path, str], list[Path]] = OrderedDict()
        for p in sorted(photo_files):
            key = (p.parent, p.stem.lower())
            groups.setdefault(key, []).append(p)

        shots: list[PhotoShot] = []
        for (parent, stem), paths in groups.items():
            sorted_paths = sorted(
                paths,
                key=lambda x: (
                    0 if x.suffix.lower() in RAW_SUFFIXES else 1,
                    0 if x.suffix.lower() in STANDARD_IMAGE_SUFFIXES else 1,
                    x.suffix.lower()
                )
            )
            shots.append(PhotoShot(primary_path=sorted_paths[0], all_paths=sorted_paths))
        return shots

    def _process_group(
        self, group: list[Any], review_dir: Path
    ) -> tuple[int, list[str]]:
        """
        对连拍组内所有照片实体进行多维度评估，综合加权后保留前 keep_count 张。
        若某照片包含 RAW+JPG 伴生文件，保留时全部保留在原目录，淘汰时全部移动至审查目录。

        各维度权重：
          AI 美学概率   : 0.6
          归一化锐度    : 0.3
          曝光评分      : 0.1
        """
        @dataclass
        class _EvaluatedShot:
            shot: PhotoShot
            sharpness: float = 0.0
            exposure: float  = 1.0
            aesthetic: float = 1.0
            failed: bool     = False
            _norm_sharp: float = 0.0
            final_score: float = -1.0

            @property
            def path(self) -> Path:
                return self.shot.primary_path

        def _to_shot(item: Any) -> PhotoShot:
            if isinstance(item, PhotoShot):
                return item
            return PhotoShot(primary_path=item, all_paths=[item])

        shots: list[PhotoShot] = [_to_shot(item) for item in group]
        evaluated_list: list[_EvaluatedShot] = []
        errors: list[str] = []

        def _evaluate_shot(shot: PhotoShot) -> _EvaluatedShot | tuple[_EvaluatedShot, str]:
            es = _EvaluatedShot(shot=shot)
            try:
                # 优先使用 primary_path（RAW 优先）提取预览与多维度打分
                preview = self._scorer.extract_preview(shot.primary_path)
                es.sharpness = self._scorer.sharpness(preview)
                es.exposure  = self._scorer.exposure_score(preview)
                es.aesthetic = self._aesthetic_scorer.score(preview)
                return es
            except Exception as exc:
                es.failed = True
                return es, f"{shot.primary_path.name}: {exc}"

        import concurrent.futures
        import os
        workers = max(1, self.max_workers)

        # ── 评分阶段并发执行 ──────────────────────────────────────────────
        # 说明：此处维持 ThreadPoolExecutor 而不采用 ProcessPoolExecutor，原因如下：
        # 1. self._aesthetic_scorer（包含 ONNX Runtime / PyTorch C++ Session）与 OpenCV
        #    Cascade 级联分类器均包含底层 C++ 对象指针，无法跨进程 Pickle 序列化；
        # 2. 在 PyInstaller 打包环境及 macOS spawn 模式下，子进程冷启动与模块重复加载开销巨大；
        # 3. ONNX Runtime 推理与 OpenCV（Laplacian/Sobel）底层 C++ 函数执行期间已主动释放 Python GIL；
        # 4. 前端 UI 响应延迟已通过 BurstWorker 层的 150ms 进度信号节流机制彻底消除。
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            for res in executor.map(_evaluate_shot, shots):
                if isinstance(res, tuple):
                    es, err_msg = res
                    warnings.warn(f"跳过 {es.shot.primary_path.name}: {err_msg}")
                    errors.append(err_msg)
                    evaluated_list.append(es)
                else:
                    evaluated_list.append(res)

        if not evaluated_list:
            return 0, errors

        # ── 阶段 2：组内归一化锐度（避免绝对值量纲差异主导结果）─────────────
        valid = [es for es in evaluated_list if not es.failed]
        if valid:
            max_s = max(es.sharpness for es in valid)
            min_s = min(es.sharpness for es in valid)
            span  = max_s - min_s + 1e-6
            for es in valid:
                es._norm_sharp = (es.sharpness - min_s) / span
        for es in evaluated_list:
            if not hasattr(es, '_norm_sharp'):
                es._norm_sharp = 0.0

        # ── 阶段 3：计算综合得分并排序 ────────────────────────────────────────
        for es in valid:
            es.final_score = (
                es.aesthetic     * 0.6
                + es._norm_sharp * 0.3
                + es.exposure    * 0.1
            )
        for es in evaluated_list:
            if not hasattr(es, 'final_score'):
                es.final_score = -1.0

        keep_n = min(self.keep_count, len(valid))
        top = sorted(valid, key=lambda x: x.final_score, reverse=True)[:keep_n]
        top_shots: set[PhotoShot] = {es.shot for es in top}

        # ── 阶段 4：移动淘汰照片（连同 RAW+JPG/HIF 等伴生文件一同移动）─────────
        moved = 0
        for es in evaluated_list:
            if es.shot in top_shots or es.failed:
                continue
            for fpath in es.shot.all_paths:
                if not fpath.exists():
                    continue
                try:
                    dest = review_dir / fpath.name
                    if dest.exists():
                        dest = review_dir / f"{fpath.stem}_dup{fpath.suffix}"
                    shutil.move(str(fpath), str(dest))
                    moved += 1
                except Exception as exc:
                    msg = f"移动 {fpath.name} 失败: {exc}"
                    warnings.warn(msg)
                    errors.append(msg)

        return moved, errors

    def _notify(self, message: str) -> None:
        if self.progress_callback:
            self.progress_callback(message)
        else:
            try:
                print(message, flush=True)
            except Exception:
                try:
                    print(message.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8"), flush=True)
                except Exception:
                    pass

