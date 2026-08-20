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
    ".nef", ".arw", ".cr3", ".cr2", ".raf", ".dng", ".rw2", ".orf", ".pef"
}

STANDARD_IMAGE_SUFFIXES = {
    ".jpg", ".jpeg", ".jpe",
    ".jxl",
    ".hif", ".heif", ".heic",
    ".png", ".webp", ".tiff", ".tif"
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

# ── 动态加载 PyTorch ──────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ── 动态加载 CLIP（transformers）─────────────────────────────────────────────
try:
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

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
    """读取拍摄时间（精确到微秒），失败时回退 mtime。适用于所有 RAW 格式。"""

    def read_datetime(self, path: Path) -> datetime:
        try:
            return self._from_exif(path)
        except Exception:
            return self._from_mtime(path)

    def _from_exif(self, path: Path) -> datetime:
        with Image.open(path) as img:
            exif = img.getexif()
            ifd  = exif.get_ifd(_EXIF_IFD_TAG) if hasattr(exif, "get_ifd") else {}

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
            try:
                return self._extract_thumb(path)
            except Exception:
                pass
            try:
                return self._half_decode(path)
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
        else:
            img = Image.fromarray(thumb.data)
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

# MLP 分类头定义（必须与训练时结构一致，用于 PyTorch Fallback）
if TORCH_AVAILABLE:
    class _AestheticMLP(nn.Module):
        """接受 CLIP 512 维特征，输出 2 分类 logits。"""
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
        self._infer_lock = threading.Lock()

        if getattr(sys, 'frozen', False):
            bundle_root = Path(sys._MEIPASS)
            exe_root = Path(sys.executable).parent
        else:
            bundle_root = project_root
            exe_root = project_root

        if (exe_root / "photo_sort_model.onnx").exists():
            onnx_path = exe_root / "photo_sort_model.onnx"
        else:
            onnx_path = bundle_root / "photo_sort_model.onnx"

        mlp_path = exe_root / "aesthetic_mlp.pth"

        # 1. 尝试初始化 ONNX 引擎
        if ONNX_AVAILABLE and onnx_path.exists():
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
                self.available = True
                return
            except Exception as exc:
                warnings.warn(f"无法加载 ONNX 模型，尝试降级: {exc}")

        # 2. 尝试回退到 PyTorch 引擎
        if TORCH_AVAILABLE and CLIP_AVAILABLE and mlp_path.exists():
            try:
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
        """返回 0.0~1.0 的 Like 概率，不可用时返回 1.0（全部通过）。"""
        if not self.available:
            return 1.0

        if self.engine == "onnx":
            try:
                inputs = self._preprocess_onnx(img_rgb)
                ort_inputs = {self._session.get_inputs()[0].name: inputs}
                with self._infer_lock:
                    probs = self._session.run(None, ort_inputs)[0]
                return float(probs[0])
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
                return prob
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
    ) -> None:
        self._exif = exif_reader
        self._previewer = preview_extractor
        self.gap_seconds = gap_seconds
        self.max_hamming_distance = max_hamming_distance
        if max_workers is None or max_workers <= 0:
            max_workers = max(1, round((os.cpu_count() or 4) * 0.8))
        self.max_workers = max_workers

    def group(self, nef_files: Sequence[Path]) -> list[list[Path]]:
        if not nef_files:
            return []

        # 1. 多线程并发读取时间与提取 dHash
        def _parse_meta(p: Path) -> tuple[Path, datetime, np.ndarray | None]:
            return p, self._exif.read_datetime(p), self._safe_dhash(p)

        import concurrent.futures
        import os
        workers = max(1, self.max_workers)

        meta_map: dict[Path, tuple[datetime, np.ndarray | None]] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            for p, dt, h in executor.map(_parse_meta, nef_files):
                meta_map[p] = (dt, h)

        timed: list[tuple[datetime, Path]] = sorted(
            ((meta_map[p][0], p) for p in nef_files),
            key=lambda x: x[0],
        )

        # 2. 首帧锚点比对聚类
        result: list[list[Path]] = []
        current_group: list[Path] = [timed[0][1]]
        anchor_hash: np.ndarray | None = meta_map[timed[0][1]][1]
        prev_time = timed[0][0]

        for cur_time, cur_path in timed[1:]:
            gap = (cur_time - prev_time).total_seconds()
            prev_time = cur_time

            # 条件 1：时间超出阈值 → 直接截断
            if gap > self.gap_seconds:
                result.append(current_group)
                current_group = [cur_path]
                anchor_hash = meta_map[cur_path][1]
                continue

            # 条件 2：与锚点帧比对 dHash 汉明距离
            cur_hash = meta_map[cur_path][1]
            if anchor_hash is None or cur_hash is None:
                # 预览提取失败，保守截断
                result.append(current_group)
                current_group = [cur_path]
                anchor_hash = cur_hash
                continue

            dist = self._hamming(anchor_hash, cur_hash)
            if dist <= self.max_hamming_distance:
                # 同一场景，加入当前子组（锚点保持第一张不变）
                current_group.append(cur_path)
            else:
                # 构图/角度变化，截断并以当前帧开启新子组
                result.append(current_group)
                current_group = [cur_path]
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
    NEF 连拍优选主控器。
      result = BurstFilter().run(Path("/path/to/nef/folder"))
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
            print(f"🚀 已启用 {engine_str} 美学评分模型！")
        else:
            print("ℹ️ 未检测到有效的美学模型，降级为纯 OpenCV 锐度过滤。")

    def run(self, input_dir: Path) -> BurstFilterResult:
        result = BurstFilterResult()
        photo_files = self._scan_raw(input_dir)
        result.total = len(photo_files)
        if not photo_files:
            return result

        self._notify(f"扫描到 {result.total} 张照片，正在分析连拍组…")
        groups = self._grouper.group(photo_files)

        burst_groups = [g for g in groups if len(g) > 1]
        result.skipped_single = sum(1 for g in groups if len(g) == 1)
        result.burst_groups = len(burst_groups)

        if not burst_groups:
            self._notify("未检测到连拍组，所有文件保留原位。")
            return result

        review_dir = input_dir / self.review_subdir
        review_dir.mkdir(parents=True, exist_ok=True)
        result.review_dir = review_dir

        for idx, group in enumerate(burst_groups, 1):
            self._notify(f"处理连拍组 {idx}/{len(burst_groups)}（{len(group)} 张）…")
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

    def _process_group(
        self, group: list[Path], review_dir: Path
    ) -> tuple[int, list[str]]:
        """
        对连拍组内所有照片进行多维度评估，综合加权后保留前 keep_count 张。

        各维度权重：
          AI 美学概率   : 0.6
          归一化锐度    : 0.3
          曝光评分      : 0.1
        """
        # ── 阶段 1：为每张照片提取三个维度的原始分数 ──────────────────────────
        @dataclass
        class _Photo:
            path: Path
            sharpness: float = 0.0
            exposure: float  = 1.0
            aesthetic: float = 1.0
            failed: bool     = False

        photos: list[_Photo] = []
        errors: list[str] = []

        def _evaluate_photo(path: Path) -> _Photo | tuple[_Photo, str]:
            p = _Photo(path=path)
            try:
                preview = self._scorer.extract_preview(path)
                p.sharpness = self._scorer.sharpness(preview)
                p.exposure  = self._scorer.exposure_score(preview)
                p.aesthetic = self._aesthetic_scorer.score(preview)
                return p
            except Exception as exc:
                p.failed = True
                return p, f"{path.name}: {exc}"

        import concurrent.futures
        import os
        workers = max(1, self.max_workers)

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            for result in executor.map(_evaluate_photo, group):
                if isinstance(result, tuple):
                    p, err_msg = result
                    warnings.warn(f"跳过 {p.path.name}: {err_msg}")
                    errors.append(err_msg)
                    photos.append(p)
                else:
                    photos.append(result)

        if not photos:
            return 0, errors

        # ── 阶段 2：组内归一化锐度（避免绝对值量纲差异主导结果）─────────────
        valid = [p for p in photos if not p.failed]
        if valid:
            max_s = max(p.sharpness for p in valid)
            min_s = min(p.sharpness for p in valid)
            span  = max_s - min_s + 1e-6
            for p in valid:
                p._norm_sharp = (p.sharpness - min_s) / span   # type: ignore[attr-defined]
        for p in photos:
            if not hasattr(p, '_norm_sharp'):
                p._norm_sharp = 0.0   # type: ignore[attr-defined]

        # ── 阶段 3：计算综合得分并排序 ────────────────────────────────────────
        for p in valid:
            p.final_score = (                                   # type: ignore[attr-defined]
                p.aesthetic     * 0.6
                + p._norm_sharp * 0.3   # type: ignore[attr-defined]
                + p.exposure    * 0.1
            )
        for p in photos:
            if not hasattr(p, 'final_score'):
                p.final_score = -1.0    # type: ignore[attr-defined]

        keep_n = min(self.keep_count, len(valid))
        top = sorted(valid, key=lambda x: x.final_score, reverse=True)[:keep_n]  # type: ignore[attr-defined]
        keep_paths: set[Path] = {p.path for p in top}

        # ── 阶段 4：移动淘汰照片 ──────────────────────────────────────────────
        moved = 0
        for p in photos:
            if p.path in keep_paths or p.failed:
                continue
            try:
                dest = review_dir / p.path.name
                if dest.exists():
                    dest = review_dir / f"{p.path.stem}_dup{p.path.suffix}"
                shutil.move(str(p.path), str(dest))
                moved += 1
            except Exception as exc:
                msg = f"移动 {p.path.name} 失败: {exc}"
                warnings.warn(msg)
                errors.append(msg)

        return moved, errors

    def _notify(self, message: str) -> None:
        if self.progress_callback:
            self.progress_callback(message)
        else:
            print(message)
