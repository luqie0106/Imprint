"""
burst_filter.py — NEF 连拍优选与冗余片自动移动

算法说明（BurstGrouper）：
  采用"首帧锚点比对"法。每个子组以第一张照片作为基准帧。
  后续照片满足以下两个条件才归入同一子组：
    1. 与前一张的时间间隔 <= gap_seconds
    2. 与当前子组基准帧的直方图相关系数 >= similarity_threshold
  任意一个条件不满足，则截断当前子组，以该照片开启新子组。
"""

from __future__ import annotations

import shutil
import sys
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

# ── EXIF 常量 ─────────────────────────────────────────────────────────────────
_EXIF_IFD_TAG = 34665
_DATETIME_ORIGINAL_TAG = 36867
_DATETIME_FMT = "%Y:%m:%d %H:%M:%S"

# ── 默认参数 ──────────────────────────────────────────────────────────────────
DEFAULT_TIME_GAP_SECONDS: float = 1.5
DEFAULT_SIMILARITY_THRESHOLD: float = 0.85
DEFAULT_REVIEW_SUBDIR: str = "审查_连拍淘汰"

_CENTER_CROP_RATIO: float = 0.6

# ── 动态加载 PyTorch ──────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

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


# ══════════════════════════════════════════════════════════════════════════════
# EXIF 时间读取
# ══════════════════════════════════════════════════════════════════════════════

class NefExifReader:
    """读取拍摄时间，失败时回退 mtime。"""

    def read_datetime(self, path: Path) -> datetime:
        try:
            return self._from_exif(path)
        except Exception:
            return self._from_mtime(path)

    def _from_exif(self, path: Path) -> datetime:
        with Image.open(path) as img:
            exif = img.getexif()
            raw = None
            if hasattr(exif, "get_ifd"):
                raw = exif.get_ifd(_EXIF_IFD_TAG).get(_DATETIME_ORIGINAL_TAG)
            if not raw:
                raw = exif.get(_DATETIME_ORIGINAL_TAG)
            if not raw:
                raise ValueError("No DateTimeOriginal")
            return datetime.strptime(str(raw).strip(), _DATETIME_FMT)

    @staticmethod
    def _from_mtime(path: Path) -> datetime:
        return datetime.fromtimestamp(path.stat().st_mtime)


# ══════════════════════════════════════════════════════════════════════════════
# 预览提取 & 锐度评分
# ══════════════════════════════════════════════════════════════════════════════

class NefSharpnessScorer:
    """
    提取 NEF 内嵌缩略图并计算中心区域锐度得分。
    得分 = Laplacian方差 + Tenengrad梯度能量（越大越清晰）。
    """

    def extract_preview(self, path: Path) -> np.ndarray:
        try:
            return self._extract_thumb(path)
        except Exception:
            pass
        try:
            return self._half_decode(path)
        except Exception as exc:
            raise RuntimeError(f"无法读取 NEF 预览: {path.name}") from exc

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

    def score(self, img: np.ndarray) -> float:
        h, w = img.shape[:2]
        mh = int(h * (1 - _CENTER_CROP_RATIO) / 2)
        mw = int(w * (1 - _CENTER_CROP_RATIO) / 2)
        crop = img[mh:h - mh, mw:w - mw]
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float64)
        lap = cv2.Laplacian(gray, cv2.CV_64F).var()
        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        return lap + float((sx ** 2 + sy ** 2).mean())


# ══════════════════════════════════════════════════════════════════════════════
# AI 美学评分器
# ══════════════════════════════════════════════════════════════════════════════

class AestheticScorer:
    """加载 MobileNetV3 偏好模型，输出 0.0~1.0 的 Like 概率。"""
    
    def __init__(self, model_path: Path):
        self.available = False
        if not TORCH_AVAILABLE or not model_path.exists():
            return
            
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if sys.platform == "darwin" and torch.backends.mps.is_available():
                self.device = torch.device("mps")
                
            weights = models.MobileNet_V3_Small_Weights.DEFAULT
            self.model = models.mobilenet_v3_small(weights=weights)
            num_features = self.model.classifier[3].in_features
            self.model.classifier[3] = nn.Linear(num_features, 2)
            
            self.model.load_state_dict(torch.load(str(model_path), map_location=self.device, weights_only=True))
            self.model.to(self.device)
            self.model.eval()
            
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            self.available = True
        except Exception as e:
            warnings.warn(f"无法加载 AI 美学模型: {e}")
            self.available = False

    def score(self, img_rgb: np.ndarray) -> float:
        if not self.available:
            return 1.0 # 如果没有模型，默认全部通过构图测试
            
        try:
            tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.model(tensor)
                probs = torch.softmax(outputs, dim=1)
                return probs[0][1].item() # 返回类别 1 (like) 的概率
        except Exception:
            return 1.0


# ══════════════════════════════════════════════════════════════════════════════
# 连拍分组（首帧锚点比令人）
# ══════════════════════════════════════════════════════════════════════════════

class BurstGrouper:
    """
    一次循环完成时间+视觉双重聚类：

    对每张照片（按拍摄时间排序后）：
      - 若与前一张时间间隔 > gap_seconds  → 截断，开新组
      - 否则与当前子组的基准帧（第一张）比对直方图：
          相关系数 >= threshold → 加入当前子组
          相关系数 <  threshold → 截断（构图变化），开新组

    返回 list[list[Path]]，单元素 = 单拍，多元素 = 连拍子组。
    """

    def __init__(
        self,
        exif_reader: NefExifReader,
        preview_extractor: NefSharpnessScorer,
        gap_seconds: float = DEFAULT_TIME_GAP_SECONDS,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ) -> None:
        self._exif = exif_reader
        self._previewer = preview_extractor
        self.gap_seconds = gap_seconds
        self.similarity_threshold = similarity_threshold

    def group(self, nef_files: Sequence[Path]) -> list[list[Path]]:
        if not nef_files:
            return []

        # 1. 读取时间并排序
        timed: list[tuple[datetime, Path]] = sorted(
            ((self._exif.read_datetime(p), p) for p in nef_files),
            key=lambda x: x[0],
        )

        # 2. 首帧锚点比对聚类
        result: list[list[Path]] = []
        current_group: list[Path] = [timed[0][1]]
        anchor_preview: np.ndarray | None = self._safe_preview(timed[0][1])
        prev_time = timed[0][0]

        for cur_time, cur_path in timed[1:]:
            gap = (cur_time - prev_time).total_seconds()
            prev_time = cur_time

            # 条件 1：时间超出阈值 → 直接截断
            if gap > self.gap_seconds:
                result.append(current_group)
                current_group = [cur_path]
                anchor_preview = self._safe_preview(cur_path)
                continue

            # 条件 2：与锚点比对视觉相似度
            cur_preview = self._safe_preview(cur_path)
            if anchor_preview is None or cur_preview is None:
                # 预览提取失败，保守截断
                result.append(current_group)
                current_group = [cur_path]
                anchor_preview = cur_preview
                continue

            corr = self._histogram_correl(anchor_preview, cur_preview)
            if corr >= self.similarity_threshold:
                # 同一场景，加入当前子组（锚点保持第一张不变）
                current_group.append(cur_path)
            else:
                # 构图/角度变化，截断并以当前帧开启新子组
                result.append(current_group)
                current_group = [cur_path]
                anchor_preview = cur_preview

        result.append(current_group)
        return result

    def _safe_preview(self, path: Path) -> np.ndarray | None:
        try:
            return self._previewer.extract_preview(path)
        except Exception:
            return None

    @staticmethod
    def _histogram_correl(a: np.ndarray, b: np.ndarray) -> float:
        ga = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
        gb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
        ha = cv2.calcHist([ga], [0], None, [64], [0, 256])
        hb = cv2.calcHist([gb], [0], None, [64], [0, 256])
        cv2.normalize(ha, ha)
        cv2.normalize(hb, hb)
        return float(cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL))


# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════

class BurstFilter:
    """
    NEF 连拍优选主控器。
      result = BurstFilter().run(Path("/path/to/nef/folder"))
    """

    _RAW_SUFFIXES = {
        ".nef", ".NEF",   # Nikon
        ".arw", ".ARW",   # Sony
        ".cr3", ".CR3",   # Canon
        ".raf", ".RAF",   # Fuji
    }

    def __init__(
        self,
        gap_seconds: float = DEFAULT_TIME_GAP_SECONDS,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        review_subdir: str = DEFAULT_REVIEW_SUBDIR,
        keep_count: int = 1,
        progress_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.gap_seconds = gap_seconds
        self.similarity_threshold = similarity_threshold
        self.review_subdir = review_subdir
        self.keep_count = max(1, keep_count)   # 至少保留 1 张
        self.progress_callback = progress_callback

        self._exif_reader = NefExifReader()
        self._scorer = NefSharpnessScorer()
        self._grouper = BurstGrouper(
            exif_reader=self._exif_reader,
            preview_extractor=self._scorer,
            gap_seconds=gap_seconds,
            similarity_threshold=similarity_threshold,
        )
        
        # 尝试加载 AI 模型 (假设在项目根目录)
        project_root = Path(__file__).resolve().parent.parent
        self._aesthetic_scorer = AestheticScorer(project_root / "aesthetic_model.pth")
        if self._aesthetic_scorer.available:
            print("🚀 已启用 AI 美学评分模型 (aesthetic_model.pth)！")
        else:
            print("ℹ️ 未检测到有效 AI 模型，降级为纯 OpenCV 锐度过滤。")

    def run(self, input_dir: Path) -> BurstFilterResult:
        result = BurstFilterResult()
        nef_files = self._scan_raw(input_dir)
        result.total = len(nef_files)
        if not nef_files:
            return result

        self._notify(f"扫描到 {result.total} 张 RAW 文件，正在分析连拍组…")
        groups = self._grouper.group(nef_files)

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
            if p.is_file() and p.suffix in self._RAW_SUFFIXES
        )

    def _process_group(
        self, group: list[Path], review_dir: Path
    ) -> tuple[int, list[str]]:
        scored: list[ScoredPhoto] = []
        errors: list[str] = []

        for path in group:
            try:
                preview = self._scorer.extract_preview(path)
                s_sharpness = self._scorer.score(preview)
                s_aesthetic = self._aesthetic_scorer.score(preview)
            except Exception as exc:
                warnings.warn(f"跳过 {path.name}: {exc}")
                errors.append(f"{path.name}: {exc}")
                scored.append(ScoredPhoto(path=path, score=-1.0))
                continue
            
            # 临时把 score 存为锐度，把 AI 偏好概率附在对象上
            photo_obj = ScoredPhoto(path=path, score=s_sharpness)
            setattr(photo_obj, 'aesthetic_prob', s_aesthetic)
            scored.append(photo_obj)

        if not scored:
            return 0, errors

        moved = 0

        # 1. AI 初筛：只保留构图得分 >= 50% 的候选者
        candidates = [p for p in scored if getattr(p, 'aesthetic_prob', 1.0) >= 0.5 and p.score >= 0]

        if not candidates:
            # 全军覆没：组内所有照片构图均不合格，全部移动
            keep_paths: set[Path] = set()
        else:
            # 2. OpenCV 终选：在构图合格的照片中按锐度降序，保留前 keep_count 张
            candidates_sorted = sorted(candidates, key=lambda x: x.score, reverse=True)
            keep_n = min(self.keep_count, len(candidates_sorted))
            keep_paths = {p.path for p in candidates_sorted[:keep_n]}

        for photo in scored:
            if photo.path in keep_paths:
                continue  # 保留
            if photo.score < 0:
                continue  # 读取失败的保守保留
            try:
                dest = review_dir / photo.path.name
                if dest.exists():
                    dest = review_dir / f"{photo.path.stem}_dup{photo.path.suffix}"
                shutil.move(str(photo.path), str(dest))
                moved += 1
            except Exception as exc:
                msg = f"移动 {photo.path.name} 失败: {exc}"
                warnings.warn(msg)
                errors.append(msg)

        return moved, errors

    def _notify(self, message: str) -> None:
        if self.progress_callback:
            self.progress_callback(message)
        else:
            print(message)
