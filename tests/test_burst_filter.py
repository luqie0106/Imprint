"""
tests/test_burst_filter.py — NEF 连拍优选核心逻辑单元测试

注意：测试不依赖真实 NEF 文件，全部使用合成图像或 mock 对象。
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
import sys

# ── 确保 src 在路径上 ─────────────────────────────────────────────────────────
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from burst_filter import (  # noqa: E402
    BurstFilter,
    BurstGrouper,
    RawExifReader,
    RawEvaluator,
    ScoredPhoto,
)


# ══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════════════════

def _make_sharp_image(size: int = 256) -> np.ndarray:
    """生成一张高频棋盘格图像（模拟清晰图）。"""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    block = size // 16
    for r in range(size // block):
        for c in range(size // block):
            if (r + c) % 2 == 0:
                img[r * block:(r + 1) * block, c * block:(c + 1) * block] = 255
    return img


def _make_blurry_image(size: int = 256, blur_ksize: int = 51) -> np.ndarray:
    """对棋盘格图像做大核高斯模糊（模拟糊片）。"""
    sharp = _make_sharp_image(size)
    return cv2.GaussianBlur(sharp, (blur_ksize, blur_ksize), 0)


def _make_nef_placeholder(directory: Path, name: str) -> Path:
    """在目录中创建一个占位的假 NEF 文件（仅用于路径测试）。"""
    p = directory / name
    p.write_bytes(b"\x00" * 16)
    return p


# ══════════════════════════════════════════════════════════════════════════════
# RawEvaluator 测试
# ══════════════════════════════════════════════════════════════════════════════

class TestRawEvaluator:
    def setup_method(self):
        self.scorer = RawEvaluator()

    def test_sharp_image_scores_higher_than_blurry(self):
        sharp = _make_sharp_image()
        blurry = _make_blurry_image()
        score_sharp = self.scorer.sharpness(sharp)
        score_blurry = self.scorer.sharpness(blurry)
        assert score_sharp > score_blurry, (
            f"清晰图得分 {score_sharp:.2f} 应高于模糊图 {score_blurry:.2f}"
        )

    def test_score_returns_positive_float(self):
        img = _make_sharp_image()
        score = self.scorer.sharpness(img)
        assert isinstance(score, float)
        assert score >= 0.0

    def test_uniform_image_scores_near_zero(self):
        """全灰图像（无纹理）的锐度得分应接近 0。"""
        flat = np.full((128, 128, 3), 128, dtype=np.uint8)
        score = self.scorer.sharpness(flat)
        assert score < 1.0, f"均一图像的得分应接近0，实际={score}"


# ══════════════════════════════════════════════════════════════════════════════
# RawExifReader 测试
# ══════════════════════════════════════════════════════════════════════════════

class TestRawExifReader:
    def test_fallback_to_mtime_on_exif_failure(self, tmp_path: Path):
        """无法读取 EXIF 时，应回退到文件系统 mtime。"""
        fake = tmp_path / "fake.NEF"
        fake.write_bytes(b"\x00")
        reader = RawExifReader()
        dt = reader.read_datetime(fake)
        assert isinstance(dt, datetime)


# ══════════════════════════════════════════════════════════════════════════════
# BurstGrouper 测试
# ══════════════════════════════════════════════════════════════════════════════

class TestBurstGrouper:
    """使用 mock 绕过真实 EXIF/rawpy，只测试分组逻辑。"""

    def _make_grouper(self, times: list[datetime], similar: bool = True) -> tuple[BurstGrouper, list[Path]]:
        """
        构造一个 BurstGrouper：
        - exif_reader.read_datetime 按索引返回对应时间
        - _safe_dhash 被 mock：
            similar=True  → 返回汉明距离=0（完全相同）
            similar=False → 返回汉明距离=30（构图很不同）
        """
        mock_exif = MagicMock(spec=RawExifReader)
        mock_scorer = MagicMock(spec=RawEvaluator)

        paths = [Path(f"img_{i:03d}.NEF") for i in range(len(times))]
        mock_exif.read_datetime.side_effect = lambda p: times[paths.index(p)]
        mock_scorer.extract_preview.return_value = _make_sharp_image()

        grouper = BurstGrouper(
            exif_reader=mock_exif,
            preview_extractor=mock_scorer,
            gap_seconds=1.5,
            max_hamming_distance=12,
        )
        # 通过 mock _safe_dhash 控制汉明距离：
        # similar=True  → 所有帧哈希相同（距离=0）
        # similar=False → 第二帧起返回不同哈希（距离=30 > 12）
        identical_hash = np.ones((8, 8), dtype=bool)
        different_hash = np.zeros((8, 8), dtype=bool)
        call_count = [0]

        def fake_dhash(path):
            i = call_count[0]
            call_count[0] += 1
            if i == 0 or similar:
                return identical_hash
            return different_hash

        grouper._safe_dhash = fake_dhash
        return grouper, paths

    def test_single_file_is_single_group(self):
        times = [datetime(2024, 1, 1, 12, 0, 0)]
        grouper, paths = self._make_grouper(times)
        groups = grouper.group(paths)
        assert len(groups) == 1
        assert len(groups[0]) == 1

    def test_two_close_similar_files_form_burst(self):
        times = [datetime(2024, 1, 1, 12, 0, 0), datetime(2024, 1, 1, 12, 0, 1)]
        grouper, paths = self._make_grouper(times, similar=True)
        groups = grouper.group(paths)
        assert len(groups) == 1
        assert len(groups[0]) == 2

    def test_two_files_beyond_gap_are_separate(self):
        times = [datetime(2024, 1, 1, 12, 0, 0), datetime(2024, 1, 1, 12, 0, 3)]
        grouper, paths = self._make_grouper(times)
        groups = grouper.group(paths)
        assert len(groups) == 2

    def test_dissimilar_images_split_into_new_subgroup(self):
        """时间满足但与锚点不相似：截断当前组，第二张作为新子组基准帧。"""
        times = [datetime(2024, 1, 1, 12, 0, 0), datetime(2024, 1, 1, 12, 0, 1)]
        grouper, paths = self._make_grouper(times, similar=False)
        groups = grouper.group(paths)
        # 锚点比对失败 → 截断 → 两个子组，每组 1 张
        assert len(groups) == 2
        assert all(len(g) == 1 for g in groups)

    def test_three_consecutive_form_one_burst(self):
        times = [
            datetime(2024, 1, 1, 12, 0, 0),
            datetime(2024, 1, 1, 12, 0, 1),
            datetime(2024, 1, 1, 12, 0, 2),
        ]
        grouper, paths = self._make_grouper(times, similar=True)
        groups = grouper.group(paths)
        assert len(groups) == 1
        assert len(groups[0]) == 3


# ══════════════════════════════════════════════════════════════════════════════
# BurstFilter 集成测试（完全 mock，不需要真实 NEF）
# ══════════════════════════════════════════════════════════════════════════════

class TestBurstFilter:
    def _make_fake_nef_dir(self, tmp_path: Path, count: int) -> tuple[Path, list[Path]]:
        nef_dir = tmp_path / "nefs"
        nef_dir.mkdir()
        paths = [_make_nef_placeholder(nef_dir, f"DSC_{i:04d}.NEF") for i in range(count)]
        return nef_dir, paths

    def test_single_shots_are_skipped(self, tmp_path: Path):
        """所有照片都是单拍时，无文件被移动。"""
        nef_dir, paths = self._make_fake_nef_dir(tmp_path, 3)

        flt = BurstFilter()
        # mock _grouper.group 返回所有单拍
        flt._grouper.group = MagicMock(
            return_value=[[p] for p in paths]
        )

        result = flt.run(nef_dir)

        assert result.total == 3
        assert result.skipped_single == 3
        assert result.burst_groups == 0
        assert result.moved == 0
        assert result.review_dir is None
        # 原文件仍然在原位
        for p in paths:
            assert p.exists()

    def test_burst_group_keeps_best_moves_rest(self, tmp_path: Path):
        """连拍组中最高分保留，其余移动到淘汰目录。"""
        nef_dir, paths = self._make_fake_nef_dir(tmp_path, 3)

        flt = BurstFilter(review_subdir="Burst_Review")

        # mock 分组：3 张均为同一连拍组
        flt._grouper.group = MagicMock(return_value=[paths])

        # mock 打分：paths[1] 最清晰
        sharp_img = _make_sharp_image()
        blurry_img = _make_blurry_image()

        def fake_extract(p: Path) -> np.ndarray:
            if p == paths[1]:
                return sharp_img
            return blurry_img

        flt._scorer.extract_preview = fake_extract
        flt._scorer.sharpness = RawEvaluator().score  # 使用真实打分
        flt._scorer.exposure_score = MagicMock(return_value=1.0)
        # mock 美学评分：全部返回 1.0（全部通过初筛），测试纯锐度选优逻辑
        flt._aesthetic_scorer.score = MagicMock(return_value=1.0)

        result = flt.run(nef_dir)

        assert result.burst_groups == 1
        assert result.moved == 2
        assert result.review_dir is not None
        assert result.review_dir.exists()

        # paths[1]（最清晰）应留在原位
        assert paths[1].exists(), "最优片应保留在原目录"
        # paths[0] 和 paths[2] 应被移走
        assert not paths[0].exists(), "淘汰片应已移出原目录"
        assert not paths[2].exists(), "淘汰片应已移出原目录"

    def test_no_nef_files_returns_empty_result(self, tmp_path: Path):
        nef_dir = tmp_path / "empty"
        nef_dir.mkdir()
        result = BurstFilter().run(nef_dir)
        assert result.total == 0
        assert result.burst_groups == 0

    def test_review_dir_created_inside_input_dir(self, tmp_path: Path):
        nef_dir, paths = self._make_fake_nef_dir(tmp_path, 2)

        flt = BurstFilter(review_subdir="审查_连拍淘汰")
        flt._grouper.group = MagicMock(return_value=[paths])

        sharp_img = _make_sharp_image()
        blurry_img = _make_blurry_image()
        flt._scorer.extract_preview = lambda p: sharp_img if p == paths[0] else blurry_img
        flt._scorer.sharpness = RawEvaluator().score
        flt._scorer.exposure_score = MagicMock(return_value=1.0)

        result = flt.run(nef_dir)

        assert result.review_dir == nef_dir / "审查_连拍淘汰"
        assert result.review_dir.exists()

    def test_multi_format_scanning(self, tmp_path: Path):
        """测试扫描支持 RAW (NEF/ARW/CR3/RAF) 及 JPEG, HIF, JXL 等格式。"""
        img_dir = tmp_path / "mixed"
        img_dir.mkdir()
        
        # 创建多种格式的假文件
        for name in ["a.NEF", "b.jpg", "c.JPEG", "d.hif", "e.jxl", "f.cr3", "g.png"]:
            (img_dir / name).write_bytes(b"\x00" * 16)
        # 创建不支持的格式
        (img_dir / "ignore.txt").write_text("hello")
        (img_dir / "ignore.mp4").write_bytes(b"\x00" * 16)

        flt = BurstFilter()
        photos = flt._scan_photos(img_dir)
        assert len(photos) == 7
        names = {p.name for p in photos}
        assert "a.NEF" in names
        assert "b.jpg" in names
        assert "d.hif" in names
        assert "e.jxl" in names
        assert "ignore.txt" not in names
        assert "ignore.mp4" not in names
