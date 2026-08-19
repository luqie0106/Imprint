"""
tests/test_model_manager.py — 测试模型管理、状态检测与路径解析
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from model_manager import (
    check_all_models,
    get_clip_model_path,
    is_clip_model_downloaded,
    CLIP_MODEL_DIR,
    MLP_WEIGHTS_PATH,
    ONNX_MODEL_PATH,
)


class TestModelManager:
    def test_get_clip_model_path_fallback(self):
        """当本地模型目录不存在时，应回退为 HuggingFace repo id"""
        with patch("model_manager.is_clip_model_downloaded", return_value=False):
            path = get_clip_model_path()
            assert path == "openai/clip-vit-base-patch32"

    def test_get_clip_model_path_local(self, tmp_path: Path):
        """当本地模型完整时，应返回本地路径"""
        with patch("model_manager.is_clip_model_downloaded", return_value=True):
            path = get_clip_model_path()
            assert path == str(CLIP_MODEL_DIR)

    def test_check_all_models_status(self):
        """测试 check_all_models 返回数据结构正确性"""
        status = check_all_models()
        assert hasattr(status, "clip_ready")
        assert hasattr(status, "mlp_ready")
        assert hasattr(status, "onnx_ready")
        assert hasattr(status, "is_fully_ready")
        assert isinstance(status.clip_ready, bool)
        assert isinstance(status.mlp_ready, bool)
        assert isinstance(status.onnx_ready, bool)

    def test_is_clip_model_downloaded_with_mock_files(self, tmp_path: Path):
        """模拟本地模型文件存在且尺寸正常"""
        test_dir = tmp_path / "clip-vit-base-patch32"
        test_dir.mkdir()
        (test_dir / "config.json").write_text("{}")
        (test_dir / "preprocessor_config.json").write_text("{}")
        
        weight_file = test_dir / "model.safetensors"
        # 权重文件太小应判断为未就绪
        weight_file.write_bytes(b"0" * 100)
        with patch("model_manager.CLIP_MODEL_DIR", test_dir):
            assert is_clip_model_downloaded() is False

        # 模拟文件大小大于 100MB
        orig_stat = Path.stat
        def fake_stat(self):
            if self.name == "model.safetensors":
                real = orig_stat(self)
                return type("MockStat", (), {"st_size": 150 * 1024 * 1024, "st_mode": real.st_mode, "st_mtime": real.st_mtime})()
            return orig_stat(self)

        with patch("model_manager.CLIP_MODEL_DIR", test_dir), \
             patch.object(Path, "stat", fake_stat):
            assert is_clip_model_downloaded() is True
