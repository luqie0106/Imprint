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
    get_active_model_mode,
    set_active_model_mode,
    get_resolved_standard_onnx_path,
    get_resolved_custom_onnx_path,
    get_active_aesthetic_model_path,
    CLIP_MODEL_DIR,
    MLP_WEIGHTS_PATH,
    STANDARD_ONNX_PATH,
    CUSTOM_ONNX_PATH,
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

    def test_model_mode_persistence(self, tmp_path: Path):
        """测试模型选择配置读写持久化"""
        fake_cfg = tmp_path / "config.json"
        with patch("model_manager.CONFIG_FILE_PATH", fake_cfg):
            assert get_active_model_mode() == "standard_b32"
            set_active_model_mode("standard_l14")
            assert get_active_model_mode() == "standard_l14"
            set_active_model_mode("custom")
            assert get_active_model_mode() == "custom"
            set_active_model_mode("invalid_mode")
            assert get_active_model_mode() == "standard_b32"

    def test_check_all_models_status(self):
        """测试 check_all_models 返回数据结构正确性"""
        status = check_all_models()
        assert hasattr(status, "clip_ready")
        assert hasattr(status, "clip_l14_ready")
        assert hasattr(status, "standard_onnx_ready")
        assert hasattr(status, "standard_l14_onnx_ready")
        assert hasattr(status, "custom_onnx_ready")
        assert hasattr(status, "custom_l14_onnx_ready")
        assert hasattr(status, "mlp_ready")
        assert hasattr(status, "mlp_l14_ready")
        assert hasattr(status, "active_mode")
        assert hasattr(status, "is_fully_ready")
        assert isinstance(status.standard_onnx_ready, bool)
        assert isinstance(status.standard_l14_onnx_ready, bool)

    def test_active_aesthetic_model_routing(self, tmp_path: Path):
        """测试模型激活路由"""
        fake_std = tmp_path / "std.onnx"
        fake_std_l14 = tmp_path / "std_l14.onnx"
        fake_custom = tmp_path / "custom.onnx"
        fake_std.write_bytes(b"0" * (120 * 1024 * 1024))
        fake_std_l14.write_bytes(b"0" * (120 * 1024 * 1024))
        fake_custom.write_bytes(b"0" * (120 * 1024 * 1024))

        with patch("model_manager.get_resolved_standard_onnx_path", return_value=fake_std), \
             patch("model_manager.get_resolved_standard_l14_onnx_path", return_value=fake_std_l14), \
             patch("model_manager.get_resolved_custom_onnx_path", return_value=fake_custom):
            with patch("model_manager.get_active_model_mode", return_value="standard_b32"):
                assert get_active_aesthetic_model_path() == fake_std
            with patch("model_manager.get_active_model_mode", return_value="standard_l14"):
                assert get_active_aesthetic_model_path() == fake_std_l14
            with patch("model_manager.get_active_model_mode", return_value="custom"):
                assert get_active_aesthetic_model_path() == fake_custom


    def test_is_clip_model_downloaded_with_mock_files(self, tmp_path: Path):
        """模拟本地模型文件存在且尺寸正常"""
        test_dir = tmp_path / "clip-vit-base-patch32"
        test_dir.mkdir()
        (test_dir / "config.json").write_text("{}")
        (test_dir / "preprocessor_config.json").write_text("{}")
        
        weight_file = test_dir / "model.safetensors"
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

