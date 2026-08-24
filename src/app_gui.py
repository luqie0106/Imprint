"""
app_gui.py — Photo Sort 综合一体化主程序界面 (PySide6 / Qt6 现代全圆角矢量设计)
整合连拍筛选、偏好训练与模型管理
"""

from __future__ import annotations

import os
import sys
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Callable

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFrame, QVBoxLayout,
    QHBoxLayout, QLabel, QPushButton, QCheckBox, QRadioButton, QProgressBar,
    QStackedWidget, QButtonGroup, QMessageBox, QScrollArea, QFileDialog
)

# ── 确保 src 在 sys.path 上 ───────────────────────────────────────────────────
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from burst_gui import BurstFilterGUI
from trainer_gui import TrainerGUI
from model_manager import (
    PROJECT_ROOT,
    BUNDLE_ROOT,
    MODELS_DIR,
    CLIP_MODEL_DIR,
    CLIP_L14_MODEL_DIR,
    MLP_WEIGHTS_PATH,
    MLP_L14_WEIGHTS_PATH,
    STANDARD_ONNX_PATH,
    STANDARD_L14_ONNX_PATH,
    CUSTOM_ONNX_PATH,
    CUSTOM_L14_ONNX_PATH,
    check_all_models,
    get_active_model_mode,
    set_active_model_mode,
    download_clip_model,
    download_clip_l14_model,
    fuse_standard_l14_onnx,
    is_clip_model_downloaded,
    is_clip_l14_model_downloaded,
    is_clip_in_hf_cache,
    is_clip_l14_in_hf_cache,
    import_from_hf_cache,
    import_from_hf_cache_l14,
)
from onnx_exporter import export_to_onnx, TORCH_EXPORT_AVAILABLE
from qt_theme import APP_STYLE, GREEN_FG, AMBER_FG, TEXT_TERT, TEXT_SEC, FAM_TITLE


class DownloadWorker(QThread):
    progress_sig = Signal(str, float)
    done_sig = Signal(bool, str)

    def __init__(self, use_mirror: bool):
        super().__init__()
        self.use_mirror = use_mirror
        self.cancel_event = threading.Event()

    def run(self):
        try:
            success = download_clip_model(
                use_mirror=self.use_mirror,
                progress_callback=lambda msg, pct: self.progress_sig.emit(msg, pct),
                cancel_event=self.cancel_event,
            )
            if success:
                self.done_sig.emit(True, "CLIP ViT-B/32 基础视觉模型已成功就绪！")
            else:
                self.done_sig.emit(False, "模型下载未完成。")
        except Exception as exc:
            self.done_sig.emit(False, f"下载出错: {exc}")


class DownloadL14Worker(QThread):
    progress_sig = Signal(str, float)
    done_sig = Signal(bool, str)

    def __init__(self, use_mirror: bool):
        super().__init__()
        self.use_mirror = use_mirror
        self.cancel_event = threading.Event()

    def run(self):
        try:
            self.progress_sig.emit("正在准备下载 CLIP ViT-L/14 (Aesthetic 3) 大模型...", 0.05)
            success = download_clip_l14_model(
                use_mirror=self.use_mirror,
                progress_callback=lambda msg, pct: self.progress_sig.emit(msg, pct * 0.7),
                cancel_event=self.cancel_event,
            )
            if not success:
                self.done_sig.emit(False, "ViT-L/14 大模型下载未完成。")
                return

            self.progress_sig.emit("正在下载官方 ViT-L/14 美学权重并熔铸 ONNX...", 0.8)
            fuse_standard_l14_onnx(progress_callback=lambda msg: self.progress_sig.emit(msg, 0.9))
            self.progress_sig.emit("✅ Aesthetic 3 专业大模型已成功熔铸就绪！", 1.0)
            self.done_sig.emit(True, "Aesthetic 3 (ViT-L/14) 专业大模型已成功就绪并熔铸为 standard_aesthetic_l14_model.onnx！")
        except Exception as exc:
            self.done_sig.emit(False, f"下载或熔铸出错: {exc}")


class ExportWorker(QThread):
    done_sig = Signal(bool, str)

    def __init__(self, mlp_path: Path | None = None):
        super().__init__()
        self.mlp_path = mlp_path

    def run(self):
        try:
            if TORCH_EXPORT_AVAILABLE:
                out = export_to_onnx(project_root=PROJECT_ROOT, mlp_path=self.mlp_path)
                self.done_sig.emit(True, f"✅ 个人专属 ONNX 模型已成功熔铸生成至：\n{out.name}\n\n连拍优选已自动切换并启用该模型！")
                return

            from trainer_gui import discover_python_environments, probe_python_environment
            envs = discover_python_environments()
            valid_py = None
            for py in envs:
                probe = probe_python_environment(py)
                if not probe.get("missing") and not probe.get("error"):
                    valid_py = py
                    break

            if not valid_py:
                self.done_sig.emit(
                    False,
                    "当前应用环境中未检测到 PyTorch，且未找到已安装 torch / transformers 的外部 Python 环境。\n\n"
                    "请先安装 Python 依赖（pip install torch transformers onnx onnxscript），或在 Conda 环境中运行。",
                )
                return

            script = f"""
import sys
from pathlib import Path
root = Path(r"{PROJECT_ROOT}")
sys.path.insert(0, str(root / "src"))
from onnx_exporter import export_to_onnx
mlp_p = Path(r"{self.mlp_path}") if r"{self.mlp_path}" != "None" else None
export_to_onnx(project_root=root, mlp_path=mlp_p)
"""
            res = subprocess.run([valid_py, "-c", script], capture_output=True, text=True, timeout=180)
            if res.returncode == 0:
                self.done_sig.emit(True, "✅ 个人专属 ONNX 模型已通过外部 Python 环境成功生成！\n\n连拍优选已自动切换并启用该模型！")
            else:
                err = res.stderr.strip() or res.stdout.strip()
                self.done_sig.emit(False, f"外部环境导出失败:\n{err}")
        except Exception as exc:
            self.done_sig.emit(False, f"ONNX 熔铸失败: {exc}")


class MainAppGUI(QMainWindow):
    """Photo Sort 综合应用主窗口 (Qt6 / PySide6 现代全圆角设计系统)"""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Photo Sort — 智能连拍优选与个人审美系统")
        self.resize(860, 660)
        self.setMinimumSize(780, 560)

        icon_path = _SRC_DIR / "assets" / "icon.png"
        if icon_path.exists():
            self.setWindowIcon(QtGui.QIcon(str(icon_path)))

        self._download_worker: DownloadWorker | None = None
        self._download_l14_worker: DownloadL14Worker | None = None
        self._export_worker: ExportWorker | None = None

        self._build_ui()
        self._switch_tab(0)

        QtCore.QTimer.singleShot(300, self._check_startup_models)


    def _build_ui(self) -> None:
        central = QWidget()
        central.setObjectName("centralWidget")
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(20, 16, 20, 16)
        main_layout.setSpacing(12)

        # ── 1. 顶部 Header ──
        hdr_layout = QHBoxLayout()
        title_lbl = QLabel("📸  Photo Sort")
        title_lbl.setStyleSheet(f"font-family: {FAM_TITLE}; font-size: 20px; font-weight: bold;")
        hdr_layout.addWidget(title_lbl)

        sub_lbl = QLabel("RAW 连拍优选 · 个人审美微调 · ONNX 极速推理")
        sub_lbl.setProperty("class", "SecondaryLabel")
        sub_lbl.setStyleSheet("margin-top: 4px;")
        hdr_layout.addWidget(sub_lbl)
        hdr_layout.addStretch()
        main_layout.addLayout(hdr_layout)

        # ── 2. iOS 风格分段控制器 (Segmented Control) ──
        seg_track = QFrame()
        seg_track.setObjectName("SegmentTrack")
        seg_layout = QHBoxLayout(seg_track)
        seg_layout.setContentsMargins(3, 3, 3, 3)
        seg_layout.setSpacing(2)

        self.tab_btn_group = QButtonGroup(self)
        self.tab_btn_group.setExclusive(True)

        tabs_info = [("📷  连拍优选", 0), ("🧠  偏好训练", 1), ("📦  模型与环境", 2)]
        for text, idx in tabs_info:
            btn = QPushButton(text)
            btn.setProperty("class", "SegmentBtn")
            btn.setCheckable(True)
            if idx == 0:
                btn.setChecked(True)
            btn.clicked.connect(lambda _, i=idx: self._switch_tab(i))
            self.tab_btn_group.addButton(btn, idx)
            seg_layout.addWidget(btn)

        main_layout.addWidget(seg_track)

        # ── 3. 堆叠内容区域 ──
        self.stack = QStackedWidget()

        # Tab 0: 连拍优选
        self.burst_gui = BurstFilterGUI(on_navigate_tab=self._switch_tab)
        self.stack.addWidget(self.burst_gui)

        # Tab 1: 偏好训练
        self.trainer_gui = TrainerGUI(on_model_updated=self._on_model_updated)
        self.stack.addWidget(self.trainer_gui)

        # Tab 2: 模型管理
        self.model_mgr_widget = self._build_model_manager_tab()
        self.stack.addWidget(self.model_mgr_widget)

        main_layout.addWidget(self.stack, stretch=1)

    def _build_model_manager_tab(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.viewport().setStyleSheet("background: transparent;")
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")

        content = QWidget()
        content.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 4, 0, 16)
        layout.setSpacing(12)

        # ── 卡片 1：美学评分模型选择与管理 ──
        c1 = QFrame()
        c1.setObjectName("cardFrame1")
        c1.setStyleSheet("#cardFrame1 { background-color: #FFFFFF; border: 1px solid #E5E5EA; border-radius: 14px; }")
        l1 = QVBoxLayout(c1)
        l1.setContentsMargins(18, 16, 18, 16)
        l1.setSpacing(10)

        t1 = QLabel("🎯  AI 美学评分模型管理与选择")
        t1.setProperty("class", "CardTitle")
        l1.addWidget(t1)

        desc1 = QLabel("连拍优选将采用当前勾选的模型进行照片画质与美学评估。出厂默认内置官方标准通用模型，开箱即用：")
        desc1.setProperty("class", "SecondaryLabel")
        desc1.setWordWrap(True)
        l1.addWidget(desc1)

        self.model_mode_group = QButtonGroup(self)

        # ── 选项 1：官方标准通用模型 (ViT-B/32) ──
        self.radio_std = QRadioButton("🌟  官方标准通用模型 (ViT-B/32 · 极速平衡)")
        self.radio_std.setStyleSheet("font-weight: bold; font-size: 13px;")
        self.model_mode_group.addButton(self.radio_std, 0)
        l1.addWidget(self.radio_std)

        std_desc = QLabel("• 官方推荐 · 全品类平衡：基于 LAION-Aesthetics / AVA 25万+ 张专业摄影数据集预训练，轻量极速，对人像、风光、街拍、生态等全题材中立公允打分。")
        std_desc.setStyleSheet("color: #636366; font-size: 12px; margin-left: 26px;")
        std_desc.setWordWrap(True)
        l1.addWidget(std_desc)

        self.std_status_label = QLabel("")
        self.std_status_label.setStyleSheet("font-size: 12px; font-weight: bold; margin-left: 26px; margin-bottom: 6px;")
        self.std_status_label.setWordWrap(True)
        l1.addWidget(self.std_status_label)

        # 分割线 1
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.HLine)
        sep1.setStyleSheet("background-color: #E5E5EA; max-height: 1px; margin: 4px 0px;")
        l1.addWidget(sep1)

        # ── 选项 2：Aesthetic 3 官方专业大模型 (ViT-L/14) ──
        self.radio_l14 = QRadioButton("💎  Aesthetic 3 官方专业大模型 (ViT-L/14 · 高精画质)")
        self.radio_l14.setStyleSheet("font-weight: bold; font-size: 13px;")
        self.model_mode_group.addButton(self.radio_l14, 1)
        l1.addWidget(self.radio_l14)

        l14_desc = QLabel("• 官方权威 · 细腻审美：基于 LAION ViT-L/14 768维大模型底座，构图、色彩、焦点与光影审美判断更细腻深入。")
        l14_desc.setStyleSheet("color: #636366; font-size: 12px; margin-left: 26px;")
        l14_desc.setWordWrap(True)
        l1.addWidget(l14_desc)

        self.l14_status_label = QLabel("")
        self.l14_status_label.setStyleSheet("font-size: 12px; font-weight: bold; margin-left: 26px; margin-bottom: 6px;")
        self.l14_status_label.setWordWrap(True)
        l1.addWidget(self.l14_status_label)

        # 分割线 2
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.HLine)
        sep2.setStyleSheet("background-color: #E5E5EA; max-height: 1px; margin: 4px 0px;")
        l1.addWidget(sep2)

        # ── 选项 3：个人专属训练模型 ──
        self.radio_custom = QRadioButton("🧠  个人专属训练模型 (Custom Trained Model)")
        self.radio_custom.setStyleSheet("font-weight: bold; font-size: 13px;")
        self.model_mode_group.addButton(self.radio_custom, 2)
        l1.addWidget(self.radio_custom)

        custom_desc = QLabel("• 个性化专属 · 审美微调：使用您在【偏好训练】中用自己的照片微调熔铸的模型，完全契合您的个人构图与色彩偏好。")
        custom_desc.setStyleSheet("color: #636366; font-size: 12px; margin-left: 26px;")
        custom_desc.setWordWrap(True)
        l1.addWidget(custom_desc)

        self.custom_status_label = QLabel("")
        self.custom_status_label.setStyleSheet("font-size: 12px; font-weight: bold; margin-left: 26px;")
        self.custom_status_label.setWordWrap(True)
        l1.addWidget(self.custom_status_label)

        r_custom_btns = QHBoxLayout()
        r_custom_btns.setContentsMargins(26, 4, 0, 0)
        r_custom_btns.setSpacing(10)

        self.import_pth_btn = QPushButton("📥  导入 .pth 权重文件")
        self.import_pth_btn.setObjectName("secondaryBtn")
        self.import_pth_btn.setProperty("class", "SecondaryBtn")
        self.import_pth_btn.setToolTip("从本地导入您或他人训练好的 PyTorch 权重文件 (.pth)")
        self.import_pth_btn.clicked.connect(self._on_import_pth)
        r_custom_btns.addWidget(self.import_pth_btn)

        self.export_onnx_btn = QPushButton("⚡  一键熔铸为 ONNX 模型")
        self.export_onnx_btn.setObjectName("primaryBtn")
        self.export_onnx_btn.setProperty("class", "PrimaryBtn")
        self.export_onnx_btn.setStyleSheet("QPushButton { background-color: #0071E3; color: #FFFFFF; border: 1px solid #0071E3; border-radius: 15px; min-height: 28px; padding: 0 18px; font-weight: bold; } QPushButton:hover { background-color: #0077ED; } QPushButton:disabled { background-color: #D1D1D6; border-color: #D1D1D6; color: #FFFFFF; }")
        self.export_onnx_btn.setToolTip("将当前导入或训练的 .pth 权重与 CLIP 视觉主干熔铸为单文件 ONNX 硬件加速模型")
        self.export_onnx_btn.clicked.connect(self._on_manual_export_onnx)
        r_custom_btns.addWidget(self.export_onnx_btn)

        to_trainer_btn = QPushButton("🎯  前往偏好训练")
        to_trainer_btn.setObjectName("secondaryBtn")
        to_trainer_btn.setProperty("class", "SecondaryBtn")
        to_trainer_btn.clicked.connect(lambda: self._switch_tab(1))
        r_custom_btns.addWidget(to_trainer_btn)

        r_custom_btns.addStretch()
        l1.addLayout(r_custom_btns)

        self.radio_std.toggled.connect(self._on_model_mode_toggled)
        self.radio_l14.toggled.connect(self._on_model_mode_toggled)
        self.radio_custom.toggled.connect(self._on_model_mode_toggled)

        layout.addWidget(c1)

        # ── 卡片 2：Aesthetic 3 / CLIP ViT-L/14 专业大模型底座 ──
        c2 = QFrame()
        c2.setObjectName("cardFrame2")
        c2.setStyleSheet("#cardFrame2 { background-color: #FFFFFF; border: 1px solid #E5E5EA; border-radius: 14px; }")
        l2 = QVBoxLayout(c2)
        l2.setContentsMargins(18, 16, 18, 16)
        l2.setSpacing(10)

        t2 = QLabel("💎  Aesthetic 3 专业大模型底座 (CLIP ViT-L/14)")
        t2.setProperty("class", "CardTitle")
        l2.addWidget(t2)

        desc2 = QLabel("提供 768 维高精度多模态特征提取。下载后可直接生成并启用 Aesthetic 3 官方标准 ONNX 模型，并支持微调专属模型。")
        desc2.setProperty("class", "SecondaryLabel")
        desc2.setWordWrap(True)
        l2.addWidget(desc2)

        self.clip_l14_status_label = QLabel("")
        self.clip_l14_status_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        self.clip_l14_status_label.setWordWrap(True)
        l2.addWidget(self.clip_l14_status_label)

        r_l14 = QHBoxLayout()
        r_l14.setSpacing(12)
        self.dl_l14_btn = QPushButton("⬇️  一键下载并熔铸 Aesthetic 3 大模型")
        self.dl_l14_btn.setObjectName("secondaryBtn")
        self.dl_l14_btn.setProperty("class", "SecondaryBtn")
        self.dl_l14_btn.clicked.connect(self._start_download_clip_l14)
        r_l14.addWidget(self.dl_l14_btn)

        self.use_l14_mirror_cb = QCheckBox("使用国内加速镜像 (hf-mirror.com)")
        self.use_l14_mirror_cb.setChecked(True)
        r_l14.addWidget(self.use_l14_mirror_cb)
        r_l14.addStretch()
        l2.addLayout(r_l14)

        self.dl_l14_progress = QProgressBar()
        self.dl_l14_progress.setRange(0, 100)
        self.dl_l14_progress.setValue(0)
        self.dl_l14_progress.setVisible(False)
        l2.addWidget(self.dl_l14_progress)

        self.dl_l14_status_lbl = QLabel("")
        self.dl_l14_status_lbl.setProperty("class", "SecondaryLabel")
        self.dl_l14_status_lbl.setVisible(False)
        l2.addWidget(self.dl_l14_status_lbl)

        layout.addWidget(c2)

        # ── 卡片 3：CLIP ViT-B/32 基础视觉底座 ──
        c3 = QFrame()
        c3.setObjectName("cardFrame3")
        c3.setStyleSheet("#cardFrame3 { background-color: #FFFFFF; border: 1px solid #E5E5EA; border-radius: 14px; }")

        l3 = QVBoxLayout(c3)
        l3.setContentsMargins(18, 16, 18, 16)
        l3.setSpacing(10)

        t3 = QLabel("📦  基础视觉主干底座 (CLIP ViT-B/32)")
        t3.setProperty("class", "CardTitle")
        l3.addWidget(t3)

        desc3 = QLabel("用于提取 512 维轻量视觉特征。训练个人标准模型时需要此底座支持；标准通用 ONNX 模型已内置此特征提取能力。")
        desc3.setProperty("class", "SecondaryLabel")
        desc3.setWordWrap(True)
        l3.addWidget(desc3)

        self.clip_status_label = QLabel("")
        self.clip_status_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        self.clip_status_label.setWordWrap(True)
        l3.addWidget(self.clip_status_label)

        r3 = QHBoxLayout()
        r3.setSpacing(12)
        self.dl_clip_btn = QPushButton("⬇️  下载/补全到本地")
        self.dl_clip_btn.setObjectName("secondaryBtn")
        self.dl_clip_btn.setProperty("class", "SecondaryBtn")
        self.dl_clip_btn.clicked.connect(self._start_download_clip)

        r3.addWidget(self.dl_clip_btn)

        self.use_mirror_cb = QCheckBox("使用国内加速镜像 (hf-mirror.com)")
        self.use_mirror_cb.setChecked(True)
        r3.addWidget(self.use_mirror_cb)
        r3.addStretch()
        l3.addLayout(r3)

        self.dl_progress = QProgressBar()
        self.dl_progress.setRange(0, 100)
        self.dl_progress.setValue(0)
        self.dl_progress.setVisible(False)
        l3.addWidget(self.dl_progress)

        self.dl_status_lbl = QLabel("")
        self.dl_status_lbl.setProperty("class", "SecondaryLabel")
        self.dl_status_lbl.setVisible(False)
        l3.addWidget(self.dl_status_lbl)

        layout.addWidget(c3)
        layout.addStretch()

        scroll.setWidget(content)
        self.refresh_model_mgr_ui()
        return scroll


    def _on_model_mode_toggled(self) -> None:
        if self.radio_custom.isChecked():
            set_active_model_mode("custom")
        elif self.radio_l14.isChecked():
            set_active_model_mode("standard_l14")
        else:
            set_active_model_mode("standard_b32")
        self._on_model_updated()

    def _switch_tab(self, idx: int) -> None:
        self.stack.setCurrentIndex(idx)
        btn = self.tab_btn_group.button(idx)
        if btn:
            btn.setChecked(True)

        if idx == 0:
            self.burst_gui.refresh_model_status()
        elif idx == 2:
            self.refresh_model_mgr_ui()

    def _on_model_updated(self) -> None:
        self.burst_gui.refresh_model_status()
        self.refresh_model_mgr_ui()

    def refresh_model_mgr_ui(self) -> None:
        status = check_all_models()

        # 0. 单选状态同步
        active_mode = get_active_model_mode()
        if active_mode in ("custom", "custom_b32", "custom_l14"):
            self.radio_custom.blockSignals(True)
            self.radio_custom.setChecked(True)
            self.radio_custom.blockSignals(False)
        elif active_mode in ("standard_l14",):
            self.radio_l14.blockSignals(True)
            self.radio_l14.setChecked(True)
            self.radio_l14.blockSignals(False)
        else:
            self.radio_std.blockSignals(True)
            self.radio_std.setChecked(True)
            self.radio_std.blockSignals(False)

        # 1. 官方标准通用模型 (ViT-B/32)
        if status.standard_onnx_ready:
            p = Path(status.standard_onnx_path)
            self.std_status_label.setText(f"🟢 状态：已就绪 (文件: {p.name}, 大小: {status.standard_onnx_size_mb:.1f} MB · ONNX 极速加速)")
            self.std_status_label.setStyleSheet(f"color: {GREEN_FG}; font-weight: bold; margin-left: 26px;")
        else:
            self.std_status_label.setText("⚪ 状态：未找到标准通用模型文件")
            self.std_status_label.setStyleSheet(f"color: {TEXT_TERT}; font-weight: bold; margin-left: 26px;")

        # 2. Aesthetic 3 官方专业大模型 (ViT-L/14)
        if status.standard_l14_onnx_ready:
            p = Path(status.standard_l14_onnx_path)
            self.l14_status_label.setText(f"🟢 状态：已就绪 (文件: {p.name}, 大小: {status.standard_l14_onnx_size_mb:.1f} MB · ONNX 极速加速)")
            self.l14_status_label.setStyleSheet(f"color: {GREEN_FG}; font-weight: bold; margin-left: 26px;")
        else:
            self.l14_status_label.setText("⚪ 状态：未就绪 (可点击下方卡片“一键下载并熔铸 Aesthetic 3 大模型”)")
            self.l14_status_label.setStyleSheet(f"color: {TEXT_TERT}; font-weight: bold; margin-left: 26px;")

        # 3. 个人专属模型状态
        if status.custom_onnx_ready or status.custom_l14_onnx_ready:
            ready_models = []
            if status.custom_onnx_ready:
                ready_models.append(f"ViT-B/32 ({status.custom_onnx_size_mb:.1f} MB)")
            if status.custom_l14_onnx_ready:
                ready_models.append(f"ViT-L/14 ({status.custom_l14_onnx_size_mb:.1f} MB)")
            self.custom_status_label.setText(f"🟢 状态：已就绪 (已熔铸: {', '.join(ready_models)} · ONNX 极速加速)")
            self.custom_status_label.setStyleSheet(f"color: {GREEN_FG}; font-weight: bold; margin-left: 26px;")
            self.export_onnx_btn.setText("⚡  重新熔铸 ONNX")
            self.export_onnx_btn.setEnabled(True)
        elif status.mlp_ready or status.mlp_l14_ready:
            ready_pths = []
            if status.mlp_ready:
                ready_pths.append("ViT-B/32")
            if status.mlp_l14_ready:
                ready_pths.append("ViT-L/14")
            self.custom_status_label.setText(f"🟡 状态：已有权重未熔铸 ({', '.join(ready_pths)}，建议点击下方按钮一键熔铸 ONNX)")
            self.custom_status_label.setStyleSheet(f"color: {AMBER_FG}; font-weight: bold; margin-left: 26px;")
            self.export_onnx_btn.setText("⚡  一键熔铸为 ONNX 模型")
            self.export_onnx_btn.setEnabled(True)
        else:
            self.custom_status_label.setText("⚪ 状态：未训练 (暂无个性化模型，可前往“偏好训练”导入照片训练)")
            self.custom_status_label.setStyleSheet(f"color: {TEXT_TERT}; font-weight: bold; margin-left: 26px;")
            self.export_onnx_btn.setEnabled(False)

        # 4. Aesthetic 3 (CLIP ViT-L/14) 底座状态
        if status.clip_l14_location == "local":
            size_mb = sum(f.stat().st_size for f in CLIP_L14_MODEL_DIR.glob("**/*") if f.is_file()) / (1024 * 1024)
            self.clip_l14_status_label.setText(f"✅ 已就绪 (项目本地: models/clip-vit-large-patch14, 共 {size_mb:.1f} MB)")
            self.clip_l14_status_label.setStyleSheet(f"color: {GREEN_FG}; font-weight: bold;")
            self.dl_l14_btn.setText("🔄  重新校验 / 重新熔铸 ONNX")
        elif status.clip_l14_location == "hf_cache":
            self.clip_l14_status_label.setText("✅ 已在系统 HuggingFace 缓存中就绪 (可直接离线秒级导入并熔铸)")
            self.clip_l14_status_label.setStyleSheet(f"color: {GREEN_FG}; font-weight: bold;")
            self.dl_l14_btn.setText("📥  秒级导入并熔铸 Aesthetic 3 ONNX")
        else:
            self.clip_l14_status_label.setText("⚪ 未下载 (点击下方按钮一键下载并自动熔铸)")
            self.clip_l14_status_label.setStyleSheet(f"color: {TEXT_TERT}; font-weight: bold;")
            self.dl_l14_btn.setText("⬇️  一键下载并熔铸 Aesthetic 3 大模型")

        # 5. CLIP ViT-B/32 底座状态
        if status.clip_location == "local":
            size_mb = sum(f.stat().st_size for f in CLIP_MODEL_DIR.glob("**/*") if f.is_file()) / (1024 * 1024)
            self.clip_status_label.setText(f"✅ 已就绪 (项目本地: models/clip-vit-base-patch32, 共 {size_mb:.1f} MB)")
            self.clip_status_label.setStyleSheet(f"color: {GREEN_FG}; font-weight: bold;")
            self.dl_clip_btn.setText("🔄  重新校验/更新")
        elif status.clip_location == "hf_cache":
            self.clip_status_label.setText("✅ 已在系统 HuggingFace 缓存中就绪 (可直接离线使用)")
            self.clip_status_label.setStyleSheet(f"color: {GREEN_FG}; font-weight: bold;")
            self.dl_clip_btn.setText("📥  秒级同步至项目 models/ 目录")
        else:
            self.clip_status_label.setText("⚪ 未下载 (训练时将自动下载)")
            self.clip_status_label.setStyleSheet(f"color: {TEXT_TERT}; font-weight: bold;")
            self.dl_clip_btn.setText("⬇️  一键下载至本地 models/ 目录")

    def _check_startup_models(self) -> None:
        status = check_all_models()
        if not status.clip_ready and not status.onnx_ready and not status.standard_onnx_ready:
            ret = QMessageBox.question(
                self, "检测到未下载基础模型",
                "欢迎使用 Photo Sort！\n\n检测到尚未下载 CLIP 基础视觉模型。\n"
                "下载后可完全离线支持照片特征提取与个人审美训练。\n\n"
                "是否立即下载至本地 ./models/ 目录？（约 340MB，推荐下载）",
                QMessageBox.Yes | QMessageBox.No
            )
            if ret == QMessageBox.Yes:
                self._switch_tab(2)
                self._start_download_clip()

    def _start_download_clip_l14(self) -> None:
        if self._download_l14_worker and self._download_l14_worker.isRunning():
            return

        self.dl_l14_btn.setEnabled(False)
        self.dl_l14_progress.setVisible(True)
        self.dl_l14_progress.setValue(0)
        self.dl_l14_status_lbl.setVisible(True)
        self.dl_l14_status_lbl.setText("正在准备下载/同步 Aesthetic 3 大模型...")

        self._download_l14_worker = DownloadL14Worker(use_mirror=self.use_l14_mirror_cb.isChecked())
        self._download_l14_worker.progress_sig.connect(self._on_dl_l14_progress)
        self._download_l14_worker.done_sig.connect(self._on_dl_l14_done)
        self._download_l14_worker.start()

    def _on_dl_l14_progress(self, msg: str, pct: float) -> None:
        self.dl_l14_status_lbl.setText(msg)
        self.dl_l14_progress.setValue(int(pct * 100))

    def _on_dl_l14_done(self, success: bool, msg: str) -> None:
        self.dl_l14_btn.setEnabled(True)
        self.dl_l14_progress.setVisible(False)
        self.dl_l14_status_lbl.setVisible(False)
        self._on_model_updated()

        if success:
            QMessageBox.information(self, "就绪完成", msg)
        else:
            QMessageBox.critical(self, "操作失败", msg)

    def _start_download_clip(self) -> None:
        if self._download_worker and self._download_worker.isRunning():
            return

        self.dl_clip_btn.setEnabled(False)
        self.dl_progress.setVisible(True)
        self.dl_progress.setValue(0)
        self.dl_status_lbl.setVisible(True)
        self.dl_status_lbl.setText("正在准备同步/下载...")

        self._download_worker = DownloadWorker(use_mirror=self.use_mirror_cb.isChecked())
        self._download_worker.progress_sig.connect(self._on_dl_progress)
        self._download_worker.done_sig.connect(self._on_dl_done)
        self._download_worker.start()

    def _on_dl_progress(self, msg: str, pct: float) -> None:
        self.dl_status_lbl.setText(msg)
        self.dl_progress.setValue(int(pct * 100))

    def _on_dl_done(self, success: bool, msg: str) -> None:
        self.dl_clip_btn.setEnabled(True)
        self.dl_progress.setVisible(False)
        self.dl_status_lbl.setVisible(False)
        self._on_model_updated()

        if success:
            QMessageBox.information(self, "同步完成", msg)
        else:
            QMessageBox.critical(self, "下载失败", msg)

    def _on_import_pth(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 PyTorch 审美权重文件 (.pth)",
            str(Path.home()),
            "PyTorch 权重 (*.pth *.pt);;所有文件 (*.*)",
        )
        if not file_path:
            return

        src_p = Path(file_path)
        if not src_p.exists() or src_p.stat().st_size < 100:
            QMessageBox.warning(self, "无效文件", "所选文件不存在或内容为空。")
            return

        try:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            is_l14 = "l14" in src_p.name.lower() or "large" in src_p.name.lower()
            target_name = "aesthetic_mlp_l14.pth" if is_l14 else "aesthetic_mlp.pth"
            dest_p = MODELS_DIR / target_name

            shutil.copy2(str(src_p), str(dest_p))
            if not is_l14:
                try:
                    shutil.copy2(str(src_p), str(PROJECT_ROOT / "aesthetic_mlp.pth"))
                except Exception:
                    pass

            self._on_model_updated()

            reply = QMessageBox.question(
                self,
                "导入成功",
                f"✅ 权重文件已成功导入至 models/{target_name}！\n\n是否立即一键熔铸为专属 ONNX 硬件加速模型并启用？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if reply == QMessageBox.Yes:
                self._on_manual_export_onnx()
            else:
                QMessageBox.information(
                    self,
                    "导入完成",
                    "权重已导入就绪。\n您可以随时在此面板点击「⚡ 一键熔铸为 ONNX 模型」生成并启用专属模型。",
                )
        except Exception as exc:
            QMessageBox.critical(self, "导入失败", f"导入权重文件时出错: {exc}")

    def _on_manual_export_onnx(self) -> None:
        pth_exists = (
            (MODELS_DIR / "aesthetic_mlp.pth").exists()
            or (MODELS_DIR / "aesthetic_mlp_l14.pth").exists()
            or (PROJECT_ROOT / "aesthetic_mlp.pth").exists()
            or (BUNDLE_ROOT / "aesthetic_mlp.pth").exists()
        )
        if not pth_exists:
            reply = QMessageBox.question(
                self,
                "未找到权重文件",
                "当前尚未检测到任何 .pth 个人训练权重文件。\n\n是否立即从本地选择并导入一个 .pth 文件？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if reply == QMessageBox.Yes:
                self._on_import_pth()
            return

        self.export_onnx_btn.setEnabled(False)
        self._export_worker = ExportWorker()
        self._export_worker.done_sig.connect(self._on_export_done)
        self._export_worker.start()

    def _on_export_done(self, success: bool, msg: str) -> None:
        self.export_onnx_btn.setEnabled(True)
        if success:
            set_active_model_mode("custom")
            self.radio_custom.setChecked(True)
        self._on_model_updated()
        if success:
            QMessageBox.information(self, "熔铸成功", msg)
        else:
            QMessageBox.critical(self, "导出失败", msg)



def launch_main_gui() -> None:
    app = QApplication(sys.argv)
    if sys.platform == "darwin":
        app.setFont(QtGui.QFont(".AppleSystemUIFont", 12))
    elif sys.platform == "win32":
        app.setFont(QtGui.QFont("Segoe UI", 10))
    
    icon_path = _SRC_DIR / "assets" / "icon.png"
    if icon_path.exists():
        app.setWindowIcon(QtGui.QIcon(str(icon_path)))

    app.setStyleSheet(APP_STYLE)
    win = MainAppGUI()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    launch_main_gui()
