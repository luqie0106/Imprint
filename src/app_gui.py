"""
app_gui.py — Photo Sort 综合一体化主程序界面 (PySide6 / Qt6 现代全圆角矢量设计)
整合连拍筛选、偏好训练与模型管理
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Callable

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFrame, QVBoxLayout,
    QHBoxLayout, QLabel, QPushButton, QCheckBox, QRadioButton, QProgressBar,
    QStackedWidget, QButtonGroup, QMessageBox
)

# ── 确保 src 在 sys.path 上 ───────────────────────────────────────────────────
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from burst_gui import BurstFilterGUI
from trainer_gui import TrainerGUI
from model_manager import (
    PROJECT_ROOT,
    MODELS_DIR,
    CLIP_MODEL_DIR,
    MLP_WEIGHTS_PATH,
    STANDARD_ONNX_PATH,
    CUSTOM_ONNX_PATH,
    check_all_models,
    get_active_model_mode,
    set_active_model_mode,
    download_clip_model,
    is_clip_model_downloaded,
    is_clip_in_hf_cache,
    import_from_hf_cache,
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
                self.done_sig.emit(True, "CLIP 基础视觉模型已成功就绪并保存在 ./models/clip-vit-base-patch32 目录！")
            else:
                self.done_sig.emit(False, "模型下载未完成。")
        except Exception as exc:
            self.done_sig.emit(False, f"下载出错: {exc}")


class ExportWorker(QThread):
    done_sig = Signal(bool, str)

    def run(self):
        try:
            export_to_onnx(project_root=PROJECT_ROOT)
            self.done_sig.emit(True, "photo_sort_model.onnx 已生成！连拍优选已自动启用极速硬件加速！")
        except Exception as exc:
            self.done_sig.emit(False, f"ONNX 熔铸失败: {exc}")


class MainAppGUI(QMainWindow):
    """Photo Sort 综合应用主窗口 (Qt6 / PySide6 现代全圆角设计系统)"""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Photo Sort — 智能连拍优选与个人审美系统")
        self.resize(840, 640)
        self.setMinimumSize(760, 560)


        icon_path = _SRC_DIR / "assets" / "icon.png"
        if icon_path.exists():
            self.setWindowIcon(QtGui.QIcon(str(icon_path)))

        self._download_worker: DownloadWorker | None = None
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
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 8, 0, 8)
        layout.setSpacing(12)

        # ── 卡片 1：美学评分模型选择与管理 ──
        c1 = QFrame()
        c1.setProperty("class", "CardFrame")
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

        # ── 选项 1：官方标准通用模型 ──
        self.radio_std = QRadioButton("🌟  官方标准通用模型 (Standard Universal Model)")
        self.radio_std.setStyleSheet("font-weight: bold; font-size: 13px;")
        self.model_mode_group.addButton(self.radio_std, 0)
        l1.addWidget(self.radio_std)

        std_desc = QLabel("• 官方推荐 · 全品类平衡：基于 LAION-Aesthetics / AVA 25万+ 张专业摄影数据集预训练，对人像、风光、街拍、生态等全题材中立公允打分。")
        std_desc.setStyleSheet("color: #636366; font-size: 12px; margin-left: 26px;")
        std_desc.setWordWrap(True)
        l1.addWidget(std_desc)

        self.std_status_label = QLabel("")
        self.std_status_label.setStyleSheet("font-size: 12px; font-weight: bold; margin-left: 26px; margin-bottom: 6px;")
        self.std_status_label.setWordWrap(True)
        l1.addWidget(self.std_status_label)

        # 分割线
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("background-color: #E5E5EA; max-height: 1px; margin: 4px 0px;")
        l1.addWidget(sep)

        # ── 选项 2：个人专属训练模型 ──
        self.radio_custom = QRadioButton("🧠  个人专属训练模型 (Custom Trained Model)")
        self.radio_custom.setStyleSheet("font-weight: bold; font-size: 13px;")
        self.model_mode_group.addButton(self.radio_custom, 1)
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

        to_trainer_btn = QPushButton("🎯  前往偏好训练")
        to_trainer_btn.setProperty("class", "SecondaryBtn")
        to_trainer_btn.clicked.connect(lambda: self._switch_tab(1))
        r_custom_btns.addWidget(to_trainer_btn)

        self.export_onnx_btn = QPushButton("⚡  从当前权重重新熔铸 ONNX")
        self.export_onnx_btn.setProperty("class", "PrimaryBtn")
        self.export_onnx_btn.clicked.connect(self._on_manual_export_onnx)
        r_custom_btns.addWidget(self.export_onnx_btn)
        r_custom_btns.addStretch()
        l1.addLayout(r_custom_btns)

        self.radio_std.toggled.connect(self._on_model_mode_toggled)
        self.radio_custom.toggled.connect(self._on_model_mode_toggled)

        layout.addWidget(c1)

        # ── 卡片 2：CLIP 基础视觉底座 ──
        c2 = QFrame()
        c2.setProperty("class", "CardFrame")
        l2 = QVBoxLayout(c2)
        l2.setContentsMargins(18, 16, 18, 16)
        l2.setSpacing(10)

        t2 = QLabel("📦  基础视觉主干底座 (CLIP ViT-B/32)")
        t2.setProperty("class", "CardTitle")
        l2.addWidget(t2)

        desc2 = QLabel("用于提取照片的多维视觉特征。训练个人模型时需要此底座支持；标准通用 ONNX 模型已内置此特征提取能力。")
        desc2.setProperty("class", "SecondaryLabel")
        desc2.setWordWrap(True)
        l2.addWidget(desc2)

        self.clip_status_label = QLabel("")
        self.clip_status_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        self.clip_status_label.setWordWrap(True)
        l2.addWidget(self.clip_status_label)

        r2 = QHBoxLayout()
        r2.setSpacing(12)
        self.dl_clip_btn = QPushButton("⬇️  下载/补全到本地")
        self.dl_clip_btn.setProperty("class", "SecondaryBtn")
        self.dl_clip_btn.clicked.connect(self._start_download_clip)
        r2.addWidget(self.dl_clip_btn)

        self.use_mirror_cb = QCheckBox("使用国内加速镜像 (hf-mirror.com)")
        self.use_mirror_cb.setChecked(True)
        r2.addWidget(self.use_mirror_cb)
        r2.addStretch()
        l2.addLayout(r2)

        self.dl_progress = QProgressBar()
        self.dl_progress.setRange(0, 100)
        self.dl_progress.setValue(0)
        self.dl_progress.setVisible(False)
        l2.addWidget(self.dl_progress)

        self.dl_status_lbl = QLabel("")
        self.dl_status_lbl.setProperty("class", "SecondaryLabel")
        self.dl_status_lbl.setVisible(False)
        l2.addWidget(self.dl_status_lbl)

        layout.addWidget(c2)
        layout.addStretch()

        self.refresh_model_mgr_ui()
        return widget


    def _on_model_mode_toggled(self) -> None:
        if self.radio_custom.isChecked():
            set_active_model_mode("custom")
        else:
            set_active_model_mode("standard")
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
        if active_mode == "custom":
            self.radio_custom.blockSignals(True)
            self.radio_custom.setChecked(True)
            self.radio_custom.blockSignals(False)
        else:
            self.radio_std.blockSignals(True)
            self.radio_std.setChecked(True)
            self.radio_std.blockSignals(False)

        # 1. 官方标准模型状态
        if status.standard_onnx_ready:
            p = Path(status.standard_onnx_path)
            self.std_status_label.setText(f"🟢 状态：已就绪 (文件: {p.name}, 大小: {status.standard_onnx_size_mb:.1f} MB · ONNX 极速加速)")
            self.std_status_label.setStyleSheet(f"color: {GREEN_FG}; font-weight: bold; margin-left: 24px;")
        else:
            self.std_status_label.setText("⚪ 状态：未找到标准通用模型文件")
            self.std_status_label.setStyleSheet(f"color: {TEXT_TERT}; font-weight: bold; margin-left: 24px;")

        # 2. 个人专属模型状态
        if status.custom_onnx_ready:
            p = Path(status.custom_onnx_path)
            self.custom_status_label.setText(f"🟢 状态：已就绪 (文件: {p.name}, 大小: {status.custom_onnx_size_mb:.1f} MB · ONNX 极速加速)")
            self.custom_status_label.setStyleSheet(f"color: {GREEN_FG}; font-weight: bold; margin-left: 24px;")
            self.export_onnx_btn.setText("⚡  重新熔铸 ONNX")
            self.export_onnx_btn.setEnabled(True)
        elif status.mlp_ready:
            p = Path(status.mlp_path)
            size_kb = p.stat().st_size / 1024 if p.exists() else 0
            self.custom_status_label.setText(f"🟡 状态：已有权重未熔铸 (文件: {p.name}, {size_kb:.1f} KB，建议点击下方按钮一键熔铸 ONNX)")
            self.custom_status_label.setStyleSheet(f"color: {AMBER_FG}; font-weight: bold; margin-left: 24px;")
            self.export_onnx_btn.setText("⚡  一键熔铸为 ONNX 模型")
            self.export_onnx_btn.setEnabled(True)
        else:
            self.custom_status_label.setText("⚪ 状态：未训练 (暂无个性化模型，可前往“偏好训练”导入照片训练)")
            self.custom_status_label.setStyleSheet(f"color: {TEXT_TERT}; font-weight: bold; margin-left: 24px;")
            self.export_onnx_btn.setEnabled(False)

        # 3. CLIP 底座状态
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
        if not status.clip_ready and not status.onnx_ready:
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

    def _on_manual_export_onnx(self) -> None:
        if not MLP_WEIGHTS_PATH.exists() and not (BUNDLE_ROOT / "aesthetic_mlp.pth").exists():
            QMessageBox.warning(self, "无法导出", "未找到 aesthetic_mlp.pth 权重文件，请先进行偏好训练。")
            return

        if not TORCH_EXPORT_AVAILABLE:
            QMessageBox.critical(self, "环境缺失", "导出 ONNX 需要 PyTorch 和 transformers，请在 py311 环境下运行。")
            return

        self.export_onnx_btn.setEnabled(False)
        self._export_worker = ExportWorker()
        self._export_worker.done_sig.connect(self._on_export_done)
        self._export_worker.start()

    def _on_export_done(self, success: bool, msg: str) -> None:
        self.export_onnx_btn.setEnabled(True)
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
