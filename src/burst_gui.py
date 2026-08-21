"""
burst_gui.py — RAW 连拍优选界面 (PySide6 / Qt6 现代全圆角矢量设计)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QWidget, QFrame, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QCheckBox, QProgressBar,
    QFileDialog, QMessageBox
)

# ── 确保 src 在 sys.path 上 ───────────────────────────────────────────────────
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from burst_filter import BurstFilter, BurstFilterResult
from model_manager import check_all_models, PROJECT_ROOT
from qt_theme import GREEN_FG, AMBER_FG, TEXT_TERT, TEXT_SEC


class BurstWorker(QThread):
    progress_sig = Signal(str)
    done_sig = Signal(object)
    error_sig = Signal(str)

    def __init__(self, input_dir: Path, gap: float, hamming: int, subdir: str, keep: int, workers: int, use_gpu: bool):
        super().__init__()
        self.input_dir = input_dir
        self.gap = gap
        self.hamming = hamming
        self.subdir = subdir
        self.keep = keep
        self.workers = workers
        self.use_gpu = use_gpu

    def run(self):
        try:
            flt = BurstFilter(
                gap_seconds=self.gap,
                max_hamming_distance=self.hamming,
                review_subdir=self.subdir,
                keep_count=self.keep,
                max_workers=self.workers,
                use_gpu=self.use_gpu,
                progress_callback=lambda msg: self.progress_sig.emit(msg),
            )
            result = flt.run(self.input_dir)
            self.done_sig.emit(result)
        except Exception as exc:
            self.error_sig.emit(str(exc))


class BurstFilterGUI(QWidget):
    """RAW 连拍优选面板 (PySide6 现代全圆角矢量界面)"""

    def __init__(self, on_navigate_tab: Callable[[int], None] | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.on_navigate_tab = on_navigate_tab
        self._running = False
        self._worker: BurstWorker | None = None

        self._build_ui()
        self.refresh_model_status()

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 10, 0, 10)
        main_layout.setSpacing(10)

        # ── 1. 模型状态胶囊条 ──
        status_card = QFrame()
        status_card.setProperty("class", "StatusPill")
        s_layout = QHBoxLayout(status_card)
        s_layout.setContentsMargins(14, 8, 14, 8)
        s_layout.setSpacing(8)

        self.model_badge_icon = QLabel("🟢")
        self.model_badge_icon.setStyleSheet("font-size: 13px;")
        s_layout.addWidget(self.model_badge_icon)

        self.model_status_lbl = QLabel("正在检测推理引擎...")
        self.model_status_lbl.setStyleSheet("font-weight: bold; font-size: 12px;")
        s_layout.addWidget(self.model_status_lbl)
        s_layout.addStretch()

        main_layout.addWidget(status_card)

        # ── 2. 照片目录选择卡片 ──
        dir_card = QFrame()
        dir_card.setProperty("class", "CardFrame")
        d_layout = QVBoxLayout(dir_card)
        d_layout.setContentsMargins(16, 12, 16, 12)
        d_layout.setSpacing(8)

        title_lbl = QLabel("照片文件夹 (支持 RAW / JPEG / JPEG XL / HIF / HEIF / PNG)")
        title_lbl.setProperty("class", "CardTitle")
        d_layout.addWidget(title_lbl)

        d_row = QHBoxLayout()
        d_row.setSpacing(10)

        self.dir_input = QLineEdit()
        self.dir_input.setPlaceholderText("选择或输入包含 RAW、JPEG、JPEG XL、HIF、HEIF 格式照片的文件夹路径...")
        d_row.addWidget(self.dir_input)

        self.pick_dir_btn = QPushButton("选择文件夹")
        self.pick_dir_btn.setProperty("class", "SecondaryBtn")
        self.pick_dir_btn.clicked.connect(self._pick_dir)
        d_row.addWidget(self.pick_dir_btn)

        d_layout.addLayout(d_row)
        main_layout.addWidget(dir_card)

        # ── 3. 参数配置卡片 ──
        param_card = QFrame()
        param_card.setProperty("class", "CardFrame")
        p_layout = QVBoxLayout(param_card)
        p_layout.setContentsMargins(16, 12, 16, 12)
        p_layout.setSpacing(10)

        p_title = QLabel("筛选与分组参数")
        p_title.setProperty("class", "CardTitle")
        p_layout.addWidget(p_title)

        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(8)

        # 第 0 行
        lbl1 = QLabel("淘汰子文件夹:")
        lbl1.setProperty("class", "SecondaryLabel")
        lbl1.setToolTip("连拍中被淘汰的废片将移动到当前照片目录下的该子文件夹中，不破坏原图。")
        grid.addWidget(lbl1, 0, 0)
        self.subdir_input = QLineEdit("审查_连拍淘汰")
        self.subdir_input.setToolTip("淘汰照片将存入该目录，方便您随时进入人工复查。")
        grid.addWidget(self.subdir_input, 0, 1)

        lbl2 = QLabel("每组保留张数:")
        lbl2.setProperty("class", "SecondaryLabel")
        lbl2.setToolTip("每个连拍序列中最终评选保留的最佳照片张数，其余照片移入审查目录。默认 1 张。")
        grid.addWidget(lbl2, 0, 2)
        self.keep_input = QLineEdit("1")
        self.keep_input.setToolTip("每个连拍组优选保留的张数（推荐 1~3 张）。")
        grid.addWidget(self.keep_input, 0, 3)

        # 第 1 行
        lbl3 = QLabel("连拍时间间隔(秒):")
        lbl3.setProperty("class", "SecondaryLabel")
        lbl3.setToolTip("【连拍时间阈值】\n相邻照片在 EXIF 亚秒时间轴上的最大间隔秒数。\n在此时间窗口内连续按下的快门会被判定为同一个连拍序列。\n默认 1.5 秒（推荐 1.0 ~ 3.0 秒）。")
        grid.addWidget(lbl3, 1, 0)
        self.gap_input = QLineEdit("1.5")
        self.gap_input.setToolTip("【连拍时间阈值】\n相邻照片拍摄间隔小于此值时参与连拍比对，超出此值自动开启新的一组。")
        grid.addWidget(self.gap_input, 1, 1)

        max_cpus = os.cpu_count() or 4
        default_workers = str(max(1, round(max_cpus * 0.8)))
        lbl4 = QLabel(f"并发处理线程数:")
        lbl4.setProperty("class", "SecondaryLabel")
        lbl4.setToolTip("多线程并发读取 RAW 预览与计算清晰度，默认自动设置为电脑 CPU 核心数的 80%。")
        grid.addWidget(lbl4, 1, 2)
        self.workers_input = QLineEdit(default_workers)
        self.workers_input.setToolTip(f"并发计算线程数（推荐 2 ~ {max_cpus}，过高可能占用较多内存）。")
        self.workers_input.textChanged.connect(self._validate_workers)
        grid.addWidget(self.workers_input, 1, 3)

        # 第 2 行
        lbl5 = QLabel("构图容差(汉明距离):")
        lbl5.setProperty("class", "SecondaryLabel")
        hamming_tip = (
            "【画面构图相似度容差 / 汉明距离 (dHash)】\n"
            "用于衡量连续快门之间画面主体与构图的相似程度（取值范围 1 ~ 64，默认 12）：\n"
            "· 数值越小（如 6~8）：要求画面构图极其严格一致（适合三脚架定点摆拍）。\n"
            "· 默认值 12：适合绝大多数手持防抖与常规追焦连拍。\n"
            "· 数值越大（如 16~20）：允许剧烈的大幅度甩镜头与奔跑运动，不易被拆分组。"
        )
        lbl5.setToolTip(hamming_tip)
        grid.addWidget(lbl5, 2, 0)
        self.hamming_input = QLineEdit("12")
        self.hamming_input.setToolTip(hamming_tip)
        grid.addWidget(self.hamming_input, 2, 1)

        self.gpu_cb = QCheckBox("启用 显卡/NPU 硬件加速 (CoreML / DirectML)")
        self.gpu_cb.setToolTip("使用系统 GPU 或 NPU 极速运行 AI 美学模型推理，提升连拍选优速度。")
        self.gpu_cb.setChecked(True)
        grid.addWidget(self.gpu_cb, 2, 2, 1, 2)

        p_layout.addLayout(grid)

        # ── 参数常驻通俗说明条 ──
        tip_frame = QFrame()
        tip_frame.setStyleSheet(
            "background-color: #F8F8FA; border: 1px solid #E5E5EA; border-radius: 8px; padding: 6px 10px;"
        )
        tip_layout = QVBoxLayout(tip_frame)
        tip_layout.setContentsMargins(8, 6, 8, 6)
        tip_layout.setSpacing(5)

        tip_lbl1 = QLabel("💡 <b>构图容差 (汉明距离，1~64，默认 12)</b>：衡量连拍中画面构图与机位相似度。定点摆拍建议调小 (6~8)；奔跑追焦/大幅甩镜头建议调大 (16~20)。")
        tip_lbl1.setStyleSheet("color: #636366; font-size: 12px; line-height: 1.4;")
        tip_lbl1.setWordWrap(True)
        tip_layout.addWidget(tip_lbl1)

        tip_lbl2 = QLabel("💡 <b>连拍时间间隔 (默认 1.5s)</b>：相邻快门时间间隔小于此值归为同一连拍组。<b>RAW+JPG 伴生照片</b>将自动绑定为整体同步保留或淘汰。")
        tip_lbl2.setStyleSheet("color: #636366; font-size: 12px; line-height: 1.4;")
        tip_lbl2.setWordWrap(True)
        tip_layout.addWidget(tip_lbl2)

        p_layout.addWidget(tip_frame)

        main_layout.addWidget(param_card)


        # ── 4. 底部执行行 ──
        br_layout = QHBoxLayout()
        br_layout.setSpacing(12)

        self.run_btn = QPushButton("▶  开始智能筛选")
        self.run_btn.setProperty("class", "PrimaryBtn")
        self.run_btn.clicked.connect(self._on_run)
        br_layout.addWidget(self.run_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedWidth(160)
        br_layout.addWidget(self.progress_bar)

        self.status_lbl = QLabel("")
        self.status_lbl.setProperty("class", "SecondaryLabel")
        br_layout.addWidget(self.status_lbl)
        br_layout.addStretch()

        main_layout.addLayout(br_layout)
        main_layout.addStretch()

    def _validate_workers(self, text: str) -> None:
        try:
            val = int(text.strip())
            max_cpus = os.cpu_count() or 4
            if val > max_cpus or val < 1:
                self.workers_input.setStyleSheet("color: #DC2626; border-color: #DC2626;")
                self.run_btn.setEnabled(False)
            else:
                self.workers_input.setStyleSheet("")
                self.run_btn.setEnabled(True)
        except ValueError:
            self.workers_input.setStyleSheet("color: #DC2626; border-color: #DC2626;")
            self.run_btn.setEnabled(False)

    def refresh_model_status(self) -> None:
        status = check_all_models()
        if status.active_mode == "custom" and status.custom_onnx_ready:
            self.model_badge_icon.setText("🟢")
            self.model_status_lbl.setText("AI 美学引擎：个人专属训练模型 (ONNX 硬件加速已启用)")
            self.model_status_lbl.setStyleSheet(f"color: {GREEN_FG}; font-weight: bold;")
        elif status.standard_onnx_ready:
            self.model_badge_icon.setText("🟢")
            self.model_status_lbl.setText("AI 美学引擎：官方标准通用模型 (ONNX 硬件加速已启用)")
            self.model_status_lbl.setStyleSheet(f"color: {GREEN_FG}; font-weight: bold;")
        elif status.custom_onnx_ready:
            self.model_badge_icon.setText("🟢")
            self.model_status_lbl.setText("AI 美学引擎：个人专属训练模型 (ONNX 硬件加速已启用)")
            self.model_status_lbl.setStyleSheet(f"color: {GREEN_FG}; font-weight: bold;")
        elif status.mlp_ready:
            self.model_badge_icon.setText("🟡")
            self.model_status_lbl.setText("AI 美学引擎：PyTorch 模式 (建议在“模型与环境”熔铸 ONNX)")
            self.model_status_lbl.setStyleSheet(f"color: {AMBER_FG}; font-weight: bold;")
        else:
            self.model_badge_icon.setText("⚪")
            self.model_status_lbl.setText("AI 美学引擎：未加载模型 (当前降级为纯物理规则对焦清晰度模式)")
            self.model_status_lbl.setStyleSheet(f"color: {TEXT_TERT}; font-weight: bold;")


    def _pick_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "选择 RAW 照片文件夹")
        if d:
            self.dir_input.setText(d)

    def _on_run(self) -> None:
        if self._running:
            return

        input_dir_str = self.dir_input.text().strip()
        if not input_dir_str or not Path(input_dir_str).exists() or not Path(input_dir_str).is_dir():
            QMessageBox.critical(self, "路径错误", "请先选择一个有效的 RAW 文件目录。")
            return

        try:
            gap = float(self.gap_input.text().strip())
            hamming = int(self.hamming_input.text().strip())
            keep = int(self.keep_input.text().strip())
            workers = int(self.workers_input.text().strip())
            use_gpu = self.gpu_cb.isChecked()

            assert 0.0 < gap <= 30.0
            assert 1 <= hamming <= 64
            assert keep >= 1
            max_cpus = os.cpu_count() or 4
            if workers > max_cpus:
                QMessageBox.critical(self, "线程数超限", f"为防止内存溢出导致闪退，最大线程数不得超过 {max_cpus}！")
                return
            assert workers >= 1
        except (ValueError, AssertionError):
            QMessageBox.critical(self, "参数错误", "请检查填写的参数。时间阈值 0~30，汉明限制 1~64，保留张数 ≥ 1，线程数必须为合法正整数且不可越界。")
            return

        subdir = self.subdir_input.text().strip() or "审查_连拍淘汰"
        self._set_running(True)

        self._worker = BurstWorker(
            input_dir=Path(input_dir_str),
            gap=gap,
            hamming=hamming,
            subdir=subdir,
            keep=keep,
            workers=workers,
            use_gpu=use_gpu,
        )
        self._worker.progress_sig.connect(self._set_status)
        self._worker.done_sig.connect(self._on_done)
        self._worker.error_sig.connect(self._on_error)
        self._worker.start()

    def _set_status(self, text: str) -> None:
        self.status_lbl.setText(text)

    def _set_running(self, running: bool) -> None:
        self._running = running
        self.run_btn.setEnabled(not running)
        self.progress_bar.setVisible(running)
        if running:
            self._set_status("正在筛选…")

    def _on_done(self, r: BurstFilterResult) -> None:
        self._set_running(False)
        self._set_status("筛选完成")

        if r.total == 0:
            QMessageBox.information(self, "处理完成", "目录中未找到任何支持的照片文件（支持 RAW / JPEG / JXL / HIF / HEIF / PNG）。")
            return

        lines = [
            f"总照片文件数：    {r.total}",
            f"单拍跳过（保留）：  {r.skipped_single}",
            f"连拍组数：         {r.burst_groups}",
            f"已移动淘汰数：     {r.moved}",
        ]
        if r.review_dir:
            lines.append(f"\n淘汰目录：\n{r.review_dir}")
        else:
            lines.append("\n（无连拍组，所有文件保留原位）")

        if r.errors:
            lines.append(f"\n⚠️  {len(r.errors)} 个文件处理警告：")
            for e in r.errors[:5]:
                lines.append(f"  · {e}")
            if len(r.errors) > 5:
                lines.append(f"  … 共 {len(r.errors)} 条")

        msg = "\n".join(lines)
        if r.errors:
            QMessageBox.warning(self, "处理完成（含警告）", msg)
        else:
            QMessageBox.information(self, "处理完成", msg)

    def _on_error(self, message: str) -> None:
        self._set_running(False)
        self._set_status("出错")
        QMessageBox.critical(self, "运行失败", message)


def launch_burst_gui() -> None:
    from qt_theme import APP_STYLE
    app = QtWidgets.QApplication(sys.argv)
    if sys.platform == "darwin":
        app.setFont(QtGui.QFont(".AppleSystemUIFont", 12))
    elif sys.platform == "win32":
        app.setFont(QtGui.QFont("Segoe UI", 10))
    app.setStyleSheet(APP_STYLE)
    w = QtWidgets.QMainWindow()
    w.setWindowTitle("Photo Sort — RAW 连拍优选")
    w.resize(760, 600)
    w.setCentralWidget(BurstFilterGUI())
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    launch_burst_gui()
