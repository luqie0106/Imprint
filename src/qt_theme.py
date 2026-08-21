"""
qt_theme.py — Apple / iOS 现代全圆角矢量设计系统 (PySide6 / Qt6)
包含自动生成的矢量级勾选图标与下拉箭头
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QColor, QPen, QPixmap

# ── 现代苹果设计色彩 ───────────────────────────────────────────────────────────
BG           = "#F2F2F7"  # Apple Grouped Background
SURFACE      = "#FFFFFF"  # Card Surface
BORDER       = "#E5E5EA"  # 1px Separator
INPUT_BG     = "#F8F8FA"  # Inset Input Background
INPUT_BORDER = "#E0E0E6"  # Inset Input Border

ACCENT       = "#0071E3"  # Apple Blue
ACCENT_HOVER = "#0077ED"  # Hover Blue
ACCENT_PRESS = "#005BB5"  # Press Blue
ACCENT_DIS   = "#D1D1D6"

TEXT         = "#1C1C1E"  # Primary Label
TEXT_SEC     = "#636366"  # Secondary Label
TEXT_TERT    = "#8E8E93"  # Tertiary Label

GREEN_BG     = "#E8F8EE"
GREEN_FG     = "#1B8738"
AMBER_BG     = "#FFF8E6"
AMBER_FG     = "#B45309"
GRAY_BG      = "#EAEAEC"

if sys.platform == "darwin":
    FAM = ".AppleSystemUIFont, 'SF Pro Text', 'Helvetica Neue'"
    FAM_TITLE = ".AppleSystemUIFont, 'SF Pro Display', 'Helvetica Neue'"
elif sys.platform == "win32":
    FAM = "'Segoe UI Variable Text', 'Segoe UI', 'Microsoft YaHei'"
    FAM_TITLE = "'Segoe UI Variable Display', 'Segoe UI', 'Microsoft YaHei'"
else:
    FAM = "'DejaVu Sans', 'Ubuntu', 'Helvetica'"
    FAM_TITLE = "'DejaVu Sans', 'Ubuntu', 'Helvetica'"

# ── 自动生成/确保矢量级微图标 ─────────────────────────────────────────────────
_ASSETS_DIR = Path(__file__).resolve().parent / "assets"
_ASSETS_DIR.mkdir(parents=True, exist_ok=True)

_CHECK_PATH = (_ASSETS_DIR / "checkmark.png").resolve().as_posix()
_ARROW_PATH = (_ASSETS_DIR / "chevron_down.png").resolve().as_posix()
_RADIO_ON_PATH = (_ASSETS_DIR / "radio_checked.png").resolve().as_posix()
_RADIO_OFF_PATH = (_ASSETS_DIR / "radio_unchecked.png").resolve().as_posix()


def _ensure_icons():
    from PySide6.QtGui import QGuiApplication
    if not QGuiApplication.instance():
        return

    if not os.path.exists(_CHECK_PATH):
        pm = QPixmap(32, 32)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor("#FFFFFF"), 3.5, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        p.setPen(pen)
        p.drawLine(6, 16, 13, 23)
        p.drawLine(13, 23, 26, 8)
        p.end()
        pm.save(_CHECK_PATH)

    if not os.path.exists(_ARROW_PATH):
        pm = QPixmap(32, 32)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor("#8E8E93"), 3.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        p.setPen(pen)
        p.drawLine(8, 12, 16, 20)
        p.drawLine(16, 20, 24, 12)
        p.end()
        pm.save(_ARROW_PATH)

    if not os.path.exists(_RADIO_ON_PATH):
        pm = QPixmap(36, 36)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)
        p.setBrush(QColor("#0071E3"))
        p.setPen(Qt.NoPen)
        p.drawEllipse(2, 2, 32, 32)
        p.setBrush(QColor("#FFFFFF"))
        p.drawEllipse(11, 11, 14, 14)
        p.end()
        pm.save(_RADIO_ON_PATH)

    if not os.path.exists(_RADIO_OFF_PATH):
        pm = QPixmap(36, 36)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)
        p.setBrush(QColor("#FFFFFF"))
        pen = QPen(QColor("#D1D1D6"), 2.5, Qt.SolidLine)
        p.setPen(pen)
        p.drawEllipse(3, 3, 30, 30)
        p.end()
        pm.save(_RADIO_OFF_PATH)




try:
    _ensure_icons()
except Exception:
    pass


# ── 全局 QSS 样式表 ───────────────────────────────────────────────────────────
APP_STYLE = f"""
QMainWindow, QWidget#centralWidget {{
    background-color: {BG};
    font-family: {FAM};
}}

/* ── 现代悬浮卡片 ── */
QFrame[class="CardFrame"] {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 14px;
}}

/* ── 标题与标签 ── */
QLabel {{
    color: {TEXT};
    font-family: {FAM};
}}
QLabel[class="CardTitle"] {{
    font-size: 13px;
    font-weight: bold;
    color: {TEXT};
}}
QLabel[class="SecondaryLabel"] {{
    color: {TEXT_SEC};
    font-size: 12px;
}}

/* ── 状态胶囊条 ── */
QFrame[class="StatusPill"] {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 12px;
}}

/* ── 现代内嵌输入框 ── */
QLineEdit {{
    background-color: {INPUT_BG};
    color: {TEXT};
    border: 1px solid {INPUT_BORDER};
    border-radius: 8px;
    padding: 6px 12px;
    font-size: 12px;
    selection-background-color: {ACCENT};
}}
QLineEdit:focus {{
    border: 1.5px solid {ACCENT};
    background-color: #FFFFFF;
}}
QLineEdit:disabled {{
    background-color: #F0F0F2;
    color: {TEXT_TERT};
    border-color: #E5E5EA;
}}

/* ── 现代下拉选择框 ── */
QComboBox {{
    background-color: {INPUT_BG};
    color: {TEXT};
    border: 1px solid {INPUT_BORDER};
    border-radius: 8px;
    padding: 5px 30px 5px 12px;
    font-size: 12px;
}}
QComboBox:focus {{
    border: 1.5px solid {ACCENT};
    background-color: #FFFFFF;
}}
QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 26px;
    border-left: none;
}}
QComboBox::down-arrow {{
    image: url('{_ARROW_PATH}');
    width: 12px;
    height: 12px;
}}
QComboBox QAbstractItemView {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 8px;
    selection-background-color: {ACCENT};
    selection-color: #FFFFFF;
    outline: none;
    padding: 4px;
}}

/* ── 现代复选框 ── */
QCheckBox {{
    color: {TEXT};
    font-size: 12px;
    spacing: 8px;
}}
QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border: 1.5px solid {INPUT_BORDER};
    border-radius: 5px;
    background-color: {INPUT_BG};
}}
QCheckBox::indicator:hover {{
    border-color: {ACCENT};
}}
QCheckBox::indicator:checked {{
    background-color: {ACCENT};
    border-color: {ACCENT};
    image: url('{_CHECK_PATH}');
}}

/* ── 现代单选框 ── */
QRadioButton {{
    color: {TEXT};
    font-size: 13px;
    spacing: 10px;
}}
QRadioButton::indicator {{
    width: 18px;
    height: 18px;
    image: url('{_RADIO_OFF_PATH}');
    background: transparent;
    border: none;
}}
QRadioButton::indicator:checked {{
    image: url('{_RADIO_ON_PATH}');
}}

/* ── 统一交互按钮规范 (必须声明显式 border 才能激活 Qt border-radius) ── */
QPushButton {{
    font-family: {FAM};
    font-size: 12px;
    font-weight: bold;
    border-radius: 15px;
    min-height: 28px;
    padding: 0 18px;
    outline: none;
}}

QPushButton[class="PrimaryBtn"] {{
    background-color: {ACCENT};
    color: #FFFFFF;
    border: 1px solid {ACCENT};
    border-radius: 15px;
    min-height: 28px;
    padding: 0 18px;
}}
QPushButton[class="PrimaryBtn"]:hover {{
    background-color: {ACCENT_HOVER};
    border-color: {ACCENT_HOVER};
}}
QPushButton[class="PrimaryBtn"]:pressed {{
    background-color: {ACCENT_PRESS};
    border-color: {ACCENT_PRESS};
}}
QPushButton[class="PrimaryBtn"]:disabled {{
    background-color: {ACCENT_DIS};
    border-color: {ACCENT_DIS};
    color: #FFFFFF;
}}

QPushButton[class="SecondaryBtn"] {{
    background-color: {GRAY_BG};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 15px;
    min-height: 28px;
    padding: 0 16px;
}}
QPushButton[class="SecondaryBtn"]:hover {{
    background-color: #DFDFE2;
    border-color: #D0D0D5;
}}
QPushButton[class="SecondaryBtn"]:pressed {{
    background-color: #D5D5DA;
    border-color: #C5C5CA;
}}
QPushButton[class="SecondaryBtn"]:disabled {{
    background-color: #F8F8FA;
    color: {TEXT_TERT};
    border-color: #EAEAEA;
}}

/* ── 现代进度条 ── */
QProgressBar {{
    background-color: {INPUT_BG};
    border: 1px solid {BORDER};
    border-radius: 6px;
    height: 8px;
    text-align: center;
}}
QProgressBar::chunk {{
    background-color: {ACCENT};
    border-radius: 5px;
}}

/* ── 现代控制台日志文本框 ── */
QPlainTextEdit[class="LogConsole"] {{
    background-color: {INPUT_BG};
    color: #334155;
    border: 1px solid {INPUT_BORDER};
    border-radius: 10px;
    padding: 8px 12px;
    font-family: Menlo, Consolas, Courier New, monospace;
    font-size: 11px;
    selection-background-color: {ACCENT};
    selection-color: #FFFFFF;
}}

/* ── 分段选择器导航栏 ── */
QFrame#SegmentTrack {{
    background-color: #E3E3E8;
    border: 1px solid #D5D5DA;
    border-radius: 12px;
}}
QPushButton[class="SegmentBtn"] {{
    background-color: transparent;
    color: {TEXT_SEC};
    border: 1px solid transparent;
    border-radius: 9px;
    padding: 6px 18px;
    min-height: 20px;
    font-size: 12px;
    font-weight: bold;
}}
QPushButton[class="SegmentBtn"]:checked {{
    background-color: #FFFFFF;
    color: {TEXT};
    border: 0.5px solid #D0D0D5;
}}
QPushButton[class="SegmentBtn"]:hover:!checked {{
    background-color: #DEDEE3;
}}
"""
