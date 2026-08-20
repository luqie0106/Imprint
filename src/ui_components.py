"""
ui_components.py — Apple / iOS 现代全圆角超采样抗锯齿组件库 (基于 Pillow 3x Lanczos 亚像素渲染)
提供：
  - SmoothCard: 真正平滑无锯齿的大圆角悬浮卡片 (严格几何内缩，绝无尖角溢出)
  - SmoothButton: 真正平滑无锯齿的 iOS 胶囊按钮 (带 Hover 与 Click 动画反馈)
  - SmoothEntry: 真正平滑无锯齿的 iOS 内嵌圆角输入框 (带 Apple 蓝焦点反馈，无双层黑框)
  - SmoothSegmentedControl: 真正平滑无锯齿的 iOS 分段选择器
"""

from __future__ import annotations

import sys
import tkinter as tk
import tkinter.font as tkfont
from typing import Callable, Sequence
from PIL import Image, ImageDraw, ImageTk

# ── 现代苹果设计色彩系统 ───────────────────────────────────────────────────────
BG           = "#F2F2F7"  # Apple Grouped Background
SURFACE      = "#FFFFFF"  # Card Surface
BORDER       = "#E5E5EA"  # Separator / Outline
INPUT_BG     = "#F8F8FA"  # Inset Input Field Background
INPUT_BORDER = "#E0E0E6"  # Input 1px Border

ACCENT       = "#0071E3"  # Apple Primary Blue
ACCENT_HOVER = "#0077ED"
ACCENT_DIS   = "#D1D1D6"

TEXT         = "#1C1C1E"  # System Label
TEXT_SEC     = "#636366"  # Secondary Label
TEXT_TERT    = "#8E8E93"  # Tertiary Label

GREEN_BG     = "#E8F8EE"
GREEN_FG     = "#1B8738"
AMBER_BG     = "#FFF8E6"
AMBER_FG     = "#B45309"
GRAY_BG      = "#EAEAEC"
GRAY_FG      = "#636366"

FAM = "SF Pro Text" if sys.platform == "darwin" else "Segoe UI Variable Text"
FAM_TITLE = "SF Pro Display" if sys.platform == "darwin" else "Segoe UI Variable Display"

# 缓存 PIL 生成的 PhotoImage，防止重复渲染与垃圾回收
_IMAGE_CACHE: dict[tuple, ImageTk.PhotoImage] = {}


def get_rounded_rect_image(
    w: int,
    h: int,
    radius: int,
    fill: str,
    outline: str | None = None,
    outline_width: float = 1.0,
    bg_color: str = BG,
    scale: int = 3,
) -> ImageTk.PhotoImage:
    """使用 Pillow 3x 超采样 + Lanczos 算法生成超平滑抗锯齿圆角矩形"""
    key = (w, h, radius, fill, outline, outline_width, bg_color, scale)
    if key in _IMAGE_CACHE:
        return _IMAGE_CACHE[key]

    sw, sh, sr = max(1, w * scale), max(1, h * scale), max(1, radius * scale)
    img = Image.new("RGBA", (sw, sh), bg_color)
    draw = ImageDraw.Draw(img)

    draw.rounded_rectangle(
        (0, 0, sw - 1, sh - 1),
        radius=sr,
        fill=fill,
        outline=outline,
        width=int(outline_width * scale),
    )

    img = img.resize((max(1, w), max(1, h)), Image.Resampling.LANCZOS)
    photo = ImageTk.PhotoImage(img)
    _IMAGE_CACHE[key] = photo
    return photo


class SmoothCard(tk.Frame):
    """超平滑抗锯齿圆角悬浮卡片 (几何内缩保护，绝不产生角落尖刺)"""

    def __init__(
        self,
        parent: tk.Widget,
        bg_color: str = SURFACE,
        border_color: str = BORDER,
        radius: int = 14,
        padx: int = 8,
        pady: int = 6,
        **kwargs
    ) -> None:
        self.parent_bg = parent.cget("bg") or BG
        super().__init__(parent, bg=self.parent_bg, **kwargs)
        self.bg_color = bg_color
        self.border_color = border_color
        self.radius = radius
        self.margin_x = radius + 2  # 严格内缩超过半径，确保内部矩形永不触碰圆角弧线
        self.margin_y = 8

        self.canvas = tk.Canvas(self, bg=self.parent_bg, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        self.inner = tk.Frame(self.canvas, bg=self.bg_color, padx=padx, pady=pady)
        self.window_id = self.canvas.create_window((self.margin_x, self.margin_y), window=self.inner, anchor="nw")
        self.bg_img_id = None
        self._curr_img = None

        self.canvas.bind("<Configure>", self._on_resize)
        self.inner.bind("<Configure>", self._on_inner_resize)

    def _on_resize(self, event):
        w, h = event.width, event.height
        if w <= self.margin_x * 2 + 10 or h <= self.margin_y * 2 + 10:
            return
        self._curr_img = get_rounded_rect_image(
            w, h, radius=self.radius, fill=self.bg_color,
            outline=self.border_color, outline_width=1.0,
            bg_color=self.parent_bg,
        )
        if self.bg_img_id is None:
            self.bg_img_id = self.canvas.create_image(0, 0, image=self._curr_img, anchor="nw")
            self.canvas.tag_lower(self.bg_img_id)
        else:
            self.canvas.itemconfig(self.bg_img_id, image=self._curr_img)

        self.canvas.coords(self.window_id, self.margin_x, self.margin_y)
        self.canvas.itemconfigure(self.window_id, width=w - self.margin_x * 2)

    def _on_inner_resize(self, event):
        req_h = self.inner.winfo_reqheight() + self.margin_y * 2
        self.canvas.configure(height=req_h)


class SmoothButton(tk.Label):
    """超平滑抗锯齿 iOS 胶囊按钮 (动态测宽，文字永不截断，丝滑反馈)"""

    def __init__(
        self,
        parent: tk.Widget,
        text: str,
        command: Callable[[], None],
        variant: str = "primary",
        big: bool = False,
        active: bool = True,
        radius: int | None = None,
        **kwargs
    ) -> None:
        self.parent_bg = parent.cget("bg") or SURFACE
        self.text_content = text
        self.command = command
        self.variant = variant
        self.big = big
        self.active = active

        self.font_size = 11 if big else 10
        self.font_tuple = (FAM, self.font_size, "bold")
        self.font_obj = tkfont.Font(family=FAM, size=self.font_size, weight="bold")

        self.pad_x = 18 if big else 14
        self.pad_y = 8 if big else 6
        self.radius = radius or (15 if big else 12)

        self._calc_size()
        self._setup_images()

        super().__init__(
            parent,
            image=self.normal_photo,
            text=self.text_content,
            compound="center",
            bg=self.parent_bg,
            fg=self.fg_color,
            font=self.font_tuple,
            cursor="hand2" if active else "arrow",
            bd=0,
            highlightthickness=0,
            **kwargs
        )

        self.bind("<Button-1>", self._on_click)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)

    def _calc_size(self):
        tw = self.font_obj.measure(self.text_content)
        th = self.font_obj.metrics("linespace")
        self.btn_w = max(40, tw + self.pad_x * 2)
        self.btn_h = max(24, th + self.pad_y * 2)

    def _setup_images(self):
        if self.variant == "primary":
            bg_c = ACCENT if self.active else ACCENT_DIS
            hover_c = ACCENT_HOVER
            self.fg_color = "#FFFFFF"
            out_c = None
        else:  # secondary
            bg_c = GRAY_BG if self.active else INPUT_BG
            hover_c = "#DFDFE2"
            self.fg_color = TEXT if self.active else TEXT_TERT
            out_c = BORDER

        self.normal_photo = get_rounded_rect_image(
            self.btn_w, self.btn_h, self.radius, fill=bg_c, outline=out_c, bg_color=self.parent_bg
        )
        self.hover_photo = get_rounded_rect_image(
            self.btn_w, self.btn_h, self.radius, fill=hover_c, outline=out_c, bg_color=self.parent_bg
        )

    def set_active(self, active: bool):
        self.active = active
        self._setup_images()
        self.configure(
            image=self.normal_photo,
            fg=self.fg_color,
            cursor="hand2" if active else "arrow"
        )

    def set_text(self, text: str):
        self.text_content = text
        self._calc_size()
        self._setup_images()
        self.configure(image=self.normal_photo, text=self.text_content)

    def _on_click(self, e):
        if self.active and self.command:
            self.command()

    def _on_enter(self, e):
        if self.active:
            self.configure(image=self.hover_photo)

    def _on_leave(self, e):
        if self.active:
            self.configure(image=self.normal_photo)


class SmoothEntry(tk.Canvas):
    """超平滑抗锯齿 iOS 内嵌圆角输入框 (带焦点 Apple 蓝光晕反馈，无双层黑框)"""

    def __init__(
        self,
        parent: tk.Widget,
        textvariable: tk.StringVar | None = None,
        width: int | None = None,
        height: int = 34,
        radius: int = 9,
        bg_color: str = INPUT_BG,
        border_color: str = INPUT_BORDER,
        font_size: int = 11,
        **kwargs
    ) -> None:
        self.parent_bg = parent.cget("bg") or SURFACE
        super().__init__(parent, bg=self.parent_bg, height=height, highlightthickness=0, **kwargs)

        self.textvariable = textvariable
        self.radius = radius
        self.input_fill = bg_color
        self.input_border = border_color
        self.entry_height = height
        self.is_focused = False

        self.normal_img = None
        self.focus_img = None
        self.bg_img_id = None

        self.entry = tk.Entry(
            self,
            textvariable=self.textvariable,
            bg=self.input_fill,
            fg=TEXT,
            insertbackground=ACCENT,
            relief="flat",
            bd=0,
            highlightthickness=0,
            width=width,
            font=(FAM, font_size),
        )
        self.window_id = self.create_window(12, 6, window=self.entry, anchor="nw")

        self.bind("<Configure>", self._on_resize)
        self.entry.bind("<FocusIn>", self._on_focus_in)
        self.entry.bind("<FocusOut>", self._on_focus_out)

    def _on_resize(self, event):
        w, h = event.width, event.height
        if w <= 8 or h <= 8:
            return

        self.normal_img = get_rounded_rect_image(
            w, h, self.radius, fill=self.input_fill, outline=self.input_border,
            outline_width=1.0, bg_color=self.parent_bg,
        )
        self.focus_img = get_rounded_rect_image(
            w, h, self.radius, fill=self.input_fill, outline=ACCENT,
            outline_width=1.5, bg_color=self.parent_bg,
        )

        curr = self.focus_img if self.is_focused else self.normal_img
        if self.bg_img_id is None:
            self.bg_img_id = self.create_image(0, 0, image=curr, anchor="nw")
            self.tag_lower(self.bg_img_id)
        else:
            self.itemconfig(self.bg_img_id, image=curr)

        entry_h = 22
        self.coords(self.window_id, 12, int((h - entry_h) / 2))
        self.itemconfigure(self.window_id, width=w - 24, height=entry_h)

    def _on_focus_in(self, event):
        self.is_focused = True
        if self.focus_img and self.bg_img_id is not None:
            self.itemconfig(self.bg_img_id, image=self.focus_img)

    def _on_focus_out(self, event):
        self.is_focused = False
        if self.normal_img and self.bg_img_id is not None:
            self.itemconfig(self.bg_img_id, image=self.normal_img)

    def get(self) -> str:
        return self.entry.get()

    def set(self, val: str) -> None:
        if self.textvariable:
            self.textvariable.set(val)
        else:
            self.entry.delete(0, "end")
            self.entry.insert(0, val)

    def configure_entry(self, **kwargs):
        self.entry.configure(**kwargs)


class SmoothSegmentedControl(tk.Canvas):
    """超平滑抗锯齿 iOS 分段控制器 (平滑底轨 + 浮动纯白圆角胶囊滑块)"""

    def __init__(
        self,
        parent: tk.Widget,
        tabs: Sequence[str],
        on_change: Callable[[int], None],
        height: int = 36,
        **kwargs
    ) -> None:
        self.parent_bg = parent.cget("bg") or BG
        super().__init__(parent, bg=self.parent_bg, height=height, highlightthickness=0, **kwargs)

        self.tabs = tabs
        self.on_change = on_change
        self.active_idx = 0
        self.ctrl_height = height
        self.font = tkfont.Font(family=FAM, size=11, weight="bold")

        self.bind("<Configure>", self._on_resize)
        self.bind("<Button-1>", self._on_click)

    def _on_resize(self, event):
        self._redraw()

    def _redraw(self):
        w, h = self.winfo_width(), self.ctrl_height
        if w <= 10 or h <= 10:
            return

        self.delete("all")

        # 1. 绘制外部平滑灰色圆角导轨
        bg_photo = get_rounded_rect_image(
            w, h, radius=10, fill="#E3E3E8", outline="#D5D5DA", outline_width=1.0, bg_color=self.parent_bg
        )
        self.create_image(0, 0, image=bg_photo, anchor="nw")

        n = len(self.tabs)
        seg_w = (w - 6) / n

        # 2. 绘制当前选中的纯白平滑浮起胶囊
        act_x1 = 3 + self.active_idx * seg_w
        pill_photo = get_rounded_rect_image(
            int(seg_w), h - 6, radius=8, fill="#FFFFFF", outline="#D0D0D5", outline_width=0.8, bg_color="#E3E3E8"
        )
        self.create_image(int(act_x1), 3, image=pill_photo, anchor="nw")

        # 3. 绘制各个 Tab 文字
        for i, text in enumerate(self.tabs):
            center_x = 3 + i * seg_w + seg_w / 2
            center_y = h / 2
            fg_color = TEXT if i == self.active_idx else TEXT_SEC
            self.create_text(center_x, center_y, text=text, fill=fg_color, font=self.font)

    def _on_click(self, event):
        w = self.winfo_width()
        n = len(self.tabs)
        if n == 0 or w <= 0:
            return

        seg_w = (w - 6) / n
        click_idx = int((event.x - 3) / seg_w)
        click_idx = max(0, min(n - 1, click_idx))

        if click_idx != self.active_idx:
            self.set_active(click_idx)
            if self.on_change:
                self.on_change(click_idx)

    def set_active(self, idx: int):
        self.active_idx = idx
        self._redraw()
