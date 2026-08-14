"""
burst_gui.py — NEF 连拍优选图形化界面

独立运行：
    python src/burst_gui.py
"""

from __future__ import annotations

import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

# ── 确保 src 在 sys.path 上 ───────────────────────────────────────────────────
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from burst_filter import BurstFilter, BurstFilterResult  # noqa: E402

# ══════════════════════════════════════════════════════════════════════════════
# 颜色 & 字体常量
# ══════════════════════════════════════════════════════════════════════════════
_BG      = "#F8FAFC"
_SURFACE = "#FFFFFF"
_ACCENT  = "#2563EB"
_ACCENT_HOVER = "#1D4ED8"
_ACCENT_DIS   = "#CBD5E1"
_SUCCESS = "#16A34A"
_WARNING = "#D97706"
_ERROR   = "#DC2626"
_TEXT    = "#0F172A"
_TEXT_DIM = "#64748B"
_BORDER  = "#E2E8F0"

_FAM = "SF Pro Text" if sys.platform == "darwin" else "Segoe UI"
_FAM_TITLE = "SF Pro Display" if sys.platform == "darwin" else "Segoe UI"


# ══════════════════════════════════════════════════════════════════════════════
# 小工具
# ══════════════════════════════════════════════════════════════════════════════

def _entry(parent, var, **kw):
    return tk.Entry(
        parent, textvariable=var,
        bg=_SURFACE, fg=_TEXT, insertbackground=_TEXT,
        relief="flat", bd=0,
        highlightthickness=1,
        highlightbackground=_BORDER,
        highlightcolor=_ACCENT,
        font=(_FAM, 12), **kw,
    )


def _label(parent, text, size=12, color=None, bold=False):
    weight = "bold" if bold else "normal"
    return tk.Label(
        parent, text=text,
        bg=parent.cget("bg"),
        fg=color or _TEXT,
        font=(_FAM, size, weight),
    )


def _card(parent):
    return tk.Frame(parent, bg=_SURFACE, highlightthickness=1,
                    highlightbackground=_BORDER)


def _pill_button(parent, text, command, big=False):
    size = 13 if big else 11
    px = 20 if big else 14
    py = 9 if big else 6
    lbl = tk.Label(
        parent, text=text,
        bg=_ACCENT, fg="white",
        font=(_FAM, size, "bold"),
        cursor="hand2", padx=px, pady=py,
    )

    def _click(e):
        if lbl.cget("bg") != _ACCENT_DIS:
            command()

    lbl.bind("<Button-1>", _click)
    lbl.bind("<Enter>", lambda e: lbl.configure(bg=_ACCENT_HOVER)
             if lbl.cget("bg") != _ACCENT_DIS else None)
    lbl.bind("<Leave>", lambda e: lbl.configure(bg=_ACCENT)
             if lbl.cget("bg") != _ACCENT_DIS else None)
    return lbl


# ══════════════════════════════════════════════════════════════════════════════
# 主窗口
# ══════════════════════════════════════════════════════════════════════════════

class BurstFilterGUI:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("NEF 连拍优选")
        self.root.configure(bg=_BG)
        self.root.geometry("720x560")
        self.root.resizable(False, False)

        self.input_dir_var    = tk.StringVar()
        self.review_subdir_var = tk.StringVar(value="审查_连拍淘汰")
        self.gap_var          = tk.StringVar(value="1.5")
        self.similarity_var   = tk.StringVar(value="0.85")
        self._running = False

        self._build_ui()
        self._apply_style()

    # ── 构建 UI ───────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # ── 标题 ──
        hdr = tk.Frame(self.root, bg=_BG, pady=18)
        hdr.pack(fill="x", padx=28)
        tk.Label(hdr, text="📷  NEF 连拍优选", bg=_BG, fg=_TEXT,
                 font=(_FAM_TITLE, 22, "bold")).pack(anchor="w")
        tk.Label(hdr, text="自动识别连拍组 · 保留最清晰帧 · 移动冗余片至审查目录",
                 bg=_BG, fg=_TEXT_DIM, font=(_FAM, 12)).pack(anchor="w", pady=(3, 0))
        tk.Frame(self.root, bg=_BORDER, height=1).pack(fill="x", padx=28)

        body = tk.Frame(self.root, bg=_BG)
        body.pack(fill="both", expand=True, padx=28, pady=20)

        # ── 目录选择卡片 ──
        dc = _card(body)
        dc.pack(fill="x", pady=(0, 12))
        di = tk.Frame(dc, bg=_SURFACE, padx=16, pady=14)
        di.pack(fill="x")

        _label(di, "📁  NEF 文件目录", bold=True).pack(anchor="w")
        _label(di, "选择包含 .NEF 文件的文件夹", size=10,
               color=_TEXT_DIM).pack(anchor="w", pady=(2, 8))

        row = tk.Frame(di, bg=_SURFACE)
        row.pack(fill="x")
        _entry(row, self.input_dir_var).pack(side="left", fill="x",
                                             expand=True, ipady=6, padx=(0, 10))
        _pill_button(row, "选择目录", self._pick_dir).pack(side="left")

        # ── 参数卡片 ──
        pc = _card(body)
        pc.pack(fill="x", pady=(0, 16))
        pi = tk.Frame(pc, bg=_SURFACE, padx=16, pady=14)
        pi.pack(fill="x")

        _label(pi, "⚙️  处理参数", bold=True).pack(anchor="w")
        _label(pi, "可使用默认值，无需更改", size=10,
               color=_TEXT_DIM).pack(anchor="w", pady=(2, 10))

        g = tk.Frame(pi, bg=_SURFACE)
        g.pack(fill="x")
        self._param_row(g, 0, "淘汰子目录名称",   self.review_subdir_var, "相对于输入目录的子文件夹名")
        self._param_row(g, 1, "连拍时间阈值（秒）", self.gap_var,           "前后间隔 ≤ 此值视为连拍候选")
        self._param_row(g, 2, "视觉相似度阈值",   self.similarity_var,    "直方图相关系数阈值，0~1，越高越严格")

        # ── 执行行 ──
        br = tk.Frame(body, bg=_BG)
        br.pack(fill="x")
        self.run_btn = _pill_button(br, "▶  开始筛选", self._on_run, big=True)
        self.run_btn.pack(side="left")
        self.progress = ttk.Progressbar(br, mode="indeterminate", length=160)
        self.progress.pack(side="left", padx=(14, 0))
        self.status_lbl = tk.Label(br, text="", bg=_BG, fg=_TEXT_DIM,
                                   font=(_FAM, 11))
        self.status_lbl.pack(side="left", padx=(12, 0))

    def _param_row(self, parent, row, label, var, hint):
        tk.Label(parent, text=label, bg=_SURFACE, fg=_TEXT_DIM,
                 font=(_FAM, 11), anchor="w").grid(
            row=row * 2, column=0, sticky="w", padx=(0, 12), pady=(0, 2))
        _entry(parent, var, width=26).grid(
            row=row * 2, column=1, sticky="we", ipady=5, padx=(0, 20))
        tk.Label(parent, text=hint, bg=_SURFACE, fg=_TEXT_DIM,
                 font=(_FAM, 10)).grid(
            row=row * 2 + 1, column=0, columnspan=2, sticky="w", pady=(0, 8))

    def _apply_style(self):
        s = ttk.Style()
        s.theme_use("default")
        s.configure("Horizontal.TProgressbar",
                    troughcolor=_SURFACE, background=_ACCENT, thickness=6)

    # ── 事件 ─────────────────────────────────────────────────────────────────

    def _pick_dir(self):
        d = filedialog.askdirectory(title="选择 NEF 文件夹")
        if d:
            self.input_dir_var.set(d)

    def _on_run(self):
        if self._running:
            return

        input_dir = Path(self.input_dir_var.get().strip())
        if not input_dir.exists() or not input_dir.is_dir():
            messagebox.showerror("路径错误", "请先选择一个有效的 NEF 文件目录。")
            return

        try:
            gap = float(self.gap_var.get().strip())
            sim = float(self.similarity_var.get().strip())
            assert 0.0 < gap <= 30.0
            assert 0.0 < sim <= 1.0
        except (ValueError, AssertionError):
            messagebox.showerror("参数错误", "时间阈值范围 0~30，相似度阈值范围 0~1。")
            return

        subdir = self.review_subdir_var.get().strip() or "审查_连拍淘汰"
        self._set_running(True)
        threading.Thread(
            target=self._worker,
            args=(input_dir, gap, sim, subdir),
            daemon=True,
        ).start()

    def _worker(self, input_dir, gap, sim, subdir):
        try:
            flt = BurstFilter(
                gap_seconds=gap,
                similarity_threshold=sim,
                review_subdir=subdir,
                progress_callback=lambda msg: self.root.after(
                    0, self._set_status, msg),
            )
            result = flt.run(input_dir)
            self.root.after(0, self._on_done, result)
        except Exception as exc:
            self.root.after(0, self._on_error, str(exc))
        finally:
            self.root.after(0, self._set_running, False)

    # ── 结果回调 ──────────────────────────────────────────────────────────────

    def _on_done(self, r: BurstFilterResult):
        self._set_status("完成")

        if r.total == 0:
            messagebox.showinfo("处理完成", "目录中未找到任何 NEF 文件。")
            return

        # 拼接统计文字
        lines = [
            f"总 NEF 文件数：    {r.total}",
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
            for e in r.errors[:5]:          # 最多显示前 5 条
                lines.append(f"  · {e}")
            if len(r.errors) > 5:
                lines.append(f"  … 共 {len(r.errors)} 条，请查看控制台")

        msg = "\n".join(lines)

        if r.errors:
            messagebox.showwarning("处理完成（含警告）", msg)
        else:
            messagebox.showinfo("处理完成", msg)

    def _on_error(self, message: str):
        self._set_status("出错")
        messagebox.showerror("运行失败", message)

    # ── 状态控制 ──────────────────────────────────────────────────────────────

    def _set_status(self, text: str):
        self.status_lbl.configure(text=text)

    def _set_running(self, running: bool):
        self._running = running
        if running:
            self.run_btn.configure(bg=_ACCENT_DIS, cursor="arrow")
            self.progress.start(12)
            self._set_status("处理中…")
        else:
            self.run_btn.configure(bg=_ACCENT, cursor="hand2")
            self.progress.stop()

    # ── 主循环 ────────────────────────────────────────────────────────────────

    def run(self):
        self.root.mainloop()


# ══════════════════════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════════════════════

def launch_burst_gui() -> None:
    BurstFilterGUI().run()


if __name__ == "__main__":
    launch_burst_gui()
