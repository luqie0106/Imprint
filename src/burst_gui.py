"""
burst_gui.py — RAW 连拍优选图形化界面
"""

from __future__ import annotations

import os
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable

# ── 确保 src 在 sys.path 上 ───────────────────────────────────────────────────
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from burst_filter import BurstFilter, BurstFilterResult
from model_manager import check_all_models, PROJECT_ROOT

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


def _pill_button(parent, text, command, big=False, active=True):
    size = 13 if big else 11
    px = 20 if big else 14
    py = 9 if big else 6
    bg_col = _ACCENT if active else _ACCENT_DIS
    cursor = "hand2" if active else "arrow"
    lbl = tk.Label(
        parent, text=text,
        bg=bg_col, fg="white",
        font=(_FAM, size, "bold"),
        cursor=cursor, padx=px, pady=py,
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
# 连拍优选界面
# ══════════════════════════════════════════════════════════════════════════════

class BurstFilterGUI:
    def __init__(self, parent: tk.Widget | None = None, on_navigate_tab: Callable[[int], None] | None = None) -> None:
        self.is_standalone = parent is None
        self.root = tk.Tk() if self.is_standalone else parent
        self.on_navigate_tab = on_navigate_tab

        if self.is_standalone:
            self.root.title("RAW 连拍优选")
            self.root.configure(bg=_BG)
            self.root.geometry("740x780")
            self.root.resizable(True, True)

        self.input_dir_var     = tk.StringVar()
        self.review_subdir_var = tk.StringVar(value="审查_连拍淘汰")
        self.gap_var           = tk.StringVar(value="1.5")
        self.hamming_var       = tk.StringVar(value="12")
        self.keep_count_var    = tk.StringVar(value="1")
        self.workers_var       = tk.StringVar(value=str(max(1, (os.cpu_count() or 4) // 2)))
        self.gpu_var           = tk.BooleanVar(value=True)
        self._running = False

        self._build_ui()
        self._apply_style()
        self.refresh_model_status()

    # ── 构建 UI ───────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        container = tk.Frame(self.root, bg=_BG)
        container.pack(fill="both", expand=True)

        # ── 独立运行时显示标题 ──
        if self.is_standalone:
            hdr = tk.Frame(container, bg=_BG, pady=18)
            hdr.pack(fill="x", padx=28)
            tk.Label(hdr, text="📷  RAW 连拍优选", bg=_BG, fg=_TEXT,
                     font=(_FAM_TITLE, 22, "bold")).pack(anchor="w")
            tk.Label(hdr, text="支持 NEF / ARW / CR3 / RAF · 智能保留最优帧 · 移动冗余片至审查目录",
                     bg=_BG, fg=_TEXT_DIM, font=(_FAM, 12)).pack(anchor="w", pady=(3, 0))
            tk.Frame(container, bg=_BORDER, height=1).pack(fill="x", padx=28)

        body = tk.Frame(container, bg=_BG)
        body.pack(fill="both", expand=True, padx=28 if self.is_standalone else 10, pady=16)

        # ── 模型状态栏卡片 ──
        mc = _card(body)
        mc.pack(fill="x", pady=(0, 12))
        self.model_status_frame = tk.Frame(mc, bg=_SURFACE, padx=16, pady=10)
        self.model_status_frame.pack(fill="x")

        self.model_status_lbl = tk.Label(
            self.model_status_frame, text="🔍 正在检测美学模型状态...",
            bg=_SURFACE, fg=_TEXT_DIM, font=(_FAM, 11)
        )
        self.model_status_lbl.pack(side="left")

        # ── 目录选择卡片 ──
        dc = _card(body)
        dc.pack(fill="x", pady=(0, 12))
        di = tk.Frame(dc, bg=_SURFACE, padx=16, pady=14)
        di.pack(fill="x")

        _label(di, "📁  RAW 文件目录", bold=True).pack(anchor="w")
        _label(di, "选择包含 NEF / ARW / CR3 / RAF 文件的文件夹", size=10,
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
        _label(pi, "可使用推荐默认值，无需频繁更改", size=10,
               color=_TEXT_DIM).pack(anchor="w", pady=(2, 10))

        g = tk.Frame(pi, bg=_SURFACE)
        g.pack(fill="x")
        self._param_row(g, 0, "淘汰子目录名称",   self.review_subdir_var, "相对于输入目录的子文件夹名（默认：审查_连拍淘汰）")
        self._param_row(g, 1, "连拍时间阈值（秒）", self.gap_var,           "前后间隔 ≤ 此値视为连拍候选（推荐 1.0~2.0 秒）")
        self._param_row(g, 2, "dHash 汉明限制",    self.hamming_var,       "64 位感知哈希最大差异位数（推荐 8～12～20）")
        self._param_row(g, 3, "每组保留张数",     self.keep_count_var,    "1 = 仅保留最优 1 张；填 2 则保留最优 2 张，以此类推")
        
        max_cpus = os.cpu_count() or 4
        self._workers_entry = self._param_row(g, 4, "最大并发线程数", self.workers_var, f"建议值不超过 {max_cpus} (物理核心数)")
        self.workers_var.trace_add("write", self._on_workers_changed)

        # GPU 勾选框
        gpu_row = tk.Frame(g, bg=_SURFACE)
        gpu_row.grid(row=10, column=0, columnspan=2, sticky="w", pady=(5, 0))
        tk.Label(gpu_row, text="硬件加速", bg=_SURFACE, fg=_TEXT_DIM, font=(_FAM, 11), anchor="w", width=16).pack(side="left")
        tk.Checkbutton(gpu_row, text="启用 显卡/NPU 硬件加速（CoreML / CUDA / MPS）",
                       variable=self.gpu_var, bg=_SURFACE, fg=_TEXT, font=(_FAM, 11),
                       activebackground=_SURFACE).pack(side="left")

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
        ent = _entry(parent, var, width=26)
        ent.grid(row=row * 2, column=1, sticky="we", ipady=4, padx=(0, 20))
        tk.Label(parent, text=hint, bg=_SURFACE, fg=_TEXT_DIM,
                 font=(_FAM, 10)).grid(
            row=row * 2 + 1, column=0, columnspan=2, sticky="w", pady=(0, 5))
        return ent

    def _on_workers_changed(self, *args):
        try:
            val = int(self.workers_var.get().strip())
            max_cpus = os.cpu_count() or 4
            if val > max_cpus:
                self._workers_entry.configure(fg=_ERROR)
                self.run_btn.configure(bg=_ACCENT_DIS, cursor="X_cursor")
            else:
                self._workers_entry.configure(fg=_TEXT)
                self.run_btn.configure(bg=_ACCENT, cursor="hand2")
        except ValueError:
            self._workers_entry.configure(fg=_ERROR)
            self.run_btn.configure(bg=_ACCENT_DIS, cursor="X_cursor")

    def _apply_style(self):
        s = ttk.Style()
        try:
            s.configure("Horizontal.TProgressbar", background=_ACCENT, thickness=6)
        except Exception:
            pass

    def refresh_model_status(self):
        """刷新模型状态指示器"""
        status = check_all_models()
        if status.onnx_ready:
            self.model_status_lbl.configure(
                text="🟢 AI 美学引擎：ONNX 极速加速已就绪 (photo_sort_model.onnx)",
                fg=_SUCCESS
            )
        elif status.mlp_ready:
            self.model_status_lbl.configure(
                text="🟡 AI 美学引擎：PyTorch 模式 (aesthetic_mlp.pth，建议熔铸 ONNX)",
                fg=_WARNING
            )
        else:
            self.model_status_lbl.configure(
                text="⚪ AI 美学引擎：未训练 (当前降级为纯 OpenCV 锐度 + 曝光筛选)",
                fg=_TEXT_DIM
            )

    # ── 事件 ─────────────────────────────────────────────────────────────────

    def _pick_dir(self):
        d = filedialog.askdirectory(title="选择 RAW 文件夹")
        if d:
            self.input_dir_var.set(d)

    def _on_run(self):
        if self._running:
            return

        input_dir = Path(self.input_dir_var.get().strip())
        if not input_dir.exists() or not input_dir.is_dir():
            messagebox.showerror("路径错误", "请先选择一个有效的 RAW 文件目录。")
            return

        try:
            gap     = float(self.gap_var.get().strip())
            hamming = int(self.hamming_var.get().strip())
            keep    = int(self.keep_count_var.get().strip())
            workers = int(self.workers_var.get().strip())
            use_gpu = self.gpu_var.get()
            
            assert 0.0 < gap <= 30.0
            assert 1 <= hamming <= 64
            assert keep >= 1
            
            max_cpus = os.cpu_count() or 4
            if workers > max_cpus:
                messagebox.showerror("线程数超限", f"为防止内存溢出导致闪退，最大线程数不得超过 {max_cpus}！")
                return
            assert workers >= 1

        except (ValueError, AssertionError):
            messagebox.showerror("参数错误", "请检查填写的参数。时间阈値 0~30，汉明限制 1~64，保留张数 ≥ 1，线程数必须为合法正整数且不可越界。")
            return

        subdir = self.review_subdir_var.get().strip() or "审查_连拍淘汰"
        self._set_running(True)
        threading.Thread(
            target=self._worker,
            args=(input_dir, gap, hamming, subdir, keep, workers, use_gpu),
            daemon=True,
        ).start()

    def _worker(self, input_dir, gap, hamming, subdir, keep, workers, use_gpu):
        try:
            flt = BurstFilter(
                gap_seconds=gap,
                max_hamming_distance=hamming,
                review_subdir=subdir,
                keep_count=keep,
                max_workers=workers,
                use_gpu=use_gpu,
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
            messagebox.showinfo("处理完成", "目录中未找到任何 RAW 文件（NEF/ARW/CR3/RAF）。")
            return

        lines = [
            f"总 RAW 文件数：    {r.total}",
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

    def run(self):
        if self.is_standalone:
            self.root.mainloop()


def launch_burst_gui() -> None:
    BurstFilterGUI().run()


if __name__ == "__main__":
    launch_burst_gui()
