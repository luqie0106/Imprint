"""
app_gui.py — Photo Sort 综合一体化主程序界面
整合连拍筛选、偏好训练与模型管理
"""

from __future__ import annotations

import os
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

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
    ONNX_MODEL_PATH,
    check_all_models,
    download_clip_model,
    is_clip_model_downloaded,
    is_clip_in_hf_cache,
    import_from_hf_cache,
)
from onnx_exporter import export_to_onnx, TORCH_EXPORT_AVAILABLE

# ── 颜色常量 ───────────────────────────────────────────────────────────────────
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
_TAB_BG  = "#F1F5F9"
_TAB_ACTIVE_BG = "#FFFFFF"

_FAM = "SF Pro Text" if sys.platform == "darwin" else "Segoe UI"
_FAM_TITLE = "SF Pro Display" if sys.platform == "darwin" else "Segoe UI"


class MainAppGUI:
    """Photo Sort 综合应用主窗口"""

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Photo Sort — RAW 智能连拍优选与个人审美系统")
        self.root.configure(bg=_BG)
        # 紧凑尺寸，完美兼容 1080p 及高分屏缩放 (100% ~ 150%)
        self.root.geometry("760x600")
        self.root.minsize(700, 520)

        self._active_tab_idx = 0
        self._tab_buttons: list[tk.Label] = []
        self._tab_frames: list[tk.Frame] = []

        # 状态变量
        self.use_mirror_var = tk.BooleanVar(value=True)
        self._downloading_model = False
        self._cancel_download_event = threading.Event()

        self._build_shell()
        self._build_tabs()
        self._switch_tab(0)

        # 启动后异步检查基础模型
        self.root.after(300, self._check_startup_models)

    # ── 构建外层框架与顶部导航 ─────────────────────────────────────────────────

    def _build_shell(self) -> None:
        # ── 顶部 Header ──
        hdr = tk.Frame(self.root, bg=_BG, pady=10)
        hdr.pack(fill="x", padx=20)

        title_row = tk.Frame(hdr, bg=_BG)
        title_row.pack(fill="x")
        tk.Label(title_row, text="📸  Photo Sort", bg=_BG, fg=_TEXT,
                 font=(_FAM_TITLE, 20, "bold")).pack(side="left")
        tk.Label(title_row, text="RAW 连拍优选 · 审美微调 · ONNX 极速推理",
                 bg=_BG, fg=_TEXT_DIM, font=(_FAM, 11)).pack(side="left", padx=(10, 0), pady=(4, 0))

        # ── 导航栏 (Tabs Switcher) ──
        nav_container = tk.Frame(self.root, bg=_TAB_BG, highlightthickness=1,
                                 highlightbackground=_BORDER)
        nav_container.pack(fill="x", padx=20, pady=(2, 6))

        tabs_info = [
            ("📷  连拍优选", 0),
            ("🧠  偏好训练", 1),
            ("📦  模型与环境", 2),
        ]

        nav_inner = tk.Frame(nav_container, bg=_TAB_BG, padx=3, pady=3)
        nav_inner.pack(fill="x")

        for label_text, idx in tabs_info:
            btn = tk.Label(
                nav_inner, text=label_text, bg=_TAB_BG, fg=_TEXT_DIM,
                font=(_FAM, 11, "bold"), cursor="hand2", padx=16, pady=6,
            )
            btn.bind("<Button-1>", lambda e, i=idx: self._switch_tab(i))
            btn.pack(side="left", padx=2)
            self._tab_buttons.append(btn)

        # ── 内容主区域 (采用 grid 堆叠，无闪烁且防白屏) ──
        self.content_area = tk.Frame(self.root, bg=_BG)
        self.content_area.pack(fill="both", expand=True)
        self.content_area.grid_rowconfigure(0, weight=1)
        self.content_area.grid_columnconfigure(0, weight=1)

    # ── 构建各个标签页 ─────────────────────────────────────────────────────────

    def _build_tabs(self) -> None:
        # 1. 连拍优选面板
        self.burst_frame = tk.Frame(self.content_area, bg=_BG)
        self.burst_frame.grid(row=0, column=0, sticky="nsew")
        self.burst_gui = BurstFilterGUI(
            parent=self.burst_frame,
            on_navigate_tab=self._switch_tab,
        )
        self._tab_frames.append(self.burst_frame)

        # 2. 偏好训练面板
        self.trainer_frame = tk.Frame(self.content_area, bg=_BG)
        self.trainer_frame.grid(row=0, column=0, sticky="nsew")
        self.trainer_gui = TrainerGUI(
            parent=self.trainer_frame,
            on_model_updated=self._on_model_updated,
        )
        self._tab_frames.append(self.trainer_frame)

        # 3. 模型管理面板
        self.model_mgr_frame = tk.Frame(self.content_area, bg=_BG)
        self.model_mgr_frame.grid(row=0, column=0, sticky="nsew")
        self._build_model_manager_tab(self.model_mgr_frame)
        self._tab_frames.append(self.model_mgr_frame)

    # ── 构建“模型与环境”标签页 ────────────────────────────────────────────────

    def _build_model_manager_tab(self, parent: tk.Frame) -> None:
        container = tk.Frame(parent, bg=_BG)
        container.pack(fill="both", expand=True, padx=20, pady=10)

        tk.Label(container, text="模型文件状态与本地管理", bg=_BG, fg=_TEXT,
                 font=(_FAM_TITLE, 14, "bold")).pack(anchor="w")
        tk.Label(container, text="模型文件存放于程序目录下的 models/ 文件夹中，支持离线与跨设备便携使用",
                 bg=_BG, fg=_TEXT_DIM, font=(_FAM, 10)).pack(anchor="w", pady=(1, 8))

        # ── 卡片 1：CLIP 基础视觉模型 ──
        c1 = self._create_card(container)
        c1.pack(fill="x", pady=(0, 8))
        i1 = tk.Frame(c1, bg=_SURFACE, padx=14, pady=8)
        i1.pack(fill="x")

        tk.Label(i1, text="1. 基础视觉主干模型 (CLIP ViT-B/32)", bg=_SURFACE, fg=_TEXT,
                 font=(_FAM, 11, "bold")).pack(anchor="w")

        self.clip_status_label = tk.Label(i1, text="", bg=_SURFACE, font=(_FAM, 10, "bold"))
        self.clip_status_label.pack(anchor="w", pady=(2, 6))

        dl_row = tk.Frame(i1, bg=_SURFACE)
        dl_row.pack(fill="x")

        self.dl_clip_btn = self._make_button(dl_row, "⬇️  下载/补全到本地", self._start_download_clip)
        self.dl_clip_btn.pack(side="left")

        tk.Checkbutton(
            dl_row, text="使用国内加速镜像 (hf-mirror.com)", variable=self.use_mirror_var,
            bg=_SURFACE, fg=_TEXT, font=(_FAM, 10), activebackground=_SURFACE,
        ).pack(side="left", padx=12)

        self.dl_progress = ttk.Progressbar(i1, mode="determinate", length=240)
        self.dl_status_lbl = tk.Label(i1, text="", bg=_SURFACE, fg=_TEXT_DIM, font=(_FAM, 9))

        # ── 卡片 2：个人偏好 MLP 权重 ──
        c2 = self._create_card(container)
        c2.pack(fill="x", pady=(0, 8))
        i2 = tk.Frame(c2, bg=_SURFACE, padx=14, pady=8)
        i2.pack(fill="x")

        tk.Label(i2, text="2. 个人审美偏好分类头 (aesthetic_mlp.pth)", bg=_SURFACE, fg=_TEXT,
                 font=(_FAM, 11, "bold")).pack(anchor="w")

        self.mlp_status_label = tk.Label(i2, text="", bg=_SURFACE, font=(_FAM, 10, "bold"))
        self.mlp_status_label.pack(anchor="w", pady=(2, 6))

        mlp_row = tk.Frame(i2, bg=_SURFACE)
        mlp_row.pack(fill="x")
        self._make_button(mlp_row, "🎯  前往偏好训练", lambda: self._switch_tab(1)).pack(side="left")

        # ── 卡片 3：ONNX 极速加速模型 ──
        c3 = self._create_card(container)
        c3.pack(fill="x", pady=(0, 8))
        i3 = tk.Frame(c3, bg=_SURFACE, padx=14, pady=8)
        i3.pack(fill="x")

        tk.Label(i3, text="3. ONNX 端到端融合模型 (photo_sort_model.onnx)", bg=_SURFACE, fg=_TEXT,
                 font=(_FAM, 11, "bold")).pack(anchor="w")

        self.onnx_status_label = tk.Label(i3, text="", bg=_SURFACE, font=(_FAM, 10, "bold"))
        self.onnx_status_label.pack(anchor="w", pady=(2, 6))

        onnx_row = tk.Frame(i3, bg=_SURFACE)
        onnx_row.pack(fill="x")
        self.export_onnx_btn = self._make_button(
            onnx_row, "⚡  从当前权重重新熔铸 ONNX", self._on_manual_export_onnx
        )
        self.export_onnx_btn.pack(side="left")

        self.refresh_model_mgr_ui()

    # ── 辅助 UI 工具 ──────────────────────────────────────────────────────────

    def _create_card(self, parent: tk.Widget) -> tk.Frame:
        return tk.Frame(parent, bg=_SURFACE, highlightthickness=1, highlightbackground=_BORDER)

    def _make_button(self, parent, text, command, active=True):
        lbl = tk.Label(
            parent, text=text,
            bg=_ACCENT if active else _ACCENT_DIS, fg="white",
            font=(_FAM, 10, "bold"), cursor="hand2" if active else "arrow",
            padx=12, pady=5,
        )

        def _click(e):
            if lbl.cget("bg") != _ACCENT_DIS:
                command()

        lbl.bind("<Button-1>", _click)
        lbl.bind("<Enter>", lambda e: lbl.configure(bg=_ACCENT_HOVER) if lbl.cget("bg") != _ACCENT_DIS else None)
        lbl.bind("<Leave>", lambda e: lbl.configure(bg=_ACCENT) if lbl.cget("bg") != _ACCENT_DIS else None)
        return lbl

    # ── 标签页切换 ─────────────────────────────────────────────────────────────

    def _switch_tab(self, idx: int) -> None:
        self._active_tab_idx = idx
        for i, btn in enumerate(self._tab_buttons):
            if i == idx:
                btn.configure(bg=_TAB_ACTIVE_BG, fg=_ACCENT)
            else:
                btn.configure(bg=_TAB_BG, fg=_TEXT_DIM)

        self._tab_frames[idx].tkraise()

        if idx == 0:
            self.burst_gui.refresh_model_status()
        elif idx == 2:
            self.refresh_model_mgr_ui()

    # ── 模型状态刷新 ───────────────────────────────────────────────────────────

    def _on_model_updated(self) -> None:
        self.burst_gui.refresh_model_status()
        self.refresh_model_mgr_ui()

    def refresh_model_mgr_ui(self) -> None:
        status = check_all_models()

        # 1. CLIP
        if status.clip_location == "local":
            size_mb = sum(f.stat().st_size for f in CLIP_MODEL_DIR.glob("**/*") if f.is_file()) / (1024 * 1024)
            self.clip_status_label.configure(
                text=f"✅ 已就绪 (项目本地目录: models/clip-vit-base-patch32, 共 {size_mb:.1f} MB)",
                fg=_SUCCESS
            )
            self.dl_clip_btn.configure(text="🔄  重新校验/更新")
        elif status.clip_location == "hf_cache":
            self.clip_status_label.configure(
                text="✅ 已在系统 HuggingFace 缓存中就绪 (~/.cache/huggingface/hub/，可直接离线使用)",
                fg=_SUCCESS
            )
            self.dl_clip_btn.configure(text="📥  秒级同步至项目 models/ 目录")
        else:
            self.clip_status_label.configure(
                text="❌ 未下载（训练或无 ONNX 时将从远程 HuggingFace 镜像下载）",
                fg=_ERROR
            )
            self.dl_clip_btn.configure(text="⬇️  一键下载至本地 models/ 目录")

        # 2. MLP
        if status.mlp_ready:
            p = Path(status.mlp_path)
            size_kb = p.stat().st_size / 1024 if p.exists() else 0
            self.mlp_status_label.configure(
                text=f"✅ 已就绪 (文件: {p.name}, 大小: {size_kb:.1f} KB)",
                fg=_SUCCESS
            )
        else:
            self.mlp_status_label.configure(
                text="⚪ 未训练 (暂无个性化权重，可在“偏好训练”中导入照片训练)",
                fg=_TEXT_DIM
            )

        # 3. ONNX
        if status.onnx_ready:
            p = Path(status.onnx_path)
            size_mb = p.stat().st_size / (1024 * 1024) if p.exists() else 0
            self.onnx_status_label.configure(
                text=f"✅ 已就绪 (文件: {p.name}, 大小: {size_mb:.1f} MB, 支持极速硬件加速)",
                fg=_SUCCESS
            )
            self.export_onnx_btn.configure(text="⚡  重新熔铸 ONNX", bg=_ACCENT, cursor="hand2")
        else:
            if status.mlp_ready:
                self.onnx_status_label.configure(
                    text="🟡 待熔铸 (已检测到 MLP 权重，点击下方按钮即可一键熔铸 ONNX)",
                    fg=_WARNING
                )
                self.export_onnx_btn.configure(text="⚡  一键熔铸为 ONNX 模型", bg=_ACCENT, cursor="hand2")
            else:
                self.onnx_status_label.configure(
                    text="⚪ 未生成 (请先在“偏好训练”中完成审美微调)",
                    fg=_TEXT_DIM
                )
                self.export_onnx_btn.configure(bg=_ACCENT_DIS, cursor="arrow")

    # ── 启动时模型自检 ─────────────────────────────────────────────────────────

    def _check_startup_models(self) -> None:
        status = check_all_models()
        if not status.clip_ready and not status.onnx_ready:
            ans = messagebox.askyesno(
                "检测到未下载基础模型",
                "欢迎使用 Photo Sort！\n\n检测到尚未下载 CLIP 基础视觉模型。\n"
                "下载后可完全离线支持照片特征提取与个人审美训练。\n\n"
                "是否立即下载至本地 ./models/ 目录？（约 340MB，推荐下载）"
            )
            if ans:
                self._switch_tab(2)
                self._start_download_clip()

    # ── 下载与导出事件 ─────────────────────────────────────────────────────────

    def _start_download_clip(self) -> None:
        if self._downloading_model:
            return

        self._downloading_model = True
        self.dl_clip_btn.configure(bg=_ACCENT_DIS, cursor="arrow")
        self.dl_progress.pack(fill="x", pady=(6, 2))
        self.dl_progress["value"] = 0
        self.dl_status_lbl.pack(anchor="w")
        self.dl_status_lbl.configure(text="正在准备同步/下载...")

        use_mirror = self.use_mirror_var.get()
        self._cancel_download_event.clear()

        def _task():
            def _cb(msg: str, pct: float):
                self.root.after(0, lambda: self.dl_status_lbl.configure(text=msg))
                self.root.after(0, lambda: self.dl_progress.configure(value=int(pct * 100)))

            try:
                success = download_clip_model(
                    use_mirror=use_mirror,
                    progress_callback=_cb,
                    cancel_event=self._cancel_download_event,
                )
                if success:
                    self.root.after(0, lambda: messagebox.showinfo(
                        "同步完成", "CLIP 基础视觉模型已成功就绪并保存在 ./models/clip-vit-base-patch32 目录！"
                    ))
            except Exception as exc:
                self.root.after(0, lambda e=str(exc): messagebox.showerror("下载失败", f"模型下载过程中出错:\n{e}"))
            finally:
                self.root.after(0, self._finish_download_clip)

        threading.Thread(target=_task, daemon=True).start()

    def _finish_download_clip(self) -> None:
        self._downloading_model = False
        self.dl_clip_btn.configure(bg=_ACCENT, cursor="hand2")
        self.dl_progress.pack_forget()
        self.dl_status_lbl.pack_forget()
        self._on_model_updated()

    def _on_manual_export_onnx(self) -> None:
        if not MLP_WEIGHTS_PATH.exists() and not (BUNDLE_ROOT / "aesthetic_mlp.pth").exists():
            messagebox.showwarning("无法导出", "未找到 aesthetic_mlp.pth 权重文件，请先进行偏好训练。")
            return

        if not TORCH_EXPORT_AVAILABLE:
            messagebox.showerror("环境缺失", "导出 ONNX 需要 PyTorch 和 transformers，请在 py311 环境下运行。")
            return

        self.export_onnx_btn.configure(bg=_ACCENT_DIS, cursor="arrow")

        def _task():
            try:
                export_to_onnx(project_root=PROJECT_ROOT)
                self.root.after(0, lambda: messagebox.showinfo(
                    "熔铸成功", "photo_sort_model.onnx 已生成！连拍优选已自动启用极速硬件加速！"
                ))
            except Exception as exc:
                self.root.after(0, lambda e=str(exc): messagebox.showerror("导出失败", f"ONNX 熔铸失败: {e}"))
            finally:
                self.root.after(0, lambda: self.export_onnx_btn.configure(bg=_ACCENT, cursor="hand2"))
                self.root.after(0, self._on_model_updated)

        threading.Thread(target=_task, daemon=True).start()

    def run(self) -> None:
        self.root.mainloop()


def launch_main_gui() -> None:
    MainAppGUI().run()


if __name__ == "__main__":
    launch_main_gui()
