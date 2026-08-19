"""
trainer_gui.py — 个人审美偏好训练器与 ONNX 自动熔铸
支持内置 PyTorch 训练与通用 Conda / Python 环境智能探测调度
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable

# ── 确保 src 在 sys.path ───────────────────────────────────────────────────────
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from model_manager import (
    PROJECT_ROOT,
    BUNDLE_ROOT,
    get_clip_model_path,
    is_clip_model_downloaded,
    download_clip_model,
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

_FAM = "SF Pro Text" if sys.platform == "darwin" else "Segoe UI"
_FAM_TITLE = "SF Pro Display" if sys.platform == "darwin" else "Segoe UI"

# ── 动态探测 PyTorch 与 Transformers ─────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import rawpy
    from PIL import Image
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

_ALL_AVAILABLE = TORCH_AVAILABLE and CLIP_AVAILABLE


# ══════════════════════════════════════════════════════════════════════════════
# 数据集与模型逻辑 (仅在依赖齐全时可用)
# ══════════════════════════════════════════════════════════════════════════════

if _ALL_AVAILABLE:
    def _extract_preview(path: Path) -> Image.Image:
        try:
            with rawpy.imread(str(path)) as raw:
                thumb = raw.extract_thumb()
            if thumb.format == rawpy.ThumbFormat.JPEG:
                img = Image.open(io.BytesIO(thumb.data))
            else:
                img = Image.fromarray(thumb.data)
            return img.convert("RGB")
        except Exception:
            pass
        with rawpy.imread(str(path)) as raw:
            arr = raw.postprocess(half_size=True, use_camera_wb=True, output_bps=8)
        return Image.fromarray(arr).convert("RGB")

    _RAW_GLOB_PATTERNS = ["*.[nN][eE][fF]", "*.[aA][rR][wW]",
                          "*.[cC][rR]3", "*.[rR][aA][fF]"]

    class RawAestheticDataset(Dataset):
        def __init__(self, root_dir: Path, clip_processor, clip_model, device):
            self.samples: list[tuple[Path, int]] = []
            self._processor = clip_processor
            self._clip = clip_model
            self._device = device

            for label, subdir in ((1, "like"), (0, "dislike")):
                d = root_dir / subdir
                if not d.exists():
                    continue
                for pat in _RAW_GLOB_PATTERNS:
                    for p in d.glob(pat):
                        self.samples.append((p, label))

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            path, label = self.samples[idx]
            try:
                img = _extract_preview(path)
            except Exception:
                img = Image.new("RGB", (224, 224), (0, 0, 0))

            inputs = self._processor(images=img, return_tensors="pt", padding=True)
            inputs = {k: v.squeeze(0) for k, v in inputs.items()}
            return inputs, label

    def collate_fn(batch):
        inputs_list, labels = zip(*batch)
        collated = {
            k: torch.stack([d[k] for d in inputs_list])
            for k in inputs_list[0]
        }
        return collated, torch.tensor(labels, dtype=torch.long)

    class AestheticMLP(nn.Module):
        def __init__(self, input_dim: int = 512):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 2),
            )

        def forward(self, x):
            return self.net(x)


# ══════════════════════════════════════════════════════════════════════════════
# 通用环境自动发现逻辑（不硬编码环境名称）
# ══════════════════════════════════════════════════════════════════════════════

def discover_python_environments() -> list[str]:
    """
    通用、自动发现本机所有 Conda 环境及 Python 解释器。
    不硬编码任何具体环境名称，自动解析 ~/.conda/environments.txt 及各 Conda 根目录下的所有子环境。
    """
    found: list[str] = []
    seen: set[str] = set()

    def _add_env(path: Path):
        if path.is_file() and path.name.lower().startswith("python"):
            exe = path
        else:
            exe = path / ("python.exe" if sys.platform == "win32" else "bin/python")

        if exe.exists() and str(exe) not in seen:
            seen.add(str(exe))
            found.append(str(exe))

    home = Path.home()

    # 1. 解析 Conda 全局环境记录列表 (~/.conda/environments.txt)
    env_txt = home / ".conda" / "environments.txt"
    if env_txt.exists():
        try:
            for line in env_txt.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if line and Path(line).exists():
                    _add_env(Path(line))
        except Exception:
            pass

    # 2. 遍历常见 Conda 安装根目录下的 envs/ 文件夹中的所有子环境
    conda_bases = [
        home / "anaconda3",
        home / "miniconda3",
        home / ".conda",
        home / "miniforge3",
        home / "mambaforge",
        Path("C:/ProgramData/anaconda3"),
        Path("C:/ProgramData/miniconda3"),
        Path("C:/ProgramData/miniforge3"),
        Path("C:/anaconda3"),
        Path("C:/miniconda3"),
        Path("/opt/homebrew/anaconda3"),
        Path("/opt/homebrew/Caskroom/miniconda/base"),
        Path("/opt/anaconda3"),
        Path("/opt/miniconda3"),
    ]

    for base in conda_bases:
        if base.exists():
            _add_env(base)
            envs_dir = base / "envs"
            if envs_dir.exists() and envs_dir.is_dir():
                try:
                    for sub in envs_dir.iterdir():
                        if sub.is_dir():
                            _add_env(sub)
                except Exception:
                    pass

    # 3. 系统 PATH 中的 python
    for cmd in ["python3", "python"]:
        w = shutil.which(cmd)
        if w:
            _add_env(Path(w))

    # 4. 排序：优先排查带有 torch/transformers 依赖的环境置顶
    def _rank_score(p_str: str) -> int:
        score = 0
        p = Path(p_str)
        env_root = p.parent if sys.platform == "win32" else p.parent.parent
        sp_candidates = [
            env_root / "Lib" / "site-packages" / "torch",
            env_root / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "torch",
        ]
        if any(c.exists() for c in sp_candidates):
            score += 20
        low = p_str.lower()
        if "photo" in low or "torch" in low or "ai" in low or "py3" in low:
            score += 5
        return score

    found.sort(key=_rank_score, reverse=True)
    return found


# ══════════════════════════════════════════════════════════════════════════════
# 小工具
# ══════════════════════════════════════════════════════════════════════════════

def _entry(parent, var, **kw):
    return tk.Entry(
        parent, textvariable=var,
        bg=_SURFACE, fg=_TEXT, insertbackground=_TEXT,
        relief="flat", bd=0, highlightthickness=1,
        highlightbackground=_BORDER, highlightcolor=_ACCENT,
        font=(_FAM, 11), **kw,
    )


def _label(parent, text, size=11, color=None, bold=False):
    weight = "bold" if bold else "normal"
    return tk.Label(
        parent, text=text, bg=parent.cget("bg"),
        fg=color or _TEXT, font=(_FAM, size, weight),
    )


def _card(parent):
    return tk.Frame(parent, bg=_SURFACE, highlightthickness=1, highlightbackground=_BORDER)


def _pill_button(parent, text, command, big=False, active=True):
    size = 12 if big else 10
    px = 18 if big else 12
    py = 7 if big else 5
    bg_color = _ACCENT if active else _ACCENT_DIS
    cursor = "hand2" if active else "arrow"

    lbl = tk.Label(
        parent, text=text, bg=bg_color, fg="white",
        font=(_FAM, size, "bold"), cursor=cursor, padx=px, pady=py,
    )

    def _click(e):
        if lbl.cget("bg") != _ACCENT_DIS:
            command()

    lbl.bind("<Button-1>", _click)
    lbl.bind("<Enter>", lambda e: lbl.configure(bg=_ACCENT_HOVER) if lbl.cget("bg") != _ACCENT_DIS else None)
    lbl.bind("<Leave>", lambda e: lbl.configure(bg=_ACCENT) if lbl.cget("bg") != _ACCENT_DIS else None)
    return lbl


# ══════════════════════════════════════════════════════════════════════════════
# 训练界面组件
# ══════════════════════════════════════════════════════════════════════════════

class TrainerGUI:
    def __init__(self, parent: tk.Widget | None = None, on_model_updated: Callable[[], None] | None = None):
        self.is_standalone = parent is None
        self.root = tk.Tk() if self.is_standalone else parent
        self.on_model_updated = on_model_updated

        if self.is_standalone:
            self.root.title("个人审美偏好训练器")
            self.root.configure(bg=_BG)
            self.root.geometry("740x600")
            self.root.minsize(700, 520)

        self.dataset_dir_var = tk.StringVar()
        self.epochs_var = tk.StringVar(value="5")
        self.auto_onnx_var = tk.BooleanVar(value=True)
        self.external_python_var = tk.StringVar()
        self._running = False

        self._build_ui()
        self._apply_style()

    def _build_ui(self):
        container = tk.Frame(self.root, bg=_BG)
        container.pack(fill="both", expand=True)

        if self.is_standalone:
            hdr = tk.Frame(container, bg=_BG, pady=12)
            hdr.pack(fill="x", padx=24)
            tk.Label(hdr, text="🧠 审美偏好模型训练", bg=_BG, fg=_TEXT, font=(_FAM_TITLE, 20, "bold")).pack(anchor="w")
            tk.Label(hdr, text="微调轻量 MLP 分类头，训练完成后自动导出 ONNX 加速模型",
                     bg=_BG, fg=_TEXT_DIM, font=(_FAM, 11)).pack(anchor="w", pady=(2, 0))
            tk.Frame(container, bg=_BORDER, height=1).pack(fill="x", padx=24)

        body = tk.Frame(container, bg=_BG)
        body.pack(fill="both", expand=True, padx=24 if self.is_standalone else 12, pady=10)

        # ── 1. 外部环境配置卡片 (仅当打包环境未内置 Torch 时展示) ──
        if not _ALL_AVAILABLE:
            env_card = _card(body)
            env_card.pack(fill="x", pady=(0, 8))
            ei = tk.Frame(env_card, bg=_SURFACE, padx=14, pady=8)
            ei.pack(fill="x")

            tk.Label(ei, text="💡 免安装版已内置 ONNX 极速推理。若需训练微调，请选择电脑上的 Python/Conda 环境：",
                     bg=_SURFACE, fg=_ACCENT, font=(_FAM, 10, "bold")).pack(anchor="w")

            erow = tk.Frame(ei, bg=_SURFACE)
            erow.pack(fill="x", pady=(4, 0))

            detected_envs = discover_python_environments()
            if detected_envs and not self.external_python_var.get():
                self.external_python_var.set(detected_envs[0])

            self.python_combo = ttk.Combobox(
                erow, textvariable=self.external_python_var,
                values=detected_envs, font=(_FAM, 10),
            )
            self.python_combo.pack(side="left", fill="x", expand=True, ipady=2, padx=(0, 8))
            _pill_button(erow, "浏览 Python", self._pick_python).pack(side="left")

        # ── 2. 数据集选择卡片 ──
        dc = _card(body)
        dc.pack(fill="x", pady=(0, 8))
        di = tk.Frame(dc, bg=_SURFACE, padx=14, pady=8)
        di.pack(fill="x")

        _label(di, "📁 数据集根目录 (内含 'like' 与 'dislike' 文件夹)", bold=True, size=11).pack(anchor="w", pady=(0, 4))
        row = tk.Frame(di, bg=_SURFACE)
        row.pack(fill="x")
        _entry(row, self.dataset_dir_var).pack(side="left", fill="x", expand=True, ipady=4, padx=(0, 8))
        _pill_button(row, "选择目录", self._pick_dir).pack(side="left")

        # ── 3. 参数卡片 ──
        pc = _card(body)
        pc.pack(fill="x", pady=(0, 8))
        pi = tk.Frame(pc, bg=_SURFACE, padx=14, pady=8)
        pi.pack(fill="x")

        g = tk.Frame(pi, bg=_SURFACE)
        g.pack(fill="x")
        tk.Label(g, text="训练轮数 (Epochs):", bg=_SURFACE, fg=_TEXT_DIM, font=(_FAM, 10)).pack(side="left")
        _entry(g, self.epochs_var, width=6).pack(side="left", padx=(6, 16))

        tk.Checkbutton(
            g, text="训练完成后自动熔铸为 ONNX 极速加速模型（推荐）",
            variable=self.auto_onnx_var, bg=_SURFACE, fg=_TEXT, font=(_FAM, 10),
            activebackground=_SURFACE,
        ).pack(side="left")

        # ── 4. 执行行 ──
        br = tk.Frame(body, bg=_BG)
        br.pack(fill="x", pady=(0, 6))

        self.run_btn = _pill_button(br, "▶  开始训练与熔铸", self._on_run, big=True)
        self.run_btn.pack(side="left")

        self.progress = ttk.Progressbar(br, mode="determinate", length=180)
        self.progress.pack(side="left", padx=(12, 0))

        self.status_lbl = tk.Label(br, text="就绪", bg=_BG, fg=_TEXT_DIM, font=(_FAM, 11))
        self.status_lbl.pack(side="left", padx=(10, 0))

        # ── 5. 紧凑日志区 ──
        lc = _card(body)
        lc.pack(fill="both", expand=True)
        self.log_text = tk.Text(lc, bg="#F1F5F9", fg=_TEXT, relief="flat", bd=0, height=8,
                                font=("Consolas" if sys.platform != "darwin" else "Menlo", 10),
                                state="disabled")
        self.log_text.pack(fill="both", expand=True, padx=6, pady=6)

    def _apply_style(self):
        s = ttk.Style()
        try:
            s.configure("Horizontal.TProgressbar", background=_ACCENT, thickness=6)
        except Exception:
            pass

    def _pick_python(self):
        f = filedialog.askopenfilename(
            title="选择 Python 可执行文件",
            filetypes=[("Python", "python.exe" if sys.platform == "win32" else "python*"), ("All Files", "*.*")]
        )
        if f:
            self.external_python_var.set(f)

    def _pick_dir(self):
        d = filedialog.askdirectory(title="选择数据集根目录")
        if d:
            self.dataset_dir_var.set(d)

    def _log(self, msg: str):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")
        if self.is_standalone:
            self.root.update_idletasks()

    def _on_run(self):
        if self._running:
            return

        dataset_path = Path(self.dataset_dir_var.get().strip())
        if not dataset_path.exists() or not dataset_path.is_dir():
            messagebox.showerror("错误", "请先选择有效的数据集目录")
            return

        like_dir = dataset_path / "like"
        dislike_dir = dataset_path / "dislike"
        if not like_dir.exists() and not dislike_dir.exists():
            messagebox.showerror(
                "数据集格式错误",
                f"所选目录下未找到 'like' 或 'dislike' 文件夹！\n路径: {dataset_path}"
            )
            return

        try:
            epochs = int(self.epochs_var.get().strip())
            assert epochs > 0
        except Exception:
            messagebox.showerror("错误", "Epoch 必须是正整数")
            return

        self._running = True
        self.run_btn.configure(bg=_ACCENT_DIS, cursor="arrow")
        self.progress["value"] = 0
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

        if _ALL_AVAILABLE:
            self._log(f"🚀 初始化训练任务，目标 Epoch: {epochs}...")
            self._log(f"📁 数据集路径: {dataset_path}")
            threading.Thread(
                target=self._train_task_inprocess,
                args=(dataset_path, epochs, self.auto_onnx_var.get()),
                daemon=True,
            ).start()
        else:
            py_bin = self.external_python_var.get().strip()
            if not py_bin or not Path(py_bin).exists():
                messagebox.showerror("环境错误", "未找到有效的外部 Python 解释器。请在下拉列表中选择或点击“浏览 Python”指定你的 Conda / Python 环境。")
                self._finish_run()
                return

            self._log(f"🚀 调用外部 Python 环境进行训练: {py_bin}")
            self._log(f"📁 数据集路径: {dataset_path}")
            threading.Thread(
                target=self._train_task_external,
                args=(py_bin, dataset_path, epochs, self.auto_onnx_var.get()),
                daemon=True,
            ).start()

    def _train_task_external(self, py_bin: str, data_dir: Path, epochs: int, auto_onnx: bool):
        try:
            self.root.after(0, lambda: self.status_lbl.configure(text="正在启动外部训练进程…"))
            script = f"""
import sys
from pathlib import Path
data_dir = Path(r"{data_dir}")
epochs = {epochs}

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from transformers import CLIPProcessor, CLIPModel
    from PIL import Image
    import rawpy, io
except ImportError as err:
    print(f"❌ 运行环境缺少依赖: {{err}}\\n请确保所选 Python 环境已安装 torch, transformers, rawpy, Pillow", flush=True)
    sys.exit(1)

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if sys.platform == "darwin" and torch.backends.mps.is_available() else "cpu"))
print(f"💻 计算加速设备: {{device}}", flush=True)

print("正在加载 CLIP 视觉主干...", flush=True)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
for p in clip_model.parameters(): p.requires_grad_(False)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

class RawDataset:
    def __init__(self, root):
        self.samples = []
        for l, dname in ((1, "like"), (0, "dislike")):
            d = root / dname
            if d.exists():
                for pat in ("*.nef", "*.NEF", "*.arw", "*.ARW", "*.cr3", "*.CR3", "*.raf", "*.RAF"):
                    for p in d.glob(pat): self.samples.append((p, l))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        p, l = self.samples[idx]
        try:
            with rawpy.imread(str(p)) as raw: thumb = raw.extract_thumb()
            img = Image.open(io.BytesIO(thumb.data)).convert("RGB")
        except Exception:
            try:
                with rawpy.imread(str(p)) as raw: arr = raw.postprocess(half_size=True, use_camera_wb=True, output_bps=8)
                img = Image.fromarray(arr).convert("RGB")
            except Exception:
                img = Image.new("RGB", (224, 224), (0, 0, 0))
        inp = clip_processor(images=img, return_tensors="pt", padding=True)
        return {{k: v.squeeze(0) for k, v in inp.items()}}, l

def collate_fn(batch):
    inputs_list, labels = zip(*batch)
    return {{k: torch.stack([d[k] for d in inputs_list]) for k in inputs_list[0]}}, torch.tensor(labels, dtype=torch.long)

dataset = RawDataset(data_dir)
print(f"✅ 成功扫描到 {{len(dataset)}} 张照片", flush=True)
if len(dataset) == 0:
    print("❌ 数据集为空", flush=True)
    sys.exit(1)

dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
mlp = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 2)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mlp.parameters(), lr=1e-3)

print("正在训练中...", flush=True)
for ep in range(epochs):
    mlp.train()
    running_loss, correct, total = 0.0, 0, 0
    for b_in, labels in dataloader:
        b_in = {{k: v.to(device) for k, v in b_in.items()}}
        labels = labels.to(device)
        with torch.no_grad():
            vout = clip_model.vision_model(pixel_values=b_in['pixel_values'])
            feat = clip_model.visual_projection(vout.pooler_output)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        optimizer.zero_grad()
        out = mlp(feat)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, pred = torch.max(out.data, 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()
    print(f"  Epoch [{{ep+1:02d}}/{{epochs:02d}}] Loss: {{running_loss/max(1, len(dataloader)):.4f}} Acc: {{100*correct/max(1, total):.1f}}%", flush=True)

save_path = Path(r"{PROJECT_ROOT}") / "aesthetic_mlp.pth"
torch.save(mlp.state_dict(), save_path)
print(f"💾 权重已保存: {{save_path.name}}", flush=True)

if {auto_onnx}:
    print("⚡ 正在导出 ONNX 模型...", flush=True)
    class Combined(nn.Module):
        def __init__(self, clip, mlp):
            super().__init__()
            self.clip = clip
            self.mlp = mlp
        def forward(self, pixel_values):
            vout = self.clip.vision_model(pixel_values=pixel_values, return_dict=False)
            feat = self.clip.visual_projection(vout[1])
            feat = feat / feat.norm(dim=-1, keepdim=True)
            return torch.softmax(self.mlp(feat), dim=1)[:, 1]
    comb = Combined(clip_model, mlp).eval()
    onnx_path = Path(r"{PROJECT_ROOT}") / "photo_sort_model.onnx"
    torch.onnx.export(comb, (torch.randn(1, 3, 224, 224).to(device),), str(onnx_path), opset_version=14, input_names=["pixel_values"], output_names=["like_prob"], dynamic_axes={{"pixel_values": {{0: "batch_size"}}, "like_prob": {{0: "batch_size"}}}})
    print("🎉 ONNX 模型生成完毕！", flush=True)
"""
            proc = subprocess.Popen(
                [py_bin, "-c", script],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in iter(proc.stdout.readline, ''):
                l = line.strip()
                if l:
                    self.root.after(0, self._log, l)
                    if "Epoch [" in l:
                        try:
                            cur_ep = int(l.split("[")[1].split("/")[0])
                            self.root.after(0, lambda v=int(100 * cur_ep / epochs): self.progress.configure(value=v))
                        except Exception:
                            pass
            proc.stdout.close()
            ret = proc.wait()
            if ret == 0:
                self.root.after(0, lambda: self.status_lbl.configure(text="训练完成"))
                if self.on_model_updated:
                    self.root.after(0, self.on_model_updated)
                messagebox.showinfo("训练成功", "专属审美偏好模型训练已完成！\nONNX 模型已保存在当前程序目录下，连拍筛选即刻生效！")
            else:
                self.root.after(0, lambda: self.status_lbl.configure(text="训练失败"))
                messagebox.showerror("训练失败", "训练进程返回了异常退出码，请查看日志详情。")

        except Exception as exc:
            self.root.after(0, self._log, f"❌ 启动失败: {exc}")
            self.root.after(0, lambda: self.status_lbl.configure(text="执行出错"))
        finally:
            self.root.after(0, self._finish_run)

    def _train_task_inprocess(self, data_dir: Path, epochs: int, auto_onnx: bool):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if sys.platform == "darwin" and torch.backends.mps.is_available():
                device = torch.device("mps")
            self.root.after(0, self._log, f"💻 计算加速设备: {device}")

            clip_source = get_clip_model_path()
            if not is_clip_model_downloaded():
                self.root.after(0, self._log, "ℹ️ 检测到本地尚未下载完整 CLIP 视觉主干，将自动下载至当前程序 models/ 目录...")
                self.root.after(0, lambda: self.status_lbl.configure(text="正在下载基础模型…"))

                def _dl_cb(msg: str, pct: float):
                    self.root.after(0, self._log, f"  [下载进度 {pct*100:.0f}%] {msg}")
                    self.root.after(0, lambda v=int(pct * 100): self.progress.configure(value=v))

                download_clip_model(use_mirror=True, progress_callback=_dl_cb)
                clip_source = get_clip_model_path()

            self.root.after(0, self._log, f"正在加载 CLIP 视觉主干 ({clip_source}) …")
            clip_model = CLIPModel.from_pretrained(clip_source)
            clip_model.to(device).eval()
            for p in clip_model.parameters():
                p.requires_grad_(False)
            clip_processor = CLIPProcessor.from_pretrained(clip_source)
            self.root.after(0, self._log, "✅ CLIP 特征提取器加载完成。")

            self.root.after(0, self._log, "正在扫描 RAW 文件并提取预览…")
            dataset = RawAestheticDataset(data_dir, clip_processor, clip_model, device)
            if len(dataset) == 0:
                self.root.after(0, self._log, "❌ like/dislike 文件夹中未找到任何有效 RAW 文件！")
                self.root.after(0, self._finish_run)
                return
            self.root.after(0, self._log, f"✅ 成功加载 {len(dataset)} 张样本照片。")
            dataloader = DataLoader(dataset, batch_size=8, shuffle=True,
                                    num_workers=0, collate_fn=collate_fn)

            mlp = AestheticMLP(input_dim=512).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(mlp.parameters(), lr=1e-3)

            self.root.after(0, lambda: self.status_lbl.configure(text="正在训练…"))
            for epoch in range(epochs):
                mlp.train()
                running_loss, correct, total = 0.0, 0, 0

                for batch_inputs, labels in dataloader:
                    batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
                    labels = labels.to(device)

                    with torch.no_grad():
                        vision_out = clip_model.vision_model(
                            pixel_values=batch_inputs['pixel_values']
                        )
                        pooled = vision_out.pooler_output
                        features = clip_model.visual_projection(pooled)
                        features = features / features.norm(dim=-1, keepdim=True)

                    optimizer.zero_grad()
                    outputs = mlp(features)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                epoch_loss = running_loss / max(1, len(dataloader))
                epoch_acc = 100 * correct / max(1, total)
                msg = f"  Epoch [{epoch+1:02d}/{epochs:02d}]  Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.1f}%"
                self.root.after(0, self._log, msg)
                pv = int(100 * (epoch + 1) / epochs)
                self.root.after(0, lambda v=pv: self.progress.configure(value=v))

            save_path = PROJECT_ROOT / "aesthetic_mlp.pth"
            torch.save(mlp.state_dict(), save_path)
            self.root.after(0, self._log, f"\n💾 MLP 权重已保存至: {save_path.name}")

            if auto_onnx:
                self.root.after(0, lambda: self.status_lbl.configure(text="正在熔铸 ONNX…"))
                self.root.after(0, self._log, "\n⚡ 开始自动熔铸 ONNX 模型...")

                def _onnx_cb(m: str):
                    self.root.after(0, self._log, f"  {m}")

                onnx_out = export_to_onnx(
                    project_root=PROJECT_ROOT,
                    mlp_path=save_path,
                    clip_source=clip_source,
                    progress_callback=_onnx_cb,
                )
                self.root.after(0, self._log, f"🎉 ONNX 模型已生成: {onnx_out.name}，连拍筛选将自动启用极速加速！")

            self.root.after(0, lambda: self.status_lbl.configure(text="训练完成"))
            if self.on_model_updated:
                self.root.after(0, self.on_model_updated)

            messagebox.showinfo("训练成功", "个人偏好模型训练已完成！" + ("\n已自动熔铸 ONNX 加速模型。" if auto_onnx else ""))

        except Exception as exc:
            self.root.after(0, self._log, f"\n❌ 训练出错: {exc}")
            self.root.after(0, lambda: self.status_lbl.configure(text="训练失败"))
            messagebox.showerror("训练出错", str(exc))
        finally:
            self.root.after(0, self._finish_run)

    def _finish_run(self):
        self._running = False
        self.run_btn.configure(bg=_ACCENT, cursor="hand2")

    def run(self):
        if self.is_standalone:
            self.root.mainloop()


if __name__ == "__main__":
    TrainerGUI().run()
