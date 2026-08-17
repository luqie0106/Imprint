import io
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

# ── 颜色常量 (沿用亮色主题) ───────────────────────────────────────────────────
_BG = "#F8FAFC"
_SURFACE = "#FFFFFF"
_ACCENT = "#2563EB"
_ACCENT_HOVER = "#1D4ED8"
_ACCENT_DIS = "#CBD5E1"
_SUCCESS = "#16A34A"
_WARNING = "#D97706"
_ERROR = "#DC2626"
_TEXT = "#0F172A"
_TEXT_DIM = "#64748B"
_BORDER = "#E2E8F0"

_FAM = "SF Pro Text" if sys.platform == "darwin" else "Segoe UI"
_FAM_TITLE = "SF Pro Display" if sys.platform == "darwin" else "Segoe UI"

# ── 动态加载 PyTorch ───────────────────────────────────────────────────────
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

# ── 动态加载 CLIP（transformers）──────────────────────────────────────────
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
    # ---------- RAW 预览提取 ----------
    def _extract_preview(path: Path) -> Image.Image:
        """从 RAW 文件提取预览图（NEF/ARW/CR3/RAF 均可）。"""
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

    # ---------- 数据集 ----------
    _RAW_GLOB_PATTERNS = ["*.[nN][eE][fF]", "*.[aA][rR][wW]",
                          "*.[cC][rR]3", "*.[rR][aA][fF]"]

    class RawAestheticDataset(Dataset):
        """
        目录结构：
          root/
            like/      <- 喜欢的照片 (label=1)
            dislike/   <- 不喜欢的照片 (label=0)
        """
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

    # ---------- MLP 分类头（与 burst_filter.py 中结构完全一致）----------
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
# 小工具
# ══════════════════════════════════════════════════════════════════════════════

def _entry(parent, var, **kw):
    return tk.Entry(
        parent, textvariable=var,
        bg=_SURFACE, fg=_TEXT, insertbackground=_TEXT,
        relief="flat", bd=0, highlightthickness=1,
        highlightbackground=_BORDER, highlightcolor=_ACCENT,
        font=(_FAM, 12), **kw,
    )

def _label(parent, text, size=12, color=None, bold=False):
    weight = "bold" if bold else "normal"
    return tk.Label(
        parent, text=text, bg=parent.cget("bg"),
        fg=color or _TEXT, font=(_FAM, size, weight),
    )

def _card(parent):
    return tk.Frame(parent, bg=_SURFACE, highlightthickness=1, highlightbackground=_BORDER)

def _pill_button(parent, text, command, big=False, active=True):
    size = 13 if big else 11
    px = 20 if big else 14
    py = 9 if big else 6
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
# GUI 逻辑
# ══════════════════════════════════════════════════════════════════════════════

class TrainerGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("个人审美偏好训练器 (Aesthetic Model Trainer)")
        self.root.configure(bg=_BG)
        self.root.geometry("780x580")
        self.root.resizable(False, False)

        self.dataset_dir_var = tk.StringVar()
        self.epochs_var = tk.StringVar(value="5")
        self._running = False

        self._build_ui()
        self._apply_style()

    def _build_ui(self):
        # ── 标题 ──
        hdr = tk.Frame(self.root, bg=_BG, pady=18)
        hdr.pack(fill="x", padx=28)
        tk.Label(hdr, text="🧠 审美偏好模型训练", bg=_BG, fg=_TEXT, font=(_FAM_TITLE, 22, "bold")).pack(anchor="w")
        tk.Label(hdr, text="冻结 CLIP 视觉编码器 + 轻量 MLP 分类头，仅训练 MLP 权重",
                 bg=_BG, fg=_TEXT_DIM, font=(_FAM, 12)).pack(anchor="w", pady=(3, 0))
        tk.Frame(self.root, bg=_BORDER, height=1).pack(fill="x", padx=28)

        body = tk.Frame(self.root, bg=_BG)
        body.pack(fill="both", expand=True, padx=28, pady=20)

        # ── 数据集选择卡片 ──
        dc = _card(body)
        dc.pack(fill="x", pady=(0, 12))
        di = tk.Frame(dc, bg=_SURFACE, padx=16, pady=14)
        di.pack(fill="x")

        _label(di, "📁 数据集目录", bold=True).pack(anchor="w")
        _label(di, "选择包含 'like' 和 'dislike' 子文件夹的根目录", size=10, color=_TEXT_DIM).pack(anchor="w", pady=(2, 8))

        row = tk.Frame(di, bg=_SURFACE)
        row.pack(fill="x")
        _entry(row, self.dataset_dir_var).pack(side="left", fill="x", expand=True, ipady=6, padx=(0, 10))
        _pill_button(row, "选择目录", self._pick_dir).pack(side="left")

        # ── 参数卡片 ──
        pc = _card(body)
        pc.pack(fill="x", pady=(0, 16))
        pi = tk.Frame(pc, bg=_SURFACE, padx=16, pady=14)
        pi.pack(fill="x")

        _label(pi, "⚙️ 训练参数", bold=True).pack(anchor="w")
        
        g = tk.Frame(pi, bg=_SURFACE)
        g.pack(fill="x", pady=(10, 0))
        tk.Label(g, text="训练轮数 (Epochs):", bg=_SURFACE, fg=_TEXT_DIM, font=(_FAM, 11)).pack(side="left")
        _entry(g, self.epochs_var, width=8).pack(side="left", padx=10)

        # ── 执行行 ──
        br = tk.Frame(body, bg=_BG)
        br.pack(fill="x")
        
        # 判断环境
        if not _ALL_AVAILABLE:
            tk.Label(br, text="⚠️ 需要 PyTorch + transformers，请检查环境。", fg=_ERROR, bg=_BG, font=(_FAM, 11, "bold")).pack(side="left")
            self.run_btn = _pill_button(br, "环境缺失", lambda: None, big=True, active=False)
            self.run_btn.pack(side="right")
        else:
            self.run_btn = _pill_button(br, "▶  开始训练", self._on_run, big=True)
            self.run_btn.pack(side="left")
            
            self.progress = ttk.Progressbar(br, mode="determinate", length=200)
            self.progress.pack(side="left", padx=(14, 0))
            
            self.status_lbl = tk.Label(br, text="等待开始...", bg=_BG, fg=_TEXT_DIM, font=(_FAM, 11))
            self.status_lbl.pack(side="left", padx=(12, 0))

        # ── 日志区 ──
        lc = _card(body)
        lc.pack(fill="both", expand=True, pady=(16, 0))
        self.log_text = tk.Text(lc, bg="#F1F5F9", fg=_TEXT, relief="flat", bd=0, font=("Consolas", 11), state="disabled")
        self.log_text.pack(fill="both", expand=True, padx=8, pady=8)

    def _apply_style(self):
        s = ttk.Style()
        s.theme_use("default")
        s.configure("Horizontal.TProgressbar", troughcolor=_SURFACE, background=_ACCENT, thickness=6)

    def _pick_dir(self):
        d = filedialog.askdirectory(title="选择数据集根目录")
        if d:
            self.dataset_dir_var.set(d)

    def _log(self, msg: str):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")
        self.root.update_idletasks()

    def _on_run(self):
        if self._running:
            return
            
        dataset_path = Path(self.dataset_dir_var.get().strip())
        if not dataset_path.exists() or not dataset_path.is_dir():
            messagebox.showerror("错误", "请先选择有效的数据集目录")
            return
            
        try:
            epochs = int(self.epochs_var.get().strip())
            assert epochs > 0
        except:
            messagebox.showerror("错误", "Epoch 必须是正整数")
            return

        self._running = True
        self.run_btn.configure(bg=_ACCENT_DIS, cursor="arrow")
        self.progress["value"] = 0
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")
        
        self._log(f"🚀 初始化训练任务，目标 Epoch: {epochs}...")
        self._log(f"📁 数据集路径: {dataset_path}")
        
        threading.Thread(target=self._train_task, args=(dataset_path, epochs), daemon=True).start()

    def _train_task(self, data_dir: Path, epochs: int):
        try:
            # 1. 设备
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if sys.platform == "darwin" and torch.backends.mps.is_available():
                device = torch.device("mps")
            self.root.after(0, self._log, f"计算设备: {device}")

            # 2. 加载冻结的 CLIP（特征提取器）
            self.root.after(0, self._log, "正在加载 CLIP 编码器（首次运行将联网下载）…")
            clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            clip_model.to(device).eval()
            for p in clip_model.parameters():
                p.requires_grad_(False)
            clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.root.after(0, self._log, "✅ CLIP 加载完成。")

            # 3. 扫描数据集
            self.root.after(0, self._log, "正在扫描 RAW 文件…")
            dataset = RawAestheticDataset(data_dir, clip_processor, clip_model, device)
            if len(dataset) == 0:
                self.root.after(0, self._log, "❌ like/dislike 文件夹中未找到任何 RAW 文件！")
                self.root.after(0, self._finish_run)
                return
            self.root.after(0, self._log, f"✅ 找到 {len(dataset)} 张照片。")
            dataloader = DataLoader(dataset, batch_size=8, shuffle=True,
                                    num_workers=0, collate_fn=collate_fn)

            # 4. 初始化 MLP 和优化器（只训练 MLP，CLIP 已冻结）
            mlp = AestheticMLP(input_dim=512).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(mlp.parameters(), lr=1e-3)

            # 5. 训练循环
            self.root.after(0, lambda: self.status_lbl.configure(text="正在训练…"))
            for epoch in range(epochs):
                mlp.train()
                running_loss = 0.0
                correct = 0
                total = 0

                for batch_inputs, labels in dataloader:
                    batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
                    labels = labels.to(device)

                    # CLIP 提取特征（no_grad，冻结）
                    with torch.no_grad():
                        vision_out = clip_model.vision_model(
                            pixel_values=batch_inputs['pixel_values']
                        )
                        pooled = vision_out.pooler_output           # [B, hidden_dim]
                        features = clip_model.visual_projection(pooled)  # [B, 512]
                        features = features / features.norm(dim=-1, keepdim=True)

                    # MLP 前向 + 反向
                    optimizer.zero_grad()
                    outputs = mlp(features)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                epoch_loss = running_loss / len(dataloader)
                epoch_acc = 100 * correct / total
                msg = f"Epoch [{epoch+1}/{epochs}]  Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.1f}%"
                self.root.after(0, self._log, msg)
                pv = int(100 * (epoch + 1) / epochs)
                self.root.after(0, lambda v=pv: self.progress.configure(value=v))

            # 6. 只保存 MLP 权重
            save_path = Path(__file__).resolve().parent.parent / "aesthetic_mlp.pth"
            torch.save(mlp.state_dict(), save_path)
            self.root.after(0, self._log, f"\n🎉 训练完成！MLP 已保存至:\n{save_path}")
            self.root.after(0, lambda: self.status_lbl.configure(text="训练完毕"))

        except Exception as exc:
            self.root.after(0, self._log, f"\n❌ 训练出错: {exc}")
            self.root.after(0, lambda: self.status_lbl.configure(text="训练失败"))
        finally:
            self.root.after(0, self._finish_run)

    def _finish_run(self):
        self._running = False
        self.run_btn.configure(bg=_ACCENT, cursor="hand2")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    TrainerGUI().run()
