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

# ── 动态加载 PyTorch 防止在没有环境的 Mac 上崩溃 ────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from torchvision import models, transforms
    import rawpy
    from PIL import Image
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
# 数据集与模型逻辑 (仅在 TORCH_AVAILABLE 时可用)
# ══════════════════════════════════════════════════════════════════════════════

if TORCH_AVAILABLE:
    def _extract_preview(path: Path) -> Image.Image:
        """从 NEF 提取预览图供训练使用。"""
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

        # 降级：半尺寸解码
        with rawpy.imread(str(path)) as raw:
            arr = raw.postprocess(half_size=True, use_camera_wb=True, output_bps=8)
        return Image.fromarray(arr).convert("RGB")

    class NefAestheticDataset(Dataset):
        def __init__(self, root_dir: Path, transform=None):
            self.root_dir = root_dir
            self.transform = transform
            self.samples = []
            
            # 定义类别：0 为不偏好 (dislike)，1 为偏好 (like)
            dislike_dir = root_dir / "dislike"
            like_dir = root_dir / "like"
            
            if dislike_dir.exists():
                for p in dislike_dir.glob("*.[nN][eE][fF]"):
                    self.samples.append((p, 0))
            if like_dir.exists():
                for p in like_dir.glob("*.[nN][eE][fF]"):
                    self.samples.append((p, 1))

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            path, label = self.samples[idx]
            try:
                img = _extract_preview(path)
            except Exception:
                # 极端异常情况返回纯黑图片占位
                img = Image.new("RGB", (224, 224), (0, 0, 0))
                
            if self.transform:
                img = self.transform(img)
            return img, label

    def create_model():
        """创建用于二分类微调的 MobileNetV3 预训练模型"""
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        model = models.mobilenet_v3_small(weights=weights)
        # 替换全连接层，输出为2类 (0: dislike, 1: like)
        num_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_features, 2)
        return model


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
        tk.Label(hdr, text="基于你的分类（like/dislike），自动微调 MobileNet 视觉模型",
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
        if not TORCH_AVAILABLE:
            tk.Label(br, text="⚠️ 未检测到 PyTorch，请在配置好的环境中运行。", fg=_ERROR, bg=_BG, font=(_FAM, 11, "bold")).pack(side="left")
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
            # 1. 准备数据
            self.root.after(0, self._log, "正在扫描 NEF 文件...")
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            dataset = NefAestheticDataset(data_dir, transform=transform)
            
            if len(dataset) == 0:
                self.root.after(0, self._log, "❌ 错误: 在 like/dislike 文件夹中没有找到任何 NEF 文件！")
                self.root.after(0, self._finish_run)
                return
                
            self.root.after(0, self._log, f"✅ 找到 {len(dataset)} 张照片。")
            
            dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
            
            # 2. 准备模型和设备
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.root.after(0, self._log, f"使用计算设备: {device}")
            
            model = create_model().to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=1e-4)
            
            # 3. 开始训练循环
            self.root.after(0, lambda: self.status_lbl.configure(text="正在训练..."))
            for epoch in range(epochs):
                model.train()
                running_loss = 0.0
                correct = 0
                total = 0
                
                for batch_idx, (inputs, labels) in enumerate(dataloader):
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                epoch_loss = running_loss / len(dataloader)
                epoch_acc = 100 * correct / total
                
                # 汇报进度
                msg = f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.2f}%"
                self.root.after(0, self._log, msg)
                progress_val = int(100 * (epoch + 1) / epochs)
                self.root.after(0, lambda v=progress_val: self.progress.configure(value=v))

            # 4. 保存模型
            save_path = Path(__file__).resolve().parent.parent / "aesthetic_model.pth"
            torch.save(model.state_dict(), save_path)
            self.root.after(0, self._log, f"\n🎉 训练完成！模型已保存至:\n{save_path}")
            self.root.after(0, lambda: self.status_lbl.configure(text="训练完毕"))

        except Exception as e:
            self.root.after(0, self._log, f"\n❌ 训练过程发生错误: {str(e)}")
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
