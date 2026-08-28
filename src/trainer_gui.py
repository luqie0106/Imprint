"""
trainer_gui.py — 个人审美偏好训练器与 ONNX 自动熔铸 (PySide6 / Qt6 现代全圆角矢量设计)
支持 Python 3.9 ~ 3.13 广泛版本、Conda 环境智能探测与依赖一键自动安装
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable

# 确保标准输出为 UTF-8 编码，防止 Windows GBK 环境下 Emoji 引发 UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QWidget, QFrame, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QCheckBox, QComboBox,
    QProgressBar, QPlainTextEdit, QFileDialog, QMessageBox
)

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
from qt_theme import GREEN_FG, AMBER_FG, TEXT_TERT, TEXT_SEC


# ══════════════════════════════════════════════════════════════════════════════
# 通用环境自动发现与依赖探针逻辑（支持 Python 3.9 ~ 3.13）
# ══════════════════════════════════════════════════════════════════════════════

def discover_python_environments() -> list[str]:
    """通用、自动发现本机所有 Conda 环境及 Python 解释器。"""
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

    if not getattr(sys, 'frozen', False) and sys.executable:
        _add_env(Path(sys.executable))

    home = Path.home()
    env_txt = home / ".conda" / "environments.txt"
    if env_txt.exists():
        try:
            for line in env_txt.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if line and Path(line).exists():
                    _add_env(Path(line))
        except Exception:
            pass

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

    for cmd in ["python3", "python"]:
        w = shutil.which(cmd)
        if w:
            _add_env(Path(w))

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
        if "torch" in low or "ai" in low or "photo" in low or "py3" in low:
            score += 5
        return score

    found.sort(key=_rank_score, reverse=True)
    return found


def probe_python_environment(py_bin: str) -> dict:
    if not py_bin or not Path(py_bin).exists():
        return {"error": "路径不存在"}

    code = """
import sys, json, importlib.util
info = {
    "version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    "major": sys.version_info.major,
    "minor": sys.version_info.minor,
    "major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
    "error": None,
}
missing = []
for pkg, imp in [("torch", "torch"), ("transformers", "transformers"), ("rawpy", "rawpy"), ("Pillow", "PIL"), ("onnx", "onnx")]:
    try:
        if importlib.util.find_spec(imp) is None:
            missing.append(pkg)
    except Exception:
        missing.append(pkg)
info["missing"] = missing
print(json.dumps(info))
"""
    try:
        res = subprocess.run(
            [py_bin, "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=12,
        )
        if res.returncode == 0 and res.stdout.strip():
            return json.loads(res.stdout.strip())
        else:
            return {"error": res.stderr.strip() or "探测失败"}
    except subprocess.TimeoutExpired:
        return {"error": "检测超时（环境响应过慢）"}
    except Exception as exc:
        return {"error": str(exc)}


# ══════════════════════════════════════════════════════════════════════════════
# 训练后台线程
# ══════════════════════════════════════════════════════════════════════════════

class TrainWorker(QThread):
    log_sig = Signal(str)
    progress_sig = Signal(int)
    done_sig = Signal(bool, str)
    need_install_sig = Signal(str, list, str)

    def __init__(self, py_bin: str, data_dir: Path, epochs: int, auto_onnx: bool, backbone_type: str = "b32"):
        super().__init__()
        self.py_bin = py_bin
        self.data_dir = data_dir
        self.epochs = epochs
        self.auto_onnx = auto_onnx
        self.backbone_type = backbone_type


    def run(self):
        info = probe_python_environment(self.py_bin)
        if info.get("error"):
            self.done_sig.emit(False, f"Python 环境检测失败: {info['error']}")
            return

        major = info.get("major", 0)
        minor = info.get("minor", 0)
        ver_str = info.get("version", "未知")

        if major < 3 or (major == 3 and minor < 9):
            self.done_sig.emit(False, f"当前 Python 版本为 {ver_str}，建议使用 Python 3.9 ~ 3.13。")
            return

        missing = info.get("missing", [])
        if missing:
            self.need_install_sig.emit(self.py_bin, missing, ver_str)
            return

        self._execute_training()

    def run_with_install(self):
        self._execute_training()

    def _execute_training(self):
        try:
            self.log_sig.emit("🚀 启动审美偏好微调训练进程...")
            repo_id = "openai/clip-vit-base-patch32" if self.backbone_type == "b32" else "openai/clip-vit-large-patch14"
            dim = 512 if self.backbone_type == "b32" else 768
            pth_name = "aesthetic_mlp.pth" if self.backbone_type == "b32" else "aesthetic_mlp_l14.pth"
            onnx_name = "custom_aesthetic_model.onnx" if self.backbone_type == "b32" else "custom_aesthetic_l14_model.onnx"
            backbone_desc = "CLIP ViT-B/32 (标准极速 · 512维)" if self.backbone_type == "b32" else "CLIP ViT-L/14 / Aesthetic 3 (专业高精 · 768维)"

            script = f"""
import sys
from pathlib import Path
data_dir = Path(r"{self.data_dir}")
epochs = {self.epochs}
backbone_repo = "{repo_id}"
feat_dim = {dim}
pth_filename = "{pth_name}"
onnx_filename = "{onnx_name}"

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
if device.type == 'cuda':
    dev_title = f"NVIDIA CUDA 显卡硬件加速 ({{torch.cuda.get_device_name(0)}})"
elif device.type == 'mps':
    dev_title = "Apple Silicon Metal 显卡硬件加速 (MPS)"
else:
    dev_title = "CPU 多核心并行计算"

print(f"⚡ 深度学习加速设备: {{dev_title}}", flush=True)
print(f"📦 正在加载视觉主干底座: {backbone_desc} ...", flush=True)

models_local = Path(r"{PROJECT_ROOT}") / "models"
local_clip_dir = models_local / ("clip-vit-base-patch32" if feat_dim == 512 else "clip-vit-large-patch14")
clip_src = str(local_clip_dir) if local_clip_dir.exists() and (local_clip_dir / "config.json").exists() else backbone_repo

clip_model = CLIPModel.from_pretrained(clip_src).to(device).eval()
for p in clip_model.parameters(): p.requires_grad_(False)
clip_processor = CLIPProcessor.from_pretrained(clip_src)

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pass

try:
    import pillow_jxl
except Exception:
    pass

RAW_SUFFIXES = {{
    ".nef", ".nrw", ".arw", ".srf", ".sr2", ".cr2", ".cr3", ".crw",
    ".rw2", ".raw", ".dng", ".raf", ".orf", ".ori", ".pef", ".ptx",
    ".3fr", ".fff", ".iiq", ".srw", ".x3f", ".mrw", ".gpr", ".erf", ".mef", ".mos",
}}
STANDARD_SUFFIXES = {{
    ".jpg", ".jpeg", ".jpe", ".jxl", ".hif", ".heif", ".heic", ".png", ".webp", ".tiff", ".tif", ".bmp"
}}
ALL_PHOTO_SUFFIXES = RAW_SUFFIXES | STANDARD_SUFFIXES

class RawDataset:
    def __init__(self, root):
        self.samples = []
        for l, dname in ((1, "like"), (0, "dislike")):
            d = root / dname
            if d.exists():
                for p in d.iterdir():
                    if p.is_file() and p.suffix.lower() in ALL_PHOTO_SUFFIXES:
                        self.samples.append((p, l))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        p, l = self.samples[idx]
        img = None
        if p.suffix.lower() in RAW_SUFFIXES:
            try:
                with rawpy.imread(str(p)) as raw: thumb = raw.extract_thumb()
                img = Image.open(io.BytesIO(thumb.data)).convert("RGB")
            except Exception:
                try:
                    with rawpy.imread(str(p)) as raw: arr = raw.postprocess(half_size=True, use_camera_wb=True, output_bps=8)
                    img = Image.fromarray(arr).convert("RGB")
                except Exception:
                    pass
        if img is None:
            try:
                with Image.open(p) as im:
                    img = im.convert("RGB")
            except Exception:
                img = Image.new("RGB", (224, 224), (0, 0, 0))
        inp = clip_processor(images=img, return_tensors="pt", padding=True)
        return {{k: v.squeeze(0) for k, v in inp.items()}}, l

def collate_fn(batch):
    inputs_list, labels = zip(*batch)
    return {{k: torch.stack([d[k] for d in inputs_list]) for k in inputs_list[0]}}, torch.tensor(labels, dtype=torch.long)

dataset = RawDataset(data_dir)
print(f"✅ 成功扫描到 {{len(dataset)}} 张标注样本照片", flush=True)
if len(dataset) == 0:
    print("❌ 数据集为空，请检查 like/dislike 目录", flush=True)
    sys.exit(1)

batch_sz = 16 if device.type in ('cuda', 'mps') else 8
dataloader = DataLoader(dataset, batch_size=batch_sz, shuffle=True, collate_fn=collate_fn)
mlp = nn.Sequential(nn.Linear(feat_dim, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 2)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mlp.parameters(), lr=1e-3)

print("🎯 正在执行特征提取与微调训练...", flush=True)
for ep in range(epochs):
    mlp.train()
    running_loss, correct, total = 0.0, 0, 0
    for b_in, labels in dataloader:
        b_in = {{k: v.to(device) for k, v in b_in.items()}}
        labels = labels.to(device)
        with torch.no_grad():
            vout = clip_model.vision_model(pixel_values=b_in['pixel_values'], return_dict=False)
            feat = clip_model.visual_projection(vout[1])
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

save_dir = Path(r"{PROJECT_ROOT}") / "models"
save_dir.mkdir(parents=True, exist_ok=True)
save_path = save_dir / pth_filename
torch.save(mlp.state_dict(), save_path)
print(f"💾 专属模型头权重已保存至: models/{{save_path.name}}", flush=True)

if {self.auto_onnx}:
    print(f"⚡ 正在将专属模型头与视觉底座熔铸为 ONNX 硬件加速模型 ({{onnx_filename}})...", flush=True)
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
    onnx_path = save_dir / onnx_filename
    torch.onnx.export(
        comb,
        (torch.randn(1, 3, 224, 224).to(device),),
        str(onnx_path),
        opset_version=14,
        input_names=["pixel_values"],
        output_names=["like_prob"],
        dynamic_axes={{"pixel_values": {{0: "batch_size"}}, "like_prob": {{0: "batch_size"}}}},
        dynamo=False,
    )
    if feat_dim == 512:
        legacy_onnx = Path(r"{PROJECT_ROOT}") / "photo_sort_model.onnx"
        try:
            import shutil
            shutil.copy2(str(onnx_path), str(legacy_onnx))
        except Exception:
            pass
    print(f"🎉 ONNX 模型熔铸完毕！文件已就绪: models/{{onnx_filename}} ({{onnx_path.stat().st_size/(1024*1024):.1f}} MB)", flush=True)

"""
            proc = subprocess.Popen(
                [self.py_bin, "-c", script],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in iter(proc.stdout.readline, ''):
                l = line.strip()
                if l:
                    self.log_sig.emit(l)
                    if "Epoch [" in l:
                        try:
                            cur_ep = int(l.split("[")[1].split("/")[0])
                            self.progress_sig.emit(int(100 * cur_ep / self.epochs))
                        except Exception:
                            pass
            proc.stdout.close()
            ret = proc.wait()

            if ret == 0:
                self.done_sig.emit(True, "🎉 专属审美偏好模型微调与 ONNX 熔铸成功！\n模型已保存在 models/ 目录下，连拍优选中可即刻选用！")
            else:
                self.done_sig.emit(False, "训练进程返回了异常退出码，请查看下方日志详情。")
        except Exception as exc:
            self.done_sig.emit(False, f"执行出错: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# 偏好训练界面
# ══════════════════════════════════════════════════════════════════════════════

class TrainerGUI(QWidget):

    """个人审美偏好训练器 (PySide6 现代全圆角矢量界面)"""

    def __init__(self, on_model_updated: Callable[[], None] | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self.on_model_updated = on_model_updated
        self._running = False
        self._worker: TrainWorker | None = None

        self._build_ui()
        QtCore.QTimer.singleShot(100, self._refresh_python_envs)

    def _detect_gpu_device(self) -> tuple[str, str]:
        """检测当前机器的深度学习硬件加速环境"""
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "CUDA"
                return "cuda", f"⚡ 硬件显卡加速：NVIDIA CUDA ({name})"
            if sys.platform == "darwin" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps", "⚡ 硬件显卡加速：Apple Silicon Metal (MPS)"
        except Exception:
            pass
        return "cpu", "⚡ 计算设备：CPU 多核心并行"

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 10, 0, 10)
        main_layout.setSpacing(10)


        # ── 1. Python 环境与 GPU 加速卡片 ──
        env_card = QFrame()
        env_card.setProperty("class", "CardFrame")
        e_layout = QVBoxLayout(env_card)
        e_layout.setContentsMargins(16, 12, 16, 12)
        e_layout.setSpacing(8)

        env_header = QHBoxLayout()
        t1 = QLabel("Python 训练环境与计算设备")
        t1.setProperty("class", "CardTitle")
        env_header.addWidget(t1)

        # 硬件加速徽章
        gpu_type, gpu_text = self._detect_gpu_device()
        self.gpu_badge = QLabel(f" {gpu_text} ")
        self.gpu_badge.setStyleSheet(f"font-size: 11px; font-weight: bold; color: {GREEN_FG}; background-color: #E8F8EE; border: 1px solid #B8ECC7; border-radius: 4px; padding: 2px 6px;")
        env_header.addWidget(self.gpu_badge)

        self.env_status_lbl = QLabel("🔍 检测中...")
        self.env_status_lbl.setProperty("class", "SecondaryLabel")
        env_header.addWidget(self.env_status_lbl)
        env_header.addStretch()
        e_layout.addLayout(env_header)


        erow = QHBoxLayout()
        erow.setSpacing(8)

        self.python_combo = QComboBox()
        self.python_combo.setEditable(True)
        self.python_combo.currentTextChanged.connect(self._on_python_changed)
        erow.addWidget(self.python_combo, stretch=1)

        self.pick_py_btn = QPushButton("浏览 Python")
        self.pick_py_btn.setProperty("class", "SecondaryBtn")
        self.pick_py_btn.clicked.connect(self._pick_python)
        erow.addWidget(self.pick_py_btn)

        self.refresh_py_btn = QPushButton("🔄 刷新")
        self.refresh_py_btn.setProperty("class", "SecondaryBtn")
        self.refresh_py_btn.clicked.connect(self._refresh_python_envs)
        erow.addWidget(self.refresh_py_btn)

        e_layout.addLayout(erow)
        main_layout.addWidget(env_card)

        # ── 2. 数据集选择卡片 ──
        ds_card = QFrame()
        ds_card.setProperty("class", "CardFrame")
        ds_layout = QVBoxLayout(ds_card)
        ds_layout.setContentsMargins(16, 12, 16, 12)
        ds_layout.setSpacing(8)

        t2 = QLabel("训练数据集根目录 (内含 'like' 与 'dislike' 文件夹)")
        t2.setProperty("class", "CardTitle")
        ds_layout.addWidget(t2)

        ds_row = QHBoxLayout()
        ds_row.setSpacing(8)

        self.ds_input = QLineEdit()
        self.ds_input.setPlaceholderText("选择包含 like / dislike 子文件夹的数据集根目录...")
        ds_row.addWidget(self.ds_input, stretch=1)

        self.pick_ds_btn = QPushButton("选择目录")
        self.pick_ds_btn.setProperty("class", "SecondaryBtn")
        self.pick_ds_btn.clicked.connect(self._pick_dataset_dir)
        ds_row.addWidget(self.pick_ds_btn)

        ds_layout.addLayout(ds_row)
        main_layout.addWidget(ds_card)

        # ── 3. 参数配置卡片 ──
        param_card = QFrame()
        param_card.setProperty("class", "CardFrame")
        p_layout = QVBoxLayout(param_card)
        p_layout.setContentsMargins(16, 12, 16, 12)
        p_layout.setSpacing(10)

        # 视觉底座下拉选择
        p_row1 = QHBoxLayout()
        p_row1.setSpacing(12)
        bb_lbl = QLabel("训练视觉底座:")
        bb_lbl.setProperty("class", "SecondaryLabel")
        p_row1.addWidget(bb_lbl)

        self.backbone_combo = QComboBox()
        self.backbone_combo.addItem("⚡  CLIP ViT-B/32 (标准极速底座 · 512 维 · 适合日常微调)", "b32")
        self.backbone_combo.addItem("💎  CLIP ViT-L/14 / Aesthetic 3 (专业高精底座 · 768 维 · 审美更细腻)", "l14")
        p_row1.addWidget(self.backbone_combo, stretch=1)
        p_layout.addLayout(p_row1)

        # 训练轮数与 ONNX 选项
        p_row2 = QHBoxLayout()
        p_row2.setSpacing(16)
        ep_lbl = QLabel("训练轮数 (Epochs):")
        ep_lbl.setProperty("class", "SecondaryLabel")
        p_row2.addWidget(ep_lbl)

        self.epochs_input = QLineEdit("5")
        self.epochs_input.setFixedWidth(60)
        p_row2.addWidget(self.epochs_input)

        self.auto_onnx_cb = QCheckBox("训练完成后自动熔铸为 ONNX 极速加速模型（推荐）")
        self.auto_onnx_cb.setChecked(True)
        p_row2.addWidget(self.auto_onnx_cb)
        p_row2.addStretch()
        p_layout.addLayout(p_row2)

        main_layout.addWidget(param_card)


        # ── 4. 执行行 ──
        br_layout = QHBoxLayout()
        br_layout.setSpacing(12)

        self.run_btn = QPushButton("▶  开始训练与熔铸")
        self.run_btn.setProperty("class", "PrimaryBtn")
        self.run_btn.clicked.connect(self._on_run)
        br_layout.addWidget(self.run_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedWidth(160)
        br_layout.addWidget(self.progress_bar)

        self.status_lbl = QLabel("就绪")
        self.status_lbl.setProperty("class", "SecondaryLabel")
        br_layout.addWidget(self.status_lbl)
        br_layout.addStretch()

        main_layout.addLayout(br_layout)

        # ── 5. 现代化训练日志卡片 ──
        log_card = QFrame()
        log_card.setProperty("class", "CardFrame")
        l_layout = QVBoxLayout(log_card)
        l_layout.setContentsMargins(16, 12, 16, 12)
        l_layout.setSpacing(8)

        log_title = QLabel("📋 训练实时进度与指标")
        log_title.setProperty("class", "CardTitle")
        l_layout.addWidget(log_title)

        self.log_text = QPlainTextEdit()
        self.log_text.setProperty("class", "LogConsole")
        self.log_text.setReadOnly(True)
        self.log_text.setPlaceholderText("💡 准备就绪。\n点击上方“▶ 开始训练与熔铸”后，将在此实时显示：\n  · 照片样本扫描与特征提取进度\n  · 每一轮 (Epoch) 的 Loss 损失函数与准确率\n  · ONNX 硬件加速模型的自动生成与导出状态\n")
        l_layout.addWidget(self.log_text)

        main_layout.addWidget(log_card, stretch=1)

    def _refresh_python_envs(self):
        envs = discover_python_environments()
        self.python_combo.clear()
        self.python_combo.addItems(envs)
        if envs:
            self.python_combo.setCurrentIndex(0)
            self._on_python_changed(envs[0])

    def _on_python_changed(self, py_bin: str):
        py_bin = py_bin.strip()
        if not py_bin or not Path(py_bin).exists():
            self.env_status_lbl.setText("（未指定有效 Python 路径）")
            self.env_status_lbl.setStyleSheet(f"color: {TEXT_TERT};")
            return

        def _task():
            info = probe_python_environment(py_bin)
            if info.get("error"):
                self.env_status_lbl.setText(f"⚠️ 环境异常: {info['error']}")
                self.env_status_lbl.setStyleSheet("color: #DC2626;")
            else:
                ver = info.get("version", "")
                missing = info.get("missing", [])
                if not missing:
                    self.env_status_lbl.setText(f"✅ Python {ver} · 依赖齐全")
                    self.env_status_lbl.setStyleSheet(f"color: {GREEN_FG}; font-weight: bold;")
                else:
                    self.env_status_lbl.setText(f"🟡 Python {ver} · 缺少: {', '.join(missing)} (点击训练可一键安装)")
                    self.env_status_lbl.setStyleSheet(f"color: {AMBER_FG}; font-weight: bold;")

        threading.Thread(target=_task, daemon=True).start()

    def _pick_python(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "选择 Python 解释器", "",
            "Python (python.exe python*);;All Files (*.*)"
        )
        if f:
            self.python_combo.setEditText(f)

    def _pick_dataset_dir(self):
        d = QFileDialog.getExistingDirectory(self, "选择训练数据集根目录")
        if d:
            self.ds_input.setText(d)

    def _append_log(self, msg: str):
        self.log_text.appendPlainText(msg)
        self.log_text.ensureCursorVisible()

    def _on_run(self):
        if self._running:
            return

        py_bin = self.python_combo.currentText().strip()
        if not py_bin or not Path(py_bin).exists():
            QMessageBox.critical(self, "环境错误", "请先选择一个有效的 Python 解释器路径。")
            return

        dataset_path = Path(self.ds_input.text().strip())
        if not dataset_path.exists() or not dataset_path.is_dir():
            QMessageBox.critical(self, "错误", "请先选择有效的数据集目录。")
            return

        like_dir = dataset_path / "like"
        dislike_dir = dataset_path / "dislike"
        if not like_dir.exists() and not dislike_dir.exists():
            QMessageBox.critical(
                self, "数据集格式错误",
                f"所选目录下未找到 'like' 或 'dislike' 文件夹！\n路径: {dataset_path}"
            )
            return

        try:
            epochs = int(self.epochs_input.text().strip())
            assert epochs > 0
        except Exception:
            QMessageBox.critical(self, "错误", "Epoch 必须是正整数。")
            return

        self._running = True
        self.run_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.log_text.clear()
        self.status_lbl.setText("正在检测环境…")

        self._worker = TrainWorker(
            py_bin=py_bin,
            data_dir=dataset_path,
            epochs=epochs,
            auto_onnx=self.auto_onnx_cb.isChecked(),
            backbone_type=self.backbone_combo.currentData() or "b32",
        )

        self._worker.log_sig.connect(self._append_log)
        self._worker.progress_sig.connect(lambda v: self.progress_bar.setValue(v))
        self._worker.done_sig.connect(self._on_training_done)
        self._worker.need_install_sig.connect(self._on_need_install)
        self._worker.start()

    def _on_need_install(self, py_bin: str, missing: list, ver_str: str):
        missing_str = ", ".join(missing)
        ret = QMessageBox.question(
            self, "需要安装训练依赖",
            f"检测到所选 Python 环境 (Python {ver_str}) 缺少以下依赖库：\n\n"
            f"  • {missing_str}\n\n"
            f"是否立即为您自动执行 pip 安装？\n"
            f"（将使用国内镜像源高速下载安装，安装完毕后自动开始训练）",
            QMessageBox.Yes | QMessageBox.No
        )
        if ret != QMessageBox.Yes:
            self._append_log("ℹ️ 用户取消了依赖安装，训练已终止。")
            self._finish_run()
            return

        self.status_lbl.setText("正在安装依赖…")
        self._append_log(f"📦 开始为 Python {ver_str} 安装依赖: {missing_str} ...")

        def _install_task():
            pkgs = []
            for m in missing:
                if m == "Pillow":
                    pkgs.append("Pillow")
                elif m == "onnx":
                    pkgs.extend(["onnx", "onnxscript"])
                else:
                    pkgs.append(m)

            cmd = [py_bin, "-m", "pip", "install", *pkgs, "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in iter(proc.stdout.readline, ''):
                l = line.strip()
                if l:
                    self._append_log(f"  [pip] {l}")
            proc.stdout.close()
            code = proc.wait()

            if code != 0:
                self._append_log("❌ 依赖安装失败，请检查网络连接或手动执行 pip install。")
                self.done_sig_proxy(False, "依赖自动安装失败，请查看日志详情。")
            else:
                self._append_log("✅ 依赖库安装成功！即将启动训练...\n")
                self._worker._execute_training()

        threading.Thread(target=_install_task, daemon=True).start()

    def done_sig_proxy(self, success: bool, msg: str):
        QtCore.QMetaObject.invokeMethod(self, lambda: self._on_training_done(success, msg), Qt.QueuedConnection)

    def _on_training_done(self, success: bool, msg: str):
        self._finish_run()
        if success:
            self.status_lbl.setText("训练完成")
            if self.on_model_updated:
                self.on_model_updated()
            QMessageBox.information(self, "训练成功", msg)
        else:
            self.status_lbl.setText("训练失败")
            QMessageBox.critical(self, "训练失败", msg)

    def _finish_run(self):
        self._running = False
        self.run_btn.setEnabled(True)


def launch_trainer_gui():
    from qt_theme import APP_STYLE
    app = QtWidgets.QApplication(sys.argv)
    if sys.platform == "darwin":
        app.setFont(QtGui.QFont(".AppleSystemUIFont", 12))
    elif sys.platform == "win32":
        app.setFont(QtGui.QFont("Segoe UI", 10))
    app.setStyleSheet(APP_STYLE)
    w = QtWidgets.QMainWindow()
    w.setWindowTitle("Photo Sort — 个人审美偏好训练")
    w.resize(760, 600)
    w.setCentralWidget(TrainerGUI())
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    launch_trainer_gui()
