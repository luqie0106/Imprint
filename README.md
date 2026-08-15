# photo_sort (NEF 连拍优选与个人审美训练器)

本工具专为摄影师（特别是航空、体育摄影等高频连拍场景）设计，用于自动筛选和淘汰冗余的连拍照片。

## 核心功能

1. **NEF 连拍优选 (双剑合璧)**:
   - **AI 审美初筛**: 基于轻量级 MobileNetV3 视觉模型打分。如果连拍组内所有照片构图都不合格（比如没拍全、严重失焦），将触发“全军覆没”机制，整组淘汰。
   - **OpenCV 锐度终选**: 在 AI 构图及格的照片中，使用拉普拉斯方差（Laplacian Variance）进行像素级分析，精准挑出最清晰、最锐利的一张保留。
2. **个人审美偏好训练器**:
   - 随附专属图形化训练工具。只需将喜欢的和淘汰的 NEF 原图分别放入 `like` 和 `dislike` 文件夹。
   - 一键训练，自动微调出属于你个人审美的 `aesthetic_model.pth`，并与优选主程序无缝联动。

## 安装与环境配置

本项目支持 `Conda` 环境或标准的 `Python 虚拟环境`。由于涉及 OpenCV 和 PyTorch 等科学计算库，强烈推荐隔离环境运行。

### 方式一：使用 Conda 环境（推荐）

```bash
# 1. 创建并激活 conda 环境
conda create -n photo_sort python=3.10
conda activate photo_sort

# 2. 安装依赖
pip install -r requirements.txt
```

### 方式二：使用标准 Python 虚拟环境 (venv)

```bash
# 1. 创建虚拟环境
python3 -m venv .venv

# 2. 激活虚拟环境
# Mac / Linux:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# 3. 安装依赖
pip install -r requirements.txt
```

## 运行程序

### 1. 运行连拍优选主程序

主程序默认以图形化界面 (GUI) 启动：

**如果你使用 Conda：**
```bash
# 如果你已经在激活的 conda 环境里，直接运行：
python main.py

# 如果你想直接通过 conda 命令运行（无需事先激活环境）：
conda run -n photo_sort python main.py
```

**如果你使用标准虚拟环境 (venv)：**
```bash
# 请确保此时终端前缀带有 (.venv)
python main.py
```

如果你希望以**纯命令行交互模式**运行，请附加 `--cli` 参数：
```bash
python main.py --cli
```

### 2. 运行美学模型训练器

当需要更新你的“专属模型”时，运行独立的训练工具：

```bash
python src/trainer_gui.py
```

在打开的界面中，选择包含 `like` 和 `dislike` 两个子目录的数据集文件夹，设定好训练轮数并点击“开始训练”。训练产出的 `aesthetic_model.pth` 将自动保存在项目根目录，供下一次 `main.py` 运行时直接调用。
