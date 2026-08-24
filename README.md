# Photo Sort

[![Python](https://img.shields.io/badge/Python-3.9%20~%203.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![GUI](https://img.shields.io/badge/GUI-PySide6%20%2F%20Qt6-41CD52?logo=qt&logoColor=white)](https://doc.qt.io/qtforpython/)
[![Inference](https://img.shields.io/badge/Inference-ONNX%20Runtime%20%7C%20DirectML%20%7C%20CoreML-005CED?logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![Formats](https://img.shields.io/badge/Formats-RAW%20%7C%20JPEG%20%7C%20JXL%20%7C%20HIF%2FHEIF-FF6F00)]()
[![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Windows-lightgrey?logo=apple&logoColor=black)]()
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-22%2F22%20Passed-brightgreen)]()

本地连拍照片智能筛选与优选工具。

专为高频连拍场景（航空、体育、人像、生态等）设计，自动根据拍摄时间与画面相似度对连拍进行分组，结合局部清晰度（人脸优先）、曝光以及本地 AI 模型选出每组中的最佳照片，并将多余废片整理到审查目录。

---

## 主要功能

- **连拍自动分组**：结合 EXIF 亚秒级拍摄时间（1.5秒）与 dHash 图像感知哈希，自动识别同一连拍序列。长距离追焦移镜时，画面发生明显变化会自动拆分子组；三脚架定机位摆拍不会因时间不同被误合并。
- **RAW + JPG 伴生双文件智能绑定**：自动识别相机同拍伴生文件（如 `_DSC0001.ARW` + `_DSC0001.JPG`、`IMG_0001.CR2` + `IMG_0001.JPG` 等）并视为单次快门实体。单拍不会被误判为连拍组；连拍优选时优胜快门的 RAW 与 JPG 同时保留，淘汰快门的 RAW 与 JPG 同步移动，绝不拆散。
- **多维度选优打分**：
  - **清晰度（人脸优先）**：自动检测人脸区域并计算 Laplacian 梯度；无人脸时以画面中心主体区域为主。
  - **曝光评估**：统计直方图高光与暗部，对高光过曝（死白）施加惩罚。
  - **AI 美学与构图打分（可选）**：基于本地 CLIP 视觉特征，量化整体构图与美感。
- **个人审美偏好微调**：内置轻量训练器，只需提供 `like` 和 `dislike` 两个文件夹，即可在本地训练自己的审美模型，并一键导出为 ONNX 格式调用 NPU / GPU 硬件加速。
- **全格式支持**：
  - **相机 RAW**：Adobe DNG (`.DNG`，含大疆无人机/徕卡/Apple ProRAW/理光GR/宾得)、佳能 (`.CR2`/`.CR3`/`.CRW`)、松下/徕卡 (`.RW2`/`.RAW`)、尼康 (`.NEF`/`.NRW`)、索尼 (`.ARW`/`.SRF`/`.SR2`)、富士 (`.RAF`)、奥林巴斯 (`.ORF`)、宾得 (`.PEF`) 等。
  - **高效率格式**：索尼/佳能 10-bit `.HIF`、苹果 `.HEIC`/`.HEIF`、JPEG XL (`.JXL`)。
  - **通用格式**：`.JPG`、`.JPEG`、`.PNG`、`.WebP`、`.TIFF`、`.BMP`。
- **安全与非破坏性**：只读取元数据与缩略图，不修改原片，淘汰的照片移动到同目录下的 `审查_连拍淘汰` 文件夹，方便随时人工复查。

---

## 💡 核心概念与算法参数通俗释义

为了方便摄影师理解，以下对程序中涉及的技术术语与调节参数进行通俗说明：

| 术语 / 参数 | 通俗含义 | 推荐设置与调节建议 |
| :--- | :--- | :--- |
| **汉明距离 (dHash 构图容差)** | **画面结构相似度**。算法将照片压缩为 64 位结构指纹，汉明距离代表两张照片构图指纹**不同的位数**（取值 1~64）。 | · **默认值 12**：适合绝大多数手持常规连拍。<br>· **调小（6~8）**：对构图要求极严格，轻微移镜即拆分组（适合三脚架定点摆拍）。<br>· **调大（16~20）**：允许奔跑追焦、大幅度甩镜头仍聚在同一连拍组。 |
| **连拍时间间隔 (秒)** | **连拍判定时间窗口**。相邻快门在 EXIF 亚秒时间轴上的最大间隔秒数。 | · **默认 1.5 秒**：在此时间内连续按下快门被视为同一连拍序列；停顿超过该时间自动开启新组。 |
| **RAW + JPG 伴生绑定** | **同拍双格式一体化**。自动将同一次快门生成的 RAW+JPG 聚合为单一实体。 | · 自动生效。单拍不会被误判；优选时胜出者的 RAW+JPG **一同保留**，淘汰时**一同移动**，绝不拆散。 |
| **人脸优先 Laplacian 清晰度** | **对焦实不实检测**。通过计算高频边缘反差衡量清晰度，自动识别合焦最实、未脱焦的锐利照片。 | · 自动检测画面人脸，优先计算**面部与眼部区域**清晰度；无人脸时检测中心主体区域。 |
| **直方图曝光评分** | **高光过曝与欠曝检测**。统计画面像素分布，对高光死白（过曝不可挽回）施加严厉扣分惩罚。 | · 自动生效。防止算法将对焦清晰但严重过曝的大白脸选为最佳照片。 |
| **CLIP 视觉底座与 ONNX 熔铸** | **AI 构图与美感引擎**。利用 OpenAI 视觉大模型提取 512 维高阶特征，结合个人偏好微调分类头。 | · 熔铸为单个 `.onnx` 静态图后，在 CPU / GPU / NPU 上毫秒级极速推理，无需加载数十 GB 的 PyTorch 运行库。 |

---


## 运行环境与安装

建议使用 Python 3.9 ~ 3.13（推荐 Conda 环境）：

```bash
# 克隆仓库
git clone https://github.com/luqie0106/photo_sort.git
cd photo_sort

# 安装依赖
pip install -r requirements.txt
```

---

## 使用说明

### 1. 图形界面（默认）

运行主程序打开桌面界面：

```bash
python main.py
```

界面包含三个标签页：
- **连拍优选**：选择照片文件夹，设置连拍时间阈值（默认 1.5s）和每组保留张数，点击开始筛选。顶部常驻显示当前激活的 AI 美学引擎状态。
- **偏好训练**：选择包含 `like/` 与 `dislike/` 的数据集根目录，本地训练专属审美模型并自动熔铸为 ONNX 模型。
- **模型与环境**：
  - **美学评分模型选择**：在「官方标准通用模型」与「个人专属训练模型」之间一键自由切换。
  - **模型文件状态**：查看与管理本地 `models/` 目录中的标准模型、个人专属模型及 CLIP 基础底座。

### 2. 命令行模式 (CLI)

适合脚本批量处理或无界面的终端环境：

```bash
# 交互式命令行运行筛选
python main.py --cli

# 下载/同步基础模型到本地 models/ 目录
python main.py --download-models

# 从现有权重直接导出 ONNX 模型
python main.py --export-onnx
```

---

## 模型体系与个人审美偏好训练

PhotoSort 采用 **「出厂预置权威标准模型 + 随时切换个人微调模型」** 的双轨架构：

1. **🌟 官方标准通用模型（开箱即用，已内置）**：
   - 基于全球摄影美学权威数据集（LAION / AVA 25万+ 张多题材专业摄影打分）预训练。
   - 对人像、风光、街拍、生态、夜景等全品类摄影题材进行中立客观的构图与画质评估。
   - **默认内置于各平台安装包中**（`models/standard_aesthetic_model.onnx`，~335MB），下载安装后无需联网或额外配置，即可拥有开箱即用的 AI 构图打分能力。

2. **🧠 个人专属微调模型（个性化偏好，支持 GUI 一键导入 / 本地训练）**：
   - **轻量分发设计**：为了避免安装包体积过大（节省 336MB+ 下载流量与解压体积），各平台安装包**默认不捆绑**个人专属 ONNX 模型。
   - **GUI 一键导入与熔铸（推荐）**：
     - 下载 GitHub 仓库中作者预置的轻量分类头权重 [`models/aesthetic_mlp.pth`](file:///models/aesthetic_mlp.pth)（仅 500KB）或他人分享的 `.pth` 权重。
     - 打开软件切换至 **【📦 模型与环境】** 页面。
     - 在「个人专属训练模型」区域直接点击 **「📥 导入 .pth 权重文件」** 选取下载的权重。
     - 系统将自动导入并提示是否立即 **「⚡ 一键熔铸为 ONNX 模型」**，熔铸完成后自动切换并启用，**完全无需手动查找软件安装路径或运行命令行**！
   - **本地训练专属模型**：
     - 准备样本文件夹：
       ```
       dataset/
       ├── like/       # 放入满意的精选照片（建议 30~100 张以上，格式不限）
       └── dislike/    # 放入不满意的废片/构图不佳照片
       ```
     - 打开软件切换至【偏好训练】页，选择 `dataset` 目录。
     - 点击【开始训练与熔铸】，训练完成后自动生成 `models/custom_aesthetic_model.onnx`。
     - 在【模型与环境】面板勾选「个人专属训练模型」即可一键启用。


---

## 编译打包与发布

项目使用 PyInstaller 进行轻量化打包，并由 GitHub Actions Workflow 自动完成双平台安装包与便携版的构建与发布：

```bash
# 安装打包依赖
pip install -r requirements-build.txt

# 打包生成可执行文件
pyinstaller photo_sort.spec
```

### GitHub Actions 发布的产物说明

| 平台 | 产物文件名 | 类型 | 适用场景 |
| :--- | :--- | :--- | :--- |
| **macOS** | `PhotoSort-macOS-Installer.pkg` | **系统原生安装包** | 推荐使用，双击系统向导一键自动安装至 `/Applications` |
| **macOS** | `PhotoSort-macOS-Portable.dmg` | **磁盘镜像 (便携版)** | 经典 DMG 镜像，内含 App 与 Applications 快捷链接，支持拖拽运行 |
| **Windows** | `PhotoSort-Windows-Installer.exe` | **向导安装包** | 推荐使用，提供中英双语向导、自定义路径、开始菜单与桌面快捷方式、控制面板卸载 |
| **Windows** | `PhotoSort-Windows-Portable.zip` | **绿色免安装版** | 解压即用，适合放在 U 盘或临时目录运行 |

---

## ❓ 常见问题与安全/安装提示 (FAQ)

### Q: 如何解除系统拦截并正常安装使用？

#### 🪟 Windows 用户
1. **浏览器拦截**：点击下载栏提示的「保留」或「仍要下载」。
2. **SmartScreen 拦截提示“Windows 已保护你的电脑”**：
   - 点击提示窗口中的 **「更多信息」**。
   - 点击右下角的 **「仍要运行」** 即可正常进入安装向导或打开软件。

#### 🍎 macOS 用户
1. **提示“无法打开，因为无法验证开发者”或“来自身份不明的开发者”**：
   - 按住键盘 `Control` 键并右键点击应用图标，选择 **「打开」**，在弹出的确认窗口中点击 **「打开」**。
   - 或打开 **「系统设置」 $\rightarrow$ 「隐私与安全性」**，下滑至安全性区域，点击 **「仍要打开」**。
2. **提示“应用已损坏，移到废纸篓”**（因 macOS 对下载文件的隔离属性导致）：
   - 打开终端（Terminal），执行以下命令解除隔离：
     ```bash
     sudo xattr -r -d com.apple.quarantine /Applications/PhotoSort.app
     # 或
     xattr -cr /Applications/PhotoSort.app
     ```

---

### Q: 如何验证下载文件的完整性？
每次发布新版本时，GitHub Actions 均在公开透明的隔离容器中自动编译并生成 Release。您可以在 [Releases 页面](https://github.com/luqie0106/photo_sort/releases) 查看对应版本的 SHA256 校验和，并在本地使用以下命令校验：
- **Windows (PowerShell)**:
  ```powershell
  Get-FileHash PhotoSort-Windows-Installer.exe -Algorithm SHA256
  ```
- **macOS / Linux**:
  ```bash
  shasum -a 256 PhotoSort-macOS-Installer.pkg
  ```

---

### Q: 仍然担心预编译包的安全性？
如果您对二进制安装包有所顾虑，推荐直接通过 **Python 源码运行**，环境与代码完全由您自行掌控：
```bash
git clone https://github.com/luqie0106/photo_sort.git
cd photo_sort
pip install -r requirements.txt
python main.py
```


---

## 目录结构

```
photo_sort/
├── main.py                     # 程序入口（GUI / CLI）
├── photo_sort.spec             # PyInstaller 打包配置
├── requirements.txt            # 运行依赖
├── requirements-build.txt      # 构建依赖
├── src/
│   ├── app_gui.py              # 主界面 (PySide6)
│   ├── burst_gui.py            # 连拍优选面板
│   ├── trainer_gui.py          # 偏好训练面板
│   ├── qt_theme.py             # 界面样式与主题
│   ├── burst_filter.py         # 连拍聚类、EXIF 与打分核心逻辑
│   ├── model_manager.py        # 模型下载与状态管理
│   └── onnx_exporter.py        # ONNX 导出工具
└── tests/                      # 单元测试套件
```

---

## 开源协议

本项目采用 [Apache 2.0](LICENSE) 协议开源。
