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

- **连拍自动分组**：结合 EXIF 亚秒级拍摄时间（$\le 1.5$ 秒）与 dHash 图像感知哈希，自动识别同一连拍序列。长距离追焦移镜时，画面发生明显变化会自动拆分子组；三脚架定机位摆拍不会因时间不同被误合并。
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
- **连拍优选**：选择照片文件夹，设置连拍时间阈值（默认 1.5s）和每组保留张数，点击开始筛选。
- **偏好训练**：选择包含 `like/` 与 `dislike/` 的数据集根目录，本地训练专属审美模型并自动熔铸为 ONNX 模型。
- **模型与环境**：查看本地基础模型（CLIP）与 ONNX 状态，支持一键从国内镜像下载。

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

## 训练个人审美偏好

如果希望系统按照你的个人喜好（构图、色彩偏好）来选片：

1. 准备样本文件夹，结构如下：
   ```
   dataset/
   ├── like/       # 放入满意的精选照片（建议 30~100 张以上，格式不限）
   └── dislike/    # 放入不满意的废片/构图不佳照片
   ```
2. 打开软件切到【偏好训练】页，选择 `dataset` 目录。
3. 点击【开始训练与熔铸】，训练完成后会自动生成 `photo_sort_model.onnx`。
4. 返回【连拍优选】页即可直接使用新模型进行硬件加速筛选。

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
