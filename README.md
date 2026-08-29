# Imprint

[![Version](https://img.shields.io/badge/Version-2.0.6-6366F1?logo=v&logoColor=white)](https://github.com/luqie0106/Imprint/releases)
[![Tauri](https://img.shields.io/badge/Tauri-2.0-FFC131?logo=tauri&logoColor=black)](https://tauri.app/)
[![Vue 3](https://img.shields.io/badge/Frontend-Vue%203%20%7C%20TailwindCSS-4FC08D?logo=vuedotjs&logoColor=white)](https://vuejs.org/)
[![Python](https://img.shields.io/badge/Python-3.9%20~%203.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Imprint 是一个运行在本地的连拍照片智能筛选与优选工具，基于 Tauri 2.0 与 Python 构建。

主要面向人像、体育、生态、航空等高频连拍摄影场景。程序会自动根据拍摄时间与画面相似度对连拍进行分组，综合对焦清晰度（人脸优先）、曝光以及可选的美学模型选出每组中的最佳照片，并将多余废片整理到独立的审查目录中。

---

## 特性介绍

- **连拍自动分组**：结合 EXIF 亚秒级拍摄时间与图像感知哈希（dHash），精准识别连拍序列。支持长距离追焦移镜时自动拆分子组，避免三脚架定点延时被误合并。
- **RAW + JPG 伴生文件绑定**：自动识别同一次快门生成的 RAW+JPG（如 `.ARW` + `.JPG`、`.CR3` + `.JPG`），在选优或移动时保持同步，绝不拆散。
- **多维度选优评估**：
  - **对焦清晰度**：自动检测人脸区域并优先计算面部/眼部清晰度，无人脸时基于中心主体区域计算。
  - **曝光评估**：直方图高光与暗部统计，对不可挽回的高光死白施加扣分。
  - **美学与构图打分（可选）**：基于轻量化 CLIP 视觉底座（ONNX Runtime 加速），量化画面构图。
- **个人审美偏好微调**：只需提供 `like` 与 `dislike` 两个样本文件夹，即可在本地一键微调专属审美模型。
- **全格式支持**：
  - **相机 RAW**：索尼 (`.ARW`)、佳能 (`.CR2`/`.CR3`)、尼康 (`.NEF`)、富士 (`.RAF`)、松下/徕卡 (`.RW2`)、奥林巴斯 (`.ORF`)、DNG（含大疆/理光 GR/iPhone ProRAW）等。
  - **高效格式**：索尼/佳能 10-bit `.HIF`、苹果 `.HEIC`、JPEG XL (`.JXL`)。
  - **通用格式**：`.JPG`、`.JPEG`、`.PNG`、`.WebP`、`.TIFF` 等。
- **安全与非破坏性**：仅读取照片缩略图与元数据，绝不修改原图；淘汰照片统一移动至子目录 `审查_连拍淘汰`，方便随时复核。
- **现代化桌面体验**：支持深色模式 (Dark)、浅色模式 (Light) 与跟随系统自动切换，支持 Windows 11 Mica 材质与 macOS 毛玻璃效果。

---

## 常用参数说明

| 参数项 | 说明 | 调节建议 |
| :--- | :--- | :--- |
| **连拍时间间隔 (秒)** | 相邻快门的最大时间间隔 | **默认 1.5 秒**。超过该停顿时间将自动开始新的一组。 |
| **汉明距离 (构图容差)** | 画面结构感知哈希差异（1~64） | **默认 12**。调小（如 6~8）对构图变动更敏感；调大（如 16~20）允许甩镜头追焦仍保持同组。 |
| **每组保留张数** | 每个连拍组中优选出的照片数量 | **默认 1 张**。可根据需要设为保留 2~3 张备选。 |
| **并发分析线程数** | 多线程读取 RAW 预览与计算锐度 | 默认按 CPU 逻辑核心数的 80% 分配。 |

---

## 命令行模式 (CLI)

除了桌面客户端，也可以直接通过终端运行批处理：

```bash
# 1. 快速筛选指定照片目录
python main.py /path/to/photos

# 2. 自定义参数运行
python main.py /path/to/photos --gap 1.5 --hamming 12 --keep 1 --review-dir 审查_连拍淘汰

# 3. 交互式终端模式
python main.py --cli

# 4. 下载/同步基础模型到本地 models/ 目录
python main.py --download-models

# 5. 导出 ONNX 模型
python main.py --export-onnx
```

---

## 本地开发与构建

### 环境要求
- **Python**: 3.9 ~ 3.13（推荐 Conda Python 3.11）
- **Node.js**: >= 18.0.0
- **Rust**: 稳定版工具链（`cargo` / `rustc`）

### 启动开发
```bash
# 1. 克隆代码库
git clone https://github.com/luqie0106/Imprint.git
cd Imprint

# 2. 安装 Python 核心依赖
pip install -r requirements.txt

# 3. 安装前端依赖
cd tauri-frontend
npm install

# 4. 启动开发模式（自动拉起桌面窗口与本地服务）
npm run tauri dev
```

### 构建打包
```bash
cd tauri-frontend

# 打包 Python Sidecar 二进制
npm run build:api

# 构建桌面端安装包 (macOS .dmg / Windows .zip)
npm run tauri build
```

---

## 目录结构

```
Imprint/
├── src/                               # 核心算法与 FastAPI Sidecar 服务端
│   ├── app_api.py                     # FastAPI 后端 (RESTful & SSE 实时流)
│   ├── burst_filter.py                # 连拍聚类、EXIF 与多维打分引擎
│   ├── model_manager.py               # AI 视觉模型下载、激活与权重管理
│   ├── onnx_exporter.py               # 权重微调与 ONNX 导出
│   ├── exif_reader.py                 # EXIF 与亚秒时间提取
│   └── config.py                      # 全局常量配置
├── tauri-frontend/                    # Tauri 2.0 桌面端与 Vue 3 前端
│   ├── src-tauri/                     # Rust 宿主（窗口管理、Sidecar 调度、主题渲染）
│   ├── src/                           # Vue 3 页面组件与状态管理
│   └── package.json                   # 前端构建配置
├── models/                            # 本地 AI 模型与 ONNX 权重目录
├── main.py                            # 纯命令行 CLI 批处理入口
├── app_api.spec                       # PyInstaller Sidecar 打包配置
├── requirements.txt                   # Python 运行依赖
├── requirements-build.txt             # Python CI 打包依赖
└── .github/workflows/build.yml        # GitHub Actions 跨平台自动构建流
```

---

## 常见问题 (FAQ)

### 系统拦截提示如何处理？

- **Windows 用户（SmartScreen 提示）**：点击窗口中的「更多信息」→「仍要运行」即可。
- **macOS 用户（提示无法验证开发者或已损坏）**：
  - 前往「系统设置」→「隐私与安全性」，点击「仍要打开」；
  - 或在终端执行解除隔离命令：
    ```bash
    xattr -cr /Applications/Imprint.app
    ```

---

## 开源协议

本项目采用 [Apache 2.0](LICENSE) 协议开源。
