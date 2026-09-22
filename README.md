# Imprint

Imprint 是一个本地优先的照片桌面工具，提供连拍优选、明显废片分流、个人偏好训练和自然去朦胧。照片解码、评分、增强和 DNG 导出默认都在本机完成。

没有被选中的照片不会删除，而是移到原文件夹里的 `审查_连拍淘汰` 目录。你可以慢慢检查，需要的话也能恢复。照片不会上传到云端，原图内容也不会被改动。

适合用在拍人像、运动、鸟类、飞机这类一次拍很多张的场景。

## 能做什么

- 按拍摄时间和画面相似度自动识别连拍；拍到一半停下来，或画面变化很大时，会尽量分成不同组。
- 从清晰度、曝光和审美评分几个角度给同组照片排序，默认每组保留 1 张。
- 支持在结果里查看缩略图，并手动把照片改为“保留”或“移入审查”。
- RAW 和同名 JPG 会当作一套处理，不会只移动其中一个。
- 支持常见 RAW（ARW、CR2、CR3、NEF、RAF、RW2、ORF、DNG 等），以及 JPG、PNG、WebP、TIFF、HEIC、HIF、JXL 等格式。
- 可以用自己的“喜欢 / 不喜欢”样片训练一个本地审美模型。
- 可以对 RAW 或普通图片进行自然去朦胧预览，并重新读取全分辨率原图导出 Linear/Demosaiced DNG。
- RAW 导出会用 Lensfun 匹配机身和镜头，将畸变与横向色差校正实际烘焙进像素；不依赖 Camera Raw 是否愿意为 Linear DNG 匹配外部镜头配置。

## 先知道这几件事

- 程序会移动文件，但不会删除文件。建议第一次先拿一小部分照片试跑。
- `审查_连拍淘汰` 是默认的审查文件夹，可以在界面或命令行中改名。
- AI 的选择只是初筛结果，不一定符合每个人的取舍。处理完成后建议重点看一遍每组的保留照片。
- 如果照片同时有 RAW 和 JPG，请把它们放在同一目录且保留相同文件名，程序才能正确配对。
- 原始照片不会被去朦胧功能修改、移动或覆盖；默认输出到输入目录下的 `去朦胧输出`。
- 去朦胧导出是 RGB Linear/Demosaiced DNG，不是重新生成的 Bayer/CFA RAW。

## 用桌面版

打开应用后，按这个顺序就能完成一次筛选：

1. 在“连拍优选”里选择照片所在文件夹。
2. 确认每组要保留的张数；第一次用建议保持 1 张。
3. 点击开始筛选，等待分析完成。
4. 在结果中检查照片。点“保留”或“审查”可以随时调整。
5. 没有保留的照片会进入照片目录下的 `审查_连拍淘汰` 文件夹。

界面里的常用设置：

| 设置 | 默认值 | 什么时候调整 |
| --- | --- | --- |
| 连拍时间间隔 | 1.5 秒 | 两次快门相隔超过这个时间，通常会被视为新的一组。拍得更慢可调大，快速连拍可调小。 |
| 构图容差 | 12 | 数值小，画面稍有变化就更容易拆组；数值大，追焦或甩镜头拍摄时更容易留在同一组。 |
| 每组保留 | 1 张 | 想留备选就设为 2 或 3 张。 |
| 审查目录名称 | `审查_连拍淘汰` | 只影响未保留照片被移动到的子文件夹名称。 |

## 用自己的喜好训练模型（可选）

如果你希望程序更接近自己的选片习惯，可以准备一个样片目录：

```text
我的样片/
├── like/       # 你喜欢的照片
└── dislike/    # 你不喜欢的照片
```

然后打开“偏好训练”，选择 `我的样片` 这个目录并开始训练。训练完成后，到“模型管理”切换到个人模型即可。样片和训练过程都在本机，不会上传。

## 去朦胧与镜头校正

在“自然去朦胧”页面选择照片后，可以先用低分辨率预览调整参数，再进行批量导出。正式导出会重新读取全分辨率文件，不会把预览图放大后保存。

对于带有完整机身、镜头、焦距和光圈信息的 RAW，导出流程为：

```text
RAW 解码为 16-bit Linear RGB
→ 自然去朦胧
→ Lensfun 畸变、横向色差与自动裁边
→ 写入 RGB Linear DNG
```

镜头校正直接作用于输出像素，并在 DNG 的 XMP 与 `ImageDescription` 中记录匹配到的机身、镜头和已执行项目。Camera Raw 打开文件时看到的就是校正后的像素；DNG 同时关闭二次镜头配置，避免重复变形。

当前限制：

- Lensfun 数据库没有相应配置，或 RAW 缺少可靠镜头信息时，RAW 导出会失败并给出提示，不会把未校正文件误报为成功。
- 当前 uint16 路径执行畸变、横向色差和自动裁边；暗角暂不由 Lensfun 烘焙。
- 普通 JPG、PNG 等已经处理过的 RGB 输入仍可导出 Linear DNG；没有镜头元数据时不会强制进行镜头匹配。

## 命令行用法

如果你更习惯终端，也可以直接运行。下面以 macOS / Linux 的 `python3` 为例；Windows 可以把它换成 `py` 或实际的 Python 命令。

```bash
# 筛选一个照片文件夹
python3 main.py /path/to/photos

# 每组保留 2 张，并指定审查目录名称
python3 main.py /path/to/photos --keep 2 --review-dir 待检查

# 交互式输入照片文件夹
python3 main.py --cli
```

常用选项：

```bash
python3 main.py /path/to/photos \
  --gap 1.5 \
  --hamming 12 \
  --keep 1 \
  --workers 4 \
  --no-gpu
```

- `--gap`：连拍时间间隔，单位是秒。
- `--hamming`：构图容差，范围 1–64。
- `--keep`：每个连拍组保留几张。
- `--workers`：同时分析照片的线程数；不填则自动决定。
- `--no-gpu`：关闭 GPU 加速。遇到显卡兼容问题时可以用它。

首次需要下载视觉模型时，可以运行：

```bash
python3 main.py --download-models
```

## 从源码运行

推荐使用 Python 3.11、Node.js 18 或更高版本，以及 Rust 稳定版工具链。

```bash
git clone https://github.com/luqie0106/Imprint.git
cd Imprint

# 安装 Python 依赖
pip install -r requirements.txt

# 安装桌面端依赖并启动
cd tauri-frontend
npm install
npm run tauri dev
```

构建安装包：

```bash
cd tauri-frontend

# 先打包 Python 后端
npm run build:api

# 再构建桌面端安装包
npm run tauri build
```

## 目录说明

```text
Imprint/
├── main.py                 # 命令行入口
├── src/                    # 照片分组、评分、模型和本地服务
├── tauri-frontend/         # 桌面端界面
├── models/                 # 本地模型文件
├── tests/                  # 测试
├── third_party/            # Lensfun 数据库快照与第三方许可证
├── THIRD_PARTY_NOTICES.md  # 第三方组件、版本、来源和许可证说明
└── requirements.txt        # Python 依赖
```

## 遇到系统拦截

Windows 的 SmartScreen 提示可以点击“更多信息”后选择“仍要运行”。

macOS 如果提示应用无法验证开发者，请到“系统设置 → 隐私与安全性”中选择“仍要打开”。若仍无法打开，可在确认应用来源可信后执行：

```bash
xattr -cr /Applications/Imprint.app
```

## 开源协议

Imprint 自有代码使用 [Apache 2.0](LICENSE) 协议。镜头校正功能还使用以下独立第三方组件：

- Lensfun 动态库：LGPL-3.0；Imprint 不修改该库。
- Lensfun 镜头数据库：CC BY-SA 3.0。仓库内数据库由 Lensfun 官方工具转换为兼容格式，继续按相同许可证提供。
- lensfunpy：MIT。

这些许可证不会改变 Imprint 自有代码的 Apache-2.0 许可证，也不会对用户的照片或导出 DNG 施加开源要求。完整版本、来源、转换说明和许可证文本见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 与 [`third_party/licenses`](third_party/licenses)。
发布包会把 Lensfun 保留为可替换的独立动态库，不会合并进单文件可执行程序。
