# TicoRAW 接入 Imprint：C++ / Metal 生产路径

日期：2026-10-12。已接入当前项目源代码，并在本机完成原生库构建和真实 DNG 后端导出。没有打开 Photoshop、Lightroom 或 Camera Raw，没有提交、推送或发布安装包。

## 接入结果

新增 `native-raw` 原生库：C++ 负责严格码流解析、GCLI、幅值/符号、逆量化、水平/垂直小波逆变换及跨 tile 状态；Metal 重写两个完整图像的整数重建算子（4-to-2 平面合并与 Bayer/LUT 输出）。Metal 不可用时采用同等 C++ 算子，生产结果没有 Python 熵解码或重建回退。

LibRaw 保留 NEF 容器元数据、参数解析、传递曲线以及 C++ 去马赛克/颜色处理。其原有 TicoRAW precinct、IDWT、tile 和 Bayer 计算源文件从构建中排除；符号检查也没有这些公开数学函数。格式几何和参数 helpers 仍复用固定源码，不能把本次称为完全独立恢复的 Adobe 实现。

`src/raw_decoder.py` 是内存/接口适配：用 C++ 仅读元数据判断 Nikon MakerNote 的 NEFCompression 13/14，再选择原生 HE 路径。普通 RAW 仍由原 rawpy 返回。适配器将原生结果复制进自有数组，不计算这些结果。正常元数据、内嵌缩略图和方向读取保持原来的快速路径。

实际 RGB/CFA 接入点包括 `image_io.py` 的预览/完整读图及 `enhanced_dng_source_data`、`camera_profile.py`、连拍的半尺寸 RGB 解码和 CLIP 的解码 fallback。文件移动、审查、增强算法、DNG 写入规则及 UI 布局未改动。

构建使用固定 SHA-256 源码下载，已实际完成默认下载配置和本地构建，不依赖研究临时目录。PyInstaller 配置、CI 构建步骤、缓存键和许可证同步更新；没有安装/替换 rawpy 或系统依赖。

## 实测

真实样本：`test_images/LCR_3520.NEF`，6064×4040、Nikon Z6 III HE*。

| 检查 | 结果 |
| --- | --- |
| 新 C++ 全码流的 A/B scatter | 与旧研究模型 28,790,784 个 int32 全部一致；只作验收，不作生产输入 |
| 强制 C++ CPU | 实际 backend=`c++/cpu`，24,498,560 个 CFA 点差异 0 |
| 强制 Metal | Apple M2 Pro，实际 backend=`c++/metal`，同样差异 0，没有静默 CPU 回退 |
| CPU/Metal 线性 RGB 预览 | 3032×2020×3 uint16，完整数组 SHA-256 相同 |
| C++ 小平面重建 | identity LUT、±5000 输入、5 种尺寸与 strided 布局，逐值相同；避免饱和掩盖错误 |
| Metal 合成算子 | 1×1、1×19、7×1、32×64，真实 GPU 与 C++ 参照逐值相同 |
| 垂直 lifting | 两种 tile 初态、carry、输出与两次 tail flush 对照通过 |
| 普通 Lossless NEF | `LCR_1132.NEF` 仍返回原始 RawPy 对象 |
| 真实预览接口 | 返回 HTTP 200、900×600 JPEG；已实际执行新原生解码 |
| 现有真实导出 worker | `success=1, failed=0`，Lensfun 几何处理和 Metal 后端完成，原图 SHA-256 不变，XMP unchanged |
| 导出 CFA 再读 | 6064×4040，差异 0；原始 14-bit CFA 未被 RGB 渲染替换 |

CFA SHA-256：`5c76848dadbd2682b1abd444d32551c0000fe6d9a56c0e638582248cbac59ed2`。

线性半尺寸 RGB SHA-256：`805323b03564c0385b55924399714874f269a9635c6e3ee93c928ec67e917dc9`。

完整证据：[原生接入与导出 JSON](evidence/lcr3520-native-integration.json)。测试产物：[实际后端生成的 DNG](/private/tmp/imprint-native-tico-acceptance/export/LCR_3520_dehaze.dng)，输出在临时目录，未写入照片目录或加入 Git。

[预览接口证据](evidence/lcr3520-native-preview.json)；[截断 NEF 和损坏 precinct 长度拒绝检查](evidence/ticoraw-native-damaged-inputs.json)。这两个损坏流均明确失败，没有补零图像或 Python 回退。

## DNG 结构检查与边界

本轮沿用现有写入器，实际产物包含：

- DNGVersion 1.7.0.0。
- CFA SubIFD：6064×4040，14-bit，Lossless JPEG，WhiteLevel 16383。
- Enhanced Linear Raw SubIFD：6064×4040，16-bit RGB，JPEG XL，WhiteLevel 65535。
- JPEG 显示预览：4096×2729；另有 RGB thumbnail。

ExifTool 普通 validate 返回大型数组未解码的 minor 提示；启用其大型数组检查方式 `-m` 后 Validate=OK，但 ExifTool 的 PreviewJXL RawConv 对 TileByteCounts 数组另给出非数值提示。**没有将它描述成“零警告”。**CFA 已用当前 rawpy 实际解码并逐点验证。尚未进行 Adobe 实际导入，增强层的 Adobe 显示效果由用户验证。

## 回归及当前限制

相关 RAW/DNG/元数据测试 138 passed，新增适配器测试其中 11 passed；Python 语法、原生 CMake 构建、C++ 严格编译和 `git diff --check` 通过。

此前范围较大的测试运行有 161 passed、1 failed：`test_enhance_preview_passes_progressive_levels_to_native_cache` 期待 L2 为 /4，但现有实现为 /3。主代理直接加载未修改 HEAD 的 `app_api.py` 重跑该项，同样失败（20×13 vs 15×10）。未删除、跳过或改弱该测试，也未扩大本次范围去改预览尺寸策略。

真实完整图像目前仅覆盖本地这张 HE*。Windows/其他相机及更多模式未作本机运行验收；CI 已添加构建路径但尚未在远端执行。当前安装的旧版应用没有被重新打包；从当前项目启动或重启后端会加载 `native-raw/build/libimprint_raw.dylib`。原生库实际执行已验证，不能把它等同于已安装发布包的验证。

构建和后端开关见 [native-raw 指南](../../../native-raw/README.md)。`auto` 默认优先 Metal，CPU 回退仍是 C++；设置 `IMPRINT_TICORAW_BACKEND=metal` 时 GPU 不可用则明确失败。用户可用当前项目版 Imprint 导入样本并正常导出 DNG；Camera Raw/Lightroom 实际视觉验收仍待用户进行。
