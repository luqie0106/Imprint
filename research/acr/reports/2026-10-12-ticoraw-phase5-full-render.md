# TicoRAW：完整熵解码、CFA 重建与彩色渲染

日期：2026-10-12。已找到并实际运行能解码及渲染本地 Nikon Z6 III HE* 样本的路径。研究保持隔离，没有接入 Imprint，没有修改照片、Adobe 二进制或 Ghidra 的函数名、类型及注释。

## 结果与证据

输入为 `test_images/LCR_3520.NEF`，SHA-256：`6c2276300688850beaae7c997f39ad0a8470333f6ffdf3f33876fa658ae0f313`。它是本地 14 张 NEF 中唯一的 HE* 样本，其余为 Lossless。因此本次结果不能泛化为所有 Nikon 相机、HE/HE* 配置或 TicoRAW profile。

| 验证层 | 实测结果 |
| --- | --- |
| 独立 Python 熵解码 | 全部 1,010 个不同 precinct，64 个 tile，共 1,136 次调用 |
| 完整 A/B scatter 与公开 C++ 参照 | 28,790,784 个 int32，差异 0；包括缓冲填充槽 |
| Python 系数 → 公开小波重建 → CFA | 6064×4040，24,498,560 个 photosite，与完整公开解码基线差异 0 |
| CFA 范围 | 974–3551，原始 14-bit 传感器域；黑/白电平为 1008/16383 |
| CFA → LibRaw 彩色渲染 | 6064×4040，3 通道，16-bit RGB；使用相机 WB、AHD、相机矩阵与 sRGB 输出空间 |
| 彩色输出与基线 | 完整 PPM 逐 byte 相同，不仅是缩略图相似 |
| ASan / UBSan | 主代理另链接实际 `build/san/libraw.a` 重跑完整重建和渲染，stderr 0 byte，输出哈希相同；macOS 上关闭 LeakSanitizer |
| 错误处理 | 真实 magnitude 子流截断时明确报错，未补零；错误长度的外部 CFA 被拒绝；产物拒绝覆盖 |
| 可见结果 | 已检查整图及跨 tile 行位置的 800×600 原尺寸裁剪，海面、飞鸟、岸线连贯；这不是 Adobe 颜色等价验证 |

CFA SHA-256：`5c76848dadbd2682b1abd444d32551c0000fe6d9a56c0e638582248cbac59ed2`。

完整 RGB16 PPM SHA-256：`44532e6278e752be24b67a57f5dde9369deb18fd901bdbdca1ba96954b9a2afc`。

证据：

- [连续熵解码比较](evidence/lcr3520-continuous-entropy.json)。
- [最终 CFA 像素比较](evidence/lcr3520-python-entropy-cfa.json)。
- [完整重建 sanitizer](evidence/lcr3520-python-reconstruction-sanitizer.json)。
- [彩色输出、完整 sanitizer 与 TIFF 往返比较](evidence/lcr3520-ticoraw-render-comparison.json)。
- [系数注入有效性检查](evidence/ticoraw-scatter-injection-check.json)。
- [重建输入长度与不覆盖检查](evidence/ticoraw-reconstruction-rejections.json)。

## 这条路径具体实现了什么

```text
真实 NEF 压缩 strip
  → Python 严格 bit reader：GCLI / 幅值 / 符号 / 逆量化
  → 独立计算的 A/B scatter（每调用两个 12672-int32 buffer）
  → 固定公开实现的小波逆变换、tile carry、transfer curve、Bayer 重建
  → 全尺寸 RGGB CFA
  → LibRaw 相机 WB、去马赛克、颜色矩阵、默认 gamma / 自动亮度
  → 全尺寸 RGB16 PPM / TIFF，另生成 JPEG 浏览图
```

Python 从压缩载荷计算系数，参照数组只在结果生成后参与比较。熵模型使用从 ACR 已保存工程读取的两张表；profile 几何、GTLI/WGT 映射、预测路由、mode 0x70 的处理及逆量化 shift 仍由公开源码辅助。**本次不是完全独立恢复的 Adobe 解码器。**

公开来源为此前固定的 [redlamp-libraw 4abfcd2](https://github.com/pdcgomes/redlamp-libraw/tree/4abfcd2ab82f2211b1f7e7c3cd83974c87e0d812)，源码来自已核对 SHA-256 的固定压缩包，目录位于 `/private/tmp/imprint-tico-4abfcd2`。该目录没有 Git 元数据，不把 harness 内的 commit 字符串当作单独的源码身份校验。优化静态库 SHA-256 为 `8707f43a497f9ca9596b2f9dd138aa53428c05b8180b29aa3fbb219e7bc52dbf`。

`reconstruct_ticoraw_scatter.cpp` 在链接时提供自己的 `decode_precinct`，只消费 Python 输出，不执行公开熵解码函数。主代理检查了符号：重建程序有替代的 `decode_precinct`，没有系数解包、GCLI 或逆量化函数。将输入第一个逆量化系数从 -12576 改为 -8480 后，输出有 292 个 photosite 改变，最大变化 344；这用于确认程序确实使用了外部系数。

渲染 harness 先解包 NEF 取得完整 LibRaw 工作状态，再按 `raw_pitch` 逐行用外部 CFA 覆盖内存，之后调用 `dcraw_process`。所有覆盖只发生于内存；NEF 不写入。其普通解包仍调用公开解码器，因此这里没有声称整个渲染程序不包含公开熵解码；**实际用于颜色渲染的 CFA 是独立 Python 系数重建产物**。

曝光 multiplier 为 1.0（0 EV），自动亮度开启。颜色是 LibRaw 的渲染，不是 ACR Camera Standard 或 Adobe Color 的匹配结果。PPM/TIFF 是已渲染 RGB，不能当作可编辑 Bayer RAW。

## 连续块与边界发现

文件共有 64 个 slice，前 63 个各 16 precinct，末个 2 precinct。重建 tile 则从 file `tile*16` 开始，最多读取 18 precinct，因此 126 次调用来自重叠读取。

- tile 0 local 16 对应 file 16，触发 tile-local 预测 reset。
- tile 1 local 0 也对应 file 16，但以新 tile 的初始状态开始，reset=false。
- 总共 63 次 local-index-16 reset；不能把 Bp/Br 当作位置标志。
- sb12 使用前一个 precinct 的 sb23，sb23 使用本 precinct 的 sb12；其他 band 保存自己的预测状态。

36,352 个子流 reader 的未消费 padding bit 全部为零，全部 precinct 的子流尾部 padding 也为零。样本所有 LB 均使用 coded GCLI，没有真实 raw-GCLI 覆盖；所有调用 sb0–24 mode 为 0x73，sb25 为 0x70。因此 mode 0x71、raw-GCLI、其他 profile 仍未用真实文件验证。

## MCP 恢复与维护

先检查端口、已有进程、锁和日志，确认没有运行实例或锁占用后，使用专用 `pyghidra-mcp` 环境打开已保存 `.gpr`。没有传入二进制，没有使用 force-analysis，也没有删除锁。日志明确显示 `Skipping full-project analysis on startup; using existing project state`。

通过实际 MCP 调用复核 75,502 个函数、`Analyzed=true`、Ghidra 12.1.4；重新读取 `07790200` 的 8192-byte 预测表和 `07792200` 的 64-byte scale 表，与已保存表逐 byte 一致。[恢复证据](evidence/2026-10-12-ghidra-restored.json)。服务在本次交付时监听 `127.0.0.1:8000`。

用户提供的维护指南已保存为独立记忆更新笔记：`/Users/hualaiwu/.codex/memories/extensions/ad_hoc/notes/2026-10-12T172600-ghidra-mcp-maintenance.md`。没有直接改写记忆 registry。

新增 [防重复启动脚本](../../../scripts/start-ghidra-mcp.sh)，使用 `bash scripts/start-ghidra-mcp.sh`。它在已有 listener、已有 MCP 进程或锁占用时不启动第二实例；只打开原工程、前台运行，Ctrl+C 由 MCP 的 clean save-and-close handler 处理。已验证语法以及已有 listener 时的退出行为。当前用户在 AGENTS.md 中的未提交修改保留。

## 产物

- [完整 16-bit RGB TIFF](/private/tmp/imprint-tico-experiment/LCR_3520_TicoRAW_render_RGB16.tiff)，与 PPM 的像素往返比较完全相同。
- [彩色浏览图](/private/tmp/imprint-tico-experiment/LCR_3520_python_entropy_render_thumb.jpg)。
- [完整 RGB16 PPM](/private/tmp/imprint-tico-experiment/LCR_3520_python_entropy_render.ppm)。
- [Python 熵解码重建 CFA](/private/tmp/imprint-tico-experiment/LCR_3520-python-entropy.cfa.u16)，无容器头，6064×4040、`<u2`。
- [跨 tile 行位置的原尺寸裁剪](/private/tmp/imprint-tico-experiment/LCR_3520_tile_boundary_crop.jpg)。

大数组和图像保留在临时目录，未加入 Git；临时目录可能被系统清理。研究工具、参数、哈希及 JSON 证据保存在项目中。

## 复现

先准备全新的临时输出目录，例如 `/private/tmp/ticoraw-phase5-new`，不要将输出指向照片或已有产物。以下假定之前固定源码及静态库仍存在，不安装或替换 Imprint 依赖。

```bash
clang++ -std=c++17 -O2 -I/private/tmp/imprint-tico-4abfcd2 research/acr/tools/ticoraw_continuous_reference.cpp /private/tmp/imprint-tico-4abfcd2/redlamp/bench/build/opt/libraw.a -lz -o /private/tmp/ticoraw-phase5-new/reference
/private/tmp/ticoraw-phase5-new/reference test_images/LCR_3520.NEF 2477568 15311360 /private/tmp/ticoraw-phase5-new/cpp-ab.i32le /private/tmp/ticoraw-phase5-new/cpp.json
/Users/hualaiwu/.conda/envs/py311/bin/python research/acr/tools/probe_ticoraw_continuous.py test_images/LCR_3520.NEF --tables research/acr/reports/evidence/acr-entropy-tables.json --cpp-output /private/tmp/ticoraw-phase5-new/cpp-ab.i32le --cpp-meta /private/tmp/ticoraw-phase5-new/cpp.json --python-output /private/tmp/ticoraw-phase5-new/python-ab.i32le --python-meta /private/tmp/ticoraw-phase5-new/python.json --report /private/tmp/ticoraw-phase5-new/comparison.json
clang++ -std=c++17 -O2 -I/private/tmp/imprint-tico-4abfcd2 research/acr/tools/reconstruct_ticoraw_scatter.cpp /private/tmp/imprint-tico-4abfcd2/redlamp/bench/build/opt/libraw.a -lz -o /private/tmp/ticoraw-phase5-new/reconstruct
/private/tmp/ticoraw-phase5-new/reconstruct test_images/LCR_3520.NEF 2477568 15311360 /private/tmp/ticoraw-phase5-new/python-ab.i32le /private/tmp/ticoraw-phase5-new/reconstructed.cfa.u16
clang++ -std=c++17 -O2 -I/private/tmp/imprint-tico-4abfcd2 research/acr/tools/render_ticoraw_reference.cpp /private/tmp/imprint-tico-4abfcd2/redlamp/bench/build/opt/libraw.a -lz -o /private/tmp/ticoraw-phase5-new/render
/private/tmp/ticoraw-phase5-new/render test_images/LCR_3520.NEF /private/tmp/ticoraw-phase5-new/render.ppm /private/tmp/ticoraw-phase5-new/render.json --cfa /private/tmp/ticoraw-phase5-new/reconstructed.cfa.u16
/Users/hualaiwu/.conda/envs/py311/bin/python research/acr/tools/inspect_ticoraw_render.py /private/tmp/ticoraw-phase5-new/render.ppm --thumbnail /private/tmp/ticoraw-phase5-new/review.jpg
```

本轮还执行了新增 Python 工具的语法检查、C++ 编译、真实全码流对照、截断和覆盖保护检查、脚本语法与 `git diff --check`。没有修改产品代码，所以没有把前端构建或全项目模型测试列为此次解码研究的验证。

## 当前产品边界

Imprint 使用的本机 `rawpy 0.27.0 / LibRaw 0.22.1` 仍对这个文件抛出 `LibRawFileUnsupportedError: Unsupported file format or not RAW file`。可行的隔离解码与渲染路径已经实测；产品依赖替换、接口接入、跨平台构建、更多相机/尺寸及完整损坏码流验证仍未进行。

没有运行 Adobe 解码函数，没有在 Lightroom/Camera Raw 实际导入比较，也没有得到 Adobe 逐像素或颜色等价结论。若后续继续追求完全独立的 ACR 复现，需要恢复全部 lifting 与 transfer curve 调用参数、25 个 Dpb 逻辑字段的内部映射，以及更多真实模式的覆盖；这些不阻止本样本已经完成的解码与渲染。
