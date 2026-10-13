# Nikon HE*：首次完整 CFA 解码与 ACR 重建公式验证

日期：2026-10-12。范围仅压缩码流、整数系数和 CFA 重建；未做颜色处理、去马赛克、GPU 研究或 Imprint 接入。

## 本轮结果

`test_images/LCR_3520.NEF` 已在独立实验中成功解出 **6064 × 4040 RGGB CFA**。14-bit 样本用 little-endian uint16 存放，共 24,498,560 个 photosite、48,997,120 字节。使用公开 LibRaw 分支，而不是调用 Adobe、读取嵌入预览，或声称已独立复现整个 ACR 解码器。

| 验证 | 结果 | 意义与限制 |
| --- | --- | --- |
| 文件打开 / 解包 | `open=0, unpack=0`，`nikon_he_load_raw()` | 真实完整 CFA 解包成功 |
| 模式与尺寸 | NEFCompression=14，6064×4040，RGGB，裁剪边距 0 | 本样本 HE*；不是低分辨率 RAW 预览 |
| 像素范围 | 974–3551，零值 0，超出 16383 的值 0 | 原始传感器域，未减黑电平、未归一化 |
| 黑 / 白电平 | 1008 / 16383 | 解码元数据，不把实际最大值 3551 当作白电平 |
| 普通优化编译 | 单次 unpack 425.1 ms | 不包含进程启动、文件打开或产品流程；不是完整性能基准 |
| ASan + UBSan 编译 | unpack 1532.0 ms，stderr 0 字节 | 本样本未报告内存越界 / 未定义行为；不是所有恶意码流的证明 |
| 两种编译的 CFA | SHA-256 完全相同 | 确定性复核，不等于 Adobe 像素一致性 |
| CFA 平面目视检查 | 四个独立灰度平面均能看到海面、飞鸟、岸线和前景人物 | 使用每平面 1–99 分位拉伸，只用于结构检查，无颜色渲染 |
| ACR 最终重建公式模型 | 10 组合成输入，共 17,664 个输出值，与公开 C++ step2 相同 | 未执行 Adobe 原函数；未覆盖跨 tile 调度、SIMD 与完整熵解码 |

CFA SHA-256：`5c76848dadbd2682b1abd444d32551c0000fe6d9a56c0e638582248cbac59ed2`。

样本本轮重算 SHA-256 仍为 `6c2276300688850beaae7c997f39ad0a8470333f6ffdf3f33876fa658ae0f313`，与此前记录相同。G1/G2 均值分别为 1801.7827 / 1802.2230，相差约 0.0244%；均值接近不能单独证明所有像素正确。

## 公开实现的来源和隔离

固定源码：[pdcgomes/redlamp-libraw 4abfcd2ab82f2211b1f7e7c3cd83974c87e0d812](https://github.com/pdcgomes/redlamp-libraw/tree/4abfcd2ab82f2211b1f7e7c3cd83974c87e0d812)。其 [REDLAMP.md](https://github.com/pdcgomes/redlamp-libraw/blob/4abfcd2ab82f2211b1f7e7c3cd83974c87e0d812/REDLAMP.md) 记录了上游基础、HE/HE* 改动与附加修复。依照其源文件列表编译，未执行应用安装脚本或下载别人的照片。

压缩包 SHA-256 已与 Redlamp 固定配置核对：`2ca1d9673531fccabcc7c8bce589444163868bcde625aa1f09eb442df1bcce2d`。源码、静态库和实验输出均在 `/private/tmp`；未替换 Imprint 的 rawpy / LibRaw，未安装依赖，未向任何服务上传用户照片。

[LibRaw PR #826](https://github.com/LibRaw/LibRaw/pull/826) 本轮查询时未合并，说明文字仍写着拒绝 HE*；但是其 head `499bfd4` 的实际 `nikon_he_decoder.cpp` 已接受模式 13 和 14。因此不能用旧说明判断当前代码能力。本轮验收只针对上面的固定 Redlamp 版本和本地样本。

实验输出：

- [完整 CFA uint16 数据](/private/tmp/imprint-tico-experiment/LCR_3520.cfa.u16)，**无容器头**，读取需使用本报告的尺寸和 `<u2` 类型。
- [四 CFA 平面灰度图](/private/tmp/imprint-tico-experiment/LCR_3520-cfa-planes.png)。不代表最终照片颜色或 ACR 渲染。
- [本轮解码 / 构建证据](evidence/lcr3520-ticoraw-decode.json)。
- [CFA 数值统计](evidence/lcr3520-ticoraw-cfa-metrics.json)。

临时目录可能被系统清理；固定来源、哈希、工具与命令已留在研究文档中，49 MB 像素输出未加入仓库。

## Ghidra 工程复核与调用关系

继续通过 `ghidra-project` 读取持久化程序 `/Camera_Raw_arm64.analysis-383152`。本轮 metadata 再次返回 **75,502 函数、Analyzed=true、Ghidra 12.1.4**；`analysis_complete=true`，`code_indexed=false`、`strings_indexed=false`。未重新导入、未启动全量分析，也未修改函数名、类型、注释或 Adobe 原始文件。

引用类别继续保持此前区分：

- `ReadNikonCompressed_Scalar` 是诊断字符串，`PARAM` 引用不能当作函数调用。
- `081985a0`、`08197950` 是任务函数指针槽，数据引用不能当作 `BL`。
- `014d5b0c: BLR x9` 是真正 worker 执行任务的间接调用。
- `013e1bc8` 与 `013e2414` 是 **FUN_013e0b84 内的调用指令地址**，均为 `BL 013e3d84`；不是两个独立 helper。按内部地址反编译会返回包含它的 77,519 字符函数。
- `0157e940` 是重建任务通过函数指针调用的点；经 thunk `013c951c` 到 `013c8bc8`。本轮仍是静态关联，没有记录 Adobe 对本照片的运行轨迹。

本轮对 `013c8bc8` 只查看参数计算、边界控制和 scalar 尾部，保留约数千字符摘录与关键指令；没有一次输出大型函数全部伪代码。见 [Ghidra 摘录](evidence/2026-10-12-ticoraw-stage3-excerpts.md)。

## 从实际码流确认的布局

新独立工具按长度字段遍历压缩段，**没有在熵载荷中搜索 ff20 来猜块边界**。重新运行的 JSON 与 worker 产物完全相同。见 [全部边界证据](evidence/lcr3520-ticoraw-packets.json)。

- RAW strip：文件偏移 2,477,568，长度 15,311,360，直到文件结尾。
- 头部依次为 SOC、CAP、PIH、CDT、WGT。首 SLH 在 strip +149，长度值 4，编号 0；首 precinct 从 +155 开始。
- PIH：Lcod=15,311,360，宽高=6064×4040，组件数=4，系数组=4，显著性组=8，Bw=18，Hsl=16。**组件数 4 不是输出 RAW 位深；Bw=18 也不是输出位深。**这些字段名称沿用公开实现，未全部通过 ACR 类型恢复证明。
- WGT 长度 52，body 50 字节，25 对 gain / priority 候选。
- 首 precinct：24-bit 长度 15,147，完整长度 15,159；Bp=3、Br=15，Dpb 字节为 `ffffffffffffc0`。
- 共 **1,010 precinct**。Bp=2 有 544 个，Bp=3 有 466 个；不能把 HE* 简化成所有 precinct 的 Bp 都是 1 或 2。
- 共 **64 个顺序 slice**：前 63 个各 16 precinct，末个 2 precinct。EOC `ff11` 位于 strip +15,311,358，即最后两个字节；未消费字节 0、尾部 padding 0。

ACR `FUN_013e0b84` 的块大小公式为：

```text
BE24 length + ((context[0x94] * 2 + 0x2f) >> 3)
```

`FUN_013d3268` 的 WGT 长度检查为 `context[0x94] * 2 + 2`，并逐对写入表。结合本样本 WGT 长度 52，可推得这个配置的 count=25，前缀开销 `(25*2+47)>>3=12`，与独立遍历完整压缩段吻合。此处是静态公式与真实码流的交叉核对，未测量 Adobe 的运行时 context。

PIH 的 13 字节扩展在公开实现中解释为 transfer curve 参数，样本得到 B=1008、A=84429、C=16255。B 与黑电平相同是观察到的事实；B/A/C 提取公式和传递函数仍需在 ACR 中独立验证，不能用数值巧合代替证明。公式实际位于公开源码 `nikon_he_decoder.cpp`，头解析器只确认扩展位置。

## 已提取的 ACR 最终像素重建公式

记 P1..P4 为 `FUN_013c8bc8` 的前四个整数平面，r/c 为当前行列。已查看的 scalar 路径在内部列可整理为：

```text
top[2c]     = F(P1[r,c] + bias
                  + ((P2[r,c-1] + P3[r-1,c] + P2[r,c] + P3[r,c]) >> 2))
top[2c+1]   = F(P2[r,c] + bias)
bottom[2c]  = F(P3[r,c] + bias)
bottom[2c+1]= F(P4[r,c] + bias
                  + ((P3[r,c+1] + P2[r,c] + P3[r,c] + P2[r+1,c]) >> 2))
```

`F` 在 LUT-enabled 分支先夹紧索引并查 int32 LUT，再加 rounding、算术右移 shift、夹紧到输出范围、以 `STRH` 写出两字节。ARM64 的 `ASR #2` 证实这里采用有符号向负无穷舍入的移位，不能替换为 C/C++ 对负数向零截断的除法。左右端和首末行需要邻域延拓。

偏置 / 裁剪的参数关系为 `bias=1<<(shift+bits-1)`、`rounding=shift>0 ? 1<<(shift-1) : 0`、`clip=(1<<bits)-1`（非零位数分支）。公开实现配置 shift=2、bits=14；**尚未完成 ACR 调用参数追溯，不能宣布这些常数已在 Adobe 样本路径得到证明**。

独立 Python 模型按上述 scalar 公式实现。合成实验调用外部 C++ `step2_bayer_rows` 作参照，覆盖 1×1、1×19、7×1、7×19、32×64 平面、两种幅度，包含负数、边界索引、LUT 上下界、舍入和输出裁剪，10 组全部逐值相同。证据：[合成重建实验](evidence/ticoraw-reconstruction-synthetic.json)。这证明提取的局部公式与参照一致，**不是整张样本的自主熵解码证明，也不是执行 Adobe 代码的结果**。

## 复现入口

在已有固定源码目录下，构建与解码命令：

```bash
/Users/hualaiwu/.conda/envs/py311/bin/python -c 'import runpy; d=runpy.run_path("/private/tmp/imprint-tico-4abfcd2/redlamp/bench/bench.py"); d["build"](("opt","san"))'
# dump_raw 会写输出；请使用全新的输出路径，不能指向原图。
/private/tmp/imprint-tico-4abfcd2/redlamp/bench/build/opt/dump_raw test_images/LCR_3520.NEF /private/tmp/LCR_3520-new.cfa.u16 > /private/tmp/LCR_3520-new.json
/Users/hualaiwu/.conda/envs/py311/bin/python research/acr/tools/inspect_ticoraw_cfa.py /private/tmp/LCR_3520-new.cfa.u16 /private/tmp/LCR_3520-new.json --output /private/tmp/LCR_3520-new-metrics.json --planes /private/tmp/LCR_3520-new-planes.png
/Users/hualaiwu/.conda/envs/py311/bin/python research/acr/tools/probe_ticoraw_packets.py test_images/LCR_3520.NEF --output /private/tmp/LCR_3520-new-packets.json
```

重建参照 dylib 的最小 wrapper（独立临时文件，无 Adobe 链接）：

```cpp
#include "src/decoders/nikon_he/nikon_he_bayer.h"
extern "C" void run_step2(const int32_t* a,const int32_t* b,const int32_t* c,const int32_t* d,
                          int rows,int cols,const int32_t* lut,uint16_t* out) {
  nikon_he::step2_bayer_rows(a,b,c,d,rows,cols,cols,cols,lut,out,cols*2,0,true,true);
}
```

```bash
clang++ -std=c++17 -O2 -dynamiclib -I/private/tmp/imprint-tico-4abfcd2 /private/tmp/imprint-tico-experiment/step2_reference.cpp /private/tmp/imprint-tico-4abfcd2/src/decoders/nikon_he/nikon_he_bayer.cpp -o /private/tmp/imprint-tico-experiment/step2_reference-new.dylib
/Users/hualaiwu/.conda/envs/py311/bin/python research/acr/tools/probe_ticoraw_reconstruction.py /private/tmp/imprint-tico-experiment/step2_reference-new.dylib --output /private/tmp/ticoraw-reconstruction-new.json
```

研究工具输出拒绝覆盖。外部 `dump_raw` 是原仓库工具，没有同样的覆盖保护，故本轮只向全新临时路径写入。

## 结论与下一步

**已证实：**本照片可完整解码；输出是 CFA；压缩段布局可独立按长度走通；ACR 最终重建的局部 scalar 算术有可执行模型；优化与 sanitizer 两种输出一致。

**推测：**ACR 候选任务链与公开实现对应同一 Nikon TicoRAW profile；组件 / 子带与已见整数平面的完整映射。仍需静态参数追溯和真实中间系数对照。

**待验证：**从 ACR 独立恢复 GCLI / coefficient / sign、逆量化、全部 lifting、跨 tile 边界与 transfer curve；把首 precinct 的中间结果与已成功解码的基线比较；本照片与 Adobe 逐像素一致性；更多相机 / 尺寸和损坏流；产品接入及跨平台打包。本机未发现 Adobe DNG Converter，未安装它或运行 Lightroom/Camera Raw 实际导入。

优先继续逆向首 precinct 与重建参数，保留公开实现作为实验参照。Imprint 产品代码没有改动，暂不做接入。
