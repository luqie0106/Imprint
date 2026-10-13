# Nikon HE* 首个 precinct：熵解码与逆量化实验

日期：2026-10-12。沿用已成功完整解码的 LCR_3520 HE* 样本与固定公开实现 `redlamp-libraw 4abfcd2`。本轮研究中间系数，不接入 Imprint，不处理颜色或 GPU，不修改 Adobe 文件或 Ghidra 数据。

## 已取得的 ACR 证据

继续通过 ghidra-project 读取已有程序。取得 `FUN_013e0b84` 与 `FUN_013e3d84` 的选定分段，以及两张数据表；没有触发导入或分析。会话中断恢复后 ghidra-project 工具不再提供，因此后续实验使用中断前已取得的字节。没有改用 REA 重新打开或分析。

| ACR 地址 | 内容 | 实验结果 |
| --- | --- | --- |
| 07790200 | 8192byte GCLI 预测表 | 独立计算的8192条预测公式与实际表全部一致 |
| 07792200 | 16项uint32逆量化scale | 与公开实现16项常数全部一致 |

两表均通过 `FUN_013e3d84` 的 DATA / PARAM 引用定位，**不是函数调用目标**。预测表按 `(gtli<<9)|(previous_gcli<<5)|unary` 索引，包含 zigzag / escape 和255 sentinel。逆量化表为：

```text
87381,74898,69905,67650,66576,66052,65793,65664,
65600,65568,65552,65544,65540,65538,65537,0
```

ACR 伪代码显示按索引选择scale、四lane乘法和移位；常量表的相等比“看起来像某种小波”更具体，但不能单独证明所有profile配置。完整来源及有限摘录：[stage4 static evidence](evidence/2026-10-12-ticoraw-stage4-static.md)。原始表：[acr-entropy-tables.json](evidence/acr-entropy-tables.json)，核对摘要：[table validation](evidence/ticoraw-entropy-table-validation.json)。

## 真实首块的参照

新增独立 C++ API harness，链接临时目录里已经编译的固定 LibRaw 静态库，只调用其解码函数，不调用 Adobe。先按阶段分别取得 GCLI、无符号幅值、带符号系数和逆量化系数，再与该库的 `decode_precinct` 完整首块流程比较；scatter后的两个缓冲区全部相同。它不是把某个返回成功状态当作系数正确性的证据。

普通编译与 ASan/UBSan 编译的完整 JSON 相同，sanitizer stderr 为0字节。参照JSON SHA-256：`f7bc647efdf976725572b61d8518fa60bb18205115ddc32d10cc0b073783a5ff`。

- 首块位于 RAW strip +155，完整长度15159，Bp=3、Br=15。
- 8个 line block，每个都有独立且持久的 sig / GCLI / data / sign 读指针。子带之间不能重建 reader 后丢失不足一byte的位状态。
- 共26个子带，6070个4系数组，包含分组填充位置，共24280个系数槽。这些不是24280个最终photosite。
- 首块实际 GCLI 范围0–10，逆量化系数范围-12688–1584。
- 最后一个sign子流结束于首块+15008；剩余151byte全零。这里是 **precinct内padding**，不同于此前验证的整个strip末尾padding=0。

公开解码器的解码部分参考：[precinct orchestration](https://github.com/pdcgomes/redlamp-libraw/blob/4abfcd2ab82f2211b1f7e7c3cd83974c87e0d812/src/decoders/nikon_he/nikon_he_precinct_decode.cpp)、[GCLI decoding](https://github.com/pdcgomes/redlamp-libraw/blob/4abfcd2ab82f2211b1f7e7c3cd83974c87e0d812/src/decoders/nikon_he/nikon_he_gcli_decode.cpp)。本轮真实数据结果来自本机运行，不是引用该项目的验收结论。

## 独立 Python 模型的边界

模型以真正的压缩载荷为输入，不读取C++参照中的GCLI或系数来“生成”结果。头信息、几何和WGT给出子带分组数量与GTLI；从bitstream计算GCLI和幅值，只在计算完成后与C++数组逐项比较。预测表与scale使用已读取的ACR字节。

这是 **ACR证据和公开源码辅助的首块模型**。GTLI/WGT映射、完整profile几何、跨band predecessor映射和shift=4仍借助公开实现，尚未全部从ACR参数流独立恢复。没有执行Adobe原函数，不能把匹配公开参照说成已证明Adobe运行时等价。

系数的处理公式为：

```text
每4系数组：GCLI-GTLI个nibble，每nibble的bit3..0分别写入4个幅值的下一bit
幅值完成后 << GTLI
只有非零幅值消费1个sign bit
逆量化：sign * ((((abs(coefficient)>>GTLI)*scale[GCLI-GTLI-1])>>(16-GTLI))<<4)
```

零或 `GCLI<=GTLI` 的组为零。本轮按固定实现的正常样本范围验证；没有复制其EOF时补零策略，独立reader越界会明确失败。首块全部raw_flag=0，因此raw-GCLI分支、其他模式和后续precinct状态还不算真实样本验证。

特别待验证：公开harness中sb25的mode=0x70，其余0x73，而ACR按25个Dpb逻辑字段映射到内部子带。首块之前的预测状态为零，容易让不同映射产生相同结果，故即便本轮系数一致，也不能跳过后续块和mode映射验证。

## 独立模型的实测对照结果

主代理审查代码并重新运行真实样本。Python与C++按以下数组逐项比较，差异全部为零：

| 阶段 | 比较值数 | 不同值 |
| --- | ---: | ---: |
| GCLI | 6070 | 0 |
| 幅值 | 24280 | 0 |
| 带符号系数 | 24280 | 0 |
| 逆量化系数 | 24280 | 0 |

共78910项实测比较。26个子带的ng / GTLI / mode配置、8组LB子流长度与偏移也全部匹配。每个reader记录consumed bits、未消费位及其值；首块各子流未消费位全零，末尾151byte也全零。比较摘要、逐band哈希和首8值：[lcr3520-first-precinct-comparison.json](evidence/lcr3520-first-precinct-comparison.json)。

数值数组存于独立临时产物 [first-precinct-review-arrays.npz](/private/tmp/imprint-tico-experiment/first-precinct-review-arrays.npz)。它含104个数组，每个子带四阶段；GCLI为uint8，其余为int32。这是中间系数，不是可以当RAW打开的图片文件。

## 复现与检查

工具：[C++参照harness](../tools/ticoraw_first_precinct_reference.cpp)、[Python首块模型](../tools/probe_ticoraw_first_precinct.py)。在仓库根目录，使用全新的输出路径：

```bash
clang++ -std=c++17 -O2 -I/private/tmp/imprint-tico-4abfcd2 research/acr/tools/ticoraw_first_precinct_reference.cpp /private/tmp/imprint-tico-4abfcd2/redlamp/bench/build/opt/libraw.a -lz -o /private/tmp/imprint-tico-experiment/first-precinct-reference-new
/private/tmp/imprint-tico-experiment/first-precinct-reference-new test_images/LCR_3520.NEF 2477568 15311360 > /private/tmp/imprint-tico-experiment/first-precinct-reference-new.json
/Users/hualaiwu/.conda/envs/py311/bin/python research/acr/tools/probe_ticoraw_first_precinct.py test_images/LCR_3520.NEF --tables research/acr/reports/evidence/acr-entropy-tables.json --reference /private/tmp/imprint-tico-experiment/first-precinct-reference-new.json --output /private/tmp/imprint-tico-experiment/first-precinct-model-new.json --arrays /private/tmp/imprint-tico-experiment/first-precinct-model-new-arrays.npz
```

C++只读输入，参照JSON由重定向写入；务必选新路径。Python拒绝覆盖原图、表、参照和已有产物。主代理复跑真实输入；worker另验证了截断位流拒绝、已有产物拒绝覆盖。没有安装依赖、提交或推送。

## 接下来优先验证

1. 从首块扩展到连续precinct的LL前驱预测状态，尤其第16 / 17块的slice边界与状态重置。仅首块一致尚不足以证明模式映射正确。
2. 接着恢复 / 核对inverse lifting与跨tile carry，使独立模型产出CFA；与上一阶段完整公开解码CFA基线对比。
3. Ghidra MCP恢复后继续追踪Dpb映射、GTLI参数和shift=4，保留ACR静态已证实与公开源码辅助的区别。

当前仍未独立解完整照片，未执行Adobe解码或做Adobe像素一致性验证。Imprint接入继续放在独立解码和边界验证之后。
