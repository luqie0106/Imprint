# ACR 熵解码表与分段证据

通过 ghidra-project 从 `/Camera_Raw_arm64.analysis-383152` 只读取得。未修改工程。本轮恢复会话后 ghidra-project 工具不再可用；表字节已提前保存，以下记录仅使用中断前已实际返回的结果，不通过 REA 补取。

## 数据引用，不是函数调用

| 地址 | 大小 | 用途候选 | 已查 xref |
| --- | --- | --- | --- |
| 07790200 | 8192 bytes | `(gtli, previous_gcli, unary)` 预测表 | DATA：013e47ec、013e48dc、013e5574 等，均位于 FUN_013e3d84 |
| 07792200 | 64 bytes | 16 项逆量化 scale | DATA：013e4f98；PARAM：013e4d90，均位于 FUN_013e3d84 |
| 0779223c | 表基址是最后一项 0；更大范围未取 | 紧邻逆量化表的 packed sign lookup 候选 | 伪代码中按 presence-mask / bit-prefix 组合索引；未完整恢复 |

`0779223c` 和 `07792200+60` 是同一地址，不能把它当作独立函数或仅凭名字认定表边界。此处零值是逆量化表的最后一项，也可能是下一表的起点；后者仍待读取更多数据确认。

## 预测表全部条目

表索引：`(gtli << 9) | (previous_gcli << 5) | unary`，范围分别 0..15、0..15、0..31，共 8192 项。

定义 `m=max(previous_gcli,gtli)`、`t=m-gtli`：

```text
u=0                 delta=0
1<=u<=2*t 且 u为奇数 delta=-(u+1)/2
1<=u<=2*t 且 u为偶数 delta=u/2
u>2*t               delta=u-t
value=m+delta；若 value>15+t，存 255 sentinel
```

按这个公式独立生成的全部条目，与 ACR 读取的实际表逐 byte 相同。公式亦与固定公开实现的预测表构建代码相符。这验证的是表行为，尚不证明所有 profile 的调用参数或越界码流处理相同。

原始表及哈希：[acr-entropy-tables.json](acr-entropy-tables.json)、[ticoraw-entropy-table-validation.json](ticoraw-entropy-table-validation.json)。

## 逆量化表

将 `07792200` 起 64 bytes 按 little-endian uint32 解读：

```text
87381, 74898, 69905, 67650, 66576, 66052, 65793, 65664,
65600, 65568, 65552, 65544, 65540, 65538, 65537, 0
```

与公开实现的 16 项 scale 表相同。已查看的 FUN_013e3d84 分支有以下乘法 / 移位摘录：

```c
if (uVar32 < 0x11) {
  iVar83 = *(int *)(&DAT_07792200 + (ulong)(uVar58 - 1) * 4);
  auVar65._0_4_ = iVar83 * auVar48._0_4_;
  auVar65._4_4_ = iVar83 * auVar48._4_4_;
  auVar65._8_4_ = iVar83 * auVar48._8_4_;
  auVar65._12_4_ = iVar83 * auVar48._12_4_;
  auVar65 = NEON_ushl(auVar65,auVar79,4);
}
if (param_6 < 0x20) {
  auVar118._4_4_ = param_6;
  auVar118._0_4_ = param_6;
  auVar118._8_4_ = param_6;
  auVar118._12_4_ = param_6;
  auVar48 = NEON_sshl(auVar65,auVar118,4);
}
```

这支持按 bitplane-count 选 scale、乘法及移位的结构判断。`uVar58` / `uVar32` 的全部定义和 `param_6` 的样本配置，尚未完整追溯。本轮独立模型中的 `shift=4` 来自公开实现配置，不能说已在 Adobe 运行时得到证实。

## 子流解析片段

FUN_013e0b84 已查看片段依序读取 1bit flag、20bit 长度、20bit 长度、15bit 长度；根据 flag 将 significance size 置零或向上取整到 byte，并构造四个连续子流指针。对应公开实现的 56bit mini-header 和 sig / GCLI / coefficient / sign 分离布局。层内四个子流的数学角色通过公开实现和真实输入模型进一步核对，仍保留静态对应关系的限制。

大函数长度：FUN_013e0b84 为 77,519 字符，FUN_013e3d84 为 78,346 字符。此次只查看 selected 3–5k 字符分段，未输出全部函数。

## 尚未接受的 ABI 或语义假设

- FUN_013e2a2c 通过 DAT_083afba8 间接调度，FUN_013e2cd8 通过 DAT_083aeae8 间接调度；不能据函数清单就命名为 coefficient 或 dequantization。
- DAT_083aeae8 初始 8byte 值为 `01404de8`，这是函数指针 DATA，不是这个数据地址被 BL 调用。
- 公开首 precinct harness 的最后一个 band mode=0x70，其他为0x73。ACR 的 25 个 Dpb 逻辑字段如何映射到26子带，需继续追踪；首块 parity 不能掩盖这种语义待验证项。
