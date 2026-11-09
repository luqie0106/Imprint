# ACR CPU 重建与本机验证

更新：2026-11-09。本文记录已验证的局部CPU能力及产品入口；完整ACR
高光/RAW管线尚未完成。GPU实现等待ACR GPU pipeline分析完成。

## 用户可验证的入口

Imprint的去朦胧、理光页面右侧有 **ACR CPU 颜色实验**：

1. 加载支持Profile的RAW，展开该区并启用实验。
2. 可调整新版Profile曲线、HSL八色范围/自然饱和度、暗部/中间调/高光/
   全局色轮，以及混合和平衡。
3. 此入口作用于预览。当前导出仍使用既有处理；original模式保持原图。

当前正式开发库的真实HTTP检查：

|输入|Profile|预览尺寸|结果|
|---|---|---|---|
|`test_images/DJI_0539.DNG`|Embedded|640×480|Profile/HSL/色轮使用`cpp_cpu_acr`|
|`test_images/LCR_1132.NEF`|Camera Standard|640×426|Profile/HSL/色轮使用`cpp_cpu_acr`|

20次HTTP请求通过；8组重复请求响应及缓存一致；original开/关实验一致；
两张RAW的size/mtime/SHA-256保持；结束后无新增缓存worker线程。
这是本机开发库验证，未重打安装包或验证Windows运行。

## 当前CPU合同

- `acr_guide_camera_black.hpp/.cpp`：prepared Camera→black/pin→固定灰度/
  条件EV90/log→local mask组合，失败原子提交；完整零原点图像、显式
  boundary mode2/4。source level选择、独立base cache槽、transfer分类分别处理。
- `acr_tone_kernels.cpp`：保存指令确定的灰度FMA、raw分位数/分组统计、
  Gaussian权重和横纵求和顺序、ceil remap采样和sigma；统一内部统计层。
- `acr_profile_hsl_controls.hpp/.cpp`：Profile/current HSL24非线性控制
  合并，显式解析好的单色predicate/灰度byte、完整int32输入、skip/wrote mask。
  正式ABI和Python `native_acr_profile_hsl_controls`可调用，尚未自动从
  真实Profile元数据生成这些控制输入。
- `acr_basic_flag_slots.hpp/.cpp`：`b69664` 的132槽flag-reset前缀，正式ABI
  与Python `native_acr_basic_flag_slots` 可调用。仅零group byte写mode2默认，
  非零保留；不包含后续header/curve、sentinel与Look消费，详见本日附录。

不能把上述prepared输入合同称为实际Stage4身份证明，或把各局部helper
相加当作完成原调度器。Profile合并可直接改写ToneMapStrength；仍需跟踪
真实请求的字段时序、其余Prepare writers和像素来源。

## 2026-11-08验证摘要

正式库SHA-256：
`5f43495c1407ca28c9a67d6ebb5b81ca9c6b02ae73b3a6780f5d26397a0f3376`。

|检查|实际结果|
|---|---|
|Profile HSL控制 source/object/ABI与旧ID130回归|125 passed|
|guide/mask/source-route相关回归|136 passed|
|Profile/HTTP/metadata相关测试|93 passed|
|native CPU CTest|30/30 passed；GPU contract排除|
|前端类型检查及生产构建|`npm run build`成功|
|Profile HSL保存ARM64算术对照|497,088 lane，double bits与整数一致；512组gate一致|
|Profile HSL ASan/UBSan|1,200,000 ABI lane通过，无诊断；leak detector关闭|
|正式prepared tone ABI/Python|9组最大误差0；9项非法输入保持目标|

已有TIFF没有拟合参数，当前库重跑结果：

|TIFF范围|组均值平均ΔE76|最大ΔE76|
|---|---:|---:|
|HSL Hue，38组|0.0476400241344|0.818496874636|
|HSL Saturation，16组|0.0478528321362|0.790666517291|
|HSL Luminance，16组|0.0510153126009|0.798820999226|
|Color Grading，13组灰阶|0.0100741159024|0.555168910055|

HSL C++/独立公式最大通道误差`5.960464477539063e-08`；grading为
`2.9801294876286022e-08`。85张参与TIFF的前后指纹保持，逐组数值与历史
一致。Grading沿用既有工作空间转换与少量端点clip，部分controls来自
README/命名假设，不是新观察的完整ACR状态。

## 复现与研究证据

```bash
cmake --build native-renderer/build -j 4
ctest --test-dir native-renderer/build --output-on-failure -E native_renderer_gpu_contract
/Users/hualaiwu/.conda/envs/py311/bin/python -m pytest -q \
  tests/test_acr_profile_hsl_controls.py tests/test_acr_profile_hsl_controls_api.py \
  tests/test_acr_profile_gain_control130.py tests/test_acr_profile_gain_control130_api.py
```

未纳入Git的持久研究目录（保留现有原始证据与运行记录）：

- `research/acr/reports/2026-11-08-guide-source-routing.md`
- `research/acr/reports/2026-11-08-profile-hsl-controls.md`
- `research/acr/reports/evidence/2026-11-08-profile-hsl-blend/`
- `research/acr/reports/evidence/2026-11-08-cpu-color-refresh-final/`
- `research/acr/reports/evidence/2026-11-08-cpu-preview-refresh/`

## 剩余目标

真实source/Stage4/Camera/EV90/epsilon的同请求绑定、profile准备与其余
Prepare effects、跨source层mask、非零原点/ROI、完整BuildStats
及最终高光联动尚未闭合。现有fixture目录未找到带可信controls记录的
Profile切换或基础Highlights2012成对样本；高光色轮TIFF不是基础高光样本。
mode4 prepared算子已完成；profile递归的局部正常返回边界见本日附录，
仍不是完整Prepare成功终止或真实请求绑定。
当前CPU检查点不代表整体目标完成，也没有启动GPU pipeline分析。

## 2026-11-08 继续：DNG 源白点绑定

现有 ACR CPU 颜色预览对同文件、scene-referred DNG 优先读取原始
`AsShotNeutral`（50728，三个 RATIONAL），原样送入既有 native 白点准备。
max-normalization 和 1e6 量化仍在 C++ 执行。旁侧参考 DNG 的白点不作为
NEF 的源白点；缺少该 tag 时继续使用原 caller neutral。显式非零
ColorimetricReference 尚不支持此 scene 路径，会拒绝解析。

Profile fingerprint 与 preview version 更新为 `acr-huesatmap-v5-source-neutral`，
缓存包含原 tag payload。新增及相关测试 **166 passed**，原 Profile 路径
回归 **30 passed**；两张 RAW 的 **20 次 HTTP 预览**通过，响应 hash 与前一
检查点一致，源 RAW/sidecar 未变，未留下新 worker 线程。

实图边界：`DJI_0539.DNG` 绑定 Embedded 的同文件原始 neutral；
`LCR_1132.NEF` 继续使用自身 LibRaw neutral，未误用参考 DNG 的白点。
`DJI_0523.DNG` 的选中名称 Adobe Standard 与内嵌 Embedded 冲突，仍按
既有规则拒绝，未换用猜测 Profile。

同时补核了高光 guide 白点：`dae4e4` 内独立参数的 mode8/xy setter 写入
D50 `(0.3457,0.3585)`，不是直接复用源 AsShotNeutral 准备的 Camera。
保存指令 404 条校验通过，14 条局部原指令探针通过。用同一实图 Profile
作数值见证时，AsShot 与 D50 的 CameraWhite 不同；这解释了为什么当前
颜色预览 Camera 参数还不能直接当高光 guide 参数。

本次补齐的是已接入颜色预览的源白点来源；实际 guide 的 Profile、Stage4
像素、EV90/epsilon、可选阶段仍需同请求绑定。详细证据：
`research/acr/reports/evidence/2026-11-08-raw-request-source/`。

## 2026-11-08：Profile HSL 合并正式接入像素预览

新增正式 CPU 源文件 `native-renderer/src/acr_profile_hsl_pixels.cpp`，导出
`im_native_acr_profile_hsl_sdr_rgbf32`。它先使用已恢复的 Profile HSL24
非线性合并和整数舍入，再执行既有 SDR HSL 像素算子；自然饱和度保持独立。
输入错误、非法像素或输出重叠会在写入前拒绝，目标保持不变。

用户入口：去朦胧／理光页面 → **ACR CPU 颜色实验** → 启用 → HSL 内
**Profile HSL 片段（CPU）**。可显式选择：

|片段|已验证的稀疏默认值，其余24槽保留sentinel|
|---|---|
|Modern 03 · 仅HSL|橙色饱和度−14、橙色明亮度−7|
|Modern 04 · 仅HSL|橙色饱和度−10|
|Vintage 07 · 仅HSL|蓝色饱和度−29|

强度0–200%，与手动HSL按恢复公式合并。**这不是完整Adobe Profile：不含
其LUT、曲线、局部修正或其它设置，不自动推断照片选中了哪个Look。** 此入口
仍只影响预览，不写XMP、不影响现有导出；原图模式忽略实验，理光预设启用时
实验暂停。关闭片段维持原来的手动HSL路径，默认关闭。

Python接入位于 `src/native_renderer.py`、`src/camera_profile.py`；片段纯数值
清单位于 `src/acr_hsl_fragments.py`。缓存包含片段ID、强度和版本，低分辨率、
全分辨率及最终显示层隔离。没有增加任意本地文件GET接口或新增Python依赖；
PyInstaller已加入新模块，现有macOS/Windows CMake流程会构建新CPU文件。

实际验证：

- Python产品路由／渲染／metadata／缓存相关 **238 passed**；控制算子等另一组
  **196 passed**（两组有重叠，不相加）。
- 正式native非GPU CTest **31/31 passed**，新增组合测试120组逐bit一致及失败
  原子性；前端 `npm run build` 成功。
- 两张现有RAW的 **28次真实HTTP请求**通过；0%与关闭像素相同，100%/200%
  有实际变化，重复请求响应一致，original不受影响；4个RAW/sidecar指纹保持。
- 新组合入口在已有70组HSL TIFF上与分开执行的结果逐bit一致；平均ΔE76为
  Hue `0.0476400241344`、Saturation `0.0478528321362`、Luminance
  `0.0510153126009`。13组色轮灰阶TIFF平均ΔE76 `0.0100741159024`，指标与
  前检查点一致。85张TIFF指纹保持。该TIFF路径仍使用既有固定tone/工作空间
  转换，不冒充完整RAW渲染或基础高光回归。
- 两个真实编辑页的新增控件已通过浏览器布局检查：同窗口有溢出／无溢出
  时卡片、直方图、select和slider横向差异0 CSS px；gutter回退同样稳定。
  已连接真实后端打开DJI0539，界面报告“已应用”并显示实际照片。

本机开发库SHA-256：
`eb3d7ab47ceab652f606df9557cdfd4f76d48ce09cca03fc850d9fd915aece90`。
未重打安装包，未验证Windows实际运行。证据位于
`research/acr/reports/evidence/2026-11-08-hsl-fragment-product/`。

全量Python测试实际为 **5287 passed、5 failed**（149.68秒），不能称全量通过。
五项为combined preview色彩转换、physical shadow budget、progressive preview
尺寸、shared XMP preset保持及per-photo Ricoh export测试。另在内存加载HEAD的
`native_renderer.py`、`camera_profile.py`、`app_api.py`重跑这五项，同样全部
失败于相同断言；此对照仍使用当前native库，不宣称完整旧二进制环境基线。
本轮未修改这些测试或对应去朦胧/XMP/导出实现来制造通过结果。

上述是可用的Profile HSL局部功能交付，不是实际照片自动Look选择或完整高光
链完成；GPU分析与“CPU整体完成”提交条件仍未满足。

## 2026-11-08：真实照片 XMP HSL 读取与限定预览

正式新增 `src/acr_photo_hsl.py`，将已恢复的字段规则用于**明确选择的单个照片
packet**，不复用首个 Description 或递归搜索的宽松语义。顶层 current24 与
`Look/Parameters` defaults24 分开；缺项分别投影为0与−999999，显式0有效。
category7关闭先置0，nested gray再置sentinel。整数、bool和结构不在支持子集
时明确拒绝，不吞成“缺少Look”。该模块不执行Adobe Prepare或资源查找。

新增会话限定 `POST /api/enhance/acr-hsl`，只接受 session_id、photo_id、
packet_source=`embedded|sidecar`，不接受路径。它可返回白名单HSL字段和局部
native合并：Amount使用既有C++百分位量化，defaults/current使用既有C++合并；
返回始终包含 `consumer_bound=false`。这个量化值是loader投影，**不是已经证明的
消费时P+b60**。CameraProfile缺失时不猜单色谓词、不计算local_merge。

用户可验证入口：两个编辑页 → ACR CPU颜色实验 → **照片HSL来源（仅顶层）**。
可明确选择内嵌或同名侧车XMP，只将top-level HSL24送入既有CPU像素算子；
自然饱和度、色轮仍由现有手动参数提供，导出和XMP写入不变。
带显式Look的packet全部拒绝像素应用，灰度/明确单色设置同样拒绝。缺少磁盘
Look不推导Adobe消费时null pair；该入口只是本实验的显式current-controls导入。
两份来源不自动合并、不推断优先级，不使用UUID/key相等来认证实际选中资源。

照片来源模式禁止同时手动HSL24或片段；切入该模式会清零手动HSL24，界面显示
“由XMP提供”。backend每请求只读一次快照，低/全分辨率显示缓存均带来源、
解析版本和原始packet SHA；前端绕过此模式的blob缓存，避免阻止后端检查XMP
变动，HTTP使用no-store。原图/关闭模式不读packet，默认仍为手动模式。

安全与打包：只读，源文件/侧车读取前后检查stat身份；4MiB上限、UTF8、禁止
DTD/entity、重复/复杂scalar/多顶层Description拒绝。仅embedded允许实图的
精确subject标签 `DJI Meta Data`，不允许任意非空subject或URI。PyInstaller已
包含新模块；无需新增依赖，现有CI的`src/*.py`哈希覆盖它，未改全局环境。

实际验证与限制：

- 新模块/API/现有HSL路由一组 **364 passed**；跨模块一组 **221 passed**，
  两组有重叠。覆盖scope、sentinel、非法字段、失败不使用旧缓存、来源互斥、
  native失败不退回近似公式、同请求快照及低/全分辨率缓存失效。
- 前端类型检查与生产构建成功；非GPU CTest **31/31**。两个编辑页有/无溢出，
  stable与auto回退均测，卡片/滑块/直方图/select左右差异0 CSS px。
- 四张RAW的8份packet请求：6份成功读取，2份Nikon内嵌packet因多顶层
  Description明确422。DJI0523的内嵌current有**11槽非零**（此前摘要误计10），
  nested defaults全部sentinel、局部native wrote_mask=0；带Look像素请求422。
- DJI0539与LCR1132侧车导入各完成真实/重复/同控件手动请求，JPEG逐byte相同、
  后端cpp_cpu_acr；两份实际侧车HSL均全零，**不是非零实图效果证明**。另真实
  浏览器连接DJI0539后端显示照片与“已应用”，与UI fixture布局截图分开保存。
- LCR9472侧车可读取，但现有相机Profile不可用，预览422；未绕过resolver。
  8个RAW/sidecar、85张既有TIFF指纹保持，native库SHA保持上个检查点。未改DNG或导出流程。
- 最终全量pytest实际 **5605 passed、5 failed**（150.02秒）；仍为前述五项断言，
  没有改弱、跳过或顺手修改它们。不能称全量通过。

本轮静态复核仅复用缓存与原指令：3,068个word匹配固定切片，新反编译0、
新反汇编0；`b657dc:mov x2,0`、`b657e0:mov w3,0`接`b657e4:b67ef0`，
IDs43–66的version分支均为true、准入均转mono/gray分支。后续Look替换、默认
安装以及消费时Amount仍未闭合，不能将该局部数据导入称为真实自动Profile→HSL。
证据：`research/acr/reports/evidence/2026-11-08-photo-hsl-product/`。
完整高光Stage4/Camera/EV90/epsilon同请求绑定与最终高光TIFF回归仍未完成；
GPU分析/实现未开始。

## 2026-11-09：高光 guide 的 mode4 双遍正式实现

在正式 `native-renderer/src/acr_tone_kernels.cpp/.hpp` 实现保存函数
`a0855c → a08b54/a08cc8/a08e8c → a09134/a09264` 的prepared子域：

1. 普通局部mask pass；
2. guide每行向左循环位移`ceil(width/2)`，不是左右翻转；
3. 第二次普通pass，重新计算统计与sigma；
4. 第二幅collapse结果逆向位移`floor(width/2)`，奇数宽度不能用同一位移；
5. 按水平权重混合，再与原guide生成delta。

`t=clamp(float(x)*(1.f/float(width-1)),0,1)`，`a=abs(fmaf(t,2,-1))`；
权重在double计算`(a*a)*fma(-2,a,3)`再转float，最终为
`fmaf(weight, second-first, first)`。端点也执行减法与FMA，不能直接copy second。
最终source/mask统计和sigma保留**第二pass**的写入，不使用第一pass，也不对
blend重新BuildStats。boundary4且width1依原gate使用普通pass；其它unsupported
boundary仍拒绝。现有boundary2运算顺序、旧测试和默认行为保持。

`native-renderer/src/acr_guide_camera_black.cpp` 的prepared Camera→log→mask
同调用组合已经消费新分支；失败保持整个旧result、输入与request不变。

新增正式 `native-renderer/src/acr_guide_mask_api.cpp`：
`im_native_acr_guide_mask_prepared_f32`，以及 `src/native_renderer.py` 的
`native_acr_guide_mask_prepared`，可直接调用而非只存在研究probe：

```python
from native_renderer import native_acr_guide_mask_prepared

# log_guide 是已正确准备的 float32 HxW log guide，epsilon 来自同一准备状态；
# 不是直接将普通预览RGB或LibRaw数组当作Adobe Stage4。
delta, diagnostics = native_acr_guide_mask_prepared(
    log_guide, epsilon, sigma_mode=1, boundary_mode=4,
)
```

支持完整zero-origin、每轴≤65535、2..2^26个finite像素；明确传epsilon、sigma
和boundary，不猜source请求。返回delta及20项诊断：三组六项统计
`[center,low,high,median,count,fallback]`，然后sigma与statistics_level。
不是完整33项Adobe BuildStats。buffer misalignment、地址回绕、重叠、非法参数
及晚NaN失败时两个输出均保持；Python无算术fallback。

实际验证：

- 保存指令/新bounded raw核对1,060个唯一word；七个callee均来自已有SQLite
  反编译缓存，**新反编译0**。新增7份有界反汇编，未启动Ghidra/Adobe或改工程。
- 保留原blend完整body（219个word＋显式adapter）执行264,112像素，与独立
  公式逐bit一致；scalar片段128值一致；swap构造28组通过。链接产物另外核
  219＋69个保留word与原切片一致，wrapper替换与覆盖限制单独记录。
- 相关source/正式object/公开ABI/Python测试 **313 passed**；新fixed-bit
  CTest见证、15组双遍mask、width1与非法输入检查通过；非GPU CTest **32/32**。
- ASan+UBSan检查swap/blend、双遍mask及正式ABI失败原子性，exit0、stderr空；
  macOS leak detector关闭，不认领完整闭源函数的sanitizer证明。
- 新库重跑已有70组HSL和13组色轮TIFF，指标与上一检查点相同。没有mode4
  Adobe配对TIFF，不能把这组颜色回归称为mode4或完整高光等价验证。
- 四RAW×两来源inspection及既有预览接受/拒绝回归保持；两张可用RAW的6次
  导入/重复/同控件手动预览响应hash与前检查点相同。8个RAW/sidecar与85张
  TIFF指纹保持。新库SHA-256：
  `06e0b8df963e1fbf0540d871bc695b98853cad8cd3bd5e6f4ccce8e89c9e5a92`。
- 全量pytest实际 **5782 passed、5 failed**（162.85秒），仍为前述五项相同
  失败断言。没有删除、跳过或弱化测试。CMake构建、py_compile、diff检查通过。

本轮未改Vue、HTTP高光开关或DNG导出；未装新依赖。打包沿用现有native library
收集和`native_renderer` hidden import；CI CMake源与cache哈希自动包含新文件。
没有重打安装包或Windows实机验证。证据目录：
`research/acr/reports/evidence/2026-11-09-guide-mode4/`。

**完成的是mode4 prepared算子，不是完整CPU高光验收。** 真实source/Stage4、
guide默认Profile/Camera、EV90/epsilon同请求绑定、非零原点/ROI、跨层mask和
最终高光TIFF仍未完成；没有据此开始GPU分析或实现。

## 2026-11-09 继续：Prepare 的 Basic flag-reset 前缀

沿 `db01a4 → b757d0 → b69664/b93244` 复用旧缓存/原指令，新增正式
`native-renderer/src/acr_basic_flag_slots.hpp/.cpp`、安全ABI与
`src/native_renderer.py:native_acr_basic_flag_slots`。这不是重复mode4工作。

恢复范围仅为 `b69664:b69684..b696c8` 的132槽前缀：

- 输入是132个int32 Basic槽，以及实际参数头 `P+4..13`（十六进制）的16字节。
- IDs18..118按九组读取对应byte；**恰为0**才写 `b5df18(ID,2)` 默认值，
  1..255一律保留。不是只认1，也不先clamp或把sentinel当特殊情况。
- HSL IDs43..66全部读取 `header[7] = P+b`；为0时24槽写0。前缀不会清
  后续独立Look的defaults，后续合并仍可产生非零HSL。
- ID129 ToneMapStrength、ID130 ProfileGainTableMap没有此flag映射，原样保留，
  包括sentinel；**不是证明它们在完整Prepare里不变**。后续sentinel/Look writer
  仍须单独执行，不能用本前缀将最终guide sigma设为0。
- 返回5个uint32的store mask，即使写入值相同也记录；高28个unused bits为0。
  原header不修改；后面的curve/vector/header置1不在这个入口里。

```python
from native_renderer import native_acr_basic_flag_slots

# current: int32 shape(132,)，header: uint8 shape(16,)，均为显式参数快照。
values, reset_mask = native_acr_basic_flag_slots(current, header)
```

这是正式Python→C ABI→C++入口，无Python算术fallback；未将照片XMP缺失字段
猜成header，未改变两个编辑页当前HSL行为，也未启用不完整高光。
ABI检查精确counts、null/alignment/地址回绕及四个buffer两两重叠；所有非法
输入都在读取前拒绝，并保持两个输出。Python可接strided/unaligned/read-only
数组，输入相互共享时建立分离快照，不触碰原数组。

递归边界的新局部核对：保守普通CFG验证 `b757d0` 正常返回必过末尾 `P+b50`
检查；非空则进入 `b93244`，其正常非空返回必经merge、pair clear和Amount=1。
在**常规独立参数/owner对象、clear后无隐藏回写**协议下，`db0218` 的递归
入口读到空pair，进入直接选层；限定的是显式 `db01a4` 递归，不是内部
`b83674` 递归的终止保证。选层仍读取准备后副本的ad9/adb，不能复用原caller。
`b75454` 仍能重新准备/安装Profile；未证明其最终字段和像素来源。

实际验证（本轮新库）：

- 六份既有指令文件共638个唯一word匹配固定切片；新反编译0，新反汇编0。
  保留原前缀及原mapping/default leaves，6处显式call/literal重定位，人工
  epilogue停在前缀末尾；链接后81个保留word再次匹配。
- 原前缀5,120组完整内存对照，15,073,280字节一致。正式ABI与原前缀另做
  5,120组：**675,840槽及全部store mask一致**。
- 新算子/组合与HSL/guide边界定向 **573 passed**；不重叠的guide组件
  **276 passed**；最终合并重跑 **849 passed**。CPU CTest **33/33**。
  ASan/UBSan组件及ABI检查通过、stderr空，
  macOS leak detector关闭。CMake、py_compile和diff检查通过。
- 70组已有HSL TIFF：明确synthetic header保留路径经过新前缀及零defaults
  merge后，与直接HSL **1,260,000像素逐bit一致**；清零路径另1,260,000像素
  等于identity。Hue/Saturation/Luminance平均ΔE76保持
  `0.0476400241344 / 0.0478528321362 / 0.0510153126009`。
- 13组色轮TIFF新库回归平均ΔE76保持 `0.0100741159024`。这些不是完整
  Prepare/照片Profile联动或基础高光等价证明，未拟合参数。
- 四RAW两来源检查和预览接受/拒绝结果保持；DJI0539/LCR1132的已有六次预览
  响应hash保持。8个RAW/sidecar及85张TIFF指纹保持，未写照片。
- 全量pytest实际 **6301 passed、5 failed**（163.10秒），仍是同五项断言，
  没有删除、跳过或弱化测试。新库SHA-256：
  `7fe7e3be1ec6981bf7a93f29ad07225c60e03b9607fe36b2e542d58e4460f6cf`。

打包/依赖配置已核，无新依赖，无需改PyInstaller收集；未重打安装包、未测
Windows实机。持久证据：
`research/acr/reports/evidence/2026-11-09-basic-flag-slots/`。

**下一缺口仍是同请求绑定**：实际递归后ad9/adb、`b75454` 最终状态以及对应
Stage4/Camera/EV90/epsilon。当前结果只闭合并实现其中一段确定的前置算子，
不据此声称完整高光已完成，不启动GPU。
