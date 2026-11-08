# ACR CPU 重建与本机验证

更新：2026-11-08。本文记录已验证的局部CPU能力及产品入口；完整ACR
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
  boundary mode2。source level选择、独立base cache槽、transfer分类分别处理。
- `acr_tone_kernels.cpp`：保存指令确定的灰度FMA、raw分位数/分组统计、
  Gaussian权重和横纵求和顺序、ceil remap采样和sigma；统一内部统计层。
- `acr_profile_hsl_controls.hpp/.cpp`：Profile/current HSL24非线性控制
  合并，显式解析好的单色predicate/灰度byte、完整int32输入、skip/wrote mask。
  正式ABI和Python `native_acr_profile_hsl_controls`可调用，尚未自动从
  真实Profile元数据生成这些控制输入。

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

真实source/Stage4/Camera/EV90/epsilon的同请求绑定、profile递归与其余
Prepare effects、跨source层mask、mode4、非零原点/ROI、完整BuildStats
及最终高光联动尚未闭合。现有fixture目录未找到带可信controls记录的
Profile切换或基础Highlights2012成对样本；高光色轮TIFF不是基础高光样本。
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
