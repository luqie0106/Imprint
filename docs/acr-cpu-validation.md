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
