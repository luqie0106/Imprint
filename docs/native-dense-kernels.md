# 去朦胧 C++ CPU 算子

`native-renderer` 除 Metal / D3D12 浮点反演外，还提供不需要 GPU 实例的 C++ CPU 算子：

- 物理去朦胧浮点像素反演，供 CPU 模式及 GPU 不可用时使用。
- 暗背景保护，包括亮度包络和源颜色上限。
- 非局部实验算法的标量 RGB 三维 LUT 查询，供 conservative / strong 使用。
- strong 模式的光学厚度 LUT 加权累积。
- 镜头畸变/色差的 RGB16 双线性重采样；Lensfun 生成坐标计划。
- 基础八项的 RGB8 / RGB16 逐像素调整，避免多份全图浮点临时数组。
- 自动曝光的全图输入验证与 RGB 峰值归约。
- 引导滤波系数生成后的全图透射率重建。
- conservative 模式的全图非局部支持度与透射率 relief 合成。

Python 通过 `native_dense.py` 调用这些可选 C ABI。旧动态库缺少相应符号或算子执行失败时，仍使用原 Python 实现；无需另装依赖。非局部开关仍默认关闭。新源文件随已有动态库一起构建、打包，不需要单独的库。

`auto/native` 首先尝试原有 GPU 反演，再尝试 C++ CPU，最后回退 Python CPU；`cpu` 直接尝试 C++ CPU。暗背景保护及非局部 LUT 运算独立使用 C++ CPU。渲染诊断的 `pixel_backend` 和 `dark_guard_backend` 记录实际完成运算的路径；状态接口的 `cpu_available` 表示新 CPU 反演符号可用。

场景估计、自动曝光 EV 决策、非局部射线拟合、WLS 求解、低分辨率引导滤波系数和 LUT 高斯平滑仍由 Python / NumPy / OpenCV 负责。解码、色彩管理、理光 LUT 生成、DNG 写入和任务管理也仍由 Python 负责。这些改动不能表述为“全部处理已原生化”。

## 验证

用同一输入、参数和色彩空间对比原 NumPy 公式。`IMPRINT_NATIVE_DENSE=0` 可暂时关闭新算子，便于对比；恢复默认后应重启后端，避免继续使用此前的预览缓存。算法版本已经更新。

```bash
cmake -S native-renderer -B /private/tmp/imprint-dense-build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build /private/tmp/imprint-dense-build --parallel
IMPRINT_NATIVE_RENDERER_LIB=/private/tmp/imprint-dense-build/libimprint_renderer.dylib /Users/hualaiwu/.conda/envs/py311/bin/python -m pytest -q native-renderer/tests/test_dense_kernels.py native-renderer/tests/test_lens_remap.py
```

Windows 使用对应的 `Release/imprint_renderer.dll`，Linux 使用 `.so`。新的 CPU 算子与 GPU 可用性无关。macOS 数值对比和编译成功不能替代 Windows 编译、目标硬件 GPU 验证或发行包扫描。

## 本机结果（2026-10-07）

macOS 上用项目 `py311`、Release 动态库对比 `IMPRINT_NATIVE_DENSE=0/1`。以下是同一全分辨率解码数组的物理去朦胧阶段耗时，包含场景估计及暗背景保护，不含 RAW 解码、其他调整和 DNG 写出。非局部模式为 `off`。

| 样片 | 像素 | Python | C++ 接入后 |
| --- | --- | --- | --- |
| DJI_0523 | 2090 万 | 6.07 秒 | 0.77 秒 |
| DJI_0539 | 2090 万 | 5.66 秒 | 0.72 秒 |
| LCR_9472 | 2450 万 | 6.49 秒 | 0.65 秒 |
| LCR_1132 | 2450 万 | 6.79 秒 | 0.68 秒 |
| LXD_5899 | 2450 万 | 8.95 秒 | 0.90 秒 |

五张照片分别对比 `off`、`conservative`、`strong`，共 15 组。最大 float32 绝对差为 `1.78e-6`，量化到 RGB16 后最大相差 1 个码值。太阳、建筑和烟花对比图中未见新增差异。大多数样片的非局部可靠性门槛仍会回退；LCR_9472 的 `conservative` 实际激活，耗时从 9.93 秒降到 1.78 秒。`strong` 实际算子接入由合成场景和 65³ LUT 数值测试覆盖，不能据这些回退实图认定强模式画质已获接受。

216 万像素独立算子的三次运行中位数：浮点反演约 12.1 倍、暗背景保护约 7.2 倍、标量 LUT 查询约 58.4 倍提速。这些是本机测量，不能代替其他设备上的测试。

CPU 与接口回归：306 项通过，32 项因 GPU/可选环境跳过，1 项阴影预算对比度断言失败；该失败用 HEAD 原始 `_physical_pixels` 也已复现，保留测试和公式。独立 Metal 强制硬件套件未执行。CMake Release 构建、ABI CTest、Python 语法及 diff 检查通过。发布流程新增了两平台的 dense CPU 数值测试，但尚未运行远端 Actions、扫描发行包或在外部 RAW 编辑器导入。

镜头计划、内存缓存、完整预览计时见 [preview-latency.md](preview-latency.md)。镜头算子额外保留两种坐标精度，以匹配当前 OpenCV 优化路径；完整实图 RGB16 对比最大差异为 1 个码值。

基础 CPU 路径沿用原 NumPy 公式，旧库保留 Python 回退；`IMPRINT_NATIVE_DENSE=0` 可对照原始实现。CPU 模式的历史请求值 `basic_backend=python` 保持兼容，界面显示为 CPU。理光默认处理、场景估计与透射率构建仍没有全部迁入 C++，不能据基础八项原生化声称整条图像链路都是 C++。

基础八项的直接 ABI / NumPy 对照测试 41 项通过。五张全分辨率样片分别以 RGB8 与 RGB16 比较一组混合参数（曝光 +0.85、对比度 +14、高光 -28、阴影 +22、白色 +3、黑色 -11、自然饱和度 +37、饱和度 -12），10 组输出逐像素相同；随机与极值测试最大差 1 个码值。镜头、基础、缓存和接口等定向后端回归共 191 项通过；Vue 调度行为测试 5 项、生产构建和 ABI CTest 通过，GPU 合约按环境跳过。尚未测实际 Tauri 输入到显示的延迟。

## 本轮接入门槛与结果（2026-10-08）

新路径保持原 Python 公式。全分辨率实图验收要求 float32 最大绝对差不超过 `3e-6`，量化 RGB16 后最大差不超过 1 个码值；自动曝光输出、EV 和原因必须完全相同。数值通过后还要测速度，更慢的路径不接入默认流程。旧库缺少符号或执行失败仍回退 Python。

800 万像素的独立 CPU 算子、三次中位数（包括 ABI 必须的输入验证）：

| 算子 | Python / NumPy | C++ | 决定 |
| --- | --- | --- | --- |
| 输入验证及 RGB 峰值 | 18.81 ms | 3.83 ms | 接入 |
| 全图透射率重建 | 47.08 ms | 22.23 ms | 接入 |
| conservative relief 合成 | 314.85 ms | 21.12 ms | 接入 |
| 独立曝光增益乘法 | 9.38 ms | 15.74 ms | 不接入，保留 NumPy |

五张全分辨率样片分别对比 `off / conservative / strong`，15 组新 CPU 路径与关闭原生后的 Python 参考全部满足门槛，最大 float 差 `1.58e-6`，RGB16 最大差 1 个码值；五张自动曝光输出、EV 与原因完全一致。`conservative` 仅风机样片实际激活，其他非局部实图会按既有门槛回退。非恒定透射率重建另由随机数值与集成测试覆盖；这些实图的恒定场不会执行该算子。

与本轮修改前已有 C++ 算子的 HEAD 相比，风机样片 conservative 全分辨率 CPU 处理三次中位数从 `1.662 s` 降到 `0.830 s`。同张普通 off 路径为 `0.655 / 0.641 s`，烟花 off 为 `0.900 / 0.887 s`，不应把微小波动当作新增显著提速。新 relief 的收益只出现在实际激活的照片和模式中。

Metal 暗背景保护与浮点反演合并为一次 GPU pass。五张全分辨率 RAW/DNG 对比原 Python 反演及保护公式，最大 float 差 `1.61e-6`，RGB16 最大差 1 个码值。烟花 `LXD_5899` 实际触发暗背景保护；预热后交替各测五次，中位数从原 GPU 反演加 C++ CPU 保护的 `0.571 s` 降到合并的 `0.397 s`。其余四张的保护阈值为零，保持原反演路径。Metal 新路径默认启用，`IMPRINT_NATIVE_GUARDED_FLOAT=0` 可关闭。

D3D12 已实现相同合并公式与 ABI，但本轮没有 Windows 编译和实机 GPU 验证，因此默认仍用原 GPU 反演加 CPU 保护；仅目标硬件验收时可设 `IMPRINT_NATIVE_GUARDED_FLOAT=1`。macOS 编译成功和公式对照不能替代 Windows 验收。

理光现有 65³ LUT 插值方案没有扩大默认接入：四张实图、三个预设对照中，部分彩色预设 RGB8 最大差达到 23–48 个码值，超过本轮门槛。理光默认 Python 路径保持不变，现有用户主动选择原生的选项保留。

这些测量是算子或已解码数组上的处理时间，不能代表 Tauri 输入到画面显示的延迟。需重启 Sidecar 加载新动态库；已安装发行包需要重新构建。

本轮 CPU / 接口 / fallback / 图像保护回归 252 项通过；沙箱外真实 Metal 新合并算子 10 项通过，连同原浮点路径回归共 46 项通过。Release 构建、ABI CTest、Python 语法和 diff 检查通过。本地测试目录遵循仓库现有忽略策略，不随本轮变更提交。本轮没有运行远端 Actions、扫描发行包或在 Camera Raw / Lightroom 验证导入。
