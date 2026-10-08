# 去朦胧 C++ CPU 算子

`native-renderer` 除 Metal / D3D12 浮点反演外，还提供不需要 GPU 实例的 C++ CPU 算子：

- 物理去朦胧浮点像素反演，供 CPU 模式及 GPU 不可用时使用。
- 暗背景保护，包括亮度包络和源颜色上限。
- 非局部实验算法的标量 RGB 三维 LUT 查询，供 conservative / strong 使用。
- strong 模式的光学厚度 LUT 加权累积。
- 镜头畸变/色差的 RGB16 双线性重采样；Lensfun 生成坐标计划。
- 基础八项的 RGB8 / RGB16 逐像素调整，避免多份全图浮点临时数组。

Python 通过 `native_dense.py` 调用这些可选 C ABI。旧动态库缺少相应符号或算子执行失败时，仍使用原 Python 实现；无需另装依赖。非局部开关仍默认关闭。新源文件随已有动态库一起构建、打包，不需要单独的库。

`auto/native` 首先尝试原有 GPU 反演，再尝试 C++ CPU，最后回退 Python CPU；`cpu` 直接尝试 C++ CPU。暗背景保护及非局部 LUT 运算独立使用 C++ CPU。渲染诊断的 `pixel_backend` 和 `dark_guard_backend` 记录实际完成运算的路径；状态接口的 `cpu_available` 表示新 CPU 反演符号可用。

这里没有迁移场景估计、自动曝光、非局部射线拟合、WLS 求解、引导滤波或 LUT 高斯平滑。解码、色彩管理、理光 LUT 生成、DNG 写入和任务管理也仍由 Python 负责。这些改动不能表述为“全部处理已原生化”。

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
