# Windows CUDA 原生渲染诊断

此脚本收集少量系统与 GPU 信息，在 Release 模式编译 CUDA native renderer，运行 CTest GPU contract 和 L2 benchmark；环境满足条件时，再运行 Python 对比测试。

## 运行

解压诊断包时保留以下目录结构：

```text
<包根目录>/
  native-renderer/
  src/dehaze.py
```

在 `<包根目录>` 打开 PowerShell，运行一条命令：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\native-renderer\tools\Run-WindowsCudaDiagnostics.ps1
```

也可用 PowerShell 7 执行：

```powershell
pwsh -NoProfile -File .\native-renderer\tools\Run-WindowsCudaDiagnostics.ps1
```

脚本不会安装 Python 包、CUDA Toolkit、Visual Studio 组件或其他依赖。CMake 配置使用 `IMPRINT_ENABLE_CUDA=ON`、`BUILD_TESTING=ON` 和 `CMAKE_BUILD_TYPE=Release`。

默认在包根目录生成 `cuda-diagnostics-<timestamp>.zip`。也可以指定输出文件：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\native-renderer\tools\Run-WindowsCudaDiagnostics.ps1 -OutputPath .\cuda-diagnostics.zip
```

## 诊断结果

构建目录位于系统临时目录，打包后会清理。ZIP 只包含 `summary.txt`、`build-config-summary.txt` 和 `logs/`，不包含 build 目录、DLL/EXE、完整 `CMakeCache.txt`、照片或其他用户文件。

日志收集 Windows 版本与架构、CPU/GPU 名称、显卡驱动版本、`nvidia-smi` 的 GPU 名称和驱动信息，以及可用的 `nvcc`、CMake、MSVC/Visual Studio 版本。脚本不扫描照片目录，也不枚举正在运行的进程。

- **CTest GPU contract: PASS**：CUDA renderer contract 实际运行并通过。
- **CTest GPU contract: SKIP**：CTest 按项目配置将退出码 77 识别为 `SKIP_RETURN_CODE`；这不表示 CUDA contract 通过。
- **L2 benchmark: PASS**：详见日志中的 backend、渲染尺寸、耗时和吞吐率。退出码 77 会显示为无可用 GPU backend 的跳过。
- **Python CUDA comparison**：当前 `PATH` 中的 `python.exe` 同时具备 NumPy、pytest 且构建出 DLL 时，脚本设置 `IMPRINT_NATIVE_RENDERER_LIB` 和 `IMPRINT_NATIVE_RENDERER_BACKEND=CUDA`，运行 `tests/test_metal_dehaze.py`，将 CUDA shader 与 Python CPU 参考对比。缺少 Python 或依赖时会跳过，不会安装依赖。
- **Python basic bridge comparison**：包中存在 `src/native_renderer.py`、`src/ricoh_filter.py`、`src/image_io.py`、`native-renderer/tests/test_basic_bridge.py`，且同一 Python 环境具备 NumPy、pytest、OpenCV（`cv2`）、Pillow 和 rawpy 时，运行 basic 调整桥接对比。缺少文件或依赖、pytest 报告有跳过项时，摘要会标为 SKIP/PARTIAL。

工具缺失或配置、编译、测试失败时，脚本仍会先生成诊断 ZIP。归档完成后进程退出码为：`0` 表示 CUDA contract 和 benchmark 均通过；`2` 表示 contract 或 benchmark 被按退出码 77 跳过；`1` 表示必需阶段失败/未执行，或已运行的 Python 对比失败。把生成的 ZIP 回传即可继续排查。
