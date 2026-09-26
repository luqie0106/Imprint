# Imprint macOS 原生界面

这是一个独立的 SwiftUI macOS 客户端，沿用项目现有 FastAPI Sidecar。支持连拍优选、去朦胧、理光风格、模型管理、偏好训练和计算后端设置；原有 Tauri 客户端仍可使用。

## 开发运行

需要 macOS 14 或更新版本、Swift 5.9 或更新版本，以及项目原有的 Python 依赖环境。进入 `native-macos` 目录后运行：

```bash
swift run ImprintMac
```

当前开发机的 Command Line Tools 默认 SDK 与 Swift 补丁版本不匹配时，指定已安装的 macOS 15.4 SDK：

```bash
SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX15.4.sdk swift run ImprintMac
```

程序会在当前目录或其父目录寻找 `src/app_api.py`，优先使用 `IMPRINT_PYTHON`，其次使用当前 Conda/venv 环境，最后尝试 `~/.conda/envs/py311/bin/python`。从其他目录运行时可设置 `IMPRINT_PROJECT_ROOT` 为项目根目录。Sidecar 监听 `127.0.0.1` 的随机端口，原生客户端从其启动输出读取端口。

Swift Package 是开发运行入口，尚未配置签名、沙盒权限或 `.app` 发布打包。macOS 系统文件选择器负责照片与目录选择。
