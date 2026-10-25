# Geographic Overlay 交付与验证记录

日期：2026-10-25。开发分支：`feature/geographic-overlay`。独立工作树：`/Users/hualaiwu/.codex/worktrees/geographic-overlay/photo_sort`。

## 使用入口

从该工作树启动 Tauri 桌面端，点击顶部“地理叠加”，选择 JPG/JPEG、TIF/TIFF、PNG 或 DNG，启用地图叠加。有 EXIF GPS 时自动读取，否则手动输入；(0,0) 是有效坐标。可查看原图、合成预览及透明层，保存/恢复项目，并导出原尺寸 PNG/JPEG 或透明 PNG。

```sh
cd /Users/hualaiwu/.codex/worktrees/geographic-overlay/photo_sort/tauri-frontend
npm run tauri dev -- --config '{"bundle":{"resources":[]}}'
```

此开发命令通过配置覆盖排除工作树未生成的 `dist-python` 发布资源，不改变主配置。

单独 `npm run dev` 的浏览器页面没有 Tauri 原生文件与渲染桥，会明确提示需要桌面版，不会伪造预览成功。其他照片工具仍可能需要已有 Python Sidecar；Geographic Overlay 的 IO、地图、合成和项目存储不调用 Python。

## 文件变更

| 文件/目录 | 职责 |
| --- | --- |
| `native-geo/include/imprint_geo.h` | RGB/RGBA C ABI 与配置 |
| `native-geo/include/imprint_geo.hpp` | 地理坐标投影、Provider、可替换天空接口 |
| `native-geo/src/geo.cpp` | Mock 几何、米制投影、AA 线稿、导航与文字、天空布局、Alpha 合成及缓存 |
| `native-geo/include/image_io.h`, `native-geo/src/image_io.mm` | macOS 原生全尺寸输入、sRGB 转换、DNG Core Image 路径 |
| `native-geo/CMakeLists.txt`, `tests/native_geo_test.cpp`, `README.md` | 独立 C++ 构建、测试与接口说明 |
| `tauri-frontend/src-tauri/src/geographic_overlay.rs` | 会话、图像 IO、后台调度、项目持久化、全尺寸导出、安全写入 |
| `tauri-frontend/src-tauri/{Cargo.toml,Cargo.lock,build.rs,src/lib.rs}` | 原生静态编译与 Tauri 命令注册 |
| `tauri-frontend/src/views/GeographicOverlayPage.vue` | Vue 独立工具页面 |
| `tauri-frontend/src/stores/geographicOverlay.ts` | 配置、历史、预览竞态保护及项目调用 |
| `tauri-frontend/src/App.vue` | 导航与本页 Sidecar 离线隔离 |
| `scripts/geographic-overlay/ui-smoke.js`, `ui-contract.js` | Playwright CLI DOM 与显式测试桩契约断言 |
| `docs/features/geographic-overlay*.md` | 使用、边界和验证记录 |

SwiftUI、ACR 核心、RAW 解码模块、DNG 写入器均未修改。

## 已完成的验证

- Vue：`npm run build` 通过，包含 vue-tsc 与 Vite 生产构建。
- 桌面可执行文件：`TAURI_CONFIG='{"bundle":{"resources":[]}}' cargo build` 通过；未代替用户启动桌面程序或验收视觉效果。
- Rust：`cargo check` 通过；9 项单元测试通过，两个按本地夹具启用的 smoke 已手动运行通过，覆盖方向、半透明 PNG、RGB/16-bit TIFF 和 DNG 全尺寸导出。检查使用 `TAURI_CONFIG='{"bundle":{"resources":[]}}'`。
- C++：独立 Release CMake 构建与 CTest 通过（1 个测试套件）。覆盖坐标校验、无 GPS 与 (0,0)、投影、窄图、开关恒等、Alpha、配置错误、可替换天空接口及合成。
- Playwright CLI：先通过 `lsof` 确认 1421 监听 PID 的 cwd 为最终 feature 工作树，再执行 DOM/程序断言；1100×780 下折叠前后卡片 x=785、宽度=295，保持一致；默认值、离线隔离、坐标零值、范围、Overpass/Terrain 明确错误通过，Console/Page errors 为空，没有 Overpass 网络请求。
- UI IPC 契约：显式测试桩验证选择照片、传递 snake_case 配置、预览与透明层接入、撤销、保存/恢复、合成与透明图层导出调用。该项不声称是真实原生渲染验收。
- 原生 IO 数值检查：已验证红顶蓝底 PNG 没有上下翻转；RGB Linear DNG 返回 160×120 有效像素；真实 DJI CFA DNG 返回 5272×3948 有效像素，非嵌入缩略图。CFA 尺寸为系统解码器输出的有效照片区域，区别于包含传感器边缘的 TIFF 原始阵列尺寸。
- `git diff --check` 通过；开发期间多次检查本地 main 的已提交状态，基线保持 `7ff7ff8`，无需引入新的 main 提交。

## Playwright CLI 复现

先在本工作树启动 `npm run dev -- --host 127.0.0.1 --port 1421`。使用不同会话，避免测试桩残留影响无桩检查：

```sh
playwright-cli -s=geo-dom open http://127.0.0.1:1421
playwright-cli -s=geo-dom run-code --filename=../scripts/geographic-overlay/ui-smoke.js
playwright-cli -s=geo-dom close
playwright-cli -s=geo-contract open http://127.0.0.1:1421
playwright-cli -s=geo-contract run-code --filename=../scripts/geographic-overlay/ui-contract.js
playwright-cli -s=geo-contract close
```

测试结束后停止自己启动的 Vite，并检查 1420/1421 是否还在监听。本次测试创建的 Vite 服务与 `imprint-geo` Playwright 会话已关闭。

## 未验收与限制

- 天空羽化、主体保护、排版视觉质量由用户通过实际预览验收，未作为本次通过项。
- Mock 是可运行的测试地图，不是真实位置地图。Overpass 尚未联网，Terrain 无 DEM，二者均明确报错；后续接入步骤见功能说明。
- 当前内部工作图与输出为 8-bit sRGB，不是 16-bit 保真编辑器，也不保留全部源 EXIF/ICC 元数据。透明输入以白底形成工作照片。
- DNG 的色彩呈现使用系统解码器，不宣称与 Photoshop/ACR 参数渲染一致；部分相机或压缩格式可能被系统拒绝。
- Windows/Linux 尚未做运行验证，当前 DNG 读取仅在 macOS 实现；嵌入 ICC 的输入在缺少转换能力的平台明确拒绝。BigTIFF 不在当前 TIFF 解析支持范围。
- 未打包、签名、发布安装程序。交付仅提交并推送 feature 分支，不合并 main。
- 未来 main 若同时修改 App.vue 导航、Tauri lib.rs 注册或 Cargo/build.rs，可能需要局部合并；独立 native-geo 对现有 ACR 模块无改动。
