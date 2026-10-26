# Geographic Overlay 交付与验证记录

日期：2026-10-26。目标分支：`feature/geographic-overlay`。最终整合工作树：`/Users/hualaiwu/.codex/worktrees/geographic-overlay/photo_sort`。

## 使用入口

从最终工作树启动 Tauri 桌面端，点击顶部“地理叠加”，选择 JPG/JPEG、TIF/TIFF、PNG 或 DNG，启用地图叠加。有 EXIF GPS 时自动读取，否则可手动输入；(0,0) 是有效坐标。预览与导出使用原尺寸像素，界面可切换“100% 原尺寸”。可查看合成预览及透明层，保存/恢复项目，并导出原尺寸 PNG/JPEG 或透明 PNG。

```sh
cd /Users/hualaiwu/.codex/worktrees/geographic-overlay/photo_sort/tauri-frontend
npm run tauri dev -- --config '{"bundle":{"resources":[]}}'
```

此开发命令通过配置排除工作树未生成的 `dist-python` 发布资源，不改变主配置。该工具的输入、地图、合成和项目存储不调用 Python；Imprint 其他工具仍可能使用原有 Sidecar。浏览器单独运行没有 Tauri 文件和渲染桥，不会伪造原生预览成功。

## 本轮变更

- 预览改为原尺寸像素，不再固定缩到 1600px，并加入 100% 查看控制。
- 成品 RGB8 底图可安全调用既有 native renderer 的 neutral `im_renderer_render_full`，并要求返回像素与输入逐字节相同。库不可用、GPU 不可用或结果不一致时保留原像素，地图仍走 C++ CPU 路径。报告的“Metal”仅指底图渲染后端；地图核心几何和绘制在 C++ CPU 执行。
- 新增 Rust OSM/Overpass HTTP、JSON 与 C ABI 几何接入。道路、水系、水域边界、海岸线和英文地名传给 C++ 投影、布局与透明图层渲染。新照片默认 OSM；旧项目缺少 GeoConfig 时保持旧行为。
- 查询只发送经纬度与半径，不发送照片内容；用户已明确授权此数据外发。默认 Overpass endpoint 为 `https://overpass-api.de/api/interpreter`，符合切换条件时最多尝试一次公共备用实例 `https://overpass.private.coffee/api/interpreter`。请求有 25 秒超时和 20 MB 响应上限；公共查询半径限制到 20 km，并带正向缓存、失败缓存、串行请求和限流冷却处理。读取 HTTP(S) 环境代理和 macOS 静态系统代理，也支持 `IMPRINT_GEO_HTTP_PROXY`。
- 修复标签无法放下时清空整层的问题；关闭 OSM、没有地图数据时可直接得到空透明层恒等结果。共享 SkyMask 缓存限制为 32 MP（约 192 MB）；全图 session RGB 内存预算为 256 MB。预览串行执行并抑制过期请求。

## 文件变更

| 文件/目录 | 职责 |
| --- | --- |
| `native-geo/include/imprint_geo.h`, `native-geo/include/imprint_geo.hpp`, `native-geo/src/geo.cpp` | RGB/RGBA C ABI、几何投影、道路与地理要素绘制、天空布局与 Alpha 合成 |
| `native-geo/include/native_preview.h`, `native-geo/src/native_preview.cpp` | 安全调用既有 renderer 的 neutral full-resolution RGB8 适配器 |
| `native-geo/include/image_io.h`, `native-geo/src/image_io.mm` | macOS 原生全尺寸输入、sRGB 转换及 DNG Core Image 路径 |
| `native-geo/CMakeLists.txt`, `native-geo/tests/native_geo_test.cpp`, `native-geo/README.md` | 独立 C++ 构建、测试与接口说明 |
| `tauri-frontend/src-tauri/src/geo_osm.rs` | Overpass HTTP、代理、OSM JSON、缓存与限流退避 |
| `tauri-frontend/src-tauri/src/native_preview.rs` | 既有 renderer 库发现及安全调用 |
| `tauri-frontend/src-tauri/src/geographic_overlay.rs` | 会话、原尺寸图像 IO、预览调度、项目持久化、全尺寸导出及安全写入 |
| `tauri-frontend/src-tauri/{Cargo.toml,Cargo.lock,build.rs,src/lib.rs}` | 原生构建和 Tauri 命令注册 |
| `tauri-frontend/src/views/GeographicOverlayPage.vue`, `tauri-frontend/src/stores/geographicOverlay.ts`, `tauri-frontend/src/App.vue` | 工具界面、状态、参数与导航 |
| `scripts/geographic-overlay/ui-smoke.js`, `scripts/geographic-overlay/ui-contract.js` | Playwright CLI DOM 检查和显式 IPC 测试桩契约检查 |
| `docs/features/geographic-overlay*.md` | 功能说明、边界和验证记录 |

SwiftUI、ACR 核心、RAW 解码模块和 DNG 写入器未纳入本功能改动范围。

## 当前验证记录

- Rust：`cargo check` 通过；Rust 单元测试 20 passed、0 failed、3 ignored。3 项依赖本机环境的专用 smoke 已分别显式运行通过：Metal 1801×905 逐字节恒等、真实 DJI DNG 全尺寸读取/PNG 导出、图像方向/透明 PNG/TIFF 夹具检查。常规测试命令保留 ignored 标记以支持没有本机夹具或 GPU 的环境。
- C++：Release CMake 构建与 CTest 通过，1 个测试套件，约 0.36 秒。
- Native renderer：主机 Metal 实测 1801×905，neutral full render 输出与输入 RGB8 逐字节相同，用时约 69.6 ms。该结果不代表地图图层由 Metal 绘制。
- Playwright CLI：经 `lsof` 确认测试服务来自最终 feature 工作树，使用独立端口 1422。DOM、Console、Network 检查与显式 IPC 测试桩契约通过；100% 查看控件已测。IPC 桩检查不代表实际原生渲染或真实 OSM 联网验收。
- Overpass：此前真实公共 endpoint 请求均超时，没有一次成功响应。按用户要求不再重试，等待用户联网验证；因此本记录不宣称真实 OSM 联网通过。
- 最终 `npm run build` 与 `TAURI_CONFIG='{"bundle":{"resources":[]}}' cargo build --locked` 均通过。
- 本次 Vite 1422 服务与独立 Playwright 会话已关闭；检查 1420/1421/1422 均无监听。未启动或终止用户的桌面应用、Python 后端或其他服务。
- 本地 main 保持 `7ff7ff8` 且工作树干净；远端 main 是当前 feature 的祖先，无需额外合并。交付仅提交并推送 feature，不合并或推送 main。

### Playwright CLI 复现

在本工作树启动 Vite；1422 是独立测试端口，不占用用户服务：

```sh
cd /Users/hualaiwu/.codex/worktrees/geographic-overlay/photo_sort/tauri-frontend
npm run dev -- --host 127.0.0.1 --port 1422
```

另开终端运行两组隔离会话：

```sh
cd /Users/hualaiwu/.codex/worktrees/geographic-overlay/photo_sort/tauri-frontend
playwright-cli -s=geo-dom open http://127.0.0.1:1422
playwright-cli -s=geo-dom run-code --filename=../scripts/geographic-overlay/ui-smoke.js
playwright-cli -s=geo-dom close
playwright-cli -s=geo-contract open http://127.0.0.1:1422
playwright-cli -s=geo-contract run-code --filename=../scripts/geographic-overlay/ui-contract.js
playwright-cli -s=geo-contract close
```

测试后停止本次启动的 Vite 服务，并确认端口 1422 已释放。最终开发启动命令见本页“使用入口”。

## 未验收与限制

- 天空羽化及主体保护的实际视觉质量由用户验收，当前不宣称已通过实图验收。
- 真实公共 Overpass 请求超时；联网能力须由用户验证。Terrain 尚无 DEM 数据，不支持地形渲染。
- 工作图和输出为 8-bit sRGB，不是 16-bit 保真编辑器，也不保留全部源 EXIF/ICC 元数据。透明输入以白底形成工作照片。
- DNG 色彩呈现使用系统解码器，不宣称与 Photoshop/ACR 参数渲染一致；DNG 读取当前仅在 macOS 实现，部分相机或压缩格式可能被系统拒绝。BigTIFF 不在当前 TIFF 解析支持范围。
- Windows/Linux 尚未做运行验证；未打包、签名或发布安装程序。
- 不修改 ACR、RAW 解码或 SwiftUI 路径。
- 当前完成源码与开发可执行文件构建，未代替用户启动桌面软件或验收真实地图效果。
