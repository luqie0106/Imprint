# Geographic Overlay

Geographic Overlay 是 Vue / Tauri 的独立成品图工具。输入 JPG/JPEG、TIF/TIFF、PNG 或 DNG，最终叠加地图并另存 JPEG/PNG 或透明 PNG。原照片不写回，不进入 ACR 数值链；SwiftUI 本阶段不修改。

## 预览与原生接口

预览和导出都使用原尺寸像素。界面默认适应窗口，可切换“100% 原尺寸”；图片实际尺寸不再固定到 1600px。

成品底图优先调用现有 `native-renderer` 稳定 C ABI：RGB8/16 转换、neutral `im_renderer_render_full` 与输出复制。只接受与输入 RGB8 逐字节相同的结果，防止对 Photoshop 成品再次应用色彩调整。库缺失、GPU 不可用或中性输出不一致时，保留原像素并使用 C++ CPU 地图路径，明确显示实际后端。

开发工作树自动寻找自己的原生构建库及共同 checkout 中已有的构建库；可通过现有变量 `IMPRINT_NATIVE_RENDERER_LIB` 指定路径。原生库运行中更新后应重启软件。Geographic Overlay 不调用 Python 进程；Imprint 其他工具仍可能使用原有 Sidecar。

## OSM / Overpass

新照片默认选择真实 OSM，图层默认关闭。有 GPS 时读取 EXIF，否则允许手动输入；(0,0) 是有效坐标。

启用 OSM 时，仅将经纬度、查询半径发送到配置的 Overpass endpoint，不上传照片内容。该外发已获得用户明确授权。默认 endpoint：`https://overpass-api.de/api/interpreter`；默认服务不可用时，只尝试一次官方公共备用实例 `https://overpass.private.coffee/api/interpreter`。用户自设 endpoint 不自动切换。

网络与 JSON 解析由 Rust 原生适配层负责，C++ 接收 WGS84 要素数组，统一执行投影、筛选/简化、布局和透明图层渲染。支持道路、水系、水域边界、海岸线与英文地名点；不将 OSM 当作地形 DEM。Terrain 尚不支持。

公共查询范围限制为 0.05–20 km，不静默缩小请求；过大的旧项目需用户调整半径。响应、要素与点数均有上限；HTTP 错误、限流、超时、解析错误与无数据明确提示，绝不回落为未标记的 Mock。参数透明度/颜色/线宽变化复用已有地理数据缓存。数据未加载或需要重试时可点击“刷新地图”。

读取既有 HTTP(S) 环境代理与 macOS 静态系统代理；也可设置 `IMPRINT_GEO_HTTP_PROXY`。不修改系统代理，不处理 PAC 自动代理脚本。来源与版权信息随预览返回，界面提供 OpenStreetMap 版权链接；渲染器在安全空间允许时绘制署名。

Mock 保留为明确标识的测试模式。Mock 的道路、水系、海岸线与地名均为虚构，不代表目标位置。

## 天空与图层

天空保护仍是保守的颜色/边缘连通检测，未作为可靠语义分割模型。采用二维主体边界距离羽化，地图随可用天空空间布局，不固定于顶部 15%。标签放不下时只跳过标签，不再清空已绘制地图。没有可靠天空时允许留空，并给出可诊断信息；用户可以调整手动边界与自动天空开关。

天空视觉效果由用户验收，本次开发不宣称通过实际建筑保护或羽化质量验收。

输入工作图是 sRGB RGB8，输出是 straight-alpha sRGB RGBA8；source-over 在 sRGB 编码值中合成。Alpha 为零时保留工作图对应像素。关闭图层不需要地图数据或网络，恢复工作底图。PNG/JPEG 输出为 8-bit；不宣称保留源 16-bit 精度或全部 EXIF/ICC。

地图采用米制局部等距投影及一致缩放，比例尺对应真实投影比例。查询半径与图层显示缩放分开。

## 源码与验证

- `native-geo/`：C++ 地图、天空接口、共享缓存、图层与旧 renderer 适配器。
- `tauri-frontend/src-tauri/src/geo_osm.rs`：HTTP、代理、OSM JSON 与数据缓存。
- `tauri-frontend/src-tauri/src/native_preview.rs`：旧 renderer 库发现和安全调用。
- `tauri-frontend/src-tauri/src/geographic_overlay.rs`：输入、会话、预览、项目和导出。
- `tauri-frontend/src/views/GeographicOverlayPage.vue` 与 `stores/geographicOverlay.ts`：工具界面、参数、历史与调用竞态保护。

配置与照片项目关联保存，旧项目缺少 GeoConfig 时保持关闭。最终导出重新读取全尺寸源图，并以临时文件与原子无覆盖提交输出。

验证与启动命令见 [交付记录](geographic-overlay-validation.md)。Overpass 数据格式与公共实例参考 [OSM 官方 Wiki](https://wiki.openstreetmap.org/wiki/Overpass_API)、[Overpass 输出格式](https://dev.overpass-api.de/overpass-doc/en/targets/formats.html)。
