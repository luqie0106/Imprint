# Geographic Overlay

Geographic Overlay 是 Vue / Tauri 中独立的成品图工具。读取 Photoshop 等软件处理完成的 JPG/JPEG、TIF/TIFF、PNG 或 DNG，在显示变换之后叠加地图，另存 JPEG/PNG 或独立透明 PNG 图层。它不进入 ACR 算子链，不写入原照片，不裁剪照片。SwiftUI 入口本阶段不修改。

## 模块边界

- `tauri-frontend/src/views/GeographicOverlayPage.vue`：参数、预览、历史、文件选择和保存。
- `tauri-frontend/src/stores/geographicOverlay.ts`：前端配置模型和原生调用契约。
- `tauri-frontend/src-tauri/src/geographic_overlay.rs`：原生图像 IO、会话、项目、全尺寸导出及阻塞工作调度。
- `native-geo/`：独立 C++17 Provider、投影、布局、天空保护、抗锯齿地图和透明图层渲染。

功能核心通过 Cargo 编译静态链接，不需要 Python 或单独部署 C++ 动态库。Imprint 的其他工具仍沿用既有后端；本页不以该后端就绪作为可用条件。

## 数据与真实性

默认 Provider 是 **Mock 测试数据**。道路、水系、海岸线和英文地名是渲染测试几何，不是真实地理信息。坐标与查询半径用于投影和尺度验证，不能据此把 Mock 当作目标位置的真实地图。

Overpass 接口地址可保存，当前 Provider 返回明确的未实现错误，不发起联网请求。Terrain 没有 DEM，当前明确不支持，不生成虚构等高线。项目应保留版本、Provider 与来源元数据。

未来接入 Overpass 时，需要实现查询构造、HTTP 超时和取消、Overpass JSON 转 GeoData、限流和服务错误、缓存时效及 OSM 署名；复用现有投影、样式、布局、渲染和 UI。不能将网络失败回退为未标识的 Mock 成功结果。

## 天空与布局

本阶段使用保守天空估计，而非语义分割模型。天空接口独立，便于后续替换。本地检测、连通空间和主体边缘羽化共同约束 Alpha；手动保护边界可进一步收紧可用区域。覆盖范围是用户上限，地图不固定于顶部 15%。查询半径与显示缩放分别控制地理数据范围和版式。

无可靠天空时应保守留空。高楼、树木、山脊、灰白立面和细小风机存在识别局限，使用者应检查透明层和成品图，必要时收紧手动边界。不能将启发式检测称作可靠语义分割或宣称所有建筑均受保护。

## 渲染规则

核心输入是已完成显示变换的 sRGB RGB8，输出是 straight Alpha 的 sRGB RGBA8 透明图层；source-over 在 sRGB 编码值上合成，是最终图层的明确定义，不参与场景线性曝光/色彩运算。Alpha 为零时保持对应输入像素逐字节不变。关闭图层的核心结果为空层；JPEG 本身有损重编码，不代表关闭导出 JPEG 与原文件逐字节一致。

地图使用局部米制投影及统一缩放，比例尺由投影到画布比例计算。透明底色无边框。文字和导航元素使用独立安全区域布局，空间不足时应减少内容。

## 开发验证

C++ 核心可单独构建和测试。Rust 测试验证 IO、配置和持久化边界，Vue 使用生产类型检查和构建。Vue 界面通过 Playwright CLI 执行 DOM、Console、Network 和程序化断言。天空羽化与主体保护的视觉验收由用户进行，开发交付不将这项视觉验收列为已通过。

最终验证记录见同目录的验收记录。真实 RAW 和照片不应加入 Git。
