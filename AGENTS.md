# Imprint — Agent 接手指南

Imprint 是一个本地优先的照片桌面工具，使用 Vue 3 + Tauri 2 构建界面，由 Python FastAPI Sidecar 提供 RAW 解码、连拍筛选、模型管理、偏好训练和去朦胧处理。照片与模型默认都在用户本机处理。

## 项目入口与关键文件

- 命令行入口：`main.py`。
- Python Sidecar：`src/app_api.py`。
- 连拍分组、评分与文件移动：`src/burst_filter.py`。
- 去朦胧算法：`src/dehaze.py`。
- RAW/普通图片读取：`src/image_io.py`。
- Linear/Demosaiced DNG 写入：`src/dng_writer.py`。
- 模型管理：`src/model_manager.py`。
- EXIF 读取：`src/exif_reader.py`。
- 桌面端入口：`tauri-frontend/src/App.vue`。
- 连拍页面：`tauri-frontend/src/views/BurstPage.vue`。
- 去朦胧页面：`tauri-frontend/src/views/EnhancePage.vue`。
- API 连接状态：`tauri-frontend/src/stores/api.ts`。
- SSE 封装：`tauri-frontend/src/composables/useSse.ts`。
- Tauri Sidecar 生命周期：`tauri-frontend/src-tauri/src/lib.rs`。
- PyInstaller 配置：`app_api.spec`。
- 跨平台构建：`.github/workflows/build.yml`。

## 架构边界

- Vue 只负责交互与展示；图像算法不要放进前端。
- Rust 负责启动、监控 Python Sidecar 和桌面能力接入；图像处理算法不要放进 Rust。
- FastAPI 负责本地 HTTP/SSE 接口、任务状态和文件访问边界。
- `BurstFilter` 只负责连拍识别、评分和审查目录移动；不要把去朦胧或其他增强算法写入其中。
- 去朦胧是独立能力，不得依赖连拍筛选结果或复用会移动原图的 decision 流程。
- 不要为了新增功能大规模重构连拍分组、人工复核、模型管理或偏好训练。

## 必须保持的产品行为

### 连拍优选

- 未保留照片只移动到审查目录，不删除。
- RAW 与同名 JPG 等伴生文件必须作为一组处理。
- 人工“保留/审查”操作会改变文件位置；修改相关逻辑时必须保留失败回滚和同名文件保护。
- 预览会话只能通过随机会话 ID 和照片 ID 访问，不得新增可读取任意本地路径的公开 GET 接口。

### 去朦胧与 DNG

- 预览使用低分辨率快速路径；最终导出必须重新读取全分辨率图像。
- 原始照片始终不修改、不移动、不覆盖。
- 默认输出目录是输入目录下的 `去朦胧输出`，目录扫描必须排除该目录。
- 去朦胧预览缓存键必须包含输入照片、算法版本、完整参数、输出尺寸和色彩模式。
- 批处理必须使用独立 job ID，并保留逐文件等待、处理、成功、失败或取消状态。
- “停止”必须通知后端；当前照片可以完成，但不能仅关闭前端连接后继续处理全部照片。
- 输出是 16-bit RGB Linear/Demosaiced DNG，不是原始 Bayer RAW。普通 JPG/PNG 输入生成的也是 RGB Linear DNG。
- 禁止把 TIFF 改名为 `.dng`，禁止在未校验 DNG 结构时声称导出成功。
- DNG 写入必须使用临时文件并在成功后原子提交；失败不得留下损坏的 `.dng`。
- 同名输出不得覆盖，使用 `_dehaze.dng`、`_dehaze_2.dng` 等安全名称。
- 修改 DNG 写入器后，至少检查 DNGVersion、PhotometricInterpretation、Compression、BitsPerSample、嵌入预览和基础 EXIF。

## 工作树与安全

- 工作树可能包含用户未提交的修改。开始前先运行 `git status --short` 并查看定向 diff。
- 用户已有修改默认属于用户；不要重置、checkout、覆盖或顺手整理无关文件。
- 禁止使用 `git reset --hard`、无明确目标的递归删除或其他破坏性命令。
- 照片、RAW、模型权重、训练样本和生成结果可能很大且不可恢复，不要将其加入提交、测试夹具或发布包。
- 不要在日志、回复、测试产物或示例配置中泄露本地照片路径之外的个人信息、密钥或令牌。
- 未经用户明确要求，不要创建分支、提交、推送、发布 Release 或修改远端状态。

## 开发与验证

本机开发优先使用项目已配置的 `py311` Conda 环境：

```bash
/Users/hualaiwu/.conda/envs/py311/bin/python
```

若该路径不存在，先查找项目实际使用的 Python 环境；不要因系统 Python 编包而直接判定项目无法运行，也不要未经允许安装或升级依赖。

按修改范围执行最小但充分的验证：

```bash
# Python 语法
/Users/hualaiwu/.conda/envs/py311/bin/python -m py_compile src/app_api.py

# Python 测试
/Users/hualaiwu/.conda/envs/py311/bin/python -m pytest -q

# 前端类型检查与生产构建
cd tauri-frontend && npm run build

# 通用 diff 检查
git diff --check
```

- 去朦胧算法测试至少覆盖尺寸、通道、dtype、数值范围、强度为零、局部对比度、NaN/Inf、16-bit 和不修改输入数组。
- DNG 测试至少覆盖真实 DNG 标签、同名不覆盖、失败无残留，并在本机有 `exiftool` 时执行 `exiftool -validate`。
- 修改 `src/app_api.py`、新增 Python 模块或依赖时，同时检查 `app_api.spec`、`requirements.txt`、`requirements-build.txt` 和 `.github/workflows/build.yml`。
- Python 测试无法执行时必须说明具体环境原因；不要删除、跳过或改弱测试来制造通过结果。
- 模型相关完整测试可能耗时数分钟；不要因为短时间没有输出就误判为挂起。

## 主代理与 `luna-worker` 分工

主代理负责理解需求、阅读相关代码、技术方案与边界决策、任务拆分、最终 diff 审查和对用户交付。不要为了委派而委派；很小、低风险、无需独立验证的改动可以由主代理直接完成。

当方案已经明确，任务边界清晰，工作主要是代码实现、局部修复、测试补充或构建验证时，优先调用 `luna-worker`。`luna-worker` 是执行型子代理，不负责替主代理决定架构、扩大需求范围或批准高风险操作。

### 标准调用方式

调用 `spawn_agent` 时使用以下约定：

```text
agent_type: "luna-worker"
fork_turns: "none"
task_name: 使用简短 snake_case 名称
message: 自包含的实现任务说明
```

不要为 `luna-worker` 额外指定模型或 reasoning effort；该角色已有固定配置。默认不要传递完整会话历史。

委派消息必须包含：

- 明确的任务目标和已经确定的实现方案；
- 该 worker 独占负责的文件或模块；
- 必须保持的现有行为和本指南中的相关约束；
- 允许修改与禁止修改的范围；
- 需要执行的测试、构建或静态检查；
- 要求汇报修改内容、验证结果和遗留风险；
- 明确说明它不是独自在代码库工作，不得回退或覆盖其他代理及用户的修改，发现并行变化时应主动适配。

推荐的自包含委派模板：

```text
你负责实现【具体目标】。

所有权范围：仅修改【文件/模块清单】。你不是独自在代码库工作；不要回退、覆盖或整理其他人和用户的修改，遇到并行变化时请基于当前文件适配。

已确定方案：【主代理给出的实现方案】。
必须保持：【现有行为与项目约束】。
禁止修改：【明确的排除范围】。
验证要求：【具体命令或验收条件】。

完成后请报告：修改文件、关键实现、实际运行的验证及结果、尚存风险。不要自行提交、推送或发布。
```

### 推荐协作流程

1. 主代理先读取相关代码，确定问题、方案、影响范围和验收条件。
2. 将一个边界清晰的实现任务交给 `luna-worker`；不同 worker 必须拥有互不冲突的文件范围。
3. `luna-worker` 在自己的范围内实现并执行直接相关的验证。
4. worker 返回后，主代理检查实际文件和 diff，不仅依赖文字汇报。
5. 主代理补做跨模块验证，确认没有破坏连拍、去朦胧、DNG、打包或用户未提交修改。
6. 若只是同一任务的边界明确修复，可继续向原 `luna-worker` 发送 follow-up；新的独立任务使用新的 worker。

### 不应委派给 `luna-worker` 的事项

- 架构设计、技术路线选择或需求仍有重大歧义的工作。
- 最终代码审查、是否接受实现以及是否发布的决定。
- 删除、移动或批量修改用户照片、RAW、训练集、模型权重或生成结果。
- 可能覆盖用户未提交修改的操作。
- Git 提交、推送、创建 PR、发布安装包或 Release，除非用户明确要求且主代理已限定范围。
- 安装系统依赖、修改开发机全局环境或执行需要额外权限的操作。
- 任何需要主代理重新取得用户授权的高风险或范围扩张事项。

### 子代理上下文策略

- 默认使用 `fork_turns: "none"`，由主代理在消息中提供完整且最小必要的上下文。
- 只有任务确实依赖最近交互、重新描述会明显损失关键信息时，才使用有限的最近 1–3 个 turns。
- 除非有明确且必要的理由，不要使用 `fork_turns: "all"`。
- 子代理应自行阅读任务所需的项目文件，但不得自行扩大文件所有权或任务范围。

## 对外沟通

- 默认使用中文，先说结果，再说关键验证和限制。
- 不把“代码已写”“测试通过”“DNG 标签有效”和“已在 Lightroom/Camera Raw 实际导入”混为一谈；每项只报告真实完成的验证。
- 若有未完成事项或兼容性限制，应明确列出，不要用模糊措辞掩盖。
