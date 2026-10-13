# REA / Hopper 启动故障只读诊断

**当前已确认的阻塞：** 主代理随后观察到 Hopper 弹出 critical alert：`Demo period has expired. Document will be closed.`，窗口只有 `OK` 按钮；弹窗尚未点击，进程和 session 均未清理。当前无法安全完成“等待分析结束并保存数据库”，应先恢复 Hopper 的有效授权，再继续分析和保存；不应尝试绕过试用限制。

前序 REA 故障仍表现为 bridge readiness 超时：保存下来的 session 运行目录没有 `bridge.sock`，而 Hopper 的 `-Y` 参数会等到初始分析结束后才执行 bootstrap。这支持“初始分析期间 bridge 尚未启动”这一机制，但不能倒推授权弹窗就是先前 120 秒超时的原因：主代理观察到 Hopper 初始分析推进到 pass 10，无法确认弹窗出现时间与启动超时的对应关系。

原始调用顺序由主代理复核：`open_binary` 在约 6.1 秒后返回成功，目标为 Camera Raw 主 Mach-O、架构 arm64；随后 `binary_overview` 等待约 120 秒，返回 `provider_timeout`，阶段为 `startup`，timeout 为 120000 ms，launcher exit code 为 0。该次错误的 cleanup 信息为 `cleanup_incomplete`，operation 为 `open_binary`。之后对同一目标的 CLI `xrefs` 请求被拒绝，错误指出目标由 REA session `61f19de4-83ac-494d-a281-f11e0d38dcb1` 持有，应复用该 session 或先由它关闭目标。

现有 MCP 检查记录也绑定到同一个 run ID 和 arm64 Hopper profile；四项后续能力（procedures、assembly、pseudocode、xrefs）都返回 `unreadable_output`。连接检查里的 `open: true` 和 binding 字段记录了目标及 provider/profile，但它们本身不等于 bridge 健康握手成功。

## 工具结果与源码路径的差异

当前安装的 REA 6.4.0 源码对冷打开的处理是：`BinarySession.open` 创建 provider client 后执行 `health`，只有该调用成功才把新 client 保存为 active；Hopper provider 会把这个调用交给 `HopperClient.callTool`，而后者先调用 `start()`，以 Unix socket 和 authenticated health handshake 等待 bridge。因此，冷打开通常应在 `open_binary` 阶段完成启动。

同一源码也有一个例外：若当前 session 已有同一 target 和同一 profile，`BinarySession` 会直接返回 target，不创建新 client，也不重复 health 调用。原始 `open_binary` 成功与之后 `binary_overview` 才等待启动，在这个 fast path 下可以相容；但目前没有该调用之前的 session history，无法确认当时是否命中了 fast path。报告因此保留原始观测时序，并将启动发生在哪次调用中的具体映射标为未确定。

## Bridge、Python 接口与等待语义

REA 的 Hopper launcher 在 macOS 上传入 `--analysis -Y <bootstrap.py> --executable <target>`。本机 Hopper launcher 的只读 `strings` 检查显示：`-y | --python-command CMD` 会在初始分析结束后执行 Python 命令；`-Y | --python script.py` 会在初始分析结束后执行 Python 脚本。REA 正是通过 `-Y` 注入自己的 bridge bootstrap。若 Hopper 的初始分析持续超过 REA 的 120 秒启动期限，bridge socket 可能尚未创建，REA 就会超时。

本次连接证据的 `python_mcp_tool` 数组为空；doctor 保存的 170 项 MCP tool catalog 中，没有名称或 surface 含 `python` 的工具。REA 自带的 Python bridge 不是任意 Python MCP：它运行在 Hopper 的 Python 线程中，按一个固定方法集合分派请求，并通过随机 token 验证连接；检查中没有执行任意 Python 源码的 MCP 操作。不要把 Hopper 的 `-y/-Y` launcher 功能理解成当前 MCP 暴露了 Python console。

在超时后保留的 `/tmp/rea-Nnwiq8` 目录目前只有 `bootstrap.py` 和 `ownership.json`，没有 `bridge.sock`。bootstrap 的 run ID 与故障 session 一致，并标记 Hopper 进程生命周期归外部应用所有。没有 socket 表示此次检查时 bridge listener 不存在；它与 bootstrap 尚未执行相符，也可能由脚本在建 socket 前失败造成，单凭这一点不能证明 Hopper 的初始分析状态。

## 进程与目标租约

只读 `ps` 快照中，Hopper PID 72991 和 External API PID 73008 的启动时间分别为 12:29:34 和 12:29:36，早于两次 `npm exec rea-agents@6.4.0 mcp`（12:31:49、12:38:09）。这支持 Hopper GUI 是已有进程的解释。REA 源码说明 macOS LaunchServices 创建或复用的 Hopper GUI 进程不归 REA 所有；本次仅做只读进程检查，未关闭或重启。

`/tmp/rea-hopper-501` 中当前可见一个目标租约 socket，`lsof` 显示该 listener 由 PID 74678 持有；该 Node 进程是第二个 npm MCP 包装进程 PID 74443 的子进程。保留的 `ownership.json` 也将故障 run `61f19de4-83ac-494d-a281-f11e0d38dcb1` 关联到 parent PID 74678。第一次 npm MCP 的 PID 73521/子进程 73741 没有可指向该目标租约的路径或 peer 证据，因此不能断言它也绑定了这个 Camera Raw target。CLI xrefs 的 owner-run 错误则明确证明目标租约归当前故障 run 持有。

启动记录里的 launcher PID 75091 在后续 `ps` 快照中已不存在，但这不代表 Hopper document 已关闭或 cleanup 已确认。REA cleanup 源码要求拿到 Hopper shutdown acknowledgement 才释放 macOS 外部应用场景下的 lease；若没有确认，会保留 private runtime 并记录 `hopper-document` 清理问题。本次 runtime 目录和目标租约仍在，故不要手动删除目录/socket，也不要通过新 session 重开同一 target。

## 诊断依据与限制

REA doctor 报告 Hopper executable 检查 healthy；`environment_healthy: false` 来自缺少 Ghidra/IDA 配置及其他 client 注册项 drift，不能解释本次 Hopper bridge readiness timeout。doctor 的 Hopper 可用性检查只验证环境路径，不能证明当前 Hopper 文档已分析完成、脚本已运行或 bridge 已连通。

主要源码位置：

- [BinarySession.js](/Users/hualaiwu/.npm/_npx/dea9475797100b20/node_modules/rea-agents/dist/application/binary/BinarySession.js:150)：同目标/同 profile fast path；新 client 的 health 检查见第 170 行。
- [HopperProvider.js](/Users/hualaiwu/.npm/_npx/dea9475797100b20/node_modules/rea-agents/dist/hopper/HopperProvider.js:181)：provider operation 到 HopperClient 的路由。
- [HopperClient.js](/Users/hualaiwu/.npm/_npx/dea9475797100b20/node_modules/rea-agents/dist/hopper/HopperClient.js:95)：默认 120000 ms；socket polling、health handshake 与超时见第 214、282、393 行；所有工具调用先启动见第 323 行。
- [BridgeLauncher.js](/Users/hualaiwu/.npm/_npx/dea9475797100b20/node_modules/rea-agents/dist/hopper/BridgeLauncher.js:71)：launcher 参数包含 `-Y`；macOS target lease 获取见第 131 行。
- [HopperCleanup.js](/Users/hualaiwu/.npm/_npx/dea9475797100b20/node_modules/rea-agents/dist/hopper/HopperCleanup.js:18)：关闭、保留未确认 runtime 和释放 lease 的条件见第 18、41、57、178 行。
- [hopper_bridge.py](/Users/hualaiwu/.npm/_npx/dea9475797100b20/node_modules/rea-agents/bridge/hopper_bridge.py:730)：bridge 的固定分派及 shutdown 行为；listener 在第 1118 行启动。
- Hopper CLI 参数说明来自本机 `/Applications/Hopper Disassembler.app/Contents/MacOS/hopper` 的只读 `strings -a` 检查。主代理另运行了不带目标的 `hopper --help`；launcher 返回 `unrecognized option` 和 usage 文本，没有传入目标或 Python 脚本，也没有执行分析脚本。

本子任务未等待或操作 Hopper 文档，未点击 license alert，未清理 retained runtime/lease，未改动配置，未安装或启动分析会话。授权弹窗和 pass 10 进度由主代理直接观察；本子任务没有读取或控制 Hopper UI。截至本报告完成时，数据库没有保存，没有重连，也没有终止进程或清理 retained runtime/lease。

## 建议的安全后续顺序

先恢复 Hopper 的有效授权，并确认授权弹窗不再要求关闭当前文档；在获得可保存的 Hopper 环境前，不要点击当前 alert 的 `OK`。之后再等待初始分析明确结束，用 Hopper 自身的 Save / Save As 保存数据库并确认保存路径。分析仍在运行时不要另开第二个 REA session。保存后先在持有该 lease 的现有 REA session 中重试一次只读查询；若要改连保存的数据库，先通过 owner session 的正常关闭流程取得 cleanup/shutdown 确认并观察 lease 释放，再用数据库文件路径打开。若原 session 无法完成有确认的关闭，停止在该处并交由用户决定下一步，不要手工删 socket 或 retained runtime。
