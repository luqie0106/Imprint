import SwiftUI

private struct ManagedModel: Identifiable {
    let mode: String
    let title: String
    let shortTitle: String
    let family: String
    let description: String
    let readyKey: String

    var id: String { mode }
}

private struct FusionTarget: Identifiable {
    let type: String
    let title: String
    let pathKey: String
    let readyKey: String
    let deployedKey: String

    var id: String { type }
}

struct ModelsView: View {
    @EnvironmentObject private var engine: Engine
    @State private var modelStatus: [String: Any] = [:]
    @State private var isRefreshing = false
    @State private var busyOperation: String?
    @State private var useMirror = true
    @State private var progress: Double?
    @State private var messages: [String] = []
    @State private var errorMessage: String?
    @State private var notice: String?

    private let models = [
        ManagedModel(mode: "standard", title: "官方通用模型", shortTitle: "标准", family: "ViT-B/32", description: "速度与画质均衡，适合日常连拍筛选。", readyKey: "standard_onnx_ready"),
        ManagedModel(mode: "standard_l14", title: "Aesthetic 3 专业模型", shortTitle: "专业", family: "ViT-L/14", description: "细节感知更强，适合高质量摄影工作流。", readyKey: "standard_l14_onnx_ready"),
        ManagedModel(mode: "custom", title: "个人偏好模型", shortTitle: "个人", family: "ViT-B/32", description: "根据喜欢与不喜欢的样片学习审美。", readyKey: "custom_onnx_ready"),
        ManagedModel(mode: "custom_l14", title: "个人偏好专业模型", shortTitle: "个人专业", family: "ViT-L/14", description: "使用更高精度的专业底座学习个人偏好。", readyKey: "custom_l14_onnx_ready"),
    ]

    private var activeMode: String { string(modelStatus, "mode").isEmpty ? "standard" : string(modelStatus, "mode") }
    private var activeModel: ManagedModel { models.first(where: { $0.mode == activeMode }) ?? models[0] }
    private func ready(_ key: String) -> Bool { modelStatus[key] as? Bool ?? false }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                HStack(alignment: .top) {
                    SectionTitle(title: "模型管理", subtitle: "选择连拍筛选使用的审美模型，并管理本地模型资源")
                    Spacer()
                    Button(isRefreshing ? "正在刷新…" : "刷新状态", systemImage: "arrow.clockwise") {
                        Task { await refreshStatus() }
                    }
                    .disabled(isRefreshing || !engine.ready)
                }

                Panel(title: "筛选模型") {
                    LazyVGrid(columns: [GridItem(.adaptive(minimum: 245), alignment: .leading)], spacing: 12) {
                        ForEach(models) { model in
                            Button {
                                Task { await setMode(model.mode) }
                            } label: {
                                VStack(alignment: .leading, spacing: 10) {
                                    HStack {
                                        Text(model.shortTitle)
                                            .font(.caption.bold())
                                            .padding(.horizontal, 8).padding(.vertical, 5)
                                            .background(activeMode == model.mode ? Color.accentColor.opacity(0.14) : Color.secondary.opacity(0.12), in: RoundedRectangle(cornerRadius: 6))
                                        Spacer()
                                        Label(ready(model.readyKey) ? "可用" : "未就绪", systemImage: ready(model.readyKey) ? "checkmark.circle.fill" : "exclamationmark.circle")
                                            .font(.caption2)
                                            .foregroundStyle(ready(model.readyKey) ? .green : .orange)
                                    }
                                    HStack(alignment: .firstTextBaseline, spacing: 8) {
                                        Text(model.title).font(.subheadline.weight(.semibold))
                                        Text(model.family).font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)
                                    }
                                    Text(model.description).font(.caption).foregroundStyle(.secondary).multilineTextAlignment(.leading)
                                }
                                .padding(13)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .background(activeMode == model.mode ? Color.accentColor.opacity(0.08) : Color.secondary.opacity(0.035), in: RoundedRectangle(cornerRadius: 11))
                                .overlay(RoundedRectangle(cornerRadius: 11).stroke(activeMode == model.mode ? Color.accentColor : Color.secondary.opacity(0.18), lineWidth: activeMode == model.mode ? 1.5 : 1))
                            }
                            .buttonStyle(.plain)
                            .disabled(busyOperation != nil || !engine.ready)
                        }
                    }
                }

                Panel(title: "当前启用") {
                    HStack(spacing: 14) {
                        Image(systemName: "gauge.with.dots.needle.67percent").font(.title2).foregroundStyle(Color.accentColor)
                        VStack(alignment: .leading, spacing: 3) {
                            Text(activeModel.title).font(.headline)
                            Text("\(activeModel.family) · 连拍筛选审美评分").font(.caption).foregroundStyle(.secondary)
                        }
                        Spacer()
                        Label("仅保存在本机", systemImage: "lock.shield").font(.caption).foregroundStyle(.secondary)
                    }
                    if let notice { Text(notice).font(.caption).foregroundStyle(.secondary) }
                }

                Panel(title: "部署个人模型") {
                    Text("将偏好训练权重转换为 ONNX 推理模型。训练完成后，可在上方切换使用。")
                        .font(.caption).foregroundStyle(.secondary)
                    let targets = [
                        FusionTarget(type: "b32", title: "个人模型 · ViT-B/32", pathKey: "mlp_path", readyKey: "mlp_ready", deployedKey: "custom_onnx_ready"),
                        FusionTarget(type: "l14", title: "个人模型 · ViT-L/14", pathKey: "mlp_l14_path", readyKey: "mlp_l14_ready", deployedKey: "custom_l14_onnx_ready"),
                    ]
                    ForEach(targets) { target in
                        HStack(spacing: 12) {
                            Image(systemName: "shippingbox").foregroundStyle(.secondary)
                            VStack(alignment: .leading, spacing: 3) {
                                Text(target.title).font(.subheadline.weight(.medium))
                                Text(string(modelStatus, target.pathKey).isEmpty ? (ready(target.readyKey) ? "个人模型已部署" : "尚无训练权重") : string(modelStatus, target.pathKey))
                                    .font(.caption2).foregroundStyle(.secondary).lineLimit(2).textSelection(.enabled)
                            }
                            Spacer()
                            Button(ready(target.deployedKey) ? "已部署" : "熔铸", systemImage: "flame") {
                                Task { await runSSE(path: "/api/models/fuse-onnx", body: ["model_type": target.type], operation: "fuse_\(target.type)") }
                            }
                            .disabled(busyOperation != nil || !ready(target.readyKey) || !engine.ready || ready(target.deployedKey))
                        }
                        .padding(10)
                        .background(Color.secondary.opacity(0.045), in: RoundedRectangle(cornerRadius: 9))
                    }
                }

                Panel(title: "模型资源") {
                    Toggle("使用 HuggingFace 国内镜像", isOn: $useMirror)
                    Text("模型和照片处理均保留在本机。首次下载需要稳定的网络连接。")
                        .font(.caption).foregroundStyle(.secondary)
                    downloadRow(title: "CLIP ViT-B/32", detail: "标准底座 · 约 335 MB", model: "clip_b32", readyKey: "clip_b32_ready")
                    downloadRow(title: "CLIP ViT-L/14", detail: "专业底座 · 约 900 MB", model: "clip_l14", readyKey: "clip_l14_ready")
                    downloadRow(title: "人像与闭眼检测", detail: "MediaPipe Face Landmarker · 可选", model: "face_landmarker", readyKey: "face_landmarker_ready")
                }

                if busyOperation != nil || !messages.isEmpty || errorMessage != nil {
                    Panel(title: "任务记录") {
                        if let progress { ProgressView(value: progress).tint(.accentColor) }
                        if let errorMessage { Text(errorMessage).foregroundStyle(.red).font(.callout) }
                        ForEach(messages.indices, id: \.self) { index in
                            Text(messages[index]).font(.system(.caption, design: .monospaced)).textSelection(.enabled)
                        }
                    }
                }
            }
            .padding(24)
            .frame(maxWidth: 1000, alignment: .leading)
            .frame(maxWidth: .infinity, alignment: .center)
        }
        .task(id: engine.ready) { await refreshStatus() }
    }

    @ViewBuilder private func downloadRow(title: String, detail: String, model: String, readyKey: String) -> some View {
        HStack(spacing: 12) {
            Image(systemName: "circle.fill")
                .font(.system(size: 9))
                .foregroundStyle(ready(readyKey) ? .green : .secondary)
            VStack(alignment: .leading, spacing: 3) {
                Text(title).font(.subheadline.weight(.medium))
                Text(detail).font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
            Button(ready(readyKey) ? "重新校验" : "下载", systemImage: "arrow.down.circle") {
                Task { await runSSE(path: "/api/models/download", body: ["model": model, "use_mirror": useMirror], operation: "download_\(model)") }
            }
            .disabled(busyOperation != nil || !engine.ready)
        }
        .padding(.vertical, 4)
    }

    @MainActor private func refreshStatus() async {
        guard engine.ready else { return }
        isRefreshing = true
        defer { isRefreshing = false }
        do {
            modelStatus = try await engine.api.json("/api/models/status")
            errorMessage = nil
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    @MainActor private func setMode(_ mode: String) async {
        guard busyOperation == nil else { return }
        busyOperation = "set_mode"
        defer { busyOperation = nil }
        do {
            _ = try await engine.api.json("/api/models/set-mode", body: ["mode": mode])
            modelStatus["mode"] = mode
            notice = "已启用\(activeModel.title)"
            errorMessage = nil
        } catch { errorMessage = error.localizedDescription }
    }

    @MainActor private func runSSE(path: String, body: [String: Any], operation: String) async {
        busyOperation = operation
        messages = []
        progress = nil
        errorMessage = nil
        notice = nil
        defer { busyOperation = nil }
        do {
            try await engine.api.events(path, body: body) { event in
                let message = event["msg"] as? String ?? ""
                if !message.isEmpty { messages.append(message) }
                if let pct = event["pct"] as? Double { progress = min(max(pct, 0), 1) }
                if event["type"] as? String == "error" { errorMessage = message.isEmpty ? "任务失败" : message }
                if event["type"] as? String == "done" { progress = 1 }
            }
            await refreshStatus()
            if errorMessage == nil { notice = "任务已完成" }
        } catch { errorMessage = error.localizedDescription }
    }
}

struct TrainerView: View {
    @EnvironmentObject private var engine: Engine
    @State private var photosDirectory = ""
    @State private var modelType = "standard"
    @State private var epochs = 15.0
    @State private var learningRate = 0.001
    @State private var isRunning = false
    @State private var isDone = false
    @State private var progress: Double?
    @State private var latestMessage = ""
    @State private var messages: [String] = []
    @State private var errorMessage: String?

    private var folderName: String {
        photosDirectory.split(separator: "/").last.map(String.init) ?? "尚未选择样本目录"
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                SectionTitle(title: "偏好训练", subtitle: "用喜欢与不喜欢的样片训练符合个人审美的本地模型")

                Panel(title: "训练样本") {
                    HStack(spacing: 14) {
                        Image(systemName: "folder.badge.gearshape").font(.title2).foregroundStyle(Color.accentColor)
                        VStack(alignment: .leading, spacing: 4) {
                            Text(photosDirectory.isEmpty ? "选择样本目录" : folderName).font(.headline)
                            Text(photosDirectory.isEmpty ? "目录内需要包含 like/ 和 dislike/ 子文件夹" : photosDirectory)
                                .font(.caption).foregroundStyle(.secondary).lineLimit(2).textSelection(.enabled)
                        }
                        Spacer()
                        Button("浏览…", systemImage: "folder") {
                            if let path = FilePicker.folder(title: "选择包含 like/ 与 dislike/ 的样本目录") { photosDirectory = path }
                        }
                        .disabled(isRunning)
                    }
                    HStack(spacing: 14) {
                        Label("like", systemImage: "heart.fill").foregroundStyle(.pink)
                        Text("喜欢的照片")
                        Spacer(minLength: 20)
                        Label("dislike", systemImage: "hand.thumbsdown.fill").foregroundStyle(.secondary)
                        Text("不喜欢的照片")
                    }
                    .font(.caption)
                    .padding(11)
                    .background(Color.secondary.opacity(0.05), in: RoundedRectangle(cornerRadius: 9))
                    Text("训练在本机进行，样片不会上传，也不会移动或删除。")
                        .font(.caption).foregroundStyle(.secondary)
                }

                Panel(title: "训练流程") {
                    HStack(alignment: .top, spacing: 0) {
                        stage(number: 1, title: "扫描样本", symbol: "photo.stack", active: isRunning && latestMessage.contains("扫描"))
                        stage(number: 2, title: "提取特征", symbol: "viewfinder", active: isRunning && (latestMessage.contains("特征") || latestMessage.contains("底座")))
                        stage(number: 3, title: "偏好训练", symbol: "brain", active: isRunning && (latestMessage.contains("Epoch") || latestMessage.contains("训练")))
                        stage(number: 4, title: "生成模型", symbol: "shippingbox", active: isRunning && (latestMessage.contains("ONNX") || latestMessage.contains("熔铸")))
                    }
                    HStack(spacing: 10) {
                        Circle().fill(errorMessage != nil ? Color.red : (isDone ? Color.green : (isRunning ? Color.accentColor : Color.secondary)))
                            .frame(width: 8, height: 8)
                        Text(errorMessage ?? (isDone ? "个人偏好模型已生成" : (isRunning ? (latestMessage.isEmpty ? "正在启动训练…" : latestMessage) : "选择样本目录并配置训练参数")))
                            .font(.callout).lineLimit(3)
                        Spacer()
                        if let progress { Text("\(Int(progress * 100))%").font(.caption.monospacedDigit()).foregroundStyle(.secondary) }
                    }
                    if let progress { ProgressView(value: progress).tint(.accentColor) }
                }

                if isDone {
                    Label("个人偏好模型训练完成，可在模型管理中切换使用。", systemImage: "checkmark.seal.fill")
                        .font(.callout).foregroundStyle(.green).padding(14)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(Color.green.opacity(0.08), in: RoundedRectangle(cornerRadius: 12))
                }

                if let errorMessage {
                    Label(errorMessage, systemImage: "exclamationmark.triangle.fill")
                        .foregroundStyle(.red).font(.callout).textSelection(.enabled)
                }

                Panel(title: "训练记录") {
                    HStack {
                        Text(messages.isEmpty ? "暂无记录" : "\(messages.count) 条消息").font(.caption).foregroundStyle(.secondary)
                        Spacer()
                        Button("清空记录", systemImage: "trash") { messages.removeAll() }
                            .font(.caption).disabled(messages.isEmpty || isRunning)
                    }
                    if messages.isEmpty {
                        Text("训练阶段和进度会显示在这里。")
                            .font(.caption).foregroundStyle(.tertiary)
                    } else {
                        ScrollView {
                            VStack(alignment: .leading, spacing: 5) {
                                ForEach(messages.indices, id: \.self) { index in
                                    Text(messages[index]).font(.system(.caption2, design: .monospaced)).textSelection(.enabled)
                                        .frame(maxWidth: .infinity, alignment: .leading)
                                }
                            }
                            .padding(10)
                        }
                        .frame(maxHeight: 190)
                        .background(Color.black.opacity(0.045), in: RoundedRectangle(cornerRadius: 8))
                    }
                }

                Panel(title: "训练设置") {
                    Picker("视觉底座", selection: $modelType) {
                        Text("标准 · ViT-B/32").tag("standard")
                        Text("专业 · ViT-L/14").tag("l14")
                    }
                    .pickerStyle(.segmented)
                    .disabled(isRunning)

                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Text("训练轮数")
                            Spacer()
                            Text("\(Int(epochs)) 轮").fontWeight(.semibold).foregroundStyle(Color.accentColor)
                        }
                        Slider(value: $epochs, in: 1...100, step: 1).disabled(isRunning)
                        HStack { Text("1"); Spacer(); Text("15"); Spacer(); Text("60"); Spacer(); Text("100") }
                            .font(.caption2).foregroundStyle(.tertiary)
                    }

                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Text("学习率")
                            Spacer()
                            Text(learningRate.formatted(.number.precision(.significantDigits(1...3))))
                                .font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)
                        }
                        Slider(value: $learningRate, in: 0.0001...0.01, step: 0.0001).disabled(isRunning)
                    }

                    Button {
                        Task { await startTraining() }
                    } label: {
                        Label(isRunning ? "训练进行中…" : "开始训练", systemImage: isRunning ? "hourglass" : "play.fill")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(isRunning || photosDirectory.isEmpty || !engine.ready)
                }
            }
            .padding(24)
            .frame(maxWidth: 900, alignment: .leading)
            .frame(maxWidth: .infinity, alignment: .center)
        }
    }

    @ViewBuilder private func stage(number: Int, title: String, symbol: String, active: Bool) -> some View {
        VStack(spacing: 7) {
            Image(systemName: symbol).font(.title3)
                .foregroundStyle(active ? Color.accentColor : Color.secondary)
            Text(String(format: "%02d · %@", number, title)).font(.caption.weight(active ? .semibold : .regular))
                .foregroundStyle(active ? .primary : .secondary).multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 8)
    }

    @MainActor private func startTraining() async {
        guard !photosDirectory.isEmpty else { return }
        isRunning = true
        isDone = false
        progress = nil
        latestMessage = ""
        messages = []
        errorMessage = nil
        defer { isRunning = false }
        do {
            try await engine.api.events("/api/trainer/run", body: [
                "photos_dir": photosDirectory,
                "model_type": modelType,
                "epochs": Int(epochs),
                "lr": learningRate,
            ]) { event in
                let message = event["msg"] as? String ?? ""
                if !message.isEmpty {
                    latestMessage = message
                    messages.append(message)
                }
                if let pct = event["pct"] as? Double { progress = min(max(pct, 0), 1) }
                switch event["type"] as? String {
                case "done":
                    isDone = true
                    progress = 1
                case "error":
                    errorMessage = message.isEmpty ? "训练失败" : message
                default: break
                }
            }
        } catch { errorMessage = error.localizedDescription }
    }
}

private struct BackendChoice: Identifiable {
    let id: String
    let title: String
    let detail: String
}

struct SettingsView: View {
    @EnvironmentObject private var engine: Engine
    @State private var renderStatus: [String: Any] = [:]
    @State private var isRefreshing = false
    @State private var errorMessage: String?

    private let renderChoices = [
        BackendChoice(id: "auto", title: "自动", detail: "优先原生渲染，其次 PyTorch，最后使用 Python CPU。"),
        BackendChoice(id: "native", title: "原生渲染", detail: "使用平台原生 GPU 渲染；不可用时由后端回退 Python。"),
        BackendChoice(id: "pytorch", title: "PyTorch", detail: "在可用时使用 PyTorch 加速去朦胧。"),
        BackendChoice(id: "cpu", title: "Python CPU", detail: "使用现有 Python CPU 计算。"),
    ]

    private var nativeStatus: [String: Any] { renderStatus["native"] as? [String: Any] ?? [:] }
    private var pytorchStatus: [String: Any] { renderStatus["pytorch"] as? [String: Any] ?? [:] }
    private var sortStatus: [String: Any] { renderStatus["sort"] as? [String: Any] ?? [:] }
    private var nativeLabel: String {
        let backend = string(nativeStatus, "backend")
        return (nativeStatus["available"] as? Bool ?? false) ? "原生渲染 · \(backend.isEmpty ? "GPU" : backend)" : "原生渲染 · 不可用"
    }
    private var pytorchLabel: String {
        let backends = (pytorchStatus["backends"] as? [String] ?? []).joined(separator: " / ")
        return (pytorchStatus["available"] as? Bool ?? false) ? "PyTorch · \(backends.isEmpty ? "GPU" : backends)" : "PyTorch · 不可用"
    }
    private var sortLabel: String {
        let backend = string(sortStatus, "backend")
        return (sortStatus["available"] as? Bool ?? false) ? "连拍原生算子 · \(backend.isEmpty ? "可用" : backend)" : "连拍原生算子 · 不可用"
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                HStack {
                    SectionTitle(title: "设置", subtitle: "分别选择各功能的计算方式")
                    Spacer()
                    Button(isRefreshing ? "检测中…" : "重新检测", systemImage: "arrow.clockwise") {
                        Task { await refreshStatus() }
                    }
                    .disabled(isRefreshing || !engine.ready)
                }

                Panel(title: "去朦胧") {
                    ForEach(renderChoices) { option in
                        VStack(alignment: .leading, spacing: 3) {
                            HStack {
                                Text(option.title).font(.subheadline.weight(.medium))
                                Spacer()
                                if engine.renderBackend == option.id { Image(systemName: "checkmark.circle.fill").foregroundStyle(Color.accentColor) }
                            }
                            Text(option.detail).font(.caption).foregroundStyle(.secondary)
                        }
                        .padding(11)
                        .contentShape(Rectangle())
                        .background(engine.renderBackend == option.id ? Color.accentColor.opacity(0.07) : Color.secondary.opacity(0.045), in: RoundedRectangle(cornerRadius: 9))
                        .onTapGesture { engine.renderBackend = option.id }
                    }
                }

                backendPanel(title: "基础调整", selection: $engine.basicBackend,
                             detail: "选择基础曝光、色彩与细节调整使用的计算实现。")
                backendPanel(title: "理光滤镜", selection: $engine.ricohBackend,
                             detail: "原生模式从当前预设生成查找表，颜色可能与 Python 渲染略有差异。")
                backendPanel(title: "连拍评分", selection: $engine.sortBackend,
                             detail: "原生模式计算区域锐度与曝光；美学模型、分组和文件移动仍由现有流程处理。")

                Panel(title: "可用状态") {
                    statusLine(nativeLabel, available: nativeStatus["available"] as? Bool ?? false)
                    statusLine(pytorchLabel, available: pytorchStatus["available"] as? Bool ?? false)
                    statusLine(sortLabel, available: sortStatus["available"] as? Bool ?? false)
                    if let errorMessage { Text(errorMessage).font(.caption).foregroundStyle(.red) }
                }
            }
            .padding(24)
            .frame(maxWidth: 700, alignment: .leading)
            .frame(maxWidth: .infinity, alignment: .center)
        }
        .task(id: engine.ready) { await refreshStatus() }
    }

    @ViewBuilder private func backendPanel(title: String, selection: Binding<String>, detail: String) -> some View {
        Panel(title: title) {
            Picker(title, selection: selection) {
                Text("Python").tag("python")
                Text("原生").tag("native")
            }
            .pickerStyle(.segmented)
            Text(detail).font(.caption).foregroundStyle(.secondary)
        }
    }

    @ViewBuilder private func statusLine(_ label: String, available: Bool) -> some View {
        Label(label, systemImage: available ? "checkmark.circle.fill" : "minus.circle")
            .font(.callout)
            .foregroundStyle(available ? Color.green : Color.secondary)
    }

    @MainActor private func refreshStatus() async {
        guard engine.ready else { return }
        isRefreshing = true
        defer { isRefreshing = false }
        do {
            renderStatus = try await engine.api.json("/api/enhance/render-status")
            errorMessage = nil
        } catch {
            renderStatus = [:]
            errorMessage = error.localizedDescription
        }
    }
}
