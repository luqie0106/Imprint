import AppKit
import SwiftUI

private struct BurstShot: Identifiable {
    let photoID: String
    let name: String
    var kept: Bool
    var category: String
    let rank: Int?
    let companionCount: Int
    let sharpness: Double
    let aesthetic: Double
    let exposure: Double
    let sharpest: Bool
    let aestheticBest: Bool
    let rejectReasons: [String]

    var id: String { photoID }

    init?(_ object: [String: Any]) {
        guard let photoID = object["photo_id"] as? String else { return nil }
        self.photoID = photoID
        name = string(object, "name")
        kept = object["kept"] as? Bool ?? false
        category = string(object, "category")
        rank = object["rank"] as? Int
        companionCount = int(object, "companion_count")
        sharpness = object["sharpness"] as? Double ?? 0
        aesthetic = object["aesthetic"] as? Double ?? 0
        exposure = object["exposure"] as? Double ?? 0
        sharpest = object["sharpest"] as? Bool ?? false
        aestheticBest = object["aesthetic_best"] as? Bool ?? false
        rejectReasons = object["reject_reasons"] as? [String] ?? []
    }
}

private struct BurstGroup: Identifiable {
    let index: Int
    let needsReview: Bool
    let confidenceMargin: Double
    let weightReason: String
    let weights: [String: Double]
    var shots: [BurstShot]

    var id: Int { index }

    init?(_ object: [String: Any]) {
        index = int(object, "index")
        needsReview = object["needs_review"] as? Bool ?? false
        confidenceMargin = object["confidence_margin"] as? Double ?? 0
        weightReason = string(object, "weight_reason")
        weights = object["weights"] as? [String: Double] ?? [:]
        shots = (object["shots"] as? [[String: Any]] ?? []).compactMap(BurstShot.init)
    }
}

private struct BurstResult {
    let total: Int
    let groupCount: Int
    var moved: Int
    let defectMoved: Int
    let skippedSingle: Int
    let reviewDirectory: String
    let defectDirectory: String
    let previewSession: String
    var groups: [BurstGroup]

    init(_ event: [String: Any]) {
        total = int(event, "total")
        groupCount = int(event, "burst_groups")
        moved = int(event, "moved")
        defectMoved = int(event, "defect_moved")
        skippedSingle = int(event, "skipped_single")
        reviewDirectory = string(event, "review_dir")
        defectDirectory = string(event, "defect_dir")
        previewSession = string(event, "preview_session")
        groups = (event["groups"] as? [[String: Any]] ?? []).compactMap(BurstGroup.init)
    }
}

struct BurstView: View {
    private enum ParameterField: Hashable {
        case gapSeconds
        case maxHammingDistance
        case keepCount
        case maxWorkers
    }

    @EnvironmentObject private var engine: Engine
    @State private var inputDirectory = ""
    @State private var gapSeconds = 1.5
    @State private var gapSecondsText = "1.5"
    @State private var maxHammingDistance = 12
    @State private var maxHammingDistanceText = "12"
    @State private var keepCount = 1
    @State private var keepCountText = "1"
    @State private var maxWorkers = max(1, Int((Double(ProcessInfo.processInfo.activeProcessorCount) * 0.8).rounded()))
    @State private var maxWorkersText = String(max(1, Int((Double(ProcessInfo.processInfo.activeProcessorCount) * 0.8).rounded())))
    @FocusState private var focusedParameter: ParameterField?
    @State private var useGPU = false
    @State private var gpuAvailable = false
    @State private var reviewSubdirectory = "审查_连拍淘汰"
    @State private var defectSubdirectory = "审查_明显废片"
    @State private var allBlurryAction = "keep"
    @State private var weightMode = "adaptive"
    @State private var customWeights = ["sharpness": 0.9, "aesthetic": 0.05, "exposure": 0.05]
    @State private var includePreviews = true
    @State private var eyeDetection = false
    @State private var eyeModelReady = false

    @State private var isRunning = false
    @State private var progressMessage = "选择照片目录后开始筛选"
    @State private var progressFraction: Double?
    @State private var errorMessage = ""
    @State private var messages: [String] = []
    @State private var result: BurstResult?
    @State private var selectedGroupIndex: Int?
    @State private var selectedPhotoID: String?
    @State private var previewImage: NSImage?
    @State private var detailImage: NSImage?
    @State private var previewMessage = ""
    @State private var decisionPendingID: String?
    @State private var decisionFeedback = ""
    @State private var task: Task<Void, Never>?

    private var cpuCount: Int { max(1, ProcessInfo.processInfo.activeProcessorCount) }
    private var selectedGroup: BurstGroup? {
        result?.groups.first { $0.index == selectedGroupIndex }
    }
    private var selectedShot: BurstShot? {
        guard let selectedGroup else { return nil }
        return selectedGroup.shots.first { $0.photoID == selectedPhotoID }
            ?? selectedGroup.shots.first(where: \.kept)
            ?? selectedGroup.shots.first
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                SectionTitle(
                    title: "连拍优选",
                    subtitle: "自动识别连拍序列，从清晰度、曝光与审美表现中挑出优选画面。照片仅在本机处理。"
                )
                progressStrip
                if result != nil { resultsColumn }
                settingsColumn
                if result == nil { resultsColumn }
            }
            .padding(26)
            .frame(maxWidth: .infinity, alignment: .topLeading)
        }
        .onChange(of: focusedParameter) { oldValue, newValue in
            guard let oldValue, oldValue != newValue else { return }
            commitParameter(oldValue)
        }
        .onChange(of: gapSeconds) { _, value in
            if focusedParameter != .gapSeconds { gapSecondsText = String(value) }
        }
        .onChange(of: maxHammingDistance) { _, value in
            if focusedParameter != .maxHammingDistance { maxHammingDistanceText = String(value) }
        }
        .onChange(of: keepCount) { _, value in
            if focusedParameter != .keepCount { keepCountText = String(value) }
        }
        .onChange(of: maxWorkers) { _, value in
            if focusedParameter != .maxWorkers { maxWorkersText = String(value) }
        }
        .task {
            try? await Task.sleep(for: .milliseconds(70))
            guard !Task.isCancelled else { return }
            await refreshCapabilities()
        }
        .task(id: previewTaskKey) {
            await loadSelectedPreviews()
        }
        .onDisappear {
            task?.cancel()
        }
    }

    private var settingsColumn: some View {
        HStack(alignment: .top, spacing: 16) {
            VStack(alignment: .leading, spacing: 16) {
                Panel(title: "照片目录") {
                    Button {
                        if let path = FilePicker.folder(title: "选择待筛选照片所在目录") {
                            inputDirectory = path
                            result = nil
                            errorMessage = ""
                        }
                    } label: {
                        Label(inputDirectory.isEmpty ? "选择照片目录…" : "更换照片目录…", systemImage: "folder")
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .buttonStyle(.bordered)
                    if !inputDirectory.isEmpty {
                        Text(inputDirectory)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .textSelection(.enabled)
                            .lineLimit(3)
                    } else {
                        Text("支持 RAW、JPG、HEIC、HIF 等常见摄影格式")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }

                Panel(title: "评分权重") {
                    Picker("权重模式", selection: $weightMode) {
                        Text("智能自适应").tag("adaptive")
                        Text("自定义").tag("custom")
                    }
                    .pickerStyle(.segmented)
                    if weightMode == "custom" {
                        weightSlider("清晰度", key: "sharpness")
                        weightSlider("审美", key: "aesthetic")
                        weightSlider("曝光", key: "exposure")
                        Button("清晰优先 · 90 / 5 / 5") {
                            customWeights = ["sharpness": 0.9, "aesthetic": 0.05, "exposure": 0.05]
                        }
                        .font(.caption)
                    } else {
                        Text("根据每组照片的质量差异自动调整清晰度、审美与曝光占比。")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }

                Panel(title: "审查目录名称") {
                    TextField("淘汰照片目录", text: $reviewSubdirectory)
                    TextField("明显废片目录", text: $defectSubdirectory)
                    Text("未保留照片只会移入审查目录，不会删除。RAW 与同名伴生文件由本地引擎成组处理。")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

            }
            .frame(maxWidth: .infinity, alignment: .topLeading)
            VStack(alignment: .leading, spacing: 16) {
                Panel(title: "筛选参数") {
                    VStack(alignment: .leading, spacing: 12) {
                        HStack {
                            Text("连拍间隔")
                            Spacer()
                            parameterTextField("秒", text: $gapSecondsText, field: .gapSeconds)
                                .frame(width: 74)
                            Text("秒").foregroundStyle(.secondary)
                        }
                        Stepper(value: $maxHammingDistance, in: 1...32) {
                            HStack {
                                Text("画面相似度")
                                Spacer()
                                parameterTextField("", text: $maxHammingDistanceText, field: .maxHammingDistance)
                                    .frame(width: 58)
                            }
                        }
                        Picker("预设", selection: $maxHammingDistance) {
                            Text("保守 · 8").tag(8)
                            Text("均衡 · 12").tag(12)
                            Text("激进 · 18").tag(18)
                        }
                        .pickerStyle(.segmented)
                        Stepper(value: $keepCount, in: 1...20) {
                            HStack {
                                Text("每组保留")
                                Spacer()
                                parameterTextField("", text: $keepCountText, field: .keepCount)
                                    .frame(width: 58)
                                Text("张").foregroundStyle(.secondary)
                            }
                        }
                        Stepper(value: $maxWorkers, in: 1...cpuCount) {
                            HStack {
                                Text("并发线程")
                                Spacer()
                                parameterTextField("", text: $maxWorkersText, field: .maxWorkers)
                                    .frame(width: 58)
                                Text("/ \(cpuCount)").foregroundStyle(.secondary)
                            }
                        }
                        Divider()
                        Picker("整组全糊时", selection: $allBlurryAction) {
                            Text("保留最佳").tag("keep")
                            Text("移入审查").tag("review")
                            Text("移入废片").tag("reject")
                        }
                        Toggle("筛选后提供连拍复核", isOn: $includePreviews)
                        Toggle("启用闭眼检测", isOn: $eyeDetection)
                            .disabled(!eyeModelReady)
                        if !eyeModelReady {
                            Text("闭眼检测模型尚未就绪")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                    }
                    .font(.callout)
                }

                Button {
                    if isRunning {
                        task?.cancel()
                        isRunning = false
                        progressMessage = "已停止接收进度；本地筛选任务可能仍在运行。"
                        appendMessage(progressMessage)
                    } else {
                        startFiltering()
                    }
                } label: {
                    Label(isRunning ? "停止接收进度" : "开始筛选", systemImage: isRunning ? "stop.fill" : "play.fill")
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 4)
                }
                .buttonStyle(.borderedProminent)
                .disabled(!engine.ready || (!isRunning && inputDirectory.isEmpty))
                if !engine.ready {
                    Label(engine.status, systemImage: "bolt.horizontal.circle")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .frame(maxWidth: .infinity, alignment: .topLeading)
        }
    }

    private var progressStrip: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 10) {
                if isRunning { ProgressView().controlSize(.small) }
                Text(progressMessage)
                    .font(.callout)
                    .lineLimit(1)
                Spacer(minLength: 8)
                if let progressFraction {
                    Text("\(Int(progressFraction * 100))%")
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
            }
            if isRunning || progressFraction != nil { ProgressView(value: progressFraction) }
            if !errorMessage.isEmpty {
                Label(errorMessage, systemImage: "exclamationmark.triangle.fill")
                    .font(.callout)
                    .foregroundStyle(.red)
                    .textSelection(.enabled)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
    }

    private var resultsColumn: some View {
        VStack(alignment: .leading, spacing: 16) {
            if let result {
                summaryPanel(result)
                if !result.groups.isEmpty { reviewPanel(result) }
            }
            if !messages.isEmpty {
                DisclosureGroup("处理记录 · \(messages.count) 条") {
                    LazyVStack(alignment: .leading, spacing: 5) {
                        ForEach(Array(messages.enumerated()), id: \.offset) { entry in
                            Text(entry.element)
                                .font(.system(.caption2, design: .monospaced))
                                .textSelection(.enabled)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                    }
                    .padding(.top, 8)
                }
                .padding(16)
                .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
            }
        }
    }

    private func summaryPanel(_ value: BurstResult) -> some View {
        Panel(title: "本次筛选已完成") {
            HStack(alignment: .top, spacing: 0) {
                summaryMetric("\(value.total)", label: "扫描文件")
                summaryMetric("\(value.groupCount)", label: "连拍组")
                summaryMetric("\(value.moved)", label: "移入审查")
                summaryMetric("\(value.skippedSingle)", label: "单张跳过")
            }
            if value.defectMoved > 0 {
                Text("明显废片：\(value.defectMoved) 张")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            if !value.reviewDirectory.isEmpty {
                Label("审查目录：\(value.reviewDirectory)", systemImage: "folder")
                    .font(.caption)
                    .textSelection(.enabled)
            }
            if !value.defectDirectory.isEmpty {
                Label("废片目录：\(value.defectDirectory)", systemImage: "folder.badge.minus")
                    .font(.caption)
                    .textSelection(.enabled)
            }
        }
    }

    private func reviewPanel(_ value: BurstResult) -> some View {
        let reviewCount = value.groups.reduce(into: 0) { count, group in
            if group.needsReview { count += 1 }
        }
        return Panel(title: "连拍组复核 · \(reviewCount) 组建议检查") {
            VStack(alignment: .leading, spacing: 14) {
                groupPicker(value.groups)
                if let group = selectedGroup {
                    groupReviewContent(group)
                }
            }
        }
    }

    private func groupPicker(_ groups: [BurstGroup]) -> some View {
        ScrollView(.horizontal) {
            HStack(spacing: 8) {
                ForEach(groups) { group in
                    Button {
                        selectGroup(group)
                    } label: {
                        HStack(spacing: 6) {
                            Text("第 \(group.index) 组")
                            Text("\(group.shots.count) 张").foregroundStyle(.secondary)
                            if group.needsReview {
                                Image(systemName: "circle.fill")
                                    .font(.system(size: 6))
                                    .foregroundStyle(.orange)
                            }
                        }
                    }
                    .buttonStyle(.bordered)
                    .tint(selectedGroupIndex == group.index ? .accentColor : .secondary)
                }
            }
        }
    }

    private func groupReviewContent(_ group: BurstGroup) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Divider()
            HStack {
                VStack(alignment: .leading, spacing: 3) {
                    Text("第 \(group.index) 组 · \(group.shots.count) 次快门")
                        .font(.headline)
                    Text(group.needsReview ? "建议人工复核" : "筛选置信度较高")
                        .font(.caption)
                        .foregroundStyle(group.needsReview ? .orange : .secondary)
                }
                Spacer()
                groupWeights(group)
            }
            if !group.weightReason.isEmpty {
                Text(group.weightReason).font(.caption).foregroundStyle(.secondary)
            }
            shotChooser(group)
            if let shot = selectedShot {
                selectedShotDetail(shot)
            }
        }
    }

    private func groupWeights(_ group: BurstGroup) -> some View {
        Text("清晰 \(percentage(group.weights["sharpness"])) · 审美 \(percentage(group.weights["aesthetic"])) · 曝光 \(percentage(group.weights["exposure"]))")
            .font(.caption)
            .foregroundStyle(.secondary)
    }

    private func shotChooser(_ group: BurstGroup) -> some View {
        ScrollView(.horizontal) {
            HStack(alignment: .top, spacing: 10) {
                ForEach(group.shots) { shot in
                    Button {
                        selectedPhotoID = shot.photoID
                    } label: {
                        VStack(alignment: .leading, spacing: 6) {
                            ZStack(alignment: .bottomLeading) {
                                Rectangle().fill(.quaternary)
                                if shot.photoID == selectedPhotoID, let previewImage {
                                    Image(nsImage: previewImage)
                                        .resizable()
                                        .aspectRatio(contentMode: .fill)
                                        .frame(width: 138, height: 92)
                                        .clipped()
                                } else {
                                    Image(systemName: "photo")
                                        .font(.title2)
                                        .foregroundStyle(.secondary)
                                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                                }
                                Text(shot.kept ? "保留" : shot.category == "defect" ? "明显废片" : "已移入审查")
                                    .font(.caption2.bold())
                                    .padding(.horizontal, 6)
                                    .padding(.vertical, 3)
                                    .background(.ultraThinMaterial, in: Capsule())
                                    .padding(5)
                            }
                            .frame(width: 138, height: 92)
                            Text(shot.name)
                                .font(.caption)
                                .lineLimit(1)
                                .frame(width: 138, alignment: .leading)
                            Text("清晰度 \(percentage(shot.sharpness))")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                                .frame(width: 138, alignment: .leading)
                        }
                        .padding(5)
                        .background(selectedPhotoID == shot.photoID ? Color.accentColor.opacity(0.12) : .clear, in: RoundedRectangle(cornerRadius: 8))
                        .overlay(RoundedRectangle(cornerRadius: 8).stroke(selectedPhotoID == shot.photoID ? Color.accentColor : Color.clear, lineWidth: 1.5))
                    }
                    .buttonStyle(.plain)
                }
            }
        }
    }

    private func selectedShotDetail(_ shot: BurstShot) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .top, spacing: 14) {
                previewCard(title: "完整画面", image: previewImage)
                previewCard(title: "细节检查", image: detailImage)
            }
            if !previewMessage.isEmpty {
                Text(previewMessage).font(.caption).foregroundStyle(.secondary)
            }
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 5) {
                    Text(shot.name).font(.headline).textSelection(.enabled)
                    Text("综合排名 #\(shot.rank.map { String($0) } ?? "—") · \(shot.companionCount) 个伴生文件")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    HStack(spacing: 8) {
                        if shot.sharpest { Label("本组最清晰", systemImage: "scope").foregroundStyle(.blue) }
                        if shot.aestheticBest { Label("审美最高", systemImage: "sparkles").foregroundStyle(.purple) }
                        if shot.rejectReasons.contains("severe_blur") { Text("严重虚焦").foregroundStyle(.red) }
                        if shot.rejectReasons.contains("closed_eyes") { Text("检测到闭眼").foregroundStyle(.red) }
                    }
                    .font(.caption2)
                    metricBar("清晰度", shot.sharpness)
                    metricBar("审美", shot.aesthetic)
                    metricBar("曝光", shot.exposure)
                }
                Spacer(minLength: 8)
                VStack(spacing: 8) {
                    decisionButton(shot, kept: true)
                    decisionButton(shot, kept: false)
                }
                .frame(width: 150)
            }
            if !decisionFeedback.isEmpty {
                Text(decisionFeedback)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }
        }
    }

    private func previewCard(title: String, image: NSImage?) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title).font(.caption.bold()).foregroundStyle(.secondary)
            ZStack {
                RoundedRectangle(cornerRadius: 8).fill(Color.black.opacity(0.06))
                if let image {
                    Image(nsImage: image)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .padding(4)
                } else {
                    ProgressView()
                }
            }
            .frame(maxWidth: .infinity)
            .frame(height: 185)
            .clipShape(RoundedRectangle(cornerRadius: 8))
        }
        .frame(maxWidth: .infinity)
    }

    private func decisionButton(_ shot: BurstShot, kept: Bool) -> some View {
        Button {
            Task { await setDecision(shot, kept: kept) }
        } label: {
            Label(kept ? "保留在原目录" : "移入审查目录", systemImage: kept ? "checkmark.circle" : "folder.badge.arrow.down")
                .frame(maxWidth: .infinity)
        }
        .buttonStyle(.bordered)
        .disabled(shot.kept == kept || decisionPendingID != nil || result?.previewSession.isEmpty != false)
        .overlay {
            if decisionPendingID == shot.photoID { ProgressView().controlSize(.small) }
        }
    }

    private func weightSlider(_ label: String, key: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(label)
                Spacer()
                Text("\(Int((customWeights[key] ?? 0) * 100))%")
                    .foregroundStyle(.secondary)
                    .monospacedDigit()
            }
            Slider(value: Binding(
                get: { customWeights[key] ?? 0 },
                set: { updateWeight(key, value: $0) }
            ), in: 0...1)
        }
    }

    private func parameterTextField(_ prompt: String, text: Binding<String>, field: ParameterField) -> some View {
        TextField(prompt, text: text)
            .textFieldStyle(.roundedBorder)
            .multilineTextAlignment(.trailing)
            .monospacedDigit()
            .focused($focusedParameter, equals: field)
            .onSubmit {
                commitParameter(field)
                focusedParameter = nil
            }
    }

    private func commitParameter(_ field: ParameterField) {
        switch field {
        case .gapSeconds:
            guard let entered = Double(gapSecondsText.trimmingCharacters(in: .whitespacesAndNewlines)),
                  entered.isFinite, entered > 0 else {
                gapSecondsText = String(gapSeconds)
                return
            }
            gapSeconds = entered
            gapSecondsText = String(entered)
        case .maxHammingDistance:
            guard let entered = Int(maxHammingDistanceText.trimmingCharacters(in: .whitespacesAndNewlines)) else {
                maxHammingDistanceText = String(maxHammingDistance)
                return
            }
            maxHammingDistance = min(32, max(1, entered))
            maxHammingDistanceText = String(maxHammingDistance)
        case .keepCount:
            guard let entered = Int(keepCountText.trimmingCharacters(in: .whitespacesAndNewlines)) else {
                keepCountText = String(keepCount)
                return
            }
            keepCount = min(20, max(1, entered))
            keepCountText = String(keepCount)
        case .maxWorkers:
            guard let entered = Int(maxWorkersText.trimmingCharacters(in: .whitespacesAndNewlines)) else {
                maxWorkersText = String(maxWorkers)
                return
            }
            maxWorkers = min(cpuCount, max(1, entered))
            maxWorkersText = String(maxWorkers)
        }
    }

    private func summaryMetric(_ value: String, label: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(value).font(.title2.bold().monospacedDigit())
            Text(label).font(.caption).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func metricBar(_ title: String, _ value: Double) -> some View {
        HStack(spacing: 8) {
            Text(title).font(.caption).foregroundStyle(.secondary).frame(width: 54, alignment: .leading)
            ProgressView(value: min(1, max(0, value)))
            Text(percentage(value)).font(.caption.monospacedDigit()).frame(width: 38, alignment: .trailing)
        }
        .frame(maxWidth: 360)
    }

    private var previewTaskKey: String {
        "\(result?.previewSession ?? "")/\(selectedShot?.photoID ?? "")"
    }

    @MainActor
    private func refreshCapabilities() async {
        guard engine.ready else { return }
        do {
            let status = try await engine.api.json("/api/models/status")
            gpuAvailable = status["gpu_available"] as? Bool ?? false
            useGPU = gpuAvailable
            eyeModelReady = status["face_landmarker_ready"] as? Bool ?? false
            if !eyeModelReady { eyeDetection = false }
        } catch {
            appendMessage("读取模型状态失败：\(error.localizedDescription)")
        }
    }

    @MainActor
    private func startFiltering() {
        commitParameter(.gapSeconds)
        commitParameter(.maxHammingDistance)
        commitParameter(.keepCount)
        commitParameter(.maxWorkers)
        focusedParameter = nil
        guard !inputDirectory.isEmpty else { return }
        guard (1...cpuCount).contains(maxWorkers) else {
            errorMessage = "并发线程数必须在 1 到 \(cpuCount) 之间。"
            return
        }
        guard gapSeconds > 0, gapSeconds.isFinite else {
            errorMessage = "连拍间隔必须大于 0。"
            return
        }

        result = nil
        errorMessage = ""
        progressFraction = nil
        progressMessage = "正在连接本地引擎…"
        messages = []
        selectedGroupIndex = nil
        selectedPhotoID = nil
        isRunning = true
        let body: [String: Any] = [
            "input_dir": inputDirectory,
            "gap_seconds": gapSeconds,
            "max_hamming_distance": maxHammingDistance,
            "review_subdir": reviewSubdirectory,
            "defect_subdir": defectSubdirectory,
            "keep_count": keepCount,
            "max_workers": maxWorkers,
            "use_gpu": useGPU && gpuAvailable,
            "sort_backend": engine.sortBackend,
            "include_previews": includePreviews,
            "weight_mode": weightMode,
            "custom_weights": weightMode == "custom" ? (customWeights as Any) : (NSNull() as Any),
            "all_blurry_action": allBlurryAction,
            "eye_detection": eyeDetection && eyeModelReady,
        ]
        task = Task { await receiveEvents(body: body) }
    }

    @MainActor
    private func receiveEvents(body: [String: Any]) async {
        do {
            try await engine.api.events("/api/burst/run", body: body) { event in
                let type = string(event, "type")
                let message = string(event, "msg")
                if !message.isEmpty {
                    progressMessage = message
                    appendMessage(message)
                }
                if let value = event["progress"] as? Double {
                    progressFraction = min(1, max(0, value))
                }
                if type == "done" {
                    result = BurstResult(event)
                    progressFraction = 1
                    progressMessage = "筛选已完成"
                    isRunning = false
                    if let first = result?.groups.first { selectGroup(first) }
                    appendMessage(progressMessage)
                } else if type == "error" {
                    errorMessage = message.isEmpty ? "连拍筛选执行失败。" : message
                    progressMessage = "处理遇到问题"
                    isRunning = false
                }
            }
            if isRunning {
                isRunning = false
                progressMessage = Task.isCancelled ? "已停止接收进度" : "连接已结束"
            }
        } catch is CancellationError {
            isRunning = false
            progressMessage = "已停止接收进度；本地筛选任务可能仍在运行。"
        } catch {
            isRunning = false
            errorMessage = error.localizedDescription
            progressMessage = "连接本地引擎失败"
            appendMessage(error.localizedDescription)
        }
        task = nil
    }

    @MainActor
    private func loadSelectedPreviews() async {
        guard let result, let shot = selectedShot, !result.previewSession.isEmpty else {
            previewImage = nil
            detailImage = nil
            previewMessage = ""
            return
        }
        previewImage = nil
        detailImage = nil
        previewMessage = "正在载入预览…"
        do {
            async let full = fetchPreview(sessionID: result.previewSession, photoID: shot.photoID, kind: "full")
            async let detail = fetchPreview(sessionID: result.previewSession, photoID: shot.photoID, kind: "review")
            let (fullImage, detailImage) = try await (full, detail)
            guard selectedPhotoID == shot.photoID else { return }
            previewImage = fullImage
            self.detailImage = detailImage
            previewMessage = ""
        } catch {
            previewMessage = "预览载入失败：\(error.localizedDescription)"
        }
    }

    @MainActor
    private func fetchPreview(sessionID: String, photoID: String, kind: String) async throws -> NSImage {
        guard safeID(sessionID), safeID(photoID), ["full", "review"].contains(kind) else {
            throw APIError.invalidResponse
        }
        let url = try engine.api.url("/api/burst/preview/\(sessionID)/\(photoID)?kind=\(kind)")
        let (data, response) = try await URLSession.shared.data(from: url)
        guard let response = response as? HTTPURLResponse else { throw APIError.invalidResponse }
        guard (200..<300).contains(response.statusCode) else {
            let object = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
            throw APIError.server(object?["error"] as? String ?? "预览不可用")
        }
        return try await PreviewImageDecoder.image(from: data)
    }

    @MainActor
    private func setDecision(_ shot: BurstShot, kept: Bool) async {
        guard let sessionID = result?.previewSession,
              !sessionID.isEmpty,
              safeID(sessionID), safeID(shot.photoID) else { return }
        decisionPendingID = shot.photoID
        decisionFeedback = ""
        defer { decisionPendingID = nil }
        do {
            let response = try await engine.api.json(
                "/api/burst/decision/\(sessionID)/\(shot.photoID)",
                body: ["kept": kept]
            )
            guard let groupIndex = result?.groups.firstIndex(where: { group in
                group.shots.contains(where: { $0.photoID == shot.photoID })
            }),
                  let shotIndex = result?.groups[groupIndex].shots.firstIndex(where: { $0.photoID == shot.photoID }) else { return }
            let previousKept = result?.groups[groupIndex].shots[shotIndex].kept ?? false
            result?.groups[groupIndex].shots[shotIndex].kept = response["kept"] as? Bool ?? kept
            result?.groups[groupIndex].shots[shotIndex].category = string(response, "category")
            if previousKept != kept {
                let movedFiles = int(response, "moved_files")
                result?.moved = max(0, (result?.moved ?? 0) + (kept ? -movedFiles : movedFiles))
            }
            decisionFeedback = string(response, "message")
            appendMessage("\(shot.name)：\(decisionFeedback)")
        } catch {
            decisionFeedback = error.localizedDescription
        }
    }

    private func selectGroup(_ group: BurstGroup) {
        selectedGroupIndex = group.index
        selectedPhotoID = group.shots.first(where: \.kept)?.photoID ?? group.shots.first?.photoID
        decisionFeedback = ""
    }

    private func updateWeight(_ key: String, value: Double) {
        let others = customWeights.keys.filter { $0 != key }
        let oldTotal = others.reduce(0) { $0 + (customWeights[$1] ?? 0) }
        let remaining = 1 - value
        customWeights[key] = value
        for other in others {
            customWeights[other] = oldTotal > 0 ? remaining * (customWeights[other] ?? 0) / oldTotal : remaining / Double(others.count)
        }
    }

    private func appendMessage(_ message: String) {
        messages.append(message)
        if messages.count > 500 { messages.removeFirst(messages.count - 500) }
    }

    private func safeID(_ value: String) -> Bool {
        !value.isEmpty && value.rangeOfCharacter(from: CharacterSet(charactersIn: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" ).inverted) == nil
    }

    private func percentage(_ value: Double?) -> String {
        "\(Int(((value ?? 0) * 100).rounded()))%"
    }
}
