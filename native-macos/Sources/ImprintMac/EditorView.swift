import AppKit
import SwiftUI

private struct Adjustment: Identifiable {
    let id: String
    let title: String
    let range: ClosedRange<Double>
    let step: Double
}

private let dehazeAdjustments: [Adjustment] = [
    .init(id: "strength", title: "去朦胧强度", range: 0...1, step: 0.01),
    .init(id: "naturalness", title: "自然度", range: 0...1, step: 0.01),
    .init(id: "fog_retention", title: "雾气保留", range: 0...1, step: 0.01),
    .init(id: "local_contrast", title: "局部对比度", range: 0...1, step: 0.01),
    .init(id: "color_recovery", title: "颜色恢复", range: 0...1, step: 0.01),
    .init(id: "color_protection", title: "色彩保护", range: 0...1, step: 0.01),
    .init(id: "highlight_protection", title: "高光保护", range: 0...1, step: 0.01),
    .init(id: "shadow_protection", title: "暗部保护", range: 0...1, step: 0.01),
    .init(id: "brightness_protection", title: "亮度保护", range: 0...1, step: 0.01),
]

private let basicAdjustments: [Adjustment] = [
    .init(id: "exposure", title: "曝光", range: -5...5, step: 0.05),
    .init(id: "contrast", title: "对比度", range: -100...100, step: 1),
    .init(id: "highlights", title: "高光", range: -100...100, step: 1),
    .init(id: "shadows", title: "阴影", range: -100...100, step: 1),
    .init(id: "whites", title: "白色", range: -100...100, step: 1),
    .init(id: "blacks", title: "黑色", range: -100...100, step: 1),
    .init(id: "vibrance", title: "自然饱和度", range: -100...100, step: 1),
    .init(id: "saturation", title: "饱和度", range: -100...100, step: 1),
]

private struct EditorDisclosure<Content: View>: View {
    let title: String
    @ViewBuilder let content: Content
    @State private var isExpanded = false

    var body: some View {
        DisclosureGroup(title, isExpanded: $isExpanded) {
            content.padding(.top, 10)
        }
        .font(.caption)
    }
}

struct EditorView: View, Equatable {
    enum Mode { case enhance, ricoh }
    private enum PreviewMode: String, CaseIterable, Identifiable {
        case original = "原图"
        case compare = "原图+效果图"
        case effect = "效果图"
        var id: String { rawValue }
    }
    let mode: Mode
    let isActive: Bool

    static func == (lhs: EditorView, rhs: EditorView) -> Bool {
        lhs.mode == rhs.mode && lhs.isActive == rhs.isActive
    }
    @EnvironmentObject private var engine: Engine
    @EnvironmentObject private var library: PhotoLibrary
    @State private var presets: [[String: Any]] = []
    @State private var original: NSImage?
    @State private var edited: NSImage?
    @State private var quickThumbnail: NSImage?
    @State private var thumbnailTask: Task<Void, Never>?
    @State private var thumbnailRequestID: UUID?
    @State private var thumbnailLoading = false
    @State private var originalTask: Task<Void, Never>?
    @State private var originalRequestID: UUID?
    @State private var originalLoading = false
    @State private var previewing = false
    @State private var previewMode: PreviewMode = .effect
    @State private var compareFraction = 0.5
    @State private var metadata: [String: Any] = [:]
    @State private var error = ""
    @State private var pollTask: Task<Void, Never>?
    @State private var previewTask: Task<Void, Never>?
    @State private var previewWorkerID: UUID?
    @State private var pendingPreviewLevel: Int?
    @State private var previewThrottleTask: Task<Void, Never>?
    @State private var previewRefinementTask: Task<Void, Never>?
    @State private var sliderPreviewPending = false
    @State private var sliderEditing = false
    @State private var presetTask: Task<Void, Never>?
    @State private var previewRequestID: UUID?
    @State private var metadataTask: Task<Void, Never>?
    @State private var xmpSaveTasks: [String: Task<Void, Never>] = [:]
    @State private var xmpSaveIDs: [String: UUID] = [:]
    @State private var xmpStatus = ""

    private var isRunning: Bool { ["queued", "running"].contains(string(job, "status")) }
    private var job: [String: Any] { mode == .enhance ? library.enhanceJob : library.ricohJob }
    private var title: String { mode == .enhance ? "去朦胧" : "理光风格" }
    private var selected: Photo? { library.photos.first { $0.id == library.selectedID } }
    private var selectedPreset: String { library.presetByPhoto[library.selectedID] ?? "" }

    private func selectionDidChange() {
        cancelPreviewWork()
        previewThrottleTask?.cancel()
        previewThrottleTask = nil
        previewRefinementTask?.cancel()
        previewRefinementTask = nil
        sliderPreviewPending = false
        sliderEditing = false
        original = nil
        edited = nil
        quickThumbnail = nil
        metadata = [:]
        error = ""
        xmpStatus = ""
        guard isActive, !library.selectedID.isEmpty else { return }
        loadSelectedThumbnail()
        refreshPreview(previewLevel: 2)
        loadMetadata()
    }

    private func loadSelectedThumbnail() {
        thumbnailTask?.cancel()
        thumbnailTask = nil
        thumbnailRequestID = nil
        thumbnailLoading = false

        let sessionID = library.sessionID
        let photoID = library.selectedID
        let quickImage = PhotoThumbnail.cachedImage(sessionID: sessionID, photoID: photoID, maxEdge: 1280)
        quickThumbnail = quickImage ?? PhotoThumbnail.cachedImage(sessionID: sessionID, photoID: photoID)
        guard quickImage == nil, !sessionID.isEmpty, !photoID.isEmpty, engine.ready else { return }

        let requestID = UUID()
        thumbnailRequestID = requestID
        thumbnailLoading = true
        thumbnailTask = Task {
            defer {
                if thumbnailRequestID == requestID {
                    thumbnailTask = nil
                    thumbnailRequestID = nil
                    thumbnailLoading = false
                }
            }
            do {
                let image = try await PhotoThumbnail.loadImage(
                    sessionID: sessionID,
                    photoID: photoID,
                    api: engine.api,
                    maxEdge: 1280
                )
                try Task.checkCancellation()
                guard library.sessionID == sessionID, library.selectedID == photoID,
                      thumbnailRequestID == requestID else { return }
                quickThumbnail = image
            } catch {
                guard !Task.isCancelled, library.sessionID == sessionID,
                      library.selectedID == photoID, thumbnailRequestID == requestID else { return }
            }
        }
    }

    var body: some View {
        GeometryReader { geometry in
            let isCompact = geometry.size.width / max(geometry.size.height, 1) < 1.4
            HStack(spacing: 0) {
                if !isCompact {
                    leftPane.frame(width: 250)
                    Divider()
                }
                previewPane(compact: isCompact)
                    .frame(minWidth: 0, maxWidth: .infinity, maxHeight: .infinity)
                Divider()
                rightPane.frame(width: 310)
            }
        }
        .frame(minWidth: 900, minHeight: 650)
        .task(id: engine.ready) {
            guard engine.ready else { return }
            if mode == .ricoh && presets.isEmpty { presetTask = Task { await loadPresets() } }
            guard isActive else { return }
            let initialSessionID = library.sessionID
            let initialPhotoID = library.selectedID
            if !initialPhotoID.isEmpty { loadSelectedThumbnail() }
            guard !Task.isCancelled else { return }
            guard library.sessionID == initialSessionID, library.selectedID == initialPhotoID else { return }
            if !initialPhotoID.isEmpty {
                refreshPreview(previewLevel: edited == nil ? 2 : 0)
                loadMetadata()
            }
            if isRunning { pollTask = Task { await pollJob() } }
        }
        .onChange(of: [library.sessionID, library.selectedID]) { _, _ in
            selectionDidChange()
        }
        .onChange(of: previewMode) { _, newMode in
            if newMode != .effect { loadOriginalPreviewIfNeeded() }
        }
        .onChange(of: isActive) { _, active in
            if active {
                guard engine.ready, !library.selectedID.isEmpty else { return }
                loadSelectedThumbnail()
                refreshPreview(previewLevel: edited == nil ? 2 : 0)
                loadMetadata()
                if isRunning { pollTask = Task { await pollJob() } }
            } else {
                guard !library.selectedID.isEmpty else { return }
                cancelPreviewWork()
                previewThrottleTask?.cancel()
                previewThrottleTask = nil
                previewRefinementTask?.cancel()
                previewRefinementTask = nil
                metadataTask?.cancel()
                pollTask?.cancel()
            }
        }
        .onDisappear {
            cancelPreviewWork()
            previewThrottleTask?.cancel()
            previewThrottleTask = nil
            previewRefinementTask?.cancel()
            previewRefinementTask = nil
            presetTask?.cancel()
            metadataTask?.cancel()
            pollTask?.cancel()
        }
    }

    private var leftPane: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("照片").font(.title2.bold())
            photoLoadButtons
            Text(library.message).font(.caption).foregroundStyle(.secondary).lineLimit(2)
            if library.photos.isEmpty {
                Spacer(minLength: 0)
            } else {
                List(library.photos, selection: $library.selectedID) { photo in
                    HStack(spacing: 10) {
                        PhotoThumbnail(sessionID: library.sessionID, photoID: photo.id, size: 48)
                        Text(photo.name).font(.caption).lineLimit(2)
                    }
                    .padding(.vertical, 3)
                    .tag(photo.id)
                }
                .listStyle(.plain)
            }
        }
        .padding(16)
        .background(Color.white)
    }

    private var photoLoadButtons: some View {
        HStack {
            Button("选择照片") {
                let paths = FilePicker.photos()
                if !paths.isEmpty { Task { await library.load(api: engine.api, paths: paths) } }
            }
            Button("文件夹") {
                if let folder = FilePicker.folder(title: "选择照片文件夹") {
                    Task { await library.load(api: engine.api, folder: folder) }
                }
            }
        }
        .disabled(!engine.ready || library.hasActiveJob)
    }

    private func previewPane(compact: Bool) -> some View {
        VStack(spacing: 0) {
            HStack(spacing: 12) {
                VStack(alignment: .leading, spacing: 3) {
                    Text(title).font(.title2.bold()).lineLimit(1)
                    Text(mode == .enhance ? "低分辨率预览 · 全分辨率 DNG 导出" : "相机风格预览 · DNG 导出")
                        .font(.caption).foregroundStyle(.secondary).lineLimit(1)
                }
                .frame(minWidth: 0, maxWidth: .infinity, alignment: .leading)
                Picker("预览方式", selection: $previewMode) {
                    ForEach(PreviewMode.allCases) { option in
                        Text(option.rawValue).tag(option)
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .controlSize(.small)
                .frame(width: 260)
                .disabled(selected == nil)
                Button { refreshPreview() } label: { Image(systemName: "arrow.clockwise") }
                    .help("刷新预览")
                    .disabled(selected == nil || previewing)
            }
            .padding(.horizontal, 18)
            .frame(height: 76)
            .background(Color.white)
            previewCanvas
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            VStack(alignment: .leading, spacing: 4) {
                Text(selected?.name ?? "未选择照片")
                    .font(.caption.weight(.medium)).foregroundStyle(.black).lineLimit(1)
                Text(metadataSummary)
                    .font(.caption2).foregroundStyle(Color.gray).lineLimit(1)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 18)
            .padding(.vertical, 12)
            .background(Color.white)
            if compact {
                compactPhotoPicker
            }
        }
    }

    private var compactPhotoPicker: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 10) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("照片").font(.subheadline.bold())
                    Text("\(library.photos.count) 张 · \(library.message)")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
                .frame(minWidth: 0, maxWidth: .infinity, alignment: .leading)
                photoLoadButtons
                    .fixedSize()
            }
            if !library.photos.isEmpty {
                ScrollView(.horizontal) {
                    LazyHStack(alignment: .top, spacing: 8) {
                        ForEach(library.photos) { photo in
                            let isSelected = photo.id == library.selectedID
                            Button {
                                library.selectedID = photo.id
                            } label: {
                                VStack(spacing: 4) {
                                    PhotoThumbnail(sessionID: library.sessionID, photoID: photo.id, size: 58)
                                    Text(photo.name)
                                        .font(.caption2)
                                        .lineLimit(2)
                                        .multilineTextAlignment(.center)
                                        .frame(width: 78, height: 28, alignment: .top)
                                }
                                .padding(5)
                                .background(isSelected ? Color.accentColor.opacity(0.12) : Color.clear,
                                            in: RoundedRectangle(cornerRadius: 8))
                                .overlay {
                                    RoundedRectangle(cornerRadius: 8)
                                        .stroke(isSelected ? Color.accentColor : Color.black.opacity(0.08),
                                                lineWidth: isSelected ? 2 : 1)
                                }
                            }
                            .buttonStyle(.plain)
                            .help(photo.name)
                        }
                    }
                    .padding(.horizontal, 2)
                }
                .frame(height: 104)
            }
        }
        .padding(.horizontal, 18)
        .padding(.top, 8)
        .padding(.bottom, 10)
        .background(Color.white)
    }

    private var previewCanvas: some View {
        GeometryReader { geometry in
            ZStack {
                Color.black
                if selected != nil {
                    let base = previewMode == .original
                        ? (original ?? quickThumbnail)
                        : (edited ?? original ?? quickThumbnail)
                    if let base {
                    Image(nsImage: base)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(width: geometry.size.width, height: geometry.size.height)
                    }
                    if previewMode == .compare, let original, edited != nil {
                        Image(nsImage: original)
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .frame(width: geometry.size.width, height: geometry.size.height)
                            .mask {
                                Rectangle()
                                    .frame(width: geometry.size.width * compareFraction)
                                    .frame(width: geometry.size.width, alignment: .leading)
                            }
                        Rectangle()
                            .fill(.white)
                            .frame(width: 2)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .offset(x: geometry.size.width * compareFraction)
                            .shadow(color: .black.opacity(0.4), radius: 2)
                        HStack {
                            Text("原图")
                                .padding(.horizontal, 8)
                                .padding(.vertical, 5)
                                .background(.black.opacity(0.55), in: RoundedRectangle(cornerRadius: 6))
                            Spacer()
                            Text("效果图")
                                .padding(.horizontal, 8)
                                .padding(.vertical, 5)
                                .background(.black.opacity(0.55), in: RoundedRectangle(cornerRadius: 6))
                        }
                        .font(.caption2.weight(.semibold))
                        .foregroundStyle(.white)
                        .padding(14)
                        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
                    }
                } else {
                    ContentUnavailableView("选择照片开始预览", systemImage: "photo.on.rectangle.angled", description: Text("照片只在本机处理"))
                        .foregroundStyle(.white)
                }
                if thumbnailLoading || originalLoading || previewing {
                    HStack(spacing: 7) {
                        ProgressView().controlSize(.small).tint(.white)
                        Text(previewing ? (edited != nil ? "正在细化效果…" : "正在生成效果…")
                                        : (originalLoading ? "正在加载原图…" : "正在加载快速预览…"))
                            .font(.caption2)
                    }
                    .foregroundStyle(.white)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 7)
                    .background(.black.opacity(0.62), in: Capsule())
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topTrailing)
                    .padding(14)
                }
            }
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { value in
                        guard previewMode == .compare, original != nil, edited != nil, geometry.size.width > 0 else { return }
                        compareFraction = min(1, max(0, value.location.x / geometry.size.width))
                    }
            )
        }
    }

    private var metadataSummary: String {
        var parts: [String] = []
        if let iso = metadata["iso"] as? NSNumber, iso.intValue > 0 { parts.append("ISO \(iso.intValue)") }
        if let aperture = metadata["aperture"] as? NSNumber, aperture.doubleValue > 0 {
            parts.append("f/\(aperture.doubleValue.formatted(.number.precision(.fractionLength(1))))")
        }
        if let exposure = metadata["exposure_time"] as? NSNumber, exposure.doubleValue > 0 {
            let seconds = exposure.doubleValue
            parts.append(seconds < 1 ? "1/\(Int((1 / seconds).rounded())) s" : "\(seconds.formatted(.number.precision(.fractionLength(1)))) s")
        }
        if let focalLength = metadata["focal_length"] as? NSNumber, focalLength.doubleValue > 0 {
            parts.append("\(Int(focalLength.doubleValue.rounded())) mm")
        }
        if let width = metadata["width"] as? Int, let height = metadata["height"] as? Int, width > 0, height > 0 {
            parts.append("\(width) × \(height)")
        }
        if let bytes = metadata["size_bytes"] as? Int, bytes > 0 {
            parts.append(String(format: "%.1f MB", Double(bytes) / 1_048_576))
        }
        return parts.isEmpty ? "拍摄信息读取中或不可用" : parts.joined(separator: "  ·  ")
    }

    private func loadMetadata() {
        metadataTask?.cancel()
        guard isActive, !library.sessionID.isEmpty, !library.selectedID.isEmpty, engine.ready else { return }
        let sessionID = library.sessionID
        let photoID = library.selectedID
        metadataTask = Task {
            do {
                let result = try await engine.api.json("/api/enhance/metadata/\(sessionID)/\(photoID)")
                try Task.checkCancellation()
                guard library.sessionID == sessionID, library.selectedID == photoID else { return }
                metadata = result
            } catch is CancellationError {
            } catch {
                metadata = [:]
            }
        }
    }

    private var rightPane: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 16) {
                if mode == .enhance {
                    editorCard("去朦胧", icon: "wand.and.stars") {
                        slider(dehazeAdjustments[0], basic: false)
                        EditorDisclosure(title: "高级参数") {
                            VStack(spacing: 12) {
                                ForEach(dehazeAdjustments.dropFirst()) { adjustment in
                                    slider(adjustment, basic: false)
                                }
                            }
                        }
                    }
                } else {
                    editorCard("理光预设", icon: "camera.filters") {
                        Picker("风格", selection: Binding(
                            get: { selectedPreset },
                            set: {
                                let photoID = library.selectedID
                                if $0.isEmpty { library.presetByPhoto.removeValue(forKey: photoID) }
                                else { library.presetByPhoto[photoID] = $0 }
                                scheduleAutoSave(photoID: photoID)
                            }
                        )) {
                            Text("请选择").tag("")
                            ForEach(presets.indices, id: \.self) { index in
                                let preset = presets[index]
                                Text(string(preset, "name")).tag(string(preset, "id"))
                            }
                        }
                        .onChange(of: selectedPreset) { _, _ in refreshPreview() }
                    }
                }
                editorCard("基础调整", icon: "slider.horizontal.3") {
                    ForEach(basicAdjustments.prefix(4)) { adjustment in slider(adjustment, basic: true) }
                    EditorDisclosure(title: "更多基础调整") {
                        VStack(spacing: 12) {
                            ForEach(basicAdjustments.dropFirst(4)) { adjustment in
                                slider(adjustment, basic: true)
                            }
                        }
                    }
                }
                editorCard("导出", icon: "square.and.arrow.up") {
                    TextField("输出文件夹", text: mode == .enhance ? $library.dehazeOutput : $library.ricohOutput)
                    Button("选择输出位置") {
                        if let folder = FilePicker.folder(title: "选择输出文件夹") {
                            if mode == .enhance { library.dehazeOutput = folder }
                            else { library.ricohOutput = folder }
                        }
                    }
                    HStack {
                        Button("写入 XMP") { Task { await saveXMP() } }.disabled(library.sessionID.isEmpty)
                        if isRunning {
                            Button("停止") { Task { await cancelJob() } }
                        } else {
                            Button("导出 DNG") { Task { await startJob() } }
                                .buttonStyle(.borderedProminent)
                                .disabled(library.sessionID.isEmpty || (mode == .ricoh && selectedPreset.isEmpty))
                        }
                    }
                    if !xmpStatus.isEmpty {
                        Text(xmpStatus).font(.caption).foregroundStyle(.secondary)
                    }
                    if !job.isEmpty {
                        ProgressView(value: min(1, max(0, job["progress"] as? Double ?? 0)))
                        Text("\(int(job, "processed"))/\(int(job, "total")) · 成功 \(int(job, "success")) · 失败 \(int(job, "failed"))")
                            .font(.caption)
                        ForEach(Array((job["files"] as? [[String: Any]] ?? []).enumerated()), id: \.offset) { _, file in
                            Text("\(string(file, "name")) · \(string(file, "status"))")
                                .font(.caption).lineLimit(1)
                        }
                    }
                    Text("原始照片不会修改或覆盖。")
                        .font(.caption).foregroundStyle(.secondary)
                }
                if !error.isEmpty { Text(error).foregroundStyle(.red).font(.caption) }
            }
            .padding(16)
        }
        .background(Color(red: 0.965, green: 0.967, blue: 0.971))
    }

    private func editorCard<Content: View>(
        _ title: String,
        icon: String,
        @ViewBuilder content: () -> Content
    ) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            Label(title, systemImage: icon)
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(Color.primary)
            content()
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.white, in: RoundedRectangle(cornerRadius: 14))
        .overlay(RoundedRectangle(cornerRadius: 14).stroke(Color.black.opacity(0.07)))
    }

    private func slider(_ adjustment: Adjustment, basic: Bool) -> some View {
        let value = Binding<Double>(
            get: {
                let data = basic ? library.basicByPhoto[library.selectedID] : library.dehazeByPhoto[library.selectedID]
                return data?[adjustment.id] as? Double ?? 0
            },
            set: { newValue in
                let photoID = library.selectedID
                let clamped = min(adjustment.range.upperBound, max(adjustment.range.lowerBound, newValue))
                let value = (clamped / adjustment.step).rounded() * adjustment.step
                if basic { library.basicByPhoto[photoID, default: [:]][adjustment.id] = value }
                else { library.dehazeByPhoto[photoID, default: [:]][adjustment.id] = value }
                scheduleAutoSave(photoID: photoID)
                if sliderEditing { scheduleSliderPreview() }
                else { refreshPreview() }
            }
        )
        return VStack(alignment: .leading, spacing: 3) {
            HStack {
                Text(adjustment.title).font(.caption)
                Spacer()
                TextField(adjustment.title, value: value,
                          format: .number.precision(.fractionLength(adjustment.step < 1 ? 2 : 0)))
                    .labelsHidden()
                    .multilineTextAlignment(.trailing)
                    .font(.caption.monospacedDigit())
                    .frame(width: 58)
                    .disabled(library.selectedID.isEmpty)
            }
            Slider(value: value, in: adjustment.range, step: adjustment.step) { editing in
                sliderEditing = editing
                if !editing {
                    previewThrottleTask?.cancel()
                    previewThrottleTask = nil
                    previewRefinementTask?.cancel()
                    previewRefinementTask = nil
                    sliderPreviewPending = false
                    refreshPreview(previewLevel: 0)
                }
            }
            .disabled(library.selectedID.isEmpty)
        }
    }

    private func loadPresets() async {
        do {
            let result = try await engine.api.json("/api/ricoh/presets")
            guard !Task.isCancelled else { return }
            presets = result["presets"] as? [[String: Any]] ?? []
        } catch { showError(error) }
    }

    private func scheduleSliderPreview() {
        sliderPreviewPending = true
        if pendingPreviewLevel == 1 { pendingPreviewLevel = nil }
        previewRefinementTask?.cancel()
        previewRefinementTask = Task {
            do {
                try await Task.sleep(for: .milliseconds(250))
            } catch {
                return
            }
            guard !Task.isCancelled else { return }
            previewRefinementTask = nil
            guard sliderEditing else { return }
            refreshPreview(previewLevel: 1, invalidatingInFlight: false)
        }

        guard previewThrottleTask == nil else { return }
        previewThrottleTask = Task {
            do {
                try await Task.sleep(for: .milliseconds(80))
            } catch {
                return
            }
            guard !Task.isCancelled else { return }
            previewThrottleTask = nil
            guard sliderEditing, sliderPreviewPending else { return }
            sliderPreviewPending = false
            refreshPreview(previewLevel: 2, invalidatingInFlight: false)
        }
    }

    private func refreshPreview(previewLevel: Int = 0, invalidatingInFlight: Bool = true) {
        guard isActive, !library.sessionID.isEmpty, !library.selectedID.isEmpty, engine.ready else { return }
        if previewMode != .effect { loadOriginalPreviewIfNeeded() }
        if invalidatingInFlight || previewRequestID == nil {
            previewRequestID = UUID()
        }
        pendingPreviewLevel = previewLevel
        guard previewTask == nil else { return }

        let workerID = UUID()
        previewWorkerID = workerID
        previewTask = Task { await runPreviewQueue(workerID: workerID) }
    }

    private func loadOriginalPreviewIfNeeded() {
        guard original == nil, originalTask == nil,
              !library.sessionID.isEmpty, !library.selectedID.isEmpty, engine.ready else { return }
        let sessionID = library.sessionID
        let photoID = library.selectedID
        let requestID = UUID()
        originalRequestID = requestID
        originalLoading = true
        originalTask = Task {
            defer {
                if originalRequestID == requestID {
                    originalTask = nil
                    originalRequestID = nil
                    originalLoading = false
                }
            }
            do {
                let request: [String: Any] = [
                    "session_id": sessionID,
                    "photo_id": photoID,
                    "max_edge": 1800,
                    "color_manage_srgb": true,
                    "preview_level": 0,
                    "mode": "original",
                ]
                let image = try await engine.api.image("/api/enhance/preview", body: request)
                try Task.checkCancellation()
                guard library.sessionID == sessionID, library.selectedID == photoID,
                      originalRequestID == requestID else { return }
                original = image
            } catch {
                guard !Task.isCancelled, library.sessionID == sessionID,
                      library.selectedID == photoID, originalRequestID == requestID else { return }
                showError(error)
            }
        }
    }

    private func runPreviewQueue(workerID: UUID) async {
        previewing = true
        defer {
            if previewWorkerID == workerID {
                previewTask = nil
                previewWorkerID = nil
                previewing = false
            }
        }

        while !Task.isCancelled {
            guard let previewLevel = pendingPreviewLevel,
                  let requestID = previewRequestID else { break }
            pendingPreviewLevel = nil

            let photoID = library.selectedID
            let sessionID = library.sessionID
            let dehaze = library.dehazeByPhoto[photoID] ?? [:]
            let basic = library.basicByPhoto[photoID] ?? [:]
            let preset = library.presetByPhoto[photoID]
            error = ""
            do {
                let common: [String: Any] = ["session_id": sessionID, "photo_id": photoID, "max_edge": 1800,
                                             "color_manage_srgb": true, "preview_level": previewLevel]
                guard previewRequestID == requestID else { continue }
                var request = common
                request["mode"] = "dehazed"
                request["params"] = dehaze
                request["basic_params"] = basic
                request["ricoh_preset_id"] = preset as Any? ?? NSNull()
                request["render_backend"] = engine.renderBackend
                request["basic_backend"] = engine.basicBackend
                request["ricoh_backend"] = engine.ricohBackend
                let image = try await engine.api.image("/api/enhance/preview", body: request)
                try Task.checkCancellation()
                guard library.sessionID == sessionID, library.selectedID == photoID,
                      previewRequestID == requestID else { continue }
                edited = image
                if previewLevel == 2 && !sliderEditing && pendingPreviewLevel == nil {
                    pendingPreviewLevel = 0
                }
            } catch {
                // URLSession reports Task cancellation as URLError.cancelled on
                // some paths, instead of throwing CancellationError.
                if library.sessionID == sessionID, library.selectedID == photoID,
                   previewRequestID == requestID {
                    showError(error)
                }
            }
        }
    }

    private func cancelPreviewWork() {
        previewTask?.cancel()
        previewTask = nil
        previewWorkerID = nil
        pendingPreviewLevel = nil
        previewRequestID = nil
        previewing = false
        thumbnailTask?.cancel()
        thumbnailTask = nil
        thumbnailRequestID = nil
        thumbnailLoading = false
        originalTask?.cancel()
        originalTask = nil
        originalRequestID = nil
        originalLoading = false
    }

    private func isExpectedCancellation(_ error: Error) -> Bool {
        if error is CancellationError { return true }
        let nsError = error as NSError
        return nsError.domain == NSURLErrorDomain && nsError.code == NSURLErrorCancelled
    }

    private func showError(_ error: Error) {
        guard !isExpectedCancellation(error), !Task.isCancelled else { return }
        self.error = error.localizedDescription
    }

    private func scheduleAutoSave(photoID: String) {
        guard !photoID.isEmpty, !library.sessionID.isEmpty else { return }
        xmpSaveTasks[photoID]?.cancel()
        let sessionID = library.sessionID
        let saveID = UUID()
        xmpSaveIDs[photoID] = saveID
        xmpStatus = "XMP 待保存…"
        xmpSaveTasks[photoID] = Task {
            defer {
                if xmpSaveIDs[photoID] == saveID {
                    xmpSaveIDs[photoID] = nil
                    xmpSaveTasks[photoID] = nil
                }
            }
            do {
                try await Task.sleep(for: .milliseconds(600))
                guard library.sessionID == sessionID else { return }
                try await library.savePhoto(api: engine.api, photoID: photoID)
                guard !Task.isCancelled, library.sessionID == sessionID else { return }
                if library.selectedID == photoID { xmpStatus = "已自动写入 XMP" }
            } catch {
                guard !isExpectedCancellation(error), !Task.isCancelled else { return }
                if library.selectedID == photoID { xmpStatus = "XMP 自动保存失败：\(error.localizedDescription)" }
            }
        }
    }

    private func saveAllSettings() async throws {
        for photo in library.photos { try await library.savePhoto(api: engine.api, photoID: photo.id) }
    }

    private func saveXMP() async {
        do {
            try await saveAllSettings()
            let result = try await engine.api.json("/api/enhance/xmp", body: [
                "session_id": library.sessionID,
                "params_by_photo": library.dehazeByPhoto,
                "basic_params_by_photo": library.basicByPhoto,
                "preset_ids_by_photo": library.presetByPhoto,
            ])
            library.message = "XMP 已写入 \(int(result, "written")) 张，失败 \(int(result, "failed")) 张"
        } catch { showError(error) }
    }

    private func startJob() async {
        do {
            try await saveAllSettings()
            let body: [String: Any]
            let path: String
            if mode == .enhance {
                path = "/api/enhance/run"
                body = ["session_id": library.sessionID, "output_dir": library.dehazeOutput,
                        "params_by_photo": library.dehazeByPhoto,
                        "basic_params_by_photo": library.basicByPhoto,
                        "render_backend": engine.renderBackend, "basic_backend": engine.basicBackend]
            } else {
                path = "/api/ricoh/run"
                body = ["session_id": library.sessionID, "output_dir": library.ricohOutput,
                        "preset_id": selectedPreset, "preset_ids_by_photo": library.presetByPhoto,
                        "basic_params_by_photo": library.basicByPhoto, "ricoh_backend": engine.ricohBackend]
            }
            setJob(try await engine.api.json(path, body: body))
            pollTask?.cancel()
            pollTask = Task { await pollJob() }
        } catch { showError(error) }
    }

    private func pollJob() async {
        while !Task.isCancelled && isRunning {
            do {
                let path = mode == .enhance ? "/api/enhance/job/" : "/api/ricoh/job/"
                setJob(try await engine.api.json(path + string(job, "job_id")))
                if !isRunning { break }
                try await Task.sleep(for: .milliseconds(700))
            } catch is CancellationError { break }
            catch {
                if isExpectedCancellation(error) || Task.isCancelled { break }
                self.error = error.localizedDescription
                break
            }
        }
    }

    private func cancelJob() async {
        do {
            let path = mode == .enhance ? "/api/enhance/cancel/" : "/api/ricoh/cancel/"
            _ = try await engine.api.json(path + string(job, "job_id"), body: [:])
            library.message = "已通知后端停止；当前照片可完成"
        } catch { showError(error) }
    }

    private func setJob(_ value: [String: Any]) {
        if mode == .enhance { library.enhanceJob = value }
        else { library.ricohJob = value }
    }
}
