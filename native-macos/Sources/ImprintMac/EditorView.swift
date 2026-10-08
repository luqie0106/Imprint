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

private let nonlocalFallbackReasons: [String: String] = [
    "manual_mode": "自动处理未启用",
    "zero_strength": "去朦胧强度为 0",
    "low_airlight": "当前场景不满足实验估计条件",
    "uncertain_airlight": "场景估计不稳定",
    "clipped_highlights": "高光过曝",
    "backlit_scene": "逆光场景",
    "no_reliable_rays": "可靠估计不足",
    "solver_nonconverged": "可靠估计不足",
    "reliable_estimate_unavailable": "可靠估计不足",
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
    private enum ExportScope: Equatable {
        case current
        case all

        var title: String { self == .current ? "当前照片" : "全部照片" }
    }

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
    @State private var originalFullResolution = false
    @State private var originalTaskFullResolution = false
    @State private var originalLoading = false
    @State private var previewing = false
    @State private var previewMode: PreviewMode = .effect
    @State private var nonlocalRenderStatus = "off"
    @State private var nonlocalRenderReason = ""
    @State private var renderedAutoExposureEnabled: Bool?
    @State private var autoExposureEV: Double?
    @State private var autoExposureReason = "off"
    @State private var compareFraction = 0.5
    @State private var metadata: [String: Any] = [:]
    @State private var error = ""
    @State private var pollTask: Task<Void, Never>?
    @State private var previewTask: Task<Void, Never>?
    @State private var previewWorkerID: UUID?
    @State private var pendingPreviewLevel: Int?
    @State private var pendingPreviewFullResolution = false
    @State private var previewThrottleTask: Task<Void, Never>?
    @State private var previewRefinementTask: Task<Void, Never>?
    @State private var previewIdleTask: Task<Void, Never>?
    @State private var previewIdleRequestID: UUID?
    @State private var sliderPreviewPending = false
    @State private var sliderEditing = false
    @State private var presetTask: Task<Void, Never>?
    @State private var previewRequestID: UUID?
    @State private var metadataTask: Task<Void, Never>?
    @State private var xmpSaveTasks: [String: Task<Void, Never>] = [:]
    @State private var xmpSaveIDs: [String: UUID] = [:]
    @State private var xmpStatus = ""
    @State private var showingExportSheet = false
    @State private var exportScope: ExportScope = .current
    @State private var exportSessionID = ""
    @State private var exportPhotoIDs: [String] = []
    @State private var exportOutputDir = ""
    @State private var exportBitDepth = "source"
    @State private var exportCompression = "lossless_jpeg"
    @State private var isStartingJob = false

    private var isRunning: Bool {
        string(job, "job_id") != "starting" && ["queued", "running"].contains(string(job, "status"))
    }
    private var job: [String: Any] { library.enhanceJob }
    private var title: String { mode == .enhance ? "去朦胧" : "理光风格" }
    private var selected: Photo? { library.photos.first { $0.id == library.selectedID } }
    private var selectedPreset: String { library.presetByPhoto[library.selectedID] ?? "" }
    private var nonlocalRenderStatusText: String? {
        guard mode == .enhance,
              (library.dehazeNonlocalModeByPhoto[library.selectedID] ?? "off") != "off",
              nonlocalRenderStatus != "off" else { return nil }
        if nonlocalRenderStatus == "active" { return "实际已启用" }
        let reason = nonlocalFallbackReasons[nonlocalRenderReason] ?? "可靠估计不足"
        return "已使用原去朦胧：\(reason)"
    }
    private var autoExposureStatusText: String? {
        guard mode == .enhance,
              library.dehazeAutoExposureByPhoto[library.selectedID] ?? false else { return nil }
        guard let renderedAutoExposureEnabled,
              renderedAutoExposureEnabled == (library.dehazeAutoExposureByPhoto[library.selectedID] ?? false) else {
            return "等待预览更新…"
        }
        let ev = autoExposureEV.map { String(format: "%+.2f EV", $0) } ?? "EV 未返回"
        switch autoExposureReason {
        case "target_reached": return "无需曝光补偿（\(ev)）"
        case "highlight_limited": return "高光余量不足（实际 \(ev)）"
        case "applied": return "实际 \(ev)"
        case "black": return "无法估计整体曝光（+0EV）"
        case "off": return "未进行自动曝光校正（\(ev)）"
        default: return "自动曝光状态不可用（\(ev)）"
        }
    }

    private func selectionDidChange() {
        cancelPreviewWork()
        previewThrottleTask?.cancel()
        previewThrottleTask = nil
        previewRefinementTask?.cancel()
        previewRefinementTask = nil
        sliderPreviewPending = false
        sliderEditing = false
        original = nil
        originalFullResolution = false
        edited = nil
        nonlocalRenderStatus = "off"
        nonlocalRenderReason = ""
        renderedAutoExposureEnabled = nil
        autoExposureEV = nil
        autoExposureReason = "off"
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
            if newMode != .effect {
                loadOriginalPreviewIfNeeded()
                scheduleFullResolutionPreview()
            }
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
            previewIdleTask?.cancel()
            previewIdleTask = nil
            previewIdleRequestID = nil
            presetTask?.cancel()
            metadataTask?.cancel()
            pollTask?.cancel()
        }
        .sheet(isPresented: $showingExportSheet) {
            exportConfigurationSheet
                .interactiveDismissDisabled(isStartingJob)
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
                Menu {
                    Button("当前照片") { prepareExport(scope: .current) }
                        .disabled(selected == nil)
                    Button("全部照片") { prepareExport(scope: .all) }
                } label: {
                    Image(systemName: "square.and.arrow.down")
                }
                .help("导出 DNG")
                .disabled(!engine.ready || library.sessionID.isEmpty || library.photos.isEmpty
                          || library.hasActiveJob || isStartingJob)
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
                if !xmpStatus.isEmpty {
                    Text(xmpStatus).font(.caption2).foregroundStyle(.secondary)
                }
                if !job.isEmpty, string(job, "job_id") != "starting" {
                    exportProgressView
                }
                if mode == .enhance {
                    editorCard("去朦胧", icon: "wand.and.stars") {
                        Toggle("自动处理", isOn: Binding(
                            get: { library.dehazeAutoModeByPhoto[library.selectedID] ?? false },
                            set: { enabled in
                                let photoID = library.selectedID
                                library.dehazeAutoModeByPhoto[photoID] = enabled
                                if !enabled {
                                    library.dehazeNonlocalModeByPhoto[photoID] = "off"
                                }
                                scheduleAutoSave(photoID: photoID)
                                refreshPreview()
                            }
                        ))
                        .disabled(library.selectedID.isEmpty)
                        if library.dehazeAutoModeByPhoto[library.selectedID] ?? false {
                            Text("自动分析照片并调整去朦胧效果。关闭后可手动微调高级参数。")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                        Toggle("自动曝光", isOn: Binding(
                            get: { library.dehazeAutoExposureByPhoto[library.selectedID] ?? false },
                            set: { enabled in
                                let photoID = library.selectedID
                                guard !photoID.isEmpty else { return }
                                library.dehazeAutoExposureByPhoto[photoID] = enabled
                                scheduleAutoSave(photoID: photoID)
                                refreshPreview()
                            }
                        ))
                        .disabled(library.selectedID.isEmpty)
                        if let autoExposureStatusText {
                            Text(autoExposureStatusText)
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                        VStack(alignment: .leading, spacing: 6) {
                            Text("实验去朦胧").font(.caption)
                            Picker("实验去朦胧", selection: Binding(
                                get: { library.dehazeNonlocalModeByPhoto[library.selectedID] ?? "off" },
                                set: { nonlocalMode in
                                    let photoID = library.selectedID
                                    guard !photoID.isEmpty else { return }
                                    library.dehazeNonlocalModeByPhoto[photoID] = nonlocalMode
                                    if nonlocalMode != "off" {
                                        library.dehazeAutoModeByPhoto[photoID] = true
                                    }
                                    scheduleAutoSave(photoID: photoID)
                                    refreshPreview()
                                }
                            )) {
                                Text("关闭").tag("off")
                                Text("保守").tag("conservative")
                                Text("强").tag("strong")
                            }
                            .pickerStyle(.segmented)
                            .labelsHidden()
                            .disabled(library.selectedID.isEmpty)
                            if (library.dehazeNonlocalModeByPhoto[library.selectedID] ?? "off") != "off" {
                                Text("实验模式会开启自动处理")
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                                if let nonlocalRenderStatusText {
                                    Text(nonlocalRenderStatusText)
                                        .font(.caption2)
                                        .foregroundStyle(.secondary)
                                }
                            }
                        }
                        slider(dehazeAdjustments[0], basic: false)
                        if !(library.dehazeAutoModeByPhoto[library.selectedID] ?? false) {
                            EditorDisclosure(title: "高级参数") {
                                VStack(spacing: 12) {
                                    ForEach(dehazeAdjustments.dropFirst()) { adjustment in
                                        slider(adjustment, basic: false)
                                    }
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
                            Text("无").tag("")
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
                if !error.isEmpty { Text(error).foregroundStyle(.red).font(.caption) }
            }
            .padding(16)
        }
        .background(Color(red: 0.965, green: 0.967, blue: 0.971))
    }

    private var exportProgressView: some View {
        VStack(alignment: .leading, spacing: 7) {
            HStack {
                Text("DNG 导出").font(.caption.weight(.semibold))
                Spacer()
                if isRunning {
                    Button("停止") { Task { await cancelJob() } }
                        .controlSize(.small)
                        .disabled(isStartingJob)
                }
            }
            ProgressView(value: min(1, max(0, job["progress"] as? Double ?? 0)))
            Text("\(int(job, "processed"))/\(int(job, "total")) · 成功 \(int(job, "success")) · 失败 \(int(job, "failed"))")
                .font(.caption2).foregroundStyle(.secondary)
            if !string(job, "current_file").isEmpty {
                Text(string(job, "current_file")).font(.caption2).lineLimit(1)
            }
            ForEach(Array((job["files"] as? [[String: Any]] ?? []).enumerated()), id: \.offset) { _, file in
                Text("\(string(file, "name")) · \(string(file, "status"))")
                    .font(.caption2).lineLimit(1)
            }
        }
        .padding(.vertical, 2)
    }

    private var exportConfigurationSheet: some View {
        VStack(alignment: .leading, spacing: 18) {
            HStack {
                VStack(alignment: .leading, spacing: 3) {
                    Text("导出 DNG").font(.title3.bold())
                    Text(exportScope.title).font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
                Button {
                    showingExportSheet = false
                } label: {
                    Image(systemName: "xmark")
                }
                .buttonStyle(.plain)
                .disabled(isStartingJob)
            }

            Form {
                Section("输出") {
                    HStack {
                        Text("格式")
                        Spacer()
                        Text("DNG（固定）").foregroundStyle(.secondary)
                    }
                    VStack(alignment: .leading, spacing: 8) {
                        TextField("输出目录", text: $exportOutputDir)
                        Button("选择文件夹") {
                            if let folder = FilePicker.folder(title: "选择 DNG 输出文件夹") {
                                exportOutputDir = folder
                            }
                        }
                    }
                }
                Section("图像") {
                    Picker("位深", selection: $exportBitDepth) {
                        Text("源文件位深").tag("source")
                        Text("16 bit").tag("16")
                    }
                    .disabled(exportCompression == "jpegxl")
                    Picker("压缩", selection: $exportCompression) {
                        Text("JPEG XL").tag("jpegxl")
                        Text("无损 JPEG").tag("lossless_jpeg")
                        Text("不压缩").tag("none")
                    }
                    if exportCompression == "jpegxl" {
                        Text("JPEG XL 使用 16-bit RGB，并输出 DNG 1.7。")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .formStyle(.grouped)
            .onChange(of: exportCompression) { _, compression in
                if compression == "jpegxl" { exportBitDepth = "16" }
            }
            .disabled(isStartingJob)
            if !error.isEmpty {
                Text(error).font(.caption).foregroundStyle(.red)
            }

            HStack {
                Button("取消") { showingExportSheet = false }
                    .disabled(isStartingJob)
                Spacer()
                Button {
                    Task { await startJob() }
                } label: {
                    if isStartingJob {
                        ProgressView().controlSize(.small)
                    } else {
                        Text("开始导出")
                    }
                }
                .buttonStyle(.borderedProminent)
                .disabled(isStartingJob || library.hasActiveJob || exportPhotoIDs.isEmpty
                          || exportOutputDir.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }
        .padding(20)
        .frame(width: 480, height: 430)
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
                if basic { return data?[adjustment.id] as? Double ?? 0 }
                return data?[adjustment.id] as? Double
                    ?? library.dehazeDefaults[adjustment.id] as? Double
                    ?? 0
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
        // Invalidate any response rendered from an older slider value immediately,
        // while keeping the throttled L2/L1 requests on the existing fast path.
        previewRequestID = UUID()
        nonlocalRenderStatus = "off"
        nonlocalRenderReason = ""
        pendingPreviewLevel = nil
        pendingPreviewFullResolution = false
        sliderPreviewPending = true
        scheduleFullResolutionPreview()
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
            refreshPreview(previewLevel: 1, invalidatingInFlight: false, schedulesFullResolution: false)
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
            refreshPreview(previewLevel: 2, invalidatingInFlight: false, schedulesFullResolution: false)
        }
    }

    private func refreshPreview(
        previewLevel: Int = 0,
        invalidatingInFlight: Bool = true,
        fullResolution: Bool = false,
        schedulesFullResolution: Bool = true
    ) {
        nonlocalRenderStatus = "off"
        nonlocalRenderReason = ""
        guard isActive, !library.sessionID.isEmpty, !library.selectedID.isEmpty, engine.ready else { return }
        if previewMode != .effect && !fullResolution { loadOriginalPreviewIfNeeded() }
        if invalidatingInFlight || previewRequestID == nil {
            previewRequestID = UUID()
        }
        pendingPreviewLevel = fullResolution ? 0 : previewLevel
        pendingPreviewFullResolution = fullResolution
        if schedulesFullResolution && !fullResolution { scheduleFullResolutionPreview() }
        guard previewTask == nil else { return }

        let workerID = UUID()
        previewWorkerID = workerID
        previewTask = Task { await runPreviewQueue(workerID: workerID) }
    }

    private func loadOriginalPreviewIfNeeded(fullResolution: Bool = false) {
        guard isActive, !library.sessionID.isEmpty, !library.selectedID.isEmpty, engine.ready else { return }
        if fullResolution && originalFullResolution { return }
        if originalTask != nil {
            if !fullResolution || originalTaskFullResolution { return }
            originalTask?.cancel()
            originalTask = nil
            originalRequestID = nil
            originalLoading = false
        }
        if !fullResolution && original != nil { return }
        let sessionID = library.sessionID
        let photoID = library.selectedID
        let requestID = UUID()
        originalRequestID = requestID
        originalTaskFullResolution = fullResolution
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
                    "full_resolution": fullResolution,
                    "mode": "original",
                    "auto_mode": library.dehazeAutoModeByPhoto[photoID] ?? false,
                    "auto_exposure": library.dehazeAutoExposureByPhoto[photoID] ?? false,
                    "nonlocal_mode": library.dehazeNonlocalModeByPhoto[photoID] ?? "off",
                ]
                let image = try await engine.api.image("/api/enhance/preview", body: request)
                try Task.checkCancellation()
                guard library.sessionID == sessionID, library.selectedID == photoID,
                      originalRequestID == requestID else { return }
                original = image
                originalFullResolution = fullResolution
            } catch {
                guard !Task.isCancelled, library.sessionID == sessionID,
                      library.selectedID == photoID, originalRequestID == requestID else { return }
                showError(error)
            }
        }
    }

    private func scheduleFullResolutionPreview() {
        previewIdleTask?.cancel()
        let idleID = UUID()
        previewIdleRequestID = idleID
        let sessionID = library.sessionID
        let photoID = library.selectedID
        let requestID = previewRequestID
        previewIdleTask = Task {
            do {
                try await Task.sleep(for: .milliseconds(500))
            } catch {
                return
            }
            guard !Task.isCancelled, previewIdleRequestID == idleID,
                  previewRequestID == requestID,
                  library.sessionID == sessionID, library.selectedID == photoID,
                  isActive else { return }
            previewIdleTask = nil
            previewIdleRequestID = nil
            refreshPreview(
                previewLevel: 0,
                invalidatingInFlight: false,
                fullResolution: true,
                schedulesFullResolution: false
            )
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
            let fullResolution = pendingPreviewFullResolution
            pendingPreviewLevel = nil
            pendingPreviewFullResolution = false

            let photoID = library.selectedID
            let sessionID = library.sessionID
            let dehaze = library.dehazeByPhoto[photoID] ?? [:]
            let autoMode = library.dehazeAutoModeByPhoto[photoID] ?? false
            let autoExposure = library.dehazeAutoExposureByPhoto[photoID] ?? false
            let nonlocalMode = library.dehazeNonlocalModeByPhoto[photoID] ?? "off"
            let basic = library.basicByPhoto[photoID] ?? [:]
            let preset = library.presetByPhoto[photoID]
            error = ""
            do {
                let common: [String: Any] = ["session_id": sessionID, "photo_id": photoID, "max_edge": 1800,
                                             "color_manage_srgb": true,
                                             "preview_level": fullResolution ? 0 : previewLevel,
                                             "full_resolution": fullResolution]
                guard previewRequestID == requestID else { continue }
                var request = common
                request["mode"] = "dehazed"
                request["params"] = dehaze
                request["auto_mode"] = autoMode
                request["auto_exposure"] = autoExposure
                request["nonlocal_mode"] = nonlocalMode
                request["basic_params"] = basic
                request["ricoh_preset_id"] = preset as Any? ?? NSNull()
                request["render_backend"] = engine.renderBackend
                request["basic_backend"] = engine.basicBackend
                request["ricoh_backend"] = engine.ricohBackend
                let preview = try await engine.api.imageWithResponseMetadata(
                    "/api/enhance/preview", body: request
                )
                try Task.checkCancellation()
                guard library.sessionID == sessionID, library.selectedID == photoID,
                      previewRequestID == requestID else { continue }
                edited = preview.image
                nonlocalRenderStatus = preview.headers["x-dehaze-nonlocal-status"] ?? "off"
                nonlocalRenderReason = preview.headers["x-dehaze-nonlocal-reason"] ?? ""
                renderedAutoExposureEnabled = autoExposure
                autoExposureEV = preview.headers["x-auto-exposure-ev"].flatMap(Double.init)
                autoExposureReason = preview.headers["x-auto-exposure-reason"] ?? "off"
                // Serialize full-resolution decodes so the source and processed
                // image do not compete for peak memory on large RAW files.
                if fullResolution && previewMode != .effect {
                    loadOriginalPreviewIfNeeded(fullResolution: true)
                }
                if previewLevel == 2 && !sliderEditing && pendingPreviewLevel == nil {
                    pendingPreviewLevel = 0
                    pendingPreviewFullResolution = false
                }
            } catch {
                // URLSession reports Task cancellation as URLError.cancelled on
                // some paths, instead of throwing CancellationError.
                if library.sessionID == sessionID, library.selectedID == photoID,
                   previewRequestID == requestID {
                    showError(error)
                    if fullResolution && previewMode != .effect {
                        loadOriginalPreviewIfNeeded(fullResolution: true)
                    }
                }
            }
        }
    }

    private func cancelPreviewWork() {
        previewTask?.cancel()
        previewTask = nil
        previewWorkerID = nil
        pendingPreviewLevel = nil
        pendingPreviewFullResolution = false
        previewRequestID = nil
        previewing = false
        nonlocalRenderStatus = "off"
        nonlocalRenderReason = ""
        previewIdleTask?.cancel()
        previewIdleTask = nil
        previewIdleRequestID = nil
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

    private func prepareExport(scope: ExportScope) {
        guard !library.sessionID.isEmpty, !library.photos.isEmpty,
              !library.hasActiveJob, !isStartingJob else { return }
        let photoIDs = scope == .current
            ? [library.selectedID].filter { !$0.isEmpty }
            : library.photos.map(\.id)
        guard !photoIDs.isEmpty else { return }
        exportScope = scope
        exportSessionID = library.sessionID
        exportPhotoIDs = photoIDs
        exportOutputDir = mode == .enhance ? library.dehazeOutput : library.ricohOutput
        error = ""
        showingExportSheet = true
    }

    private func saveAllSettings(photoIDs: [String], sessionID: String) async throws {
        for photoID in photoIDs {
            if let pendingSave = xmpSaveTasks[photoID] {
                await pendingSave.value
            }
            guard library.sessionID == sessionID else {
                throw NSError(domain: "Imprint", code: 1,
                              userInfo: [NSLocalizedDescriptionKey: "照片会话已更改，请重新开始导出"])
            }
            try await library.savePhoto(api: engine.api, photoID: photoID)
        }
    }

    private func startJob() async {
        guard !isStartingJob, !library.hasActiveJob,
              !exportSessionID.isEmpty, !exportPhotoIDs.isEmpty else { return }
        let sessionID = exportSessionID
        let targetPhotoIDs = exportPhotoIDs
        let allPhotoIDs = library.photos.map(\.id)
        guard library.sessionID == sessionID,
              Set(targetPhotoIDs).isSubset(of: Set(allPhotoIDs)) else {
            error = "照片会话已更改，请重新选择导出范围"
            showingExportSheet = false
            return
        }

        isStartingJob = true
        error = ""
        // The library refuses to load another session while a queued/running job exists.
        // Register the startup window there too, before the first await.
        setJob(["job_id": "starting", "status": "queued", "progress": 0.0,
                "processed": 0, "total": targetPhotoIDs.count, "success": 0, "failed": 0])
        var didStart = false
        defer {
            if !didStart, string(job, "job_id") == "starting" { setJob([:]) }
            isStartingJob = false
        }

        do {
            try await saveAllSettings(photoIDs: allPhotoIDs, sessionID: sessionID)
            guard library.sessionID == sessionID,
                  Set(targetPhotoIDs).isSubset(of: Set(library.photos.map(\.id))) else {
                throw NSError(domain: "Imprint", code: 1,
                              userInfo: [NSLocalizedDescriptionKey: "照片会话已更改，请重新开始导出"])
            }

            var presetIDsByPhoto: [String: Any] = [:]
            for photoID in allPhotoIDs {
                presetIDsByPhoto[photoID] = library.presetByPhoto[photoID] as Any? ?? NSNull()
            }
            let photoIDsPayload: Any
            if exportScope == .all { photoIDsPayload = NSNull() }
            else { photoIDsPayload = targetPhotoIDs }
            let outputDir = exportOutputDir.trimmingCharacters(in: .whitespacesAndNewlines)
            let body: [String: Any] = [
                "session_id": sessionID,
                "output_dir": outputDir,
                "photo_ids": photoIDsPayload,
                "params_by_photo": library.dehazeByPhoto,
                "auto_modes_by_photo": library.dehazeAutoModeByPhoto,
                "auto_exposures_by_photo": library.dehazeAutoExposureByPhoto,
                "nonlocal_modes_by_photo": library.dehazeNonlocalModeByPhoto,
                "basic_params_by_photo": library.basicByPhoto,
                "preset_ids_by_photo": presetIDsByPhoto,
                "render_backend": engine.renderBackend,
                "basic_backend": engine.basicBackend,
                "ricoh_backend": engine.ricohBackend,
                "compression": exportCompression,
                "bit_depth": exportBitDepth,
            ]
            if mode == .enhance { library.dehazeOutput = outputDir }
            else { library.ricohOutput = outputDir }
            let result = try await engine.api.json("/api/enhance/run", body: body)
            guard library.sessionID == sessionID else {
                throw NSError(domain: "Imprint", code: 1,
                              userInfo: [NSLocalizedDescriptionKey: "导出已提交，但照片会话已更改"])
            }
            setJob(result)
            didStart = true
            showingExportSheet = false
            pollTask?.cancel()
            pollTask = Task { await pollJob() }
        } catch {
            showError(error)
        }
    }

    private func pollJob() async {
        while !Task.isCancelled && isRunning {
            do {
                setJob(try await engine.api.json("/api/enhance/job/" + string(job, "job_id")))
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
            _ = try await engine.api.json("/api/enhance/cancel/" + string(job, "job_id"), body: [:])
            library.message = "已通知后端停止；当前照片可完成"
        } catch { showError(error) }
    }

    private func setJob(_ value: [String: Any]) {
        library.enhanceJob = value
    }
}
