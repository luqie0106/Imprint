import AppKit
import Foundation
import ImageIO
import SwiftUI

private final class LineAccumulator: @unchecked Sendable {
    private let lock = NSLock()
    private var pending = Data()

    func append(_ data: Data) -> [Data] {
        lock.lock()
        defer { lock.unlock() }
        pending.append(data)
        var lines: [Data] = []
        while let newline = pending.firstIndex(of: 10) {
            lines.append(Data(pending.prefix(upTo: newline)))
            pending.removeSubrange(...newline)
        }
        return lines
    }
}

enum APIError: LocalizedError {
    case unavailable
    case invalidResponse
    case server(String)

    var errorDescription: String? {
        switch self {
        case .unavailable: "本地引擎尚未就绪"
        case .invalidResponse: "服务响应格式无效"
        case .server(let message): message
        }
    }
}

enum PreviewImageDecoder {
    @MainActor static func image(from data: Data) async throws -> NSImage {
        let cgImage = await Task.detached(priority: .userInitiated) {
            guard let source = CGImageSourceCreateWithData(data as CFData, nil) else { return nil as CGImage? }
            let options: [CFString: Any] = [
                kCGImageSourceCreateThumbnailFromImageAlways: true,
                kCGImageSourceCreateThumbnailWithTransform: true,
                kCGImageSourceThumbnailMaxPixelSize: 2048,
                kCGImageSourceShouldCacheImmediately: true,
            ]
            return CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary)
        }.value
        try Task.checkCancellation()
        guard let cgImage else { throw APIError.invalidResponse }
        return NSImage(cgImage: cgImage, size: NSSize(width: cgImage.width, height: cgImage.height))
    }
}

@MainActor final class Engine: ObservableObject {
    @Published var port: Int?
    @Published var status = "正在启动本地引擎…"
    @Published var error: String?
    @Published var renderBackend = UserDefaults.standard.string(forKey: "renderBackend") ?? "auto" {
        didSet { UserDefaults.standard.set(renderBackend, forKey: "renderBackend") }
    }
    @Published var basicBackend = UserDefaults.standard.string(forKey: "basicBackend") ?? "python" {
        didSet { UserDefaults.standard.set(basicBackend, forKey: "basicBackend") }
    }
    @Published var ricohBackend = UserDefaults.standard.string(forKey: "ricohBackend") ?? "python" {
        didSet { UserDefaults.standard.set(ricohBackend, forKey: "ricohBackend") }
    }
    @Published var sortBackend = UserDefaults.standard.string(forKey: "sortBackend") ?? "python" {
        didSet { UserDefaults.standard.set(sortBackend, forKey: "sortBackend") }
    }
    private var process: Process?
    private let endpoint = APIClient()

    var api: APIClient { endpoint }
    var ready: Bool { port != nil }

    func start() {
        if process != nil {
            if ready { return }
            stop()
        }
        let environment = ProcessInfo.processInfo.environment
        let current = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let candidates = [
            environment["IMPRINT_PROJECT_ROOT"].map { URL(fileURLWithPath: $0) },
            current,
            current.deletingLastPathComponent(),
            Bundle.main.resourceURL,
        ].compactMap { $0 }
        guard let root = candidates.first(where: {
            FileManager.default.fileExists(atPath: $0.appendingPathComponent("src/app_api.py").path)
        }) else {
            error = "找不到 src/app_api.py。请从项目目录运行，或设置 IMPRINT_PROJECT_ROOT。"
            status = "本地引擎离线"
            return
        }
        let pythonCandidates = [
            environment["IMPRINT_PYTHON"],
            environment["CONDA_PREFIX"].map { "\($0)/bin/python" },
            environment["VIRTUAL_ENV"].map { "\($0)/bin/python" },
            "\(NSHomeDirectory())/.conda/envs/py311/bin/python",
        ].compactMap { $0 }
        guard let python = pythonCandidates.first(where: { FileManager.default.isExecutableFile(atPath: $0) }) else {
            error = "找不到 Python 环境。请设置 IMPRINT_PYTHON 为项目的 Python 路径。"
            status = "本地引擎离线"
            return
        }
        let task = Process()
        task.executableURL = URL(fileURLWithPath: python)
        task.arguments = [root.appendingPathComponent("src/app_api.py").path]
        task.currentDirectoryURL = root
        task.environment = environment.merging(["PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"]) { _, new in new }
        let pipe = Pipe()
        task.standardOutput = pipe
        task.standardError = FileHandle.standardError
        let lines = LineAccumulator()
        pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty else { return }
            for line in lines.append(data) {
                guard let json = try? JSONSerialization.jsonObject(with: line) as? [String: Any],
                      let port = json["port"] as? Int else { continue }
                Task { @MainActor [weak self] in await self?.connect(port: port) }
            }
        }
        task.terminationHandler = { [weak self, weak task] _ in
            Task { @MainActor [weak self] in
                guard let task, self?.process === task else { return }
                self?.port = nil
                self?.api.baseURL = nil
                self?.process = nil
                self?.status = "本地引擎已停止"
            }
        }
        do {
            try task.run()
            process = task
            error = nil
        } catch {
            self.error = error.localizedDescription
            status = "本地引擎离线"
        }
    }

    private func connect(port: Int) async {
        endpoint.baseURL = URL(string: "http://127.0.0.1:\(port)")
        for _ in 0..<60 {
            if (try? await endpoint.json("/api/health")) != nil {
                self.port = port
                status = "本地引擎就绪"
                error = nil
                return
            }
            try? await Task.sleep(for: .milliseconds(250))
        }
        error = "服务启动后未能响应，请检查 Python 依赖。"
        status = "本地引擎离线"
    }

    func stop() {
        process?.terminate()
        process = nil
    }
}

@MainActor final class APIClient {
    var baseURL: URL?
    private let previewCache: NSCache<NSString, NSImage> = {
        let cache = NSCache<NSString, NSImage>()
        cache.totalCostLimit = 128 * 1024 * 1024
        return cache
    }()

    func url(_ path: String) throws -> URL {
        guard let baseURL, let url = URL(string: path, relativeTo: baseURL) else { throw APIError.unavailable }
        return url
    }

    func json(_ path: String, body: [String: Any]? = nil) async throws -> [String: Any] {
        var request = URLRequest(url: try url(path))
        if let body {
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.httpBody = try JSONSerialization.data(withJSONObject: body)
        }
        let (data, response) = try await URLSession.shared.data(for: request)
        let object = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
        guard let response = response as? HTTPURLResponse else { throw APIError.invalidResponse }
        guard (200..<300).contains(response.statusCode) else {
            throw APIError.server(object?["error"] as? String ?? "HTTP \(response.statusCode)")
        }
        guard let object else { throw APIError.invalidResponse }
        return object
    }

    func image(_ path: String, body: [String: Any]) async throws -> NSImage {
        let bodyData = try JSONSerialization.data(withJSONObject: body, options: [.sortedKeys])
        let cacheKey = NSString(string: "\(path):\(bodyData.base64EncodedString())")
        if let cached = previewCache.object(forKey: cacheKey) { return cached }
        var request = URLRequest(url: try url(path))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = bodyData
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let response = response as? HTTPURLResponse else { throw APIError.invalidResponse }
        guard (200..<300).contains(response.statusCode) else {
            let object = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
            throw APIError.server(object?["error"] as? String ?? "预览失败")
        }
        let image = try await PreviewImageDecoder.image(from: data)
        previewCache.setObject(image, forKey: cacheKey,
                               cost: Int(image.size.width * image.size.height * 4))
        return image
    }

    func events(_ path: String, body: [String: Any], onEvent: @escaping @MainActor ([String: Any]) -> Void) async throws {
        var request = URLRequest(url: try url(path))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        let (bytes, response) = try await URLSession.shared.bytes(for: request)
        guard let response = response as? HTTPURLResponse, (200..<300).contains(response.statusCode) else {
            throw APIError.invalidResponse
        }
        for try await line in bytes.lines {
            try Task.checkCancellation()
            guard line.hasPrefix("data: "),
                  let event = try? JSONSerialization.jsonObject(with: Data(line.dropFirst(6).utf8)) as? [String: Any] else { continue }
            onEvent(event)
            if ["done", "error"].contains(event["type"] as? String ?? "") { break }
        }
    }
}

enum FilePicker {
    @MainActor static func folder(title: String) -> String? {
        let panel = NSOpenPanel()
        panel.title = title
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        return panel.runModal() == .OK ? panel.url?.path : nil
    }

    @MainActor static func photos() -> [String] {
        let panel = NSOpenPanel()
        panel.title = "选择照片"
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = true
        return panel.runModal() == .OK ? panel.urls.map(\.path) : []
    }
}

func string(_ object: [String: Any], _ key: String) -> String { object[key] as? String ?? "" }
func int(_ object: [String: Any], _ key: String) -> Int { object[key] as? Int ?? 0 }

struct Photo: Identifiable {
    let id: String
    let name: String
    let dehaze: [String: Any]
    let basic: [String: Any]
    let preset: String?

    init?(_ object: [String: Any]) {
        guard let id = object["photo_id"] as? String else { return nil }
        self.id = id
        name = string(object, "name")
        dehaze = object["dehaze_params"] as? [String: Any] ?? [:]
        basic = object["basic_params"] as? [String: Any] ?? [:]
        preset = object["ricoh_preset_id"] as? String
    }
}

@MainActor final class PhotoLibrary: ObservableObject {
    @Published var sessionID = ""
    @Published var photos: [Photo] = []
    @Published var selectedID = ""
    @Published var dehazeOutput = ""
    @Published var ricohOutput = ""
    @Published var message = ""
    @Published var dehazeByPhoto: [String: [String: Any]] = [:]
    @Published var basicByPhoto: [String: [String: Any]] = [:]
    @Published var presetByPhoto: [String: String] = [:]
    @Published var enhanceJob: [String: Any] = [:]
    @Published var ricohJob: [String: Any] = [:]
    var hasActiveJob: Bool {
        [enhanceJob, ricohJob].contains { ["queued", "running"].contains(string($0, "status")) }
    }

    func load(api: APIClient, paths: [String]? = nil, folder: String? = nil) async {
        guard !hasActiveJob else {
            message = "导出任务仍在运行，请等待完成或先停止任务。"
            return
        }
        do {
            let result = try await api.json("/api/enhance/session", body: folder.map { ["input_dir": $0] } ?? ["paths": paths ?? []])
            sessionID = string(result, "session_id")
            photos = (result["files"] as? [[String: Any]] ?? []).compactMap(Photo.init)
            selectedID = photos.first?.id ?? ""
            dehazeOutput = string(result, "default_output_dir")
            ricohOutput = string(result, "ricoh_default_output_dir")
            dehazeByPhoto = Dictionary(uniqueKeysWithValues: photos.map { ($0.id, $0.dehaze) })
            basicByPhoto = Dictionary(uniqueKeysWithValues: photos.map { ($0.id, $0.basic) })
            presetByPhoto = Dictionary(uniqueKeysWithValues: photos.compactMap { photo in photo.preset.map { (photo.id, $0) } })
            enhanceJob = [:]
            ricohJob = [:]
            message = "已载入 \(photos.count) 张照片"
        } catch { message = error.localizedDescription }
    }

    func savePhoto(api: APIClient, photoID: String) async throws {
        _ = try await api.json("/api/photo/settings", body: [
            "session_id": sessionID,
            "photo_id": photoID,
            "dehaze_params": dehazeByPhoto[photoID] ?? [:],
            "basic_params": basicByPhoto[photoID] ?? [:],
            "ricoh_preset_id": presetByPhoto[photoID] as Any? ?? NSNull(),
        ])
    }
}

struct Panel<Content: View>: View {
    let title: String
    @ViewBuilder let content: Content
    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(title).font(.headline)
            content
        }
        .padding(18)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16))
    }
}

struct SectionTitle: View {
    let title: String
    let subtitle: String
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title).font(.largeTitle.bold())
            Text(subtitle).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}
