import AppKit
import SwiftUI

@MainActor
struct PhotoThumbnail: View {
    @EnvironmentObject private var engine: Engine

    let sessionID: String
    let photoID: String
    var size: CGFloat = 52

    @State private var thumbnail: NSImage?
    @State private var thumbnailKey: String?
    @State private var failedKey: String?

    private static let cache: NSCache<NSString, NSImage> = {
        let cache = NSCache<NSString, NSImage>()
        cache.countLimit = 96
        cache.totalCostLimit = 96 * 1024 * 1024
        return cache
    }()

    private var cacheKey: String { Self.key(sessionID: sessionID, photoID: photoID, maxEdge: 360) }

    private static func key(sessionID: String, photoID: String, maxEdge: Int) -> String {
        "\(sessionID)/\(photoID)/\(maxEdge)"
    }

    static func cachedImage(sessionID: String, photoID: String, maxEdge: Int = 360) -> NSImage? {
        guard isSafeID(sessionID), isSafeID(photoID) else { return nil }
        return cache.object(forKey: NSString(string: key(sessionID: sessionID, photoID: photoID, maxEdge: maxEdge)))
    }

    static func loadImage(sessionID: String, photoID: String, api: APIClient, maxEdge: Int = 360) async throws -> NSImage {
        guard isSafeID(sessionID), isSafeID(photoID) else {
            throw APIError.invalidResponse
        }
        guard (160...1800).contains(maxEdge) else { throw APIError.invalidResponse }
        let key = key(sessionID: sessionID, photoID: photoID, maxEdge: maxEdge)
        if let cached = cache.object(forKey: NSString(string: key)) {
            try Task.checkCancellation()
            return cached
        }

        let url = try api.url("/api/enhance/thumbnail/\(sessionID)/\(photoID)?max_edge=\(maxEdge)")
        let (data, response) = try await URLSession.shared.data(from: url)
        try Task.checkCancellation()
        guard let response = response as? HTTPURLResponse else {
            throw APIError.invalidResponse
        }
        guard (200..<300).contains(response.statusCode) else {
            throw APIError.server("缩略图暂不可用")
        }

        let decoded = try await PreviewImageDecoder.image(from: data)
        try Task.checkCancellation()
        let cost = max(0, Int(decoded.size.width * decoded.size.height * 4))
        cache.setObject(decoded, forKey: NSString(string: key), cost: cost)
        return decoded
    }

    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 6)
                .fill(Color.secondary.opacity(0.12))

            if let thumbnail, thumbnailKey == cacheKey {
                Image(nsImage: thumbnail)
                    .resizable()
                    .scaledToFill()
            } else {
                Image(systemName: failedKey == cacheKey ? "exclamationmark.triangle" : "photo")
                    .font(.system(size: min(size * 0.38, 20)))
                    .foregroundStyle(.secondary)
            }
        }
        .frame(width: size, height: size)
        .clipShape(RoundedRectangle(cornerRadius: 6))
        .accessibilityLabel(failedKey == cacheKey ? "缩略图载入失败" : "照片缩略图")
        .task(id: cacheKey) {
            await loadThumbnail(for: cacheKey)
        }
    }

    private func loadThumbnail(for requestKey: String) async {
        thumbnail = nil
        thumbnailKey = nil
        failedKey = nil

        if let cached = Self.cachedImage(sessionID: sessionID, photoID: photoID) {
            guard !Task.isCancelled, cacheKey == requestKey else { return }
            thumbnail = cached
            thumbnailKey = requestKey
            return
        }

        do {
            let decoded = try await Self.loadImage(sessionID: sessionID, photoID: photoID, api: engine.api)
            try Task.checkCancellation()
            guard cacheKey == requestKey else { return }
            thumbnail = decoded
            thumbnailKey = requestKey
        } catch is CancellationError {
            // A reused List row may already be loading a different photo.
        } catch {
            guard !Task.isCancelled else { return }
            failedKey = requestKey
        }
    }

    private static func isSafeID(_ value: String) -> Bool {
        !value.isEmpty && value.rangeOfCharacter(
            from: CharacterSet(charactersIn: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_").inverted
        ) == nil
    }
}
