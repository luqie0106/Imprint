// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "ImprintMac",
    platforms: [.macOS(.v14)],
    products: [.executable(name: "ImprintMac", targets: ["ImprintMac"])],
    targets: [.executableTarget(name: "ImprintMac", path: "Sources/ImprintMac")]
)
