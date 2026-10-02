// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "MeetingScribe",
    platforms: [.macOS(.v15)],
    products: [.executable(name: "MeetingScribe", targets: ["MeetingScribe"])],
    targets: [.executableTarget(name: "MeetingScribe")],
    swiftLanguageModes: [.v5]
)
