import SwiftUI

private enum Page: String, CaseIterable, Identifiable {
    case burst = "连拍优选"
    case enhance = "去朦胧"
    case ricoh = "理光风格"
    case models = "模型管理"
    case trainer = "偏好训练"
    case settings = "设置"

    var id: String { rawValue }
    var icon: String {
        switch self {
        case .burst: "square.stack.3d.up"
        case .enhance: "wand.and.stars"
        case .ricoh: "camera.filters"
        case .models: "shippingbox"
        case .trainer: "brain.head.profile"
        case .settings: "gearshape"
        }
    }
}

@main struct ImprintMacApp: App {
    @StateObject private var engine = Engine()
    @StateObject private var library = PhotoLibrary()
    @State private var selectedPage: Page? = .burst
    // Build both editor layouts once while the library is empty. Switching
    // between them then changes visibility instead of constructing a page.
    @State private var openedPages: Set<Page> = [.burst, .enhance, .ricoh]

    var body: some Scene {
        WindowGroup("Imprint") {
            NavigationSplitView {
                List(Page.allCases, selection: $selectedPage) { item in
                    Label(item.rawValue, systemImage: item.icon).tag(item)
                }
                .listStyle(.sidebar)
                .safeAreaInset(edge: .top) {
                    HStack(spacing: 10) {
                        Image(systemName: "square.stack.3d.up.fill")
                            .font(.title3).foregroundStyle(.blue)
                        Text("Imprint").font(.title3.bold())
                        Spacer()
                    }
                    .padding(.horizontal, 16)
                    .padding(.vertical, 14)
                }
                .safeAreaInset(edge: .bottom) {
                    HStack(spacing: 7) {
                        Circle().fill(engine.ready ? .green : .orange).frame(width: 8, height: 8)
                        Text(engine.status).font(.caption).lineLimit(1)
                    }
                    .padding(12)
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
            } detail: {
                Group {
                    if !engine.ready && selectedPage != .enhance && selectedPage != .ricoh {
                        ContentUnavailableView {
                            Label("本地引擎尚未就绪", systemImage: "externaldrive.badge.exclamationmark")
                        } description: {
                            Text(engine.error ?? engine.status)
                        } actions: {
                            Button("重试") { engine.start() }
                        }
                    } else {
                        ZStack {
                            ForEach(Page.allCases) { page in
                                if openedPages.contains(page) {
                                    pageView(page)
                                        .opacity((selectedPage ?? .burst) == page ? 1 : 0)
                                        .allowsHitTesting((selectedPage ?? .burst) == page)
                                        .accessibilityHidden((selectedPage ?? .burst) != page)
                                }
                            }
                        }
                        .animation(nil, value: selectedPage)
                    }
                }
                .frame(minWidth: 900, minHeight: 650)
            }
            .animation(nil, value: selectedPage)
            .environmentObject(engine)
            .environmentObject(library)
            .task { engine.start() }
            .onChange(of: selectedPage) { _, selection in
                let page = selection ?? .burst
                if !openedPages.contains(page) { openedPages.insert(page) }
            }
            .onReceive(NotificationCenter.default.publisher(for: NSApplication.willTerminateNotification)) { _ in
                engine.stop()
            }
        }
        .windowStyle(.titleBar)
        .commands {
            CommandGroup(after: .appInfo) {
                Button("重新连接本地引擎") { engine.start() }
            }
        }
    }

    @ViewBuilder private func pageView(_ page: Page) -> some View {
        switch page {
        case .burst: BurstView()
        case .enhance: EditorView(mode: .enhance, isActive: !library.photos.isEmpty && selectedPage == .enhance).equatable()
        case .ricoh: EditorView(mode: .ricoh, isActive: !library.photos.isEmpty && selectedPage == .ricoh).equatable()
        case .models: ModelsView()
        case .trainer: TrainerView()
        case .settings: SettingsView()
        }
    }
}
