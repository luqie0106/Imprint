import Darwin
import Dispatch
import Foundation

/// 每个客户端实例只管理自己启动的后端，并在确认退出前保留进程所有权。
final class BackendProcess: @unchecked Sendable {
    private final class OutputReader: @unchecked Sendable {
        let pipe = Pipe()
        private let lock = NSLock()
        private var closed = false

        func read(_ handle: FileHandle) -> Data? {
            lock.lock()
            defer { lock.unlock() }
            guard !closed else { return nil }
            let data = handle.availableData
            if data.isEmpty { handle.readabilityHandler = nil }
            return data.isEmpty ? nil : data
        }

        func close() {
            pipe.fileHandleForReading.readabilityHandler = nil
            lock.lock()
            defer { lock.unlock() }
            guard !closed else { return }
            closed = true
            try? pipe.fileHandleForReading.close()
            try? pipe.fileHandleForWriting.close()
        }
    }

    private struct Child {
        let id: UUID
        let process: Process
        let output: OutputReader
        let exited: DispatchSemaphore
    }

    private let lock = NSRecursiveLock()
    private var child: Child?
    private let gracePeriod: TimeInterval

    init(gracePeriod: TimeInterval = 1.5) {
        self.gracePeriod = max(0, gracePeriod)
    }

    var isRunning: Bool {
        lock.lock()
        defer { lock.unlock() }
        return child?.process.isRunning ?? false
    }

    func isCurrent(_ id: UUID) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return child?.id == id && child?.process.isRunning == true
    }

    @discardableResult
    func start(executable: URL, arguments: [String], directory: URL? = nil,
               environment: [String: String]? = nil,
               onOutput: @escaping @Sendable (UUID, Data) -> Void,
               onExit: @escaping @Sendable (UUID, Int32) -> Void) throws -> UUID {
        lock.lock()
        defer { lock.unlock() }
        stop()

        let id = UUID()
        let process = Process()
        let output = OutputReader()
        let exited = DispatchSemaphore(value: 0)
        process.executableURL = executable
        process.arguments = arguments
        process.currentDirectoryURL = directory
        process.environment = environment
        process.standardOutput = output.pipe
        process.standardError = FileHandle.standardError
        output.pipe.fileHandleForReading.readabilityHandler = { [weak output] handle in
            if let data = output?.read(handle) { onOutput(id, data) }
        }
        process.terminationHandler = { process in
            exited.signal()
            onExit(id, process.terminationStatus)
        }
        child = Child(id: id, process: process, output: output, exited: exited)
        do {
            try process.run()
            try? output.pipe.fileHandleForWriting.close()
            return id
        } catch {
            output.close()
            child = nil
            throw error
        }
    }

    /// 过期启动任务不能停止下一次启动的后端。
    @discardableResult
    func stop(ifCurrent id: UUID? = nil) -> Int32? {
        lock.lock()
        defer { lock.unlock() }
        guard let child, id == nil || child.id == id else { return nil }
        child.output.pipe.fileHandleForReading.readabilityHandler = nil
        if child.process.isRunning {
            child.process.terminate()
            if child.exited.wait(timeout: .now() + gracePeriod) == .timedOut,
               child.process.isRunning {
                // 只终止当前仍由本实例持有的子进程，不使用进程名或进程组。
                Darwin.kill(child.process.processIdentifier, SIGKILL)
            }
        }
        child.process.waitUntilExit()
        child.output.close()
        self.child = nil
        return child.process.terminationStatus
    }

    deinit { stop() }
}

/// POSIX handler 只接收信号，实际清理由 Dispatch 队列执行。
final class BackendSignalHandler {
    private let sources: [DispatchSourceSignal]
    private let previousHandlers: [(Int32, sig_t?)]

    init(backend: BackendProcess, onSignal: @escaping @Sendable (Int32) -> Void) {
        let queue = DispatchQueue(label: "imprint.backend.signals")
        var handlers: [(Int32, sig_t?)] = []
        var sources: [DispatchSourceSignal] = []
        for number in [SIGINT, SIGTERM] {
            // 空 handler 在 exec 时恢复默认行为，避免后端继承 SIG_IGN。
            let previous: sig_t? = Darwin.signal(number, { _ in })
            handlers.append((number, previous))
            let source = DispatchSource.makeSignalSource(signal: number, queue: queue)
            source.setEventHandler {
                backend.stop()
                onSignal(number)
            }
            source.resume()
            sources.append(source)
        }
        self.sources = sources
        previousHandlers = handlers
    }

    deinit {
        for source in sources { source.cancel() }
        for (number, handler) in previousHandlers { Darwin.signal(number, handler) }
    }
}
