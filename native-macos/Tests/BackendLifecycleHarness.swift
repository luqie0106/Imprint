import Darwin
import Foundation

@main struct BackendLifecycleHarness {
    static func main() throws {
        let arguments = CommandLine.arguments
        let mode = arguments[1]
        let python = arguments[2]
        let script = arguments[3]
        let backend = BackendProcess(gracePeriod: 0.2)
        let signals = BackendSignalHandler(backend: backend) { number in
            Darwin.exit(128 + number)
        }
        let outputLock = NSLock()
        let ready = DispatchSemaphore(value: 0)
        let output: @Sendable (UUID, Data) -> Void = { _, data in
            outputLock.lock()
            defer { outputLock.unlock() }
            FileHandle.standardOutput.write(data)
            ready.signal()
        }
        func start(_ extra: [String] = []) throws -> UUID {
            try backend.start(executable: URL(fileURLWithPath: python),
                              arguments: [script, "child"] + extra,
                              onOutput: output, onExit: { _, _ in })
        }
        func waitForOutput() {
            let deadline = Date().addingTimeInterval(5)
            while ready.wait(timeout: .now()) == .timedOut {
                guard Date() < deadline else { backend.stop(); Darwin.exit(6) }
                RunLoop.current.run(until: Date().addingTimeInterval(0.01))
            }
        }
        let first = try start(mode == "stubborn" ? ["ignore-term"] : mode == "output" ? ["output"] : [])
        if mode == "signals" {
            withExtendedLifetime(signals) { dispatchMain() }
        }
        waitForOutput()
        if mode == "restart" {
            let second = try start()
            waitForOutput()
            backend.stop(ifCurrent: first)
            guard backend.isCurrent(second) else { Darwin.exit(2) }
        } else if mode == "launch-error" {
            do {
                try backend.start(executable: URL(fileURLWithPath: "/imprint/nonexistent/backend"),
                                  arguments: [], onOutput: output, onExit: { _, _ in })
                Darwin.exit(3)
            } catch {
                guard !backend.isRunning else { Darwin.exit(4) }
            }
        } else if mode == "startup-failure" {
            // 与 Engine 启动超时相同，失败任务必须带原实例 ID 清理。
            backend.stop(ifCurrent: first)
        } else if mode == "output" {
            // 让后台 stdout 写入超过管道容量后再停止，检查关闭读取端不会卡住。
            RunLoop.current.run(until: Date().addingTimeInterval(0.1))
        }
        let termination = backend.stop()
        if mode == "stubborn", termination != SIGKILL { Darwin.exit(7) }
        if mode == "stop", termination != SIGTERM { Darwin.exit(8) }
        backend.stop()
        guard !backend.isRunning else { Darwin.exit(5) }
        withExtendedLifetime(signals) {}
    }
}
