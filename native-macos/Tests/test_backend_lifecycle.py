"""用受控子进程验证 Swift 后端所有权，不启动 Imprint/FastAPI。"""
from pathlib import Path
import json
import os
import selectors
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest


def child():
    if "ignore-term" in sys.argv:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        print(json.dumps({"pid": os.getpid(), "port": listener.getsockname()[1]}), flush=True)
        if "output" in sys.argv:
            for _ in range(128):
                print(json.dumps({"log": "x" * 4096}), flush=True)
        while True:
            time.sleep(1)


class BackendLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="imprint-backend-tests-")
        cls.binary = Path(cls.directory.name) / "backend-lifecycle-tests"
        root = Path(__file__).resolve().parents[1]
        subprocess.run([
            "/usr/bin/swiftc", "-parse-as-library",
            str(root / "Sources/ImprintMac/BackendProcess.swift"),
            str(root / "Tests/BackendLifecycleHarness.swift"), "-o", str(cls.binary),
        ], check=True, timeout=90)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def run_case(self, mode, parent_signal=None):
        records = []
        process = subprocess.Popen([
            str(self.binary), mode, sys.executable, str(Path(__file__).resolve()),
        ], stdout=subprocess.PIPE, text=True)
        try:
            if parent_signal is not None:
                with selectors.DefaultSelector() as selector:
                    selector.register(process.stdout, selectors.EVENT_READ)
                    self.assertTrue(selector.select(5), "child did not publish its PID")
                records.append(json.loads(process.stdout.readline()))
                os.kill(process.pid, parent_signal)
            stdout, _ = process.communicate(timeout=8)
            records.extend(record for line in stdout.splitlines() if line
                           for record in [json.loads(line)] if "pid" in record)
            expected = 0 if parent_signal is None else 128 + parent_signal
            self.assertEqual(process.returncode, expected)
            self.assertEqual(len(records), 2 if mode == "restart" else 1)
            for record in records:
                result = subprocess.run(
                    ["/bin/ps", "-p", str(record["pid"]), "-o", "pid="],
                    capture_output=True, text=True, check=False,
                )
                self.assertEqual(result.stdout.strip(), "", "owned child still exists")
                with socket.socket() as listener:
                    listener.bind(("127.0.0.1", record["port"]))
                lsof = subprocess.run(
                    ["/usr/sbin/lsof", "-nP", "-a", "-p", str(record["pid"]), "-iTCP"],
                    capture_output=True, text=True, check=False,
                )
                self.assertEqual(lsof.stdout.strip(), "", "owned child still listens")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            process.stdout.close()
            for record in records:
                # 测试失败时也只清理本次已记录的 PID。
                command = subprocess.run(
                    ["/bin/ps", "-p", str(record["pid"]), "-o", "command="],
                    capture_output=True, text=True, check=False,
                ).stdout
                if str(Path(__file__).resolve()) + " child" in command:
                    try:
                        os.kill(record["pid"], signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    def test_normal_stop_and_repeated_stop(self):
        self.run_case("stop")

    def test_stop_escalates_when_child_ignores_term(self):
        self.run_case("stubborn")

    def test_large_stdout_does_not_block_stop(self):
        self.run_case("output")

    def test_sigterm_cleans_up_child(self):
        self.run_case("signals", signal.SIGTERM)

    def test_sigint_cleans_up_child(self):
        self.run_case("signals", signal.SIGINT)

    def test_startup_failure_cleans_up_child(self):
        self.run_case("startup-failure")

    def test_failed_launch_cleans_up_previous_child(self):
        self.run_case("launch-error")

    def test_stale_stop_does_not_stop_replacement(self):
        for _ in range(3):
            self.run_case("restart")

    def test_other_instance_is_untouched(self):
        other = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "child"],
            stdout=subprocess.PIPE, text=True,
        )
        try:
            record = json.loads(other.stdout.readline())
            self.run_case("signals", signal.SIGTERM)
            self.assertIsNone(other.poll())
            with socket.create_connection(("127.0.0.1", record["port"]), timeout=1):
                pass
        finally:
            other.terminate()
            other.wait(timeout=5)
            other.stdout.close()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "child":
        child()
    else:
        unittest.main()
