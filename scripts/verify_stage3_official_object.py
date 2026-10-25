#!/usr/bin/env python3
"""Reproducible Stage3/core3 verification against the official renderer objects.

The CMake ``imprint_stage3_test_bridge`` and
``imprint_stage3_core3_test_bridge`` targets recompile the dispatch,
fast-Bayer and core3 sources. That proves the source tree, not the objects that
ship inside ``libimprint_renderer``.

This script builds second bridges that link the actual
``CMakeFiles/imprint_renderer.dir/src/*.cpp.o`` objects, then runs the same
version-controlled oracles against the source bridge and the official-object
bridge. It records the compiler, flags, source/object/bridge hashes and mtimes
so a later run can attribute drift.

It does not modify the build tree, the checked-in evidence history or the
renderer library. Output defaults to the path passed with ``--output-dir``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUILD = ROOT / "native-renderer/build"
DEFAULT_OUTPUT = (
    ROOT
    / "research/acr/reports/evidence/2026-10-18-stage3-domain-close"
)

STAGE3 = {
    "key": "stage3",
    "oracle": ROOT / "tests/acr_support/stage3_oracle.py",
    "bridge_source": ROOT / "native-renderer/tools/acr_stage3_bridge.cpp",
    "source_bridge": DEFAULT_BUILD / "libimprint_stage3_test_bridge.dylib",
    "objects": (
        "CMakeFiles/imprint_renderer.dir/src/acr_stage3_dispatch.cpp.o",
        "CMakeFiles/imprint_renderer.dir/src/acr_fast_bayer.cpp.o",
    ),
    "sources": (
        "native-renderer/src/acr_stage3_dispatch.cpp",
        "native-renderer/src/acr_stage3_dispatch.hpp",
        "native-renderer/src/acr_fast_bayer.cpp",
        "native-renderer/src/acr_fast_bayer.hpp",
        "native-renderer/tools/acr_stage3_bridge.cpp",
    ),
}

CORE3 = {
    "key": "core3",
    "oracle": ROOT / "tests/acr_support/core3_oracle.py",
    "bridge_source": ROOT / "native-renderer/tools/acr_stage3_core3_bridge.cpp",
    "source_bridge": DEFAULT_BUILD / "libimprint_stage3_core3_test_bridge.dylib",
    "objects": (
        "CMakeFiles/imprint_renderer.dir/src/acr_stage3_core3.cpp.o",
    ),
    "sources": (
        "native-renderer/src/acr_stage3_core3.cpp",
        "native-renderer/src/acr_stage3_core3.hpp",
        "native-renderer/tools/acr_stage3_core3_bridge.cpp",
    ),
}

SUITES = (STAGE3, CORE3)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path),
        "sha256": sha256(path),
        "size": stat.st_size,
        "mtime": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }


def cmake_compiler(build_dir: Path) -> str:
    cache = build_dir / "CMakeCache.txt"
    if cache.exists():
        for line in cache.read_text().splitlines():
            if line.startswith("CMAKE_CXX_COMPILER:FILEPATH="):
                return line.split("=", 1)[1]
    for candidate in ("clang++", "g++", "c++"):
        found = shutil.which(candidate)
        if found:
            return found
    raise SystemExit("no C++ compiler found")


def build_official_bridge(build_dir: Path, output: Path, compiler: str,
                          bridge_source: Path, objects: tuple[str, ...]) -> list[str]:
    object_paths = [build_dir / relative for relative in objects]
    missing = [str(path) for path in object_paths if not path.exists()]
    if missing:
        raise SystemExit(
            "official renderer objects are missing; build native-renderer first: "
            + ", ".join(missing)
        )
    link_flags = ["-std=c++17", "-O2", "-fPIC"]
    if platform.system() == "Darwin":
        link_flags += ["-dynamiclib", "-Wl,-undefined,dynamic_lookup"]
    else:
        link_flags += ["-shared"]
    command = [
        compiler,
        *link_flags,
        str(bridge_source),
        *(str(path) for path in object_paths),
        "-o",
        str(output),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise SystemExit(
            "official-object bridge link failed:\n"
            + " ".join(command)
            + "\n"
            + completed.stdout
            + completed.stderr
        )
    return command


def run_oracle(oracle: Path, bridge: Path, output: Path) -> dict:
    completed = subprocess.run(
        [
            sys.executable,
            str(oracle),
            "--bridge",
            str(bridge),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads(output.read_text()) if output.exists() else {}
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "payload": payload,
    }


def run_suite(suite: dict, build_dir: Path, output_dir: Path,
              compiler: str, work: Path) -> dict:
    official_bridge = work / f"libimprint_{suite['key']}_official.dylib"
    command = build_official_bridge(
        build_dir, official_bridge, compiler, suite["bridge_source"],
        suite["objects"])
    if not suite["source_bridge"].exists():
        raise SystemExit(
            f"source bridge not found: {suite['source_bridge']}; build the "
            f"CMake imprint_{suite['key']}_test_bridge target first"
        )
    source_result = run_oracle(
        suite["oracle"], suite["source_bridge"],
        output_dir / f"{suite['key']}-source-bridge.json")
    official_result = run_oracle(
        suite["oracle"], official_bridge,
        output_dir / f"{suite['key']}-official-object.json")
    object_records = [file_record(build_dir / relative)
                      for relative in suite["objects"]]
    source_records = [file_record(ROOT / relative)
                      for relative in suite["sources"]]
    return {
        "compiler": compiler,
        "link_command": command,
        "sources": source_records,
        "objects": object_records,
        "source_bridge": file_record(suite["source_bridge"]),
        "official_object_bridge": {
            **file_record(official_bridge),
            "note": "temporary; rebuilt from the official objects each run",
            "hash_stability": (
                "the linked dylib hash is not guaranteed stable across links "
                "(linker UUID/signature); reproducibility is anchored to the "
                "source/object hashes and link_command"
            ),
        },
        "source_bridge_result": {
            "returncode": source_result["returncode"],
            "stdout": source_result["stdout"],
            "stderr": source_result["stderr"],
            "passed": source_result["payload"].get("passed"),
            "case_count": source_result["payload"].get("case_count"),
        },
        "official_object_result": {
            "returncode": official_result["returncode"],
            "stdout": official_result["stdout"],
            "stderr": official_result["stderr"],
            "passed": official_result["payload"].get("passed"),
            "case_count": official_result["payload"].get("case_count"),
        },
        "source_bridge_payload": source_result["payload"],
        "official_object_payload": official_result["payload"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    compiler = cmake_compiler(args.build_dir)

    work = Path(tempfile.mkdtemp(prefix="stage3-official-"))
    results = {}
    for suite in SUITES:
        results[suite["key"]] = run_suite(
            suite, args.build_dir, output_dir, compiler, work)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "compiler": compiler,
        "build_dir": str(args.build_dir),
        "suites": results,
    }
    # Keep the original top-level Stage3 fields readable for older reports.
    stage3 = results["stage3"]
    manifest.update({
        "link_command": stage3["link_command"],
        "sources": stage3["sources"],
        "objects": stage3["objects"],
        "source_bridge": stage3["source_bridge"],
        "official_object_bridge": stage3["official_object_bridge"],
        "source_bridge_result": stage3["source_bridge_result"],
        "official_object_result": stage3["official_object_result"],
    })
    (output_dir / "official-object-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    return_codes = []
    for key, result in results.items():
        print(f"{key} source bridge:   {result['source_bridge_result']['stdout']}")
        print(f"{key} official object: {result['official_object_result']['stdout']}")
        return_codes.append(result["source_bridge_result"]["returncode"])
        return_codes.append(result["official_object_result"]["returncode"])
    print(f"official object bridge sha256: "
          f"{stage3['official_object_bridge']['sha256']}")
    print(f"manifest: {output_dir / 'official-object-manifest.json'}")
    return 0 if all(code == 0 for code in return_codes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
