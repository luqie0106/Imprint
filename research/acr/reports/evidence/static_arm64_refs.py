#!/usr/bin/env python3
"""Extract conservative ADRP/ADD string-address candidates from Camera Raw arm64."""

from __future__ import annotations

import argparse
import json
import re
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Any


DEFAULT_BINARY = Path(
    "/Library/Application Support/Adobe/Plug-Ins/CC/File Formats/"
    "Camera Raw.plugin/Contents/MacOS/Camera Raw"
)
EVIDENCE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = EVIDENCE_DIR / "static-arm64-refs.json"
ARM64_STRINGS = EVIDENCE_DIR / "main-arm64-strings.json"
CODEC_STRINGS = EVIDENCE_DIR / "main-codec-strings.json"
METAL_SELECTOR_MANIFEST = EVIDENCE_DIR / "metal-selectors.json"
OBJDUMP = Path("/Library/Developer/CommandLineTools/usr/bin/llvm-objdump")
LIPO = Path("/usr/bin/lipo")

TARGETS = (
    "ReadNikonCompressed_Scalar",
    "ipxcpucodec_init_ctx_options failed",
    "ipxcpucodec_set_thread_count failed",
    "ipxcpucodec_init_decoder failed",
    "ipxcpucodec_init_image_geometry failed",
    "ipxcpucodec_init_image_format_cfg failed",
    "ipxcpucodec_init_decode_options failed",
    "standard bayer demosaic",
    "LoadBayerDemosaicModels",
    "LoadXTransDemosaicModels",
    "Stage1-ColorTransform",
    "BuildNegative-ColorTransform",
    "InitializeGPUDisplayColorTransforms",
    "MakeMetalSystem",
    'create metal lib failed: "%s"',
    "cr_gpu_tile_render_task",
)

METAL_SELECTOR_TARGETS = (
    "newComputePipelineStateWithFunction:error:",
    "newComputePipelineStateWithDescriptor:options:reflection:error:",
    "newRenderPipelineStateWithDescriptor:error:",
    "newLibraryWithData:error:",
    "newTextureWithDescriptor:",
    "setTexture:atIndex:",
    "setFragmentTexture:atIndex:",
    "copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:",
    "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toBuffer:destinationOffset:destinationBytesPerRow:destinationBytesPerImage:",
    "presentDrawable:",
    "nextDrawable",
    "commit",
)


class MachOError(ValueError):
    pass


def read_uleb(data: bytes, pos: int, limit: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while pos < limit:
        byte = data[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, pos
        shift += 7
        if shift >= 64:
            raise MachOError("LC_FUNCTION_STARTS ULEB value exceeds 64 bits")
    raise MachOError("unterminated ULEB value in LC_FUNCTION_STARTS")


def arm64_slice(data: bytes) -> tuple[int, bytes]:
    """Return (full-file slice offset, arm64 Mach-O bytes)."""
    if len(data) < 8:
        raise MachOError("input is too short for a Mach-O header")
    magic = data[:4]
    if magic in (b"\xca\xfe\xba\xbe", b"\xca\xfe\xba\xbf"):
        is_64 = magic == b"\xca\xfe\xba\xbf"
        nfat_arch = struct.unpack_from(">I", data, 4)[0]
        entry_size = 32 if is_64 else 20
        arch_pos = 8
        for _ in range(nfat_arch):
            if arch_pos + entry_size > len(data):
                raise MachOError("truncated fat architecture table")
            if is_64:
                cpu, _subtype, offset, size, _align, _reserved = struct.unpack_from(
                    ">iiQQII", data, arch_pos
                )
            else:
                cpu, _subtype, offset, size, _align = struct.unpack_from(
                    ">iiIII", data, arch_pos
                )
            if cpu == 0x0100000C:
                end = offset + size
                if end > len(data):
                    raise MachOError("arm64 slice extends beyond input file")
                return offset, data[offset:end]
            arch_pos += entry_size
        raise MachOError("fat Mach-O contains no arm64 slice")
    if magic == b"\xcf\xfa\xed\xfe":
        return 0, data
    raise MachOError(f"unsupported Mach-O/fat magic {magic.hex()}")


def parse_arm64_macho(slice_data: bytes) -> dict[str, Any]:
    if len(slice_data) < 32 or slice_data[:4] != b"\xcf\xfa\xed\xfe":
        raise MachOError("arm64 slice is not a little-endian 64-bit Mach-O")
    (_magic, cpu, _subtype, filetype, ncmds, sizeofcmds, _flags, _reserved) = (
        struct.unpack_from("<IiiIIIII", slice_data, 0)
    )
    if cpu != 0x0100000C:
        raise MachOError(f"selected slice CPU type is not arm64: {cpu:#x}")
    if 32 + sizeofcmds > len(slice_data):
        raise MachOError("load command table extends beyond the arm64 slice")

    text_segment_vm: int | None = None
    text_section: dict[str, int] | None = None
    sections: list[dict[str, Any]] = []
    function_starts: tuple[int, int] | None = None
    commands: list[dict[str, Any]] = []
    pos = 32
    for _ in range(ncmds):
        if pos + 8 > 32 + sizeofcmds:
            raise MachOError("truncated Mach-O load command")
        cmd, cmdsize = struct.unpack_from("<II", slice_data, pos)
        if cmdsize < 8 or pos + cmdsize > 32 + sizeofcmds:
            raise MachOError("invalid Mach-O load command size")
        commands.append({"cmd": cmd, "cmdsize": cmdsize})
        if cmd == 0x19:  # LC_SEGMENT_64
            if cmdsize < 72:
                raise MachOError("short LC_SEGMENT_64")
            (_cmd, _cmdsize, segname_b, vmaddr, _vmsize, _fileoff, _filesize,
             _maxprot, _initprot, nsects, _segflags) = struct.unpack_from(
                "<II16sQQQQiiII", slice_data, pos
            )
            segname = segname_b.rstrip(b"\0").decode("ascii", "replace")
            if segname == "__TEXT":
                text_segment_vm = vmaddr
            section_pos = pos + 72
            if section_pos + nsects * 80 > pos + cmdsize:
                raise MachOError("LC_SEGMENT_64 sections exceed command size")
            for _section_index in range(nsects):
                (sectname_b, section_segname_b, addr, size, offset, _align,
                 _reloff, _nreloc, _sectflags, _reserved1, _reserved2,
                 _reserved3) = struct.unpack_from("<16s16sQQIIIIIIII", slice_data, section_pos)
                sectname = sectname_b.rstrip(b"\0").decode("ascii", "replace")
                section_segname = section_segname_b.rstrip(b"\0").decode("ascii", "replace")
                section = {
                    "segment": section_segname,
                    "name": sectname,
                    "vm_address": addr,
                    "size": size,
                    "slice_file_offset": offset,
                    "reserved2": _reserved2,
                }
                sections.append(section)
                if sectname == "__text" and section_segname == "__TEXT":
                    text_section = section
                section_pos += 80
        elif cmd == 0x26:  # LC_FUNCTION_STARTS (linkedit_data_command)
            if cmdsize < 16:
                raise MachOError("short LC_FUNCTION_STARTS")
            _cmd, _cmdsize, dataoff, datasize = struct.unpack_from("<IIII", slice_data, pos)
            function_starts = (dataoff, datasize)
        pos += cmdsize

    if filetype not in (2, 6, 8):  # MH_EXECUTE, MH_DYLIB, MH_BUNDLE
        raise MachOError(f"expected a loadable Mach-O, found filetype {filetype}")
    if text_segment_vm is None or text_section is None:
        raise MachOError("could not locate __TEXT,__text")
    if function_starts is None:
        raise MachOError("LC_FUNCTION_STARTS is absent")
    dataoff, datasize = function_starts
    if dataoff + datasize > len(slice_data):
        raise MachOError("LC_FUNCTION_STARTS data exceeds arm64 slice")
    starts: list[int] = []
    cursor = dataoff
    limit = dataoff + datasize
    address = text_segment_vm
    while cursor < limit:
        delta, cursor = read_uleb(slice_data, cursor, limit)
        if delta == 0:
            break
        address += delta
        starts.append(address)
    starts.sort()
    return {
        "cpu_type": cpu,
        "file_type": filetype,
        "text_segment_vm_address": text_segment_vm,
        "text_section": text_section,
        "sections": sections,
        "function_starts": starts,
        "function_starts_data": {"slice_file_offset": dataoff, "size": datasize},
        "load_command_count": ncmds,
    }


def decode_adrp(word: int, pc: int) -> tuple[int, int] | None:
    # ADRP: 1 immlo 10000 immhi Rd
    if (word & 0x9F000000) != 0x90000000:
        return None
    immlo = (word >> 29) & 0x3
    immhi = (word >> 5) & 0x7FFFF
    immediate = (immhi << 2) | immlo
    if immediate & (1 << 20):
        immediate -= 1 << 21
    rd = word & 0x1F
    return ((pc & ~0xFFF) + (immediate << 12), rd)


def decode_add_imm64(word: int) -> tuple[int, int, int] | None:
    # 64-bit ADD (immediate), without flags, including optional LSL #12.
    if (word & 0xFF000000) != 0x91000000:
        return None
    rn = (word >> 5) & 0x1F
    rd = word & 0x1F
    imm12 = (word >> 10) & 0xFFF
    shift = 12 if (word >> 22) & 1 else 0
    return rd, rn, imm12 << shift


def load_strings(paths: tuple[Path, ...], file_size: int, slice_offset: int, raw: bytes) -> dict[str, list[dict[str, Any]]]:
    unique: dict[tuple[str, int, int], dict[str, Any]] = {}
    for path in paths:
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"expected a JSON array in {path}")
        for row in rows:
            value = row.get("value")
            if value not in TARGETS or row.get("architecture", "arm64") != "arm64":
                continue
            full_offset = int(row["file_offset"], 16)
            vm_address = int(row["vm_address"], 16)
            slice_file_offset = int(row.get("slice_file_offset", hex(full_offset - slice_offset)), 16)
            if full_offset != slice_offset + slice_file_offset:
                raise ValueError(f"inconsistent full/slice offsets for {value!r}")
            if not (0 <= full_offset < file_size):
                raise ValueError(f"string offset is outside input file for {value!r}")
            encoded = value.encode("utf-8")
            if raw[full_offset : full_offset + len(encoded)] != encoded:
                raise ValueError(f"string bytes do not match recorded offset for {value!r}")
            key = (value, vm_address, full_offset)
            location = unique.setdefault(
                key,
                {
                    "string_vm_address": vm_address,
                    "string_slice_file_offset": slice_file_offset,
                    "string_full_file_offset": full_offset,
                    "source_manifests": [],
                },
            )
            if path.name not in location["source_manifests"]:
                location["source_manifests"].append(path.name)
    by_value: dict[str, list[dict[str, Any]]] = {value: [] for value in TARGETS}
    for (value, _vm_address, _full_offset), location in unique.items():
        by_value[value].append(location)
    for locations in by_value.values():
        locations.sort(key=lambda row: (row["string_vm_address"], row["string_full_file_offset"]))
    return by_value


def find_references(slice_data: bytes, slice_offset: int, metadata: dict[str, Any], strings: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    section = metadata["text_section"]
    text_off = section["slice_file_offset"]
    text_size = section["size"]
    if text_off + text_size > len(slice_data):
        raise MachOError("__TEXT,__text exceeds arm64 slice")
    starts = metadata["function_starts"]
    address_to_targets: dict[int, list[tuple[str, dict[str, Any]]]] = {}
    for target, locations in strings.items():
        for location in locations:
            address_to_targets.setdefault(location["string_vm_address"], []).append((target, location))

    text_vm = section["vm_address"]
    instructions = [
        struct.unpack_from("<I", slice_data, off)[0]
        for off in range(text_off, text_off + text_size - (text_size % 4), 4)
    ]
    refs: list[dict[str, Any]] = []
    for index, word in enumerate(instructions):
        pc = text_vm + index * 4
        decoded = decode_adrp(word, pc)
        if decoded is None:
            continue
        page_address, rd = decoded
        for distance in range(1, 5):
            add_index = index + distance
            if add_index >= len(instructions):
                break
            add_word = instructions[add_index]
            add_decoded = decode_add_imm64(add_word)
            if add_decoded is None:
                continue
            add_rd, add_rn, immediate = add_decoded
            if add_rd != rd or add_rn != rd:
                continue
            target_address = page_address + immediate
            for target, location in address_to_targets.get(target_address, []):
                func_index = __import__("bisect").bisect_right(starts, pc) - 1
                func_start = starts[func_index] if func_index >= 0 else None
                intervening = instructions[index + 1 : add_index]
                refs.append(
                    {
                        "target": target,
                        **location,
                        "adrp_instruction_vm_address": pc,
                        "adrp_instruction_full_file_offset": slice_offset + text_off + index * 4,
                        "adrp_instruction_word": f"0x{word:08x}",
                        "adrp_page_vm_address": f"0x{page_address:x}",
                        "add_instruction_vm_address": pc + distance * 4,
                        "add_instruction_full_file_offset": slice_offset + text_off + add_index * 4,
                        "add_instruction_word": f"0x{add_word:08x}",
                        "add_immediate": immediate,
                        "computed_target_vm_address": f"0x{target_address:x}",
                        "intervening_instruction_words": [f"0x{item:08x}" for item in intervening],
                        "lc_function_start_candidate_vm_address": (
                            f"0x{func_start:x}" if func_start is not None else None
                        ),
                    }
                )
    refs.sort(key=lambda item: (item["target"], item["string_vm_address"], item["adrp_instruction_vm_address"]))
    return refs


def decode_ldr_unsigned64(word: int) -> tuple[int, int, int] | None:
    # LDR Xt, [Xn, #imm12 * 8]
    if (word & 0xFFC00000) != 0xF9400000:
        return None
    rd = word & 0x1F
    rn = (word >> 5) & 0x1F
    displacement = ((word >> 10) & 0xFFF) << 3
    return rd, rn, displacement


def load_metal_selector_data(
    raw: bytes, slice_data: bytes, slice_offset: int, metadata: dict[str, Any]
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    manifest = json.loads(METAL_SELECTOR_MANIFEST.read_text(encoding="utf-8"))
    sections = {section["name"]: section for section in metadata["sections"]}
    selref_section = sections.get("__objc_selrefs")
    stub_section = sections.get("__objc_stubs")
    if selref_section is None or stub_section is None:
        raise MachOError("could not locate __objc_selrefs or __objc_stubs")

    rows_by_selector = {row.get("selector"): row for row in manifest if row.get("selector") in METAL_SELECTOR_TARGETS}
    selrefs: list[dict[str, Any]] = []
    selref_to_selector: dict[int, dict[str, Any]] = {}
    for selector in METAL_SELECTOR_TARGETS:
        row = rows_by_selector.get(selector)
        if row is None:
            continue
        selector_vm = int(row["vm_address"], 16)
        selector_full_offset = int(row["file_offset"], 16)
        if not 0 <= selector_full_offset < len(raw):
            raise ValueError(f"selector file offset is outside input file: {selector}")
        encoded = selector.encode("utf-8")
        if raw[selector_full_offset : selector_full_offset + len(encoded)] != encoded:
            raise ValueError(f"selector bytes do not match manifest offset for {selector}")
        for address_text in row.get("selref_addresses", []):
            selref_vm = int(address_text, 16)
            relative = selref_vm - selref_section["vm_address"]
            if relative < 0 or relative + 8 > selref_section["size"] or relative % 8:
                raise ValueError(f"selref address is outside/alignment-invalid in section: {address_text}")
            slice_file_offset = selref_section["slice_file_offset"] + relative
            full_file_offset = slice_offset + slice_file_offset
            pointer = struct.unpack_from("<Q", slice_data, slice_file_offset)[0]
            if pointer != selector_vm:
                raise ValueError(
                    f"selref {address_text} points to {pointer:#x}, expected cstring {selector_vm:#x}"
                )
            info = {
                "selector": selector,
                "selector_vm_address": f"0x{selector_vm:x}",
                "selector_full_file_offset": selector_full_offset,
                "selref_vm_address": f"0x{selref_vm:x}",
                "selref_full_file_offset": full_file_offset,
                "selref_raw_pointer_value": f"0x{pointer:x}",
                "selref_pointer_matches_selector_cstring": True,
            }
            selrefs.append(info)
            selref_to_selector[selref_vm] = info

    stub_words = [
        struct.unpack_from("<I", slice_data, off)[0]
        for off in range(
            stub_section["slice_file_offset"],
            stub_section["slice_file_offset"] + stub_section["size"] - stub_section["size"] % 4,
            4,
        )
    ]
    selector_stubs: list[dict[str, Any]] = []
    for index, word in enumerate(stub_words):
        pc = stub_section["vm_address"] + index * 4
        decoded = decode_adrp(word, pc)
        if decoded is None:
            continue
        page_address, rd = decoded
        if rd != 1:
            continue
        for distance in range(1, 5):
            ldr_index = index + distance
            if ldr_index >= len(stub_words):
                break
            ldr_word = stub_words[ldr_index]
            ldr = decode_ldr_unsigned64(ldr_word)
            if ldr is None:
                continue
            ldr_rd, ldr_rn, displacement = ldr
            if ldr_rd != 1 or ldr_rn != rd:
                continue
            selref_vm = page_address + displacement
            selector_info = selref_to_selector.get(selref_vm)
            if selector_info is None:
                continue
            if (pc - stub_section["vm_address"]) % 32 != 0:
                # __objc_stubs in this binary uses 32-byte entries. Requiring
                # the observed entry boundary avoids inventing a stub identity.
                continue
            selector_stubs.append(
                {
                    **selector_info,
                    "objc_stub_vm_address": f"0x{pc:x}",
                    "adrp_instruction_vm_address": f"0x{pc:x}",
                    "adrp_instruction_full_file_offset": slice_offset + stub_section["slice_file_offset"] + index * 4,
                    "adrp_instruction_word": f"0x{word:08x}",
                    "ldr_instruction_vm_address": f"0x{pc + distance * 4:x}",
                    "ldr_instruction_full_file_offset": slice_offset + stub_section["slice_file_offset"] + ldr_index * 4,
                    "ldr_instruction_word": f"0x{ldr_word:08x}",
                    "ldr_computed_selref_vm_address": f"0x{selref_vm:x}",
                    "stub_entry_stride_bytes_observed": 32,
                }
            )
    selector_stubs.sort(key=lambda item: (item["selector"], int(item["objc_stub_vm_address"], 16)))

    stub_map = {int(row["objc_stub_vm_address"], 16): row for row in selector_stubs}
    text = metadata["text_section"]
    text_words = [
        struct.unpack_from("<I", slice_data, off)[0]
        for off in range(text["slice_file_offset"], text["slice_file_offset"] + text["size"] - text["size"] % 4, 4)
    ]
    starts = metadata["function_starts"]
    import bisect

    text_selref_loads: list[dict[str, Any]] = []
    for index, word in enumerate(text_words):
        pc = text["vm_address"] + index * 4
        decoded = decode_adrp(word, pc)
        if decoded is None:
            continue
        page_address, rd = decoded
        for distance in range(1, 5):
            ldr_index = index + distance
            if ldr_index >= len(text_words):
                break
            ldr_word = text_words[ldr_index]
            ldr = decode_ldr_unsigned64(ldr_word)
            if ldr is None:
                continue
            ldr_rd, ldr_rn, displacement = ldr
            if ldr_rn != rd:
                continue
            selref_vm = page_address + displacement
            selector_info = selref_to_selector.get(selref_vm)
            if selector_info is None:
                continue
            start_index = bisect.bisect_right(starts, pc) - 1
            function_start = starts[start_index] if start_index >= 0 else None
            text_selref_loads.append(
                {
                    **selector_info,
                    "adrp_instruction_vm_address": f"0x{pc:x}",
                    "adrp_instruction_full_file_offset": slice_offset + text["slice_file_offset"] + index * 4,
                    "adrp_instruction_word": f"0x{word:08x}",
                    "ldr_instruction_vm_address": f"0x{pc + distance * 4:x}",
                    "ldr_instruction_full_file_offset": slice_offset + text["slice_file_offset"] + ldr_index * 4,
                    "ldr_instruction_word": f"0x{ldr_word:08x}",
                    "ldr_destination_register": ldr_rd,
                    "computed_selref_vm_address": f"0x{selref_vm:x}",
                    "lc_function_start_candidate_vm_address": (
                        f"0x{function_start:x}" if function_start is not None else None
                    ),
                }
            )

    calls: list[dict[str, Any]] = []
    for index, word in enumerate(text_words):
        if (word & 0xFC000000) != 0x94000000:  # BL imm26
            continue
        immediate = word & 0x03FFFFFF
        if immediate & (1 << 25):
            immediate -= 1 << 26
        pc = text["vm_address"] + index * 4
        target = pc + (immediate << 2)
        stub = stub_map.get(target)
        if stub is None:
            continue
        start_index = bisect.bisect_right(starts, pc) - 1
        function_start = starts[start_index] if start_index >= 0 else None
        calls.append(
            {
                "selector": stub["selector"],
                "selector_vm_address": stub["selector_vm_address"],
                "selref_vm_address": stub["selref_vm_address"],
                "objc_stub_vm_address": f"0x{target:x}",
                "call_instruction_vm_address": f"0x{pc:x}",
                "call_instruction_full_file_offset": slice_offset + text["slice_file_offset"] + index * 4,
                "call_instruction_word": f"0x{word:08x}",
                "direct_bl_target_vm_address": f"0x{target:x}",
                "lc_function_start_candidate_vm_address": (
                    f"0x{function_start:x}" if function_start is not None else None
                ),
            }
        )
    calls.sort(key=lambda item: (item["lc_function_start_candidate_vm_address"] or "", item["call_instruction_vm_address"]))

    grouped: dict[str | None, dict[str, Any]] = {}
    for call in calls:
        key = call["lc_function_start_candidate_vm_address"]
        group = grouped.setdefault(key, {"lc_function_start_candidate_vm_address": key, "direct_bl_calls": []})
        group["direct_bl_calls"].append(
            {
                "selector": call["selector"],
                "call_instruction_vm_address": call["call_instruction_vm_address"],
                "direct_bl_target_vm_address": call["direct_bl_target_vm_address"],
            }
        )
    function_groups = list(grouped.values())
    return selrefs, selector_stubs, text_selref_loads, calls, function_groups


def disassemble_objc_calls(binary: Path, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = (
        (
            "pipeline state creation",
            ("newComputePipelineStateWithDescriptor:options:reflection:error:", "newRenderPipelineStateWithDescriptor:error:"),
        ),
        ("Metal library creation", ("newLibraryWithData:error:",)),
        ("texture creation", ("newTextureWithDescriptor:",)),
        (
            "texture input copy",
            ("copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:",),
        ),
        (
            "texture output copy",
            ("copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toBuffer:destinationOffset:destinationBytesPerRow:destinationBytesPerImage:",),
        ),
        ("drawable acquisition", ("nextDrawable",)),
        ("drawable presentation and command submission", ("presentDrawable:", "commit")),
    )
    selected_groups: list[tuple[str, list[dict[str, Any]]]] = []
    for label, selectors in groups:
        group_calls = [
            call for selector in selectors
            if (call := next((row for row in calls if row["selector"] == selector), None)) is not None
        ]
        if group_calls:
            selected_groups.append((label, group_calls))
    if not selected_groups:
        return []
    with tempfile.TemporaryDirectory(prefix="acr-metal-callsite-", dir="/private/tmp") as temp_dir:
        thin_binary = Path(temp_dir) / "camera-raw-arm64"
        thin_result = subprocess.run(
            [str(LIPO), "-thin", "arm64", str(binary), "-output", str(thin_binary)],
            capture_output=True,
            text=True,
            check=False,
        )
        if thin_result.returncode != 0:
            raise RuntimeError(f"lipo could not make a temporary arm64 slice: {thin_result.stderr[:1000]}")
        snippets: list[dict[str, Any]] = []
        for label, group_calls in selected_groups:
            fragments: list[dict[str, Any]] = []
            for call in group_calls:
                pc = int(call["call_instruction_vm_address"], 16)
                start = max(0, pc - 16)
                stop = pc + 20
                args = [
                    str(OBJDUMP),
                    "-d",
                    "--arch-name=arm64",
                    f"--start-address=0x{start:x}",
                    f"--stop-address=0x{stop:x}",
                    str(thin_binary),
                ]
                result = subprocess.run(args, capture_output=True, text=True, check=False)
                output = (result.stdout + result.stderr).strip()
                if result.returncode != 0:
                    raise RuntimeError(
                        f"llvm-objdump failed at {start:#x}..{stop:#x} ({result.returncode}): {output[:1000]}"
                    )
                lines = [
                    line
                    for line in result.stdout.splitlines()
                    if (match := re.match(r"^\s*([0-9a-fA-F]+):", line))
                    and start <= int(match.group(1), 16) < stop
                ]
                if not any(
                    (match := re.match(r"^\s*([0-9a-fA-F]+):", line))
                    and int(match.group(1), 16) == pc
                    for line in lines
                ):
                    raise RuntimeError(f"llvm-objdump did not show direct BL at {pc:#x}")
                fragments.append(
                    {
                        "selector": call["selector"],
                        "call_instruction_vm_address": call["call_instruction_vm_address"],
                        "direct_bl_target_vm_address": call["direct_bl_target_vm_address"],
                        "lc_function_start_candidate_vm_address": call["lc_function_start_candidate_vm_address"],
                        "arm64_vm_address_range": f"0x{start:x}-0x{stop:x}",
                        "output": "\n".join(lines),
                    }
                )
            snippets.append(
                {
                    "area": label,
                    "local_fragments": fragments,
                    "llvm_objdump_command": "llvm-objdump -d --arch-name=arm64 --start-address=... --stop-address=... <temporary arm64 thin copy>",
                }
            )
        return snippets


def disassemble_snippets(binary: Path, refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not OBJDUMP.is_file():
        raise FileNotFoundError(f"llvm-objdump not found at {OBJDUMP}")
    if not LIPO.is_file():
        raise FileNotFoundError(f"lipo not found at {LIPO}")

    # Keep eight short local windows. Related targets in the codec/model groups
    # are close enough to share one bounded range.
    groups = (
        ("Nikon compressed decoder", ("ReadNikonCompressed_Scalar",)),
        (
            "ipxcpucodec initialization failures",
            tuple(target for target in TARGETS if target.startswith("ipxcpucodec_")),
        ),
        ("standard Bayer demosaic", ("standard bayer demosaic",)),
        ("Bayer/X-Trans model loading", ("LoadBayerDemosaicModels", "LoadXTransDemosaicModels")),
        ("stage 1 color transform", ("Stage1-ColorTransform",)),
        ("negative color transform", ("BuildNegative-ColorTransform",)),
        ("GPU display color transforms", ("InitializeGPUDisplayColorTransforms",)),
        ("Metal system creation", ("MakeMetalSystem", 'create metal lib failed: "%s"')),
    )
    selected_groups: list[tuple[str, list[dict[str, Any]]]] = []
    for label, targets in groups:
        group_refs = [ref for ref in refs if ref["target"] in targets]
        if group_refs:
            selected_groups.append((label, group_refs))

    snippets: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="acr-static-arm64-", dir="/private/tmp") as temp_dir:
        thin_binary = Path(temp_dir) / "camera-raw-arm64"
        thin_result = subprocess.run(
            [str(LIPO), "-thin", "arm64", str(binary), "-output", str(thin_binary)],
            capture_output=True,
            text=True,
            check=False,
        )
        if thin_result.returncode != 0:
            raise RuntimeError(f"lipo could not make a temporary arm64 slice: {thin_result.stderr[:1000]}")

        for label, group_refs in selected_groups:
            start = max(0, min(ref["adrp_instruction_vm_address"] for ref in group_refs) - 16)
            stop = max(ref["add_instruction_vm_address"] for ref in group_refs) + 20
            args = [
                str(OBJDUMP),
                "-d",
                "--arch-name=arm64",
                f"--start-address=0x{start:x}",
                f"--stop-address=0x{stop:x}",
                str(thin_binary),
            ]
            result = subprocess.run(args, capture_output=True, text=True, check=False)
            output = (result.stdout + result.stderr).strip()
            if result.returncode != 0:
                raise RuntimeError(
                    f"llvm-objdump failed at {start:#x}..{stop:#x} ({result.returncode}): {output[:1000]}"
                )

            # A group can span more than one reference. Keep only the few exact
            # instruction lines around each matched ADRP/ADD pair.
            wanted_addresses: set[int] = set()
            for ref in group_refs:
                first = ref["adrp_instruction_vm_address"]
                last = ref["add_instruction_vm_address"]
                wanted_addresses.update(range(first - 8, last + 12, 4))
            selected_lines: list[str] = []
            for line in result.stdout.splitlines():
                match = re.match(r"^\s*([0-9a-fA-F]+):", line)
                if match and int(match.group(1), 16) in wanted_addresses:
                    selected_lines.append(line)
            if not selected_lines:
                raise RuntimeError(f"llvm-objdump returned no instruction lines for {label} at {start:#x}")
            snippets.append(
                {
                    "area": label,
                    "targets": sorted({ref["target"] for ref in group_refs}),
                    "arm64_vm_address_range": f"0x{start:x}-0x{stop:x}",
                    "reference_instruction_addresses": [
                        {
                            "target": ref["target"],
                            "adrp": f"0x{ref['adrp_instruction_vm_address']:x}",
                            "add": f"0x{ref['add_instruction_vm_address']:x}",
                        }
                        for ref in group_refs
                    ],
                    "llvm_objdump_command": "llvm-objdump -d --arch-name=arm64 --start-address=... --stop-address=... <temporary arm64 thin copy>",
                    "output": "\n".join(selected_lines),
                }
            )
    return snippets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    raw = args.binary.read_bytes()
    slice_offset, slice_data = arm64_slice(raw)
    metadata = parse_arm64_macho(slice_data)
    string_locations = load_strings((ARM64_STRINGS, CODEC_STRINGS), len(raw), slice_offset, raw)
    references = find_references(slice_data, slice_offset, metadata, string_locations)
    snippets = disassemble_snippets(args.binary, references)
    selector_refs, selector_stubs, text_selref_loads, selector_calls, selector_function_groups = load_metal_selector_data(
        raw, slice_data, slice_offset, metadata
    )
    selector_call_snippets = disassemble_objc_calls(args.binary, selector_calls)
    selectors_with_stubs = {row["selector"] for row in selector_stubs}
    selectors_with_calls = {row["selector"] for row in selector_calls}
    selector_manifest_hits = {row["selector"] for row in selector_refs}

    result = {
        "evidence_kind": "static ARM64 instruction-byte string-address candidates",
        "input": {
            "path": str(args.binary),
            "file_size": len(raw),
            "architecture": "arm64",
            "slice_full_file_offset": slice_offset,
            "slice_size": len(slice_data),
        },
        "method": {
            "text_section": "LC_SEGMENT_64 __TEXT,__text",
            "function_start_source": "LC_FUNCTION_STARTS ULEB128 offsets based at __TEXT vmaddr",
            "instruction_pattern": "ADRP followed 1-4 instructions later by 64-bit ADD (immediate), where ADRP Rd = ADD Rn = ADD Rd and computed address exactly equals a requested string VM address",
            "scope_limit": "Instructions between ADRP and ADD are recorded as raw words but are not fully decoded for writes to the base register. Matches are structural candidates, not verified register dataflow, xrefs, calls, or semantic proof.",
            "objdump_note": "The installed llvm-objdump --macho path ignored start/stop bounds on the fat input. Bounded snippets were therefore checked against a temporary lipo -thin arm64 copy using the help-documented --arch-name=arm64 mode; the temporary copy is removed after execution.",
        },
        "macho": {
            "cpu_type": f"0x{metadata['cpu_type']:x}",
            "file_type": metadata["file_type"],
            "load_command_count": metadata["load_command_count"],
            "text_segment_vm_address": f"0x{metadata['text_segment_vm_address']:x}",
            "text_section_vm_address": f"0x{metadata['text_section']['vm_address']:x}",
            "text_section_slice_file_offset": metadata["text_section"]["slice_file_offset"],
            "text_section_size": metadata["text_section"]["size"],
            "function_starts_data": metadata["function_starts_data"],
            "function_start_candidate_count": len(metadata["function_starts"]),
        },
        "requested_targets": list(TARGETS),
        "string_target_locations": {
            target: locations for target, locations in string_locations.items() if locations
        },
        "reference_candidates": references,
        "local_llvm_objdump_snippets": snippets,
        "unmatched_targets": [target for target in TARGETS if not any(r["target"] == target for r in references)],
        "metal_selector_evidence": {
            "selector_targets": list(METAL_SELECTOR_TARGETS),
            "cstring_selref_validation": selector_refs,
            "objc_stub_candidates": selector_stubs,
            "text_adrp_ldr_selref_candidates": text_selref_loads,
            "direct_bl_call_candidates": selector_calls,
            "direct_bl_calls_grouped_by_lc_function_start_candidate": selector_function_groups,
            "local_objdump_callsite_snippets": selector_call_snippets,
            "selector_targets_without_manifest_selref": [
                selector for selector in METAL_SELECTOR_TARGETS if selector not in selector_manifest_hits
            ],
            "selectors_with_selref_stub_but_no_text_direct_bl_candidate": [
                selector for selector in METAL_SELECTOR_TARGETS
                if selector in selectors_with_stubs and selector not in selectors_with_calls
            ],
            "scope_limit": "The selector cstring-to-selref pointer, __objc_stubs ADRP/LDR pair, and __text BL imm26 destination are byte-checked static edges. LC_FUNCTION_STARTS values are nearest-start candidates, not recovered symbol names. They do not prove runtime execution, receiver identity, dynamic dispatch outcome, or a complete preview/pipeline call chain.",
        },
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "references": len(references), "snippets": len(snippets), "unmatched": result["unmatched_targets"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
