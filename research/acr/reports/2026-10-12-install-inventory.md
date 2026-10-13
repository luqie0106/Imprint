# Adobe Camera Raw installation inventory (2026-10-12)

## Scope and method

Read-only inventory of the installed Camera Raw plug-in at `/Library/Application Support/Adobe/Plug-Ins/CC/File Formats/Camera Raw.plugin` and the shared and per-user `Adobe/CameraRaw` support directories. The plugin tree contained 138 regular files and 0 symlinks. Every regular file was classified with `file`; native executable formats were then inspected with `lipo`, `otool -L`, SHA-256, size, and modification time. Both plugin Info.plists were read. GPU Config XMP was parsed for selected GPU fields; only selected GPU/Metal status fields were extracted from the latest Camera Raw logs. Log bodies were not copied into the evidence file.

No Adobe file was modified and no Adobe application was launched.

## Installed plug-in

- Bundle: `/Library/Application Support/Adobe/Plug-Ins/CC/File Formats/Camera Raw.plugin`
- Version: `17.2 (2155)` (`17.2f2155`); bundle identifier `com.adobe.CameraRaw`.
- Both `Contents/Info.plist` and `Contents/Resources/Info.plist` were inspected; the full selected version metadata is in `inventory.json`.
- Native binaries found by file magic: **1**.

| Relative path | Architecture | Size (bytes) | SHA-256 | File mtime (+08:00) |
|---|---|---:|---|---|
| `Contents/MacOS/Camera Raw` | x86_64, arm64 | 290,177,856 | `c0d0bdad561b3e2948b21b6a4b8a267a86ad4f4f2d99b890632dcaa88f431234` | `2025-01-31T11:06:13+08:00` |

### Imported dependencies

**x86_64:** `/System/Library/Frameworks/AVKit.framework/Versions/A/AVKit`, `/System/Library/Frameworks/CoreML.framework/Versions/A/CoreML`, `/System/Library/Frameworks/VideoToolbox.framework/Versions/A/VideoToolbox`, `/System/Library/Frameworks/CoreMedia.framework/Versions/A/CoreMedia`, `/System/Library/Frameworks/AVFoundation.framework/Versions/A/AVFoundation`, `/System/Library/Frameworks/Metal.framework/Versions/A/Metal`, `/System/Library/Frameworks/QuartzCore.framework/Versions/A/QuartzCore`, `/System/Library/Frameworks/Carbon.framework/Versions/A/Carbon`, `/System/Library/Frameworks/ApplicationServices.framework/Versions/A/ApplicationServices`, `/usr/lib/libSystem.B.dylib`, `/System/Library/Frameworks/CoreServices.framework/Versions/A/CoreServices`, `/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate`, `/System/Library/Frameworks/Cocoa.framework/Versions/A/Cocoa`, `/System/Library/Frameworks/OpenGL.framework/Versions/A/OpenGL`, `/System/Library/Frameworks/Foundation.framework/Versions/C/Foundation`, `/usr/lib/libobjc.A.dylib`, `/usr/lib/libc++.1.dylib`, `/System/Library/Frameworks/AppKit.framework/Versions/C/AppKit`, `/System/Library/Frameworks/ColorSync.framework/Versions/A/ColorSync`, `/System/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation`, `/System/Library/Frameworks/CoreGraphics.framework/Versions/A/CoreGraphics`, `/System/Library/Frameworks/CoreVideo.framework/Versions/A/CoreVideo`
**arm64:** `/System/Library/Frameworks/AVKit.framework/Versions/A/AVKit`, `/System/Library/Frameworks/CoreML.framework/Versions/A/CoreML`, `/System/Library/Frameworks/VideoToolbox.framework/Versions/A/VideoToolbox`, `/System/Library/Frameworks/CoreMedia.framework/Versions/A/CoreMedia`, `/System/Library/Frameworks/AVFoundation.framework/Versions/A/AVFoundation`, `/System/Library/Frameworks/Metal.framework/Versions/A/Metal`, `/System/Library/Frameworks/QuartzCore.framework/Versions/A/QuartzCore`, `/System/Library/Frameworks/Carbon.framework/Versions/A/Carbon`, `/System/Library/Frameworks/ApplicationServices.framework/Versions/A/ApplicationServices`, `/usr/lib/libSystem.B.dylib`, `/System/Library/Frameworks/CoreServices.framework/Versions/A/CoreServices`, `/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate`, `/System/Library/Frameworks/Cocoa.framework/Versions/A/Cocoa`, `/System/Library/Frameworks/OpenGL.framework/Versions/A/OpenGL`, `/System/Library/Frameworks/Foundation.framework/Versions/C/Foundation`, `/usr/lib/libobjc.A.dylib`, `/usr/lib/libc++.1.dylib`, `/System/Library/Frameworks/AppKit.framework/Versions/C/AppKit`, `/System/Library/Frameworks/ColorSync.framework/Versions/A/ColorSync`, `/System/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation`, `/System/Library/Frameworks/CoreGraphics.framework/Versions/A/CoreGraphics`, `/System/Library/Frameworks/CoreVideo.framework/Versions/A/CoreVideo`

The bundle root is the only `.plugin` package found. `Contents/Frameworks` exists and is empty; no nested `.framework`, `.bundle`, `.app`, `.xpc`, helper, `.dylib`, or `.so` entries and no symlinks were found inside the plug-in tree.

## GPU Config and log evidence

Shared support directory: `/Library/Application Support/Adobe/CameraRaw`. It exists; no `Camera Raw GPU Config.txt` or `Camera Raw Log Latest` files were found there.

Per-user support directory: `~/Library/Application Support/Adobe/CameraRaw`.

- `~/Library/Application Support/Adobe/CameraRaw/GPU/Adobe Photoshop Camera Raw/Camera Raw GPU Config.txt` — mtime `2026-10-03T14:05:58+08:00`; `gpu_preferred_system=`, `gpu_compute_quick_self_test_passed=True`, `gpu_init_digest=5EF5FCFC55A72CEAF00E326A08AC9F10`, `gpu_compute_digest=5E898912B879C13867C4CAB57B321ADF`, `gpu_hdr_display_scale=1`.
- `~/Library/Application Support/Adobe/CameraRaw/GPU/Adobe Photoshop Lightroom Classic/Camera Raw GPU Config.txt` — mtime `2026-08-05T11:26:04+08:00`; `gpu_preferred_system=`, `gpu_compute_quick_self_test_passed=True`, `gpu_init_digest=562A8F3EE4E5C94981E2259E3253BEB1`, `gpu_compute_digest=60E5A3CE02D43A08A3BFDB9C5940086C`, `gpu_hdr_display_scale=1`.

Latest Camera Raw log files, summarized without copying raw log text:

- `~/Library/Application Support/Adobe/CameraRaw/Logs/Camera Raw Log Latest v1.txt` — mtime `2026-10-11T20:22:57+08:00` (645 lines); CPU `Apple M2 Pro`, GPU `Apple M2 Pro`, device count `1`, loaded system `Metal`, GPU primary driver version `15.8.1`. GPU init `I2_GPU2` → `I4_GPU4`; final GPU3/GPU4 hard and soft checks all `success`; Metal compute pipeline states `257/257` completed.
- `~/Library/Application Support/Adobe/CameraRaw/Logs/Camera Raw Log Latest v0.txt` — mtime `2026-10-11T20:22:37+08:00` (53 lines); SDK `Adobe Photoshop Camera Raw 17.2 (Macintosh), version 17.2x2155, 64-bit, Release`, ML default device `GPU`, dereflect device `GPU`.

The newer v1 log records `Loaded GPU system: Metal`, one Apple M2 Pro device, GPU3/GPU4 final hard and soft status `success`, and completion of 257 compute pipeline states. The matching v0 log records Camera Raw SDK 17.2x2155 and GPU as the ML and dereflect inference device. Filesystem modification times are shown as evidence dates; they do not prove that ACR is running now.

## Observed facts, inference, and unknowns

**Observed facts:** The installed plug-in is version 17.2 (2155), with one universal Mach-O bundle executable containing x86_64 and arm64 slices. Its direct dynamic dependencies include Metal.framework. The newest recorded Camera Raw log (mtime 2026-10-11 20:22:57 +08:00) says its Metal GPU system initialized on Apple M2 Pro and reports final success statuses; both per-user GPU Config files record `gpu_compute_quick_self_test_passed=True`.

**Inference:** The log supports that the particular recorded Camera Raw session initialized its Metal GPU path successfully at that time. The binary's Metal.framework dependency by itself would not prove runtime use; the log provides the runtime-session evidence.

**Unknown:** The inventory does not establish whether ACR or a host application is running now, which host invoked the plugin, or whether every image operation used the GPU. The persisted GPU Config files are not a current live health check.

Machine-readable evidence: [`inventory.json`](evidence/inventory.json).
