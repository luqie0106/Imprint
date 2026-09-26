# -*- mode: python ; coding: utf-8 -*-
import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_all

block_cipher = None

datas = [
    ('src/burst_filter.py',   'src'),
    ('src/model_manager.py',  'src'),
    ('src/onnx_exporter.py',  'src'),
    ('src/exif_reader.py',    'src'),
    ('src/config.py',         'src'),
    ('src/dehaze.py',         'src'),
    ('src/image_io.py',       'src'),
    ('src/dng_writer.py',     'src'),
    ('src/lens_correction.py', 'src'),
    ('src/portrait_quality.py', 'src'),
    ('src/measured_response.py', 'src'),
    ('assets/acr_measured/hsl_preview_lut.npz', 'assets/acr_measured'),
    ('assets/acr_measured/grading_preview_lut.npz', 'assets/acr_measured'),
    ('assets/ricoh/Ricoh_GR2_CameraRaw_Presets/Presets', 'assets/ricoh/Ricoh_GR2_CameraRaw_Presets/Presets'),
    ('assets/ricoh/Ricoh_GR3_CameraRaw_Presets/Presets', 'assets/ricoh/Ricoh_GR3_CameraRaw_Presets/Presets'),
    ('third_party/lensfun-db', 'third_party/lensfun-db'),
    ('third_party/licenses', 'third_party/licenses'),
    ('THIRD_PARTY_NOTICES.md', '.'),
    ('LICENSE', '.'),
]

try:
    lensfun_datas, lensfun_binaries, lensfun_hiddenimports = collect_all('lensfunpy')
    datas.extend(lensfun_datas)
except Exception:
    lensfun_binaries = []
    lensfun_hiddenimports = []

try:
    mediapipe_datas, mediapipe_binaries, mediapipe_hiddenimports = collect_all('mediapipe')
    datas.extend(mediapipe_datas)
except Exception:
    mediapipe_binaries = []
    mediapipe_hiddenimports = []

try:
    imagecodecs_datas, imagecodecs_binaries, imagecodecs_hiddenimports = collect_all('imagecodecs')
    datas.extend(imagecodecs_datas)
except Exception:
    imagecodecs_binaries = []
    imagecodecs_hiddenimports = []
# 打包标准 ONNX 模型（如果存在）
for model_rel in [
    'models/standard_aesthetic_model.onnx',
    'models/standard_aesthetic_l14_model.onnx',
]:
    if os.path.exists(model_rel):
        datas.append((model_rel, 'models'))

native_binaries = []
candidate = Path('native-renderer/build/libimprint_renderer.dylib')
if candidate.is_file():
    native_binaries.append((str(candidate), '.'))
for build_dir in (Path('native-renderer/build'), Path('native-renderer/build/Release')):
    for candidate in build_dir.glob('imprint_renderer*.dll'):
        if candidate.is_file():
            native_binaries.append((str(candidate), '.'))
for candidate in (
    Path('native-sort/build/libimprint_sort.dylib'),
    Path('native-sort/build/libimprint_sort.so'),
    Path('native-sort/build/imprint_sort.dll'),
    Path('native-sort/build/Release/imprint_sort.dll'),
):
    if candidate.is_file():
        native_binaries.append((str(candidate), '.'))

a = Analysis(
    ['src/app_api.py'],
    pathex=['src'],
    binaries=lensfun_binaries + mediapipe_binaries + imagecodecs_binaries + native_binaries,
    datas=datas,
    hiddenimports=[
        'burst_filter',
        'model_manager',
        'onnx_exporter',
        'exif_reader',
        'config',
        'dehaze',
        'dehaze_gpu',
        'native_renderer',
        'native_sort',
        'image_io',
        'dng_writer',
        'lens_correction',
        'portrait_quality',
        'ricoh_filter',
        'measured_response',
        *lensfun_hiddenimports,
        *mediapipe_hiddenimports,
        *imagecodecs_hiddenimports,
        'uvicorn',
        'uvicorn.logging',
        'uvicorn.loops',
        'uvicorn.loops.auto',
        'uvicorn.protocols',
        'uvicorn.protocols.http',
        'uvicorn.protocols.http.auto',
        'uvicorn.protocols.websockets',
        'uvicorn.protocols.websockets.auto',
        'uvicorn.lifespan',
        'uvicorn.lifespan.on',
        'fastapi',
        'cv2',
        'numpy',
        'rawpy',
        'onnxruntime',
        'onnx',
        'onnx.helper',
        'onnx.numpy_helper',
        'PIL',
        'PIL.Image',
        'PIL._imaging',
        'pillow_heif',
        'pillow_jxl',
        'huggingface_hub',
        'starlette',
        'anyio',
        'anyio._backends._asyncio',
        'h11',
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        'torch', 'torchvision', 'transformers',
        'PySide6', 'tkinter', 'matplotlib',
    ],
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# All release platforms use onedir. Besides faster startup, keeping Lensfun as a
# separate shared library lets recipients replace/relink the LGPL component.
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='imprint_api',
    debug=False,
    strip=False,
    upx=True,
    console=True,   # sidecar needs stdout to announce its port
    argv_emulation=False,
    target_arch=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='imprint_api',
)
