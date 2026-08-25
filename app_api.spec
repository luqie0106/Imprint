# -*- mode: python ; coding: utf-8 -*-
import os

block_cipher = None

datas = [
    ('src/burst_filter.py',   'src'),
    ('src/model_manager.py',  'src'),
    ('src/onnx_exporter.py',  'src'),
    ('src/exif_reader.py',    'src'),
    ('src/config.py',         'src'),
]
# 打包标准 ONNX 模型（如果存在）
for model_rel in [
    'models/standard_aesthetic_model.onnx',
    'models/standard_aesthetic_l14_model.onnx',
]:
    if os.path.exists(model_rel):
        datas.append((model_rel, 'models'))

a = Analysis(
    ['src/app_api.py'],
    pathex=['src'],
    binaries=[],
    datas=datas,
    hiddenimports=[
        'burst_filter',
        'model_manager',
        'onnx_exporter',
        'exif_reader',
        'config',
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

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='imprint_api',
    debug=False,
    strip=False,
    upx=True,
    console=True,   # sidecar 需要 stdout 输出端口信息，必须 True
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
