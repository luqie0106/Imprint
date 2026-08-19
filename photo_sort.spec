# -*- mode: python ; coding: utf-8 -*-

import os

block_cipher = None

# 收集需要打包的静态资源
datas = [('src', 'src')]
if os.path.exists('photo_sort_model.onnx'):
    datas.append(('photo_sort_model.onnx', '.'))
if os.path.exists('aesthetic_mlp.pth'):
    datas.append(('aesthetic_mlp.pth', '.'))

a = Analysis(
    ['main.py'],
    pathex=['src'],
    binaries=[],
    datas=datas,
    hiddenimports=[
        'app_gui',
        'burst_filter',
        'burst_gui',
        'trainer_gui',
        'model_manager',
        'onnx_exporter',
        'cv2',
        'numpy',
        'onnxruntime',
        'rawpy',
        'PIL',
        'PIL.Image',
        'PIL.ImageTk',
        'PIL._imaging',
        'tkinter',
        'tkinter.ttk',
        'tkinter.filedialog',
        'tkinter.messagebox',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # 排除庞大的训练框架以保持轻量化
    excludes=['torch', 'torchvision', 'transformers', 'huggingface_hub', 'safetensors'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='PhotoSort',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False, # GUI 应用无控制台窗口
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='PhotoSort',
)

app = BUNDLE(
    coll,
    name='PhotoSort.app',
    icon=None,
    bundle_identifier=None,
)
