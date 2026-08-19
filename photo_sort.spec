# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=['src'],
    binaries=[],
    datas=[
        ('photo_sort_model.onnx', '.') if os.path.exists('photo_sort_model.onnx') else None,
        ('aesthetic_mlp.pth', '.') if os.path.exists('aesthetic_mlp.pth') else None,
        ('src', 'src'),
    ],
    datas=[d for d in [
        ('photo_sort_model.onnx', '.') if os.path.exists('photo_sort_model.onnx') else None,
        ('aesthetic_mlp.pth', '.') if os.path.exists('aesthetic_mlp.pth') else None,
        ('src', 'src'),
    ] if d is not None],
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
    # Exclude PyTorch and Transformers from the lightweight distribution build
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
    console=False, # Hide console for GUI app
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
