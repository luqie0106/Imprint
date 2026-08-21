# -*- mode: python ; coding: utf-8 -*-

import os
import sys

block_cipher = None

# 收集需要打包的静态资源
datas = [('src', 'src')]
if os.path.exists('models/standard_aesthetic_model.onnx'):
    datas.append(('models/standard_aesthetic_model.onnx', 'models'))
if os.path.exists('models/custom_aesthetic_model.onnx'):
    datas.append(('models/custom_aesthetic_model.onnx', 'models'))
if os.path.exists('models/aesthetic_mlp.pth'):
    datas.append(('models/aesthetic_mlp.pth', 'models'))
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
        'qt_theme',
        'PySide6',
        'PySide6.QtCore',
        'PySide6.QtGui',
        'PySide6.QtWidgets',
        'cv2',
        'numpy',
        'onnxruntime',
        'rawpy',
        'pillow_heif',
        'pillow_jxl',
        'PIL',
        'PIL.Image',
        'PIL._imaging',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # 极致精简：排除 PyTorch 训练库以及大量未使用的 Qt6 大型子模块 (如 QML/Quick/3D/Multimedia/WebEngine/Pdf 等)
    excludes=[
        'torch', 'torchvision', 'transformers', 'huggingface_hub', 'safetensors', 'tkinter',
        'PySide6.QtNetwork',
        'PySide6.QtQml',
        'PySide6.QtQuick',
        'PySide6.QtQuickWidgets',
        'PySide6.Qt3DCore',
        'PySide6.Qt3DRender',
        'PySide6.Qt3DInput',
        'PySide6.Qt3DLogic',
        'PySide6.Qt3DAnimation',
        'PySide6.Qt3DExtras',
        'PySide6.QtMultimedia',
        'PySide6.QtMultimediaWidgets',
        'PySide6.QtPositioning',
        'PySide6.QtBluetooth',
        'PySide6.QtNfc',
        'PySide6.QtSensors',
        'PySide6.QtSerialPort',
        'PySide6.QtSerialBus',
        'PySide6.QtRemoteObjects',
        'PySide6.QtScxml',
        'PySide6.QtWebChannel',
        'PySide6.QtWebSockets',
        'PySide6.QtHttpServer',
        'PySide6.QtSpatialAudio',
        'PySide6.QtDesigner',
        'PySide6.QtHelp',
        'PySide6.QtPdf',
        'PySide6.QtPdfWidgets',
        'PySide6.QtSql',
        'PySide6.QtTest',
        'PySide6.QtXml',
        'PySide6.QtOpenGL',
        'PySide6.QtOpenGLWidgets',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# 图标配置
icon_file = 'src/assets/icon.ico' if os.name == 'nt' or sys.platform == 'win32' else 'src/assets/icon.icns'
if not os.path.exists(icon_file):
    icon_file = 'src/assets/icon.png' if os.path.exists('src/assets/icon.png') else None

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
    icon=icon_file,
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
    icon='src/assets/icon.icns' if os.path.exists('src/assets/icon.icns') else None,
    bundle_identifier='com.photosort.app',
    info_plist={
        'CFBundleDisplayName': 'PhotoSort',
        'CFBundleName': 'PhotoSort',
        'CFBundlePackageType': 'APPL',
        'CFBundleShortVersionString': '0.3.0',
        'NSHighResolutionCapable': 'True',
    },
)

