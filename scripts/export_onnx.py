#!/usr/bin/env python3
"""
export_onnx.py

合并 CLIP 视觉模型和我们自己训练的 AestheticMLP，
并将其导出为一个统一的 ONNX 模型。
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

try:
    from onnx_exporter import export_to_onnx
except ImportError as err:
    print(f"Error: 无法导入 onnx_exporter: {err}")
    sys.exit(1)


def main():
    try:
        onnx_file = export_to_onnx(project_root=_ROOT)
        print(f"\n🎉 导出成功: {onnx_file}")
    except Exception as exc:
        print(f"\n❌ 导出失败: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
