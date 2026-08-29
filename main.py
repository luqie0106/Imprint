#!/usr/bin/env python3
"""
main.py — Imprint 纯命令行 CLI 批处理工具
支持在无 GUI 环境、自动化脚本及终端中直接调用连拍优选与模型管理。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# 确保控制台输出 UTF-8 编码，防止 Windows GBK 崩溃
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# 将 src 加入模块搜索路径
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _prompt_path(prompt: str) -> Path:
    while True:
        value = input(prompt).strip().strip("'\"")
        path = Path(value).expanduser()
        if path.exists() and path.is_dir():
            return path
        print("❌ 路径无效或目录不存在，请输入正确的文件夹路径。")


def run_burst_cli(
    input_dir: Path | str | None = None,
    review_subdir: str = "审查_连拍淘汰",
    gap_seconds: float = 1.5,
    max_hamming_distance: int = 12,
    keep_count: int = 1,
    max_workers: int | None = None,
    use_gpu: bool = True,
) -> None:
    from burst_filter import BurstFilter

    if not input_dir:
        print("=== Imprint RAW 连拍优选筛选 (CLI 交互模式) ===")
        target_path = _prompt_path("请输入照片所在目录路径: ")
    else:
        target_path = Path(str(input_dir).strip().strip("'\"")).expanduser()
        if not target_path.exists() or not target_path.is_dir():
            print(f"❌ 目标目录不存在: {target_path}")
            sys.exit(1)

    print(f"📁 目标照片目录: {target_path}")
    print(f"⚙️ 参数: 时间阈值={gap_seconds}s | 相似度距离={max_hamming_distance} | 保留={keep_count}张 | GPU加速={use_gpu}")

    def _progress(msg: str):
        print(f"  [进度] {msg}")

    filter_engine = BurstFilter(
        gap_seconds=gap_seconds,
        max_hamming_distance=max_hamming_distance,
        review_subdir=review_subdir,
        keep_count=keep_count,
        max_workers=max_workers,
        use_gpu=use_gpu,
        progress_callback=_progress,
    )

    result = filter_engine.run(target_path)

    print("\n" + "═" * 45)
    print("🎉 连拍筛选处理完成！")
    print(f"  📸 总扫描照片数：    {result.total}")
    print(f"  ⏩ 单张跳过（保留）：  {result.skipped_single}")
    print(f"  👥 发现连拍组数：    {result.burst_groups}")
    print(f"  📦 已移动淘汰照片：  {result.moved}")
    if result.review_dir:
        print(f"  📂 淘汰照片目录：    {result.review_dir}")
    if result.errors:
        print("  ⚠️ 处理警告:")
        for err in result.errors:
            print(f"     - {err}")
    print("═" * 45 + "\n")


def run_download_models_cli() -> None:
    from model_manager import download_clip_model
    print("🚀 开始下载/同步 CLIP 基础视觉模型到本地 models/ 目录...")

    def _cb(msg: str, pct: float):
        print(f"  [{pct*100:3.0f}%] {msg}")

    success = download_clip_model(use_mirror=True, progress_callback=_cb)
    if success:
        print("✅ CLIP 基础模型已成功就绪！")
    else:
        print("❌ 模型下载失败，请检查网络连接。")
        sys.exit(1)


def run_export_onnx_cli() -> None:
    from onnx_exporter import export_to_onnx
    try:
        out = export_to_onnx(project_root=PROJECT_ROOT)
        print(f"✅ ONNX 模型成功生成: {out}")
    except Exception as exc:
        print(f"❌ ONNX 导出失败: {exc}")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Imprint — RAW / JPEG / JXL / HIF 智能连拍优选与个人审美微调系统 (CLI 命令行工具)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("path", nargs="?", default=None, help="待筛选的照片目录路径（可选，未提供时进入交互输入）")
    parser.add_argument("--cli", action="store_true", help="以交互命令行模式运行连拍筛选")
    parser.add_argument("--gap", type=float, default=1.5, help="连拍时间判定阈值（秒，默认 1.5）")
    parser.add_argument("--hamming", type=int, default=12, help="dHash 汉明距离阈值 1~64（默认 12）")
    parser.add_argument("--keep", type=int, default=1, help="每组连拍保留最佳张数（默认 1）")
    parser.add_argument("--review-dir", type=str, default="审查_连拍淘汰", help="淘汰照片存放子目录名称")
    parser.add_argument("--workers", type=int, default=None, help="并发分析线程数（默认自动匹配 CPU）")
    parser.add_argument("--no-gpu", action="store_true", help="禁用 GPU 硬件加速，使用纯 CPU 模式")
    parser.add_argument("--download-models", action="store_true", help="下载基础视觉底座模型至本地 models/ 目录")
    parser.add_argument("--export-onnx", action="store_true", help="从本地 MLP 权重熔铸并导出 ONNX 格式模型")

    args = parser.parse_args()

    if args.download_models:
        run_download_models_cli()
    elif args.export_onnx:
        run_export_onnx_cli()
    else:
        run_burst_cli(
            input_dir=args.path,
            review_subdir=args.review_dir,
            gap_seconds=args.gap,
            max_hamming_distance=args.hamming,
            keep_count=args.keep,
            max_workers=args.workers,
            use_gpu=not args.no_gpu,
        )


if __name__ == "__main__":
    main()
