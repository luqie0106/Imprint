import argparse
import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"


def _ensure_src_on_path() -> None:
    if str(SRC_PATH) not in sys.path:
        sys.path.insert(0, str(SRC_PATH))


def _prompt_path(prompt: str) -> Path:
    while True:
        value = input(prompt).strip()
        path = Path(value).expanduser()
        if path.exists() and path.is_dir():
            return path
        print("路径无效，请输入存在的文件夹路径。")


def _prompt_float(prompt: str, default: float) -> float:
    raw = input(prompt).strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        print("输入无效，使用默认值。")
        return default


def run_burst_console() -> None:
    _ensure_src_on_path()
    BurstFilter = importlib.import_module("burst_filter").BurstFilter

    print("=== RAW 连拍优选筛选（NEF / ARW / CR3 / RAF）===")
    input_dir = _prompt_path("RAW 文件目录路径: ")
    review_subdir = input('淘汰子目录名称（默认"审查_连拍淘汰"）: ').strip() or "审查_连拍淘汰"
    gap = _prompt_float("连拍时间阈值，秒（默认1.5）: ", 1.5)
    hamming = int(_prompt_float("dHash 汉明距离限制 1~64（默认12）: ", 12.0))
    keep_count = int(_prompt_float("每组保留张数（默认1）: ", 1.0))

    result = BurstFilter(
        gap_seconds=gap,
        max_hamming_distance=hamming,
        review_subdir=review_subdir,
        keep_count=keep_count,
    ).run(input_dir)

    print("\n── 处理完成 ────────────────────────────────")
    print(f"  总 RAW 文件数：    {result.total}")
    print(f"  单拍跳过（保留）：  {result.skipped_single}")
    print(f"  连拍组数：         {result.burst_groups}")
    print(f"  已移动淘汰数：     {result.moved}")
    if result.review_dir:
        print(f"  淘汰目录：         {result.review_dir}")
    for err in result.errors:
        print(f"  ⚠ {err}")


def run_download_models_cli() -> None:
    _ensure_src_on_path()
    download_clip_model = importlib.import_module("model_manager").download_clip_model
    print("开始下载 CLIP 基础模型到本地 models/ 目录...")
    def _cb(msg: str, pct: float):
        print(f"[{pct*100:3.0f}%] {msg}")
    success = download_clip_model(use_mirror=True, progress_callback=_cb)
    if success:
        print("✅ 模型下载并校验成功！")
    else:
        print("❌ 模型下载失败，请检查网络连接。")


def run_export_onnx_cli() -> None:
    _ensure_src_on_path()
    export_to_onnx = importlib.import_module("onnx_exporter").export_to_onnx
    try:
        out = export_to_onnx(project_root=PROJECT_ROOT)
        print(f"✅ ONNX 模型成功生成: {out}")
    except Exception as exc:
        print(f"❌ ONNX 导出失败: {exc}")


def run_gui() -> None:
    _ensure_src_on_path()
    try:
        importlib.import_module("app_gui").launch_main_gui()
    except Exception as exc:
        print(f"综合 GUI 启动遇到问题 ({exc})，尝试启动经典连拍界面...")
        try:
            importlib.import_module("burst_gui").launch_burst_gui()
        except Exception:
            print("GUI 无法启动，已切换到命令行模式。")
            run_burst_console()


def main() -> None:
    parser = argparse.ArgumentParser(description="Photo Sort — RAW 智能连拍优选与个人审美系统")
    parser.add_argument("--cli", action="store_true", help="以命令行模式运行连拍筛选")
    parser.add_argument("--download-models", action="store_true", help="下载基础 CLIP 模型至本地 models 目录")
    parser.add_argument("--export-onnx", action="store_true", help="从本地 MLP 权重熔铸并导出 photo_sort_model.onnx")
    parser.add_argument("--train-gui", action="store_true", help="直接打开偏好训练器独立窗口")
    
    args, unknown = parser.parse_known_args()

    if args.download_models:
        run_download_models_cli()
    elif args.export_onnx:
        run_export_onnx_cli()
    elif args.cli:
        run_burst_console()
    elif args.train_gui:
        _ensure_src_on_path()
        importlib.import_module("trainer_gui").TrainerGUI().run()
    else:
        run_gui()


if __name__ == "__main__":
    main()
