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
    print(f"  总 NEF 文件数：    {result.total}")
    print(f"  单拍跳过（保留）：  {result.skipped_single}")
    print(f"  连拍组数：         {result.burst_groups}")
    print(f"  已移动淘汰数：     {result.moved}")
    if result.review_dir:
        print(f"  淘汰目录：         {result.review_dir}")
    for err in result.errors:
        print(f"  ⚠ {err}")


def run_gui() -> None:
    _ensure_src_on_path()
    try:
        importlib.import_module("burst_gui").launch_burst_gui()
    except Exception:
        print("GUI 启动失败，已切换到命令行模式。")
        run_burst_console()


if __name__ == "__main__":
    if "--cli" in sys.argv:
        run_burst_console()
    else:
        run_gui()
