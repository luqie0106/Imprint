from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class BurstConfig:
    input_dir: Path
    review_subdir: str = "审查_连拍淘汰"
    time_gap_seconds: float = 1.5
    similarity_threshold: float = 0.85
