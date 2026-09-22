"""Optional local portrait-quality analysis for burst culling."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class EyeClosureResult:
    """Conservative per-photo eye state returned to the burst filter."""

    available: bool = False
    face_count: int = 0
    closed_face_count: int = 0
    uncertain_face_count: int = 0
    max_blink_score: float = 0.0


class EyeClosureEvaluator:
    """MediaPipe Face Landmarker wrapper with thread-safe, lazy loading."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        closed_threshold: float = 0.68,
        uncertain_threshold: float = 0.42,
        min_face_area_ratio: float = 0.004,
    ) -> None:
        if model_path is None:
            try:
                from model_manager import get_resolved_face_landmarker_path

                model_path = get_resolved_face_landmarker_path()
            except Exception:
                model_path = None
        self.model_path = Path(model_path) if model_path else None
        self.closed_threshold = float(closed_threshold)
        self.uncertain_threshold = float(uncertain_threshold)
        self.min_face_area_ratio = float(min_face_area_ratio)
        self._lock = threading.Lock()
        self._landmarker: Any | None = None
        self._mp: Any | None = None
        self._load_failed = False

    @property
    def available(self) -> bool:
        return bool(self.model_path and self.model_path.is_file() and not self._load_failed)

    def _ensure_landmarker(self) -> bool:
        if self._landmarker is not None:
            return True
        if not self.available:
            return False
        try:
            import mediapipe as mp

            options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(self.model_path)),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_faces=10,
                min_face_detection_confidence=0.55,
                min_face_presence_confidence=0.55,
                output_face_blendshapes=True,
            )
            self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
            self._mp = mp
            return True
        except Exception:
            self._load_failed = True
            self._landmarker = None
            self._mp = None
            return False

    @staticmethod
    def _face_area_ratio(landmarks: list[Any]) -> float:
        if not landmarks:
            return 0.0
        xs = [float(point.x) for point in landmarks]
        ys = [float(point.y) for point in landmarks]
        return max(0.0, max(xs) - min(xs)) * max(0.0, max(ys) - min(ys))

    @staticmethod
    def _blendshape_scores(categories: list[Any]) -> dict[str, float]:
        scores: dict[str, float] = {}
        for category in categories:
            name = getattr(category, "category_name", "") or getattr(category, "display_name", "")
            if name:
                scores[str(name)] = float(getattr(category, "score", 0.0))
        return scores

    def analyze(self, image_rgb: np.ndarray) -> EyeClosureResult:
        """Return closed-eye evidence; unavailable/ambiguous input never rejects."""
        with self._lock:
            if not self._ensure_landmarker():
                return EyeClosureResult(available=False)
            try:
                image = np.ascontiguousarray(image_rgb, dtype=np.uint8)
                mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=image)
                detected = self._landmarker.detect(mp_image)
            except Exception:
                return EyeClosureResult(available=False)

        landmarks_per_face = list(getattr(detected, "face_landmarks", []) or [])
        blendshapes_per_face = list(getattr(detected, "face_blendshapes", []) or [])
        result = EyeClosureResult(available=True, face_count=len(landmarks_per_face))

        for index, landmarks in enumerate(landmarks_per_face):
            if index >= len(blendshapes_per_face):
                result.uncertain_face_count += 1
                continue
            if self._face_area_ratio(landmarks) < self.min_face_area_ratio:
                result.uncertain_face_count += 1
                continue
            scores = self._blendshape_scores(blendshapes_per_face[index])
            left = scores.get("eyeBlinkLeft")
            right = scores.get("eyeBlinkRight")
            if left is None or right is None:
                result.uncertain_face_count += 1
                continue
            blink_score = max(left, right)
            result.max_blink_score = max(result.max_blink_score, blink_score)
            if blink_score >= self.closed_threshold:
                result.closed_face_count += 1
            elif blink_score >= self.uncertain_threshold:
                result.uncertain_face_count += 1

        return result

    def close(self) -> None:
        with self._lock:
            landmarker = self._landmarker
            self._landmarker = None
            if landmarker is not None and hasattr(landmarker, "close"):
                landmarker.close()
