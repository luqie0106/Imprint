from types import SimpleNamespace

import numpy as np

from src.portrait_quality import EyeClosureEvaluator


class _FakeMp:
    class ImageFormat:
        SRGB = "srgb"

    @staticmethod
    def Image(*, image_format, data):
        assert image_format == "srgb"
        return data


def _face(size: float = 0.4):
    return [
        SimpleNamespace(x=0.2, y=0.2),
        SimpleNamespace(x=0.2 + size, y=0.2 + size),
    ]


def _blendshapes(left: float, right: float):
    return [
        SimpleNamespace(category_name="eyeBlinkLeft", display_name="", score=left),
        SimpleNamespace(category_name="eyeBlinkRight", display_name="", score=right),
    ]


def _evaluator(detected):
    evaluator = EyeClosureEvaluator()
    evaluator._mp = _FakeMp
    evaluator._landmarker = SimpleNamespace(detect=lambda _image: detected)
    return evaluator


def test_clear_blink_is_reported_as_closed():
    detected = SimpleNamespace(
        face_landmarks=[_face()],
        face_blendshapes=[_blendshapes(0.12, 0.91)],
    )
    result = _evaluator(detected).analyze(np.zeros((100, 100, 3), dtype=np.uint8))
    assert result.available
    assert result.face_count == 1
    assert result.closed_face_count == 1
    assert result.uncertain_face_count == 0
    assert result.max_blink_score == 0.91


def test_small_face_is_uncertain_instead_of_closed():
    detected = SimpleNamespace(
        face_landmarks=[_face(0.02)],
        face_blendshapes=[_blendshapes(0.95, 0.95)],
    )
    result = _evaluator(detected).analyze(np.zeros((100, 100, 3), dtype=np.uint8))
    assert result.available
    assert result.closed_face_count == 0
    assert result.uncertain_face_count == 1


def test_missing_model_safely_disables_analysis():
    result = EyeClosureEvaluator(model_path=None).analyze(
        np.zeros((100, 100, 3), dtype=np.uint8)
    )
    assert not result.available
    assert result.closed_face_count == 0
