# tests/test_detections/test_detector.py
import numpy as np
import pytest
from pathlib import Path
import cv2

from src.detection.detector import PPEDetector

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"
TEST_IMAGE = PROJECT_ROOT / "data" / "processed" / "test" / "images" / "youtube-108_jpg.rf.9dc7ed5f816f07d520f3dbfaad08d40f.jpg"


@pytest.fixture(scope="module")
def detector():
    if not MODEL_PATH.exists():
        pytest.skip("Model weights not found")
    return PPEDetector(
        model_path=str(MODEL_PATH),
        conf_threshold=0.4,
        device=None
    )


def test_detector_handles_empty_frame(detector):
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    dets = detector.detect(frame)
    assert len(dets) == 0


def test_detector_none_frame_returns_empty(detector):
    dets = detector.detect(None)
    assert len(dets) == 0


def test_detector_valid_image_returns_detections(detector):
    if not Path(TEST_IMAGE).exists():
        pytest.skip("Test image not found")

    frame = cv2.imread(TEST_IMAGE)
    assert frame is not None

    dets = detector.detect(frame)
    assert len(dets) >= 1
    assert dets.xyxy is not None
    assert dets.class_id is not None
    assert dets.confidence is not None