# tests/test_detections/test_pipeline.py
from pathlib import Path
import numpy as np
import pytest

from src.services.detection_service import DetectionService
from src.processing.video_processor import get_frame_stream

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"
VIDEO_PATH = PROJECT_ROOT / "data" / "test_video.mp4"


@pytest.mark.integration
def test_full_pipeline_short_clip():
    if not MODEL_PATH.exists():
        pytest.skip("Model not found")
    if not VIDEO_PATH.exists():
        pytest.skip("Test video not found")

    service = DetectionService(
        model_path=str(MODEL_PATH),
        conf_threshold=0.4,
        device=None,
        iob_threshold=0.15,
        snapshot_cooldown_sec=10.0,
    )

    frame_count = 0
    last_stats = None

    for frame in get_frame_stream(VIDEO_PATH):
        annotated, stats = service.process_frame(frame)

        assert isinstance(annotated, np.ndarray)
        assert annotated.ndim == 3
        assert "active_workers" in stats
        assert "compliance_rate" in stats
        assert "violation_count" in stats

        last_stats = stats
        frame_count += 1

        if frame_count >= 20:
            break

    assert frame_count > 0
    assert last_stats is not None