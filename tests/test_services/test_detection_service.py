import cv2
from pathlib import Path
import numpy as np
import pytest
from src.services.detection_service import DetectionService
from src.processing.video_processor import get_frame_stream

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_detection_service_empty_frame():
    service = DetectionService(
        model_path=str(PROJECT_ROOT / "models" / "best.pt"),
        conf_threshold=0.4,
        device=None,
        iob_threshold=0.15,
        snapshot_cooldown_sec=5.0
    )
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    annotated, stats = service.process_frame(dummy_frame)
    assert annotated is not None
    assert "active_workers" in stats
    assert "compliance_rate" in stats


def main():
    service = DetectionService(
        model_path=str(PROJECT_ROOT / "models" / "best.pt"),
        conf_threshold=0.4,
        device=None,
        iob_threshold=0.15,
        snapshot_cooldown_sec=5.0
    )

    video_path = PROJECT_ROOT / "data" / "test_video1.mp4"
    if not video_path.exists():
        video_path = PROJECT_ROOT / "data" / "test_video.mp4"
    frame_id = 0

    print("===== DetectionService Test =====")
    print("Press 'q' to quit\n")

    for frame in get_frame_stream(video_path):
        frame_id += 1

        annotated, stats = service.process_frame(frame)

        # Print every 20 frames
        if frame_id % 20 == 0:
            print(
                f"Frame {frame_id:3d} | "
                f"Workers={stats['active_workers']} | "
                f"C={stats['compliant_count']} "
                f"V={stats['violation_count']} "
                f"U={stats['unknown_count']} | "
                f"Unique={stats['unique_workers_seen']} | "
                f"Snaps={stats['total_violations_recorded']} | "
                f"Rate={stats['compliance_rate']}% | "
                f"FPS={stats['fps']}"
            )

        cv2.imshow("DetectionService Test", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print("\n===== Finished =====")
    print(f"Unique workers seen: {stats['unique_workers_seen']}")
    print(f"Total snapshots saved: {stats['total_violations_recorded']}")
    print(f"Final compliance rate: {stats['compliance_rate']}%")


if __name__ == "__main__":
    main()