import cv2
import time
from pathlib import Path
import numpy as np
import supervision as sv

from src.detection.detector import PPEDetector
from src.detection.tracker import WorkerTracker
from src.detection.violation import associate_ppe_to_worker, evaluate_compliance
from src.processing.frame_processor import FrameProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_frame_processor_unit():
    processor = FrameProcessor()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    tracked_persons = sv.Detections(
        xyxy=np.array([[100, 100, 200, 300]]),
        class_id=np.array([1]),
        confidence=np.array([0.9]),
        tracker_id=np.array([1])
    )
    ppe_detections = sv.Detections(
        xyxy=np.array([[110, 100, 150, 140]]),
        class_id=np.array([0]),
        confidence=np.array([0.85])
    )
    compliance_list = [{
        "worker_id": 1,
        "has_helmet": True,
        "has_vest": False,
        "status": "VIOLATION",
        "missing_items": ["safety-vest"]
    }]

    annotated = processor.annotate(
        frame=dummy_frame,
        tracked_persons=tracked_persons,
        ppe_detections=ppe_detections,
        compliance_list=compliance_list,
        fps=30.0,
        total_workers=1,
        compliant_count=0,
        violation_count=1
    )

    assert annotated is not None
    assert annotated.shape == dummy_frame.shape


def main():
    # 1) Init modules
    detector = PPEDetector(
        model_path=str(PROJECT_ROOT / "models" / "best.pt"),
        conf_threshold=0.4,
        device=None
    )
    tracker = WorkerTracker()
    frame_processor = FrameProcessor()

    # 2) Open video
    video_path = PROJECT_ROOT / "data" / "test_video.mp4"
    if not video_path.exists():
        video_path = PROJECT_ROOT / "data" / "test_video1.mp4"
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("Cannot open video. Check the path.")
        return

    print("===== FrameProcessor Video Test =====")
    print("Press 'q' to quit\n")

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ----- Detection -----
        all_detections = detector.detect(frame)

        if len(all_detections) > 0:
            person_mask = all_detections.class_id == 1
            ppe_mask = (all_detections.class_id == 0) | (all_detections.class_id == 2)
            person_detections = all_detections[person_mask]
            ppe_detections = all_detections[ppe_mask]
        else:
            person_detections = all_detections
            ppe_detections = all_detections

        # ----- Tracking -----
        tracked_persons = tracker.update(person_detections)

        # ----- Association + Compliance -----
        if len(tracked_persons) > 0:
            assignment = associate_ppe_to_worker(
                person_detections=tracked_persons,
                ppe_detections=ppe_detections,
                iob_threshold=0.15
            )
            compliance_list = evaluate_compliance(assignment)
        else:
            compliance_list = []

        # ----- Counts for HUD -----
        total_workers = len(compliance_list)
        compliant_count = sum(1 for w in compliance_list if w["status"] == "COMPLIANT")
        violation_count = sum(1 for w in compliance_list if w["status"] == "VIOLATION")

        # ----- FPS -----
        now = time.time()
        fps = 1.0 / (now - prev_time + 1e-6)
        prev_time = now

        # ----- Draw everything -----
        annotated = frame_processor.annotate(
            frame=frame,
            tracked_persons=tracked_persons,
            ppe_detections=ppe_detections,
            compliance_list=compliance_list,
            fps=fps,
            total_workers=total_workers,
            compliant_count=compliant_count,
            violation_count=violation_count
        )

        cv2.imshow("PPE FrameProcessor Video Test", annotated)

        if cv2.waitKey(60) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("===== Test Finished =====")


if __name__ == "__main__":
    main()