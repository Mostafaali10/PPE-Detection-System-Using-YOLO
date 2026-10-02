# tests/test_processing/test_frame_processor_video.py

import cv2
import time
from src.detection.detector import PPEDetector
from src.detection.tracker import WorkerTracker
from src.detection.violation import associate_ppe_to_worker, evaluate_compliance
from src.processing.frame_processor import FrameProcessor


def main():
    # 1) Init modules
    detector = PPEDetector(
        model_path=r"C:\Users\aa683\Desktop\ppe_detection\models\best.pt",
        conf_threshold=0.4,
        device="0"
    )
    tracker = WorkerTracker()
    frame_processor = FrameProcessor()

    # 2) Open video
    video_path = r"C:\Users\aa683\Desktop\ppe_detection\data\test_video.mp4"
    cap = cv2.VideoCapture(video_path)

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