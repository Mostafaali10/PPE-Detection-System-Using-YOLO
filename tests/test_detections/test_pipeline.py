# tests/test_pipeline.py

import cv2
import numpy as np
import supervision as sv
from src.detection.detector import PPEDetector
from src.detection.tracker import WorkerTracker
from src.detection.violation import associate_ppe_to_worker, evaluate_compliance


def main():
    # 1. Load models
    detector = PPEDetector(
        model_path=r"C:\Users\aa683\Desktop\ppe_detection\models\best.pt",
        conf_threshold=0.4,
        device="0"
    )
    tracker = WorkerTracker()

    # 2. Open video
    video_path = r"C:\Users\aa683\Desktop\ppe_detection\data\test_video1.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Cannot open video. Check the path.")
        return

    # Drawing tools
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5)

    frame_count = 0
    print("\n===== Full Pipeline Test Started =====")
    print("Press 'q' to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # ---------- 1. Detection ----------
        all_detections = detector.detect(frame)

        # Split detections
        if len(all_detections) > 0:
            person_mask = all_detections.class_id == 1
            ppe_mask = (all_detections.class_id == 0) | (all_detections.class_id == 2)

            person_detections = all_detections[person_mask]
            ppe_detections = all_detections[ppe_mask]
        else:
            person_detections = all_detections
            ppe_detections = all_detections

        # ---------- 2. Tracking ----------
        tracked_persons = tracker.update(person_detections)

        # ---------- 3. Association + Compliance ----------
        if len(tracked_persons) > 0:
            assignment = associate_ppe_to_worker(
                person_detections=tracked_persons,
                ppe_detections=ppe_detections,
                iob_threshold=0.15
            )
            compliance_list = evaluate_compliance(assignment)
        else:
            compliance_list = []

        # ---------- 4. Print results every 20 frames ----------
        if frame_count % 20 == 0:
            print(f"\n----- Frame {frame_count} -----")
            if len(compliance_list) == 0:
                print("No workers detected")
            for worker in compliance_list:
                helmet = "✓" if worker["has_helmet"] else "✗"
                vest = "✓" if worker["has_vest"] else "✗"
                print(f"Worker #{worker['worker_id']:2d} | "
                      f"Helmet: {helmet} | Vest: {vest} | "
                      f"Status: {worker['status']}")

        # ---------- 5. Draw on frame ----------
        labels = []
        if tracked_persons.tracker_id is not None:
            # Create a quick lookup for status
            status_dict = {w["worker_id"]: w["status"] for w in compliance_list}

            for tid in tracked_persons.tracker_id:
                status = status_dict.get(int(tid), "UNKNOWN")
                labels.append(f"#{tid} {status}")

        annotated = box_annotator.annotate(frame.copy(), tracked_persons)
        annotated = label_annotator.annotate(annotated, tracked_persons, labels)

        # Show
        cv2.imshow("PPE Pipeline Test", annotated)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n===== Pipeline Test Finished =====")


if __name__ == "__main__":
    main()