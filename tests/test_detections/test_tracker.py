# tests/test_tracker.py

import cv2
import supervision as sv
from src.detection.detector import PPEDetector
from src.detection.tracker import WorkerTracker

# 1. Create Detector and Tracker
detector = PPEDetector(
    model_path=r"C:\Users\aa683\Desktop\ppe_detection\models\best.pt",
    conf_threshold=0.4,
    device="0"
)

tracker = WorkerTracker()

# 2. Open video
video_path = r"C:\Users\aa683\Desktop\ppe_detection\data\test_video.mp4"
cap = cv2.VideoCapture(video_path)

# Use webcam if no video:
# cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open video. Check the path.")
    exit()

# Annotators (for drawing)
box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_scale=0.5)

frame_count = 0
print("\n===== Starting Tracker Test =====")
print("Press 'q' to quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Detect
    all_detections = detector.detect(frame)

    # Keep only Person (class_id = 1)
    if len(all_detections) > 0:
        person_mask = all_detections.class_id == 1
        person_detections = all_detections[person_mask]
    else:
        person_detections = all_detections

    # Track
    tracked = tracker.update(person_detections)

    # Create labels with Tracker ID
    labels = []
    if tracked.tracker_id is not None:
        for tracker_id in tracked.tracker_id:
            labels.append(f"Worker #{tracker_id}")

    # Draw boxes + labels
    annotated_frame = box_annotator.annotate(frame.copy(), tracked)
    annotated_frame = label_annotator.annotate(annotated_frame, tracked, labels)

    # Show the frame
    cv2.imshow("Tracker Test", annotated_frame)

    # Print every 15 frames
    if frame_count % 15 == 0:
        print(f"Frame {frame_count:3d} → Tracker IDs: {tracked.tracker_id}")

    # Press 'q' to quit
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\n===== Tracker Test Finished =====")