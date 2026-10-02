# test_detector.py

import cv2
import numpy as np
import time
from src.detection.detector import PPEDetector

# 1. Create the Detector (only once)
detector = PPEDetector(
    model_path=r"C:\Users\aa683\Desktop\ppe_detection\models\best.pt",
    conf_threshold=0.4,
    device="0"          # change to "cpu" if you want
)

print("\n===== Test 1: Real Image =====")
# Put the path of any image that contains workers
image_path = r"C:\Users\aa683\Desktop\ppe_detection\data\processed\test\images\youtube-108_jpg.rf.9dc7ed5f816f07d520f3dbfaad08d40f.jpg"
frame = cv2.imread(image_path)

if frame is None:
    print("Image not found, please change the path")
else:
    detections = detector.detect(frame)
    print(f"Number of detections: {len(detections)}")
    print(f"Class IDs: {detections.class_id}")
    print(f"Confidences: {detections.confidence}")

print("\n===== Test 2: Black (Empty) Image =====")
black_frame = np.zeros((640, 640, 3), dtype=np.uint8)
detections_black = detector.detect(black_frame)
print(f"Number of detections: {len(detections_black)}")   # should be 0

print("\n===== Test 3: Speed Measurement =====")
if frame is not None:
    times = []
    for i in range(20):
        start = time.time()
        _ = detector.detect(frame)
        times.append(time.time() - start)

    avg_ms = (sum(times) / len(times)) * 1000
    print(f"Average time per frame: {avg_ms:.1f} ms")