from ultralytics import YOLO
from pathlib import Path

def main():
    # مسار أفضل نموذج تم تدريبه
    model_path = Path(r"C:\Users\aa683\Desktop\ppe_detection\models\runs\ppe_yolov8s-5\weights\best.pt")
    
    if not model_path.exists():
        print(f"Model weight not found at {model_path}. Please train first.")
        return

    model = YOLO(str(model_path))

    # التقييم على Test Set (أو Validation Set)
    metrics = model.val(
        data=r"C:\Users\aa683\Desktop\ppe_detection\data\ppe_data.yaml",
        split="test",       # تقييم على test set
        imgsz=640,
        batch=8,
        project=r"C:\Users\aa683\Desktop\ppe_detection\models\runs",
        name="eval_test"
    )

    print("\n--- Evaluation Results ---")
    print(f"Precision:    {metrics.box.mp:.4f}")
    print(f"Recall:       {metrics.box.mr:.4f}")
    print(f"mAP@50:       {metrics.box.map50:.4f}")
    print(f"mAP@50-95:    {metrics.box.map:.4f}")

if __name__ == "__main__":
    main()



# ============================================================
# YOLOv8 Training Results - Epoch 1/50
# ============================================================
#
# Training:
# Epoch       : 1/50
# GPU Memory  : 1.96 GB
# Box Loss    : 1.279  -> Bounding box localization error
# Class Loss  : 1.621  -> Object classification error
# DFL Loss    : 1.403  -> Bounding box edge/shape accuracy
# Instances   : 54     -> Objects in the current batch
# Image Size  : 640x640
# Batches     : 326/326
# Speed       : 3.3 it/s
# Epoch Time  : 1m 38s
#
# ============================================================
# Validation Results
# ============================================================
#
# Validation Images     : 114
# Validation Instances  : 286
#
# Precision (P)         : 0.524  (52.4%)
# Recall (R)            : 0.410  (41.0%)
# mAP@50                : 0.403  (40.3%)
# mAP@50-95             : 0.177  (17.7%)
#
# ============================================================
# Notes:
# - Precision: Percentage of detected objects that are correct.
# - Recall: Percentage of actual objects successfully detected.
# - mAP@50: Mean Average Precision at IoU = 0.50.
# - mAP@50-95: Average mAP across IoU thresholds from 0.50 to 0.95.
# ============================================================