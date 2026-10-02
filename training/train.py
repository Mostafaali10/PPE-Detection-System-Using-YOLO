from ultralytics import YOLO
import torch
from pathlib import Path


def main():
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    model = YOLO("yolov8s.pt")

    results = model.train(


        data = r"C:\Users\aa683\Desktop\ppe_detection\data\ppe_data.yaml",
        epochs = 50,
        batch = 8,
        imgsz = 640,
        device=device,
        project = r"C:\Users\aa683\Desktop\ppe_detection\models\runs",
        name="ppe_yolov8s",
        save=True,
        plots=True,
        workers=0

    )
    print("Training finished successfully!")
    print(f"Best Model Saved At : {results.save_dir / 'weights' / 'best.pt'}")



if __name__ == "__main__":
    main()


