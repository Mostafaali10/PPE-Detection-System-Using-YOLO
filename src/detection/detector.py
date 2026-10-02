# src/detection/detector.py
from pathlib import Path
import numpy as np
import supervision as sv
from ultralytics import YOLO


class PPEDetector:
    """
    This class is a clean wrapper around the YOLO model.
    The rest of the project will only talk to this class,
    never directly to YOLO.
    """

    def __init__(self, model_path: str, conf_threshold: float = 0.4, device: str = "cpu"):
        """
        This function runs only ONE time when we create the detector.

        Parameters:
            model_path     → path to the trained model (best.pt)
            conf_threshold → minimum confidence score to keep a detection
            device         → "cpu" or "0" (for GPU)
        """
        # Save the settings
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.device = device


        # Check if the model file really exists
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # Load the YOLO model (this is slow, so we do it only once)
        print("Loading YOLO model... please wait")
        self.model = YOLO(str(self.model_path))

        
        if self.device == "0":
            self.device = "cuda:0"
            
        # Put the model on GPU or CPU
        self.model.to(self.device)

        print(f"Model loaded successfully!")
        print(f"Using device: {self.device}")
        print(f"Confidence threshold: {self.conf_threshold}")




    def detect(self, frame: np.ndarray) -> sv.Detections:
        """
        This function runs on every single frame (image).

        Input:
            frame → a normal image (NumPy array) in BGR format

        Output:
            detections → a clean object that contains:
                         - boxes (xyxy)
                         - confidence scores
                         - class ids
        """

        # Safety check: if the frame is empty, return empty detections
        if frame is None or frame.size == 0:
            return sv.Detections.empty()

        # Run the model on the frame
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,   # only keep detections above this score
            device=self.device,
            verbose=False               # don't print messages every frame
        )

        # Ultralytics returns a list, we take the first result
        result = results[0]

        # Convert the result into Supervision format (cleaner and standard)
        detections = sv.Detections.from_ultralytics(result)

        return detections