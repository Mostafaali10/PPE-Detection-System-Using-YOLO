import cv2
from typing import Generator, Union
import numpy as np


def get_frame_stream(source: Union[str, int]) -> Generator[np.ndarray, None, None]:
    """
    Unified frame source.

    source can be:
      - path to video file (str)   → "video.mp4"
      - webcam index (int)        → 0
      - RTSP url (str)            → "rtsp://..."

    Yields:
      frames as NumPy arrays (BGR)
    """
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    print(f"[VideoProcessor] Opened source: {source}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                # End of file or stream failed
                break
            yield frame
    finally:
        cap.release()
        print("[VideoProcessor] Source closed")


def get_video_info(source: Union[str, int]) -> dict:
    """
    Optional helper: get FPS, width, height.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return info