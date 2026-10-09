import cv2
from pathlib import Path
import pytest
from src.processing.video_processor import get_frame_stream, get_video_info

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_video_info_helper():
    video_path = PROJECT_ROOT / "data" / "test_video1.mp4"
    if not video_path.exists():
        video_path = PROJECT_ROOT / "data" / "test_video.mp4"

    if not video_path.exists():
        pytest.skip("No test video found")

    info = get_video_info(str(video_path))
    assert "fps" in info
    assert "width" in info
    assert "height" in info
    assert info["width"] > 0
    assert info["height"] > 0


def main():
    video_path = PROJECT_ROOT / "data" / "test_video.mp4"
    if not video_path.exists():
        video_path = PROJECT_ROOT / "data" / "test_video1.mp4"

    # 1) Print video info
    info = get_video_info(str(video_path))
    print("Video info:", info)

    # 2) Read frames
    print("\nReading frames... Press 'q' to quit")
    frame_id = 0

    for frame in get_frame_stream(video_path):
        frame_id += 1
        cv2.imshow("VideoProcessor Test", frame)

        if frame_id % 30 == 0:
            print(f"Frame {frame_id}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()