# tests/test_processing/test_video_processor.py

import cv2
from src.processing.video_processor import get_frame_stream, get_video_info


def main():
    video_path = r"C:\Users\aa683\Desktop\ppe_detection\data\test_video.mp4"

    # 1) Print video info
    info = get_video_info(video_path)
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