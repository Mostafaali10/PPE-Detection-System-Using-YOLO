import argparse
import subprocess
import sys
from pathlib import Path

import cv2

from src.services.detection_service import DetectionService
from src.processing.video_processor import get_frame_stream


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"
DEFAULT_SNAPSHOT_DIR = PROJECT_ROOT / "data" / "outputs" / "violations"


def build_service(
    model_path: str = None,
    conf: float = 0.4,
    device: str = None
) -> DetectionService:
    if model_path is None:
        model_path = str(DEFAULT_MODEL_PATH)
    return DetectionService(
        model_path=model_path,
        conf_threshold=conf,
        device=device,
        iob_threshold=0.15,
        snapshot_dir=str(DEFAULT_SNAPSHOT_DIR),
        snapshot_cooldown_sec=5.0
    )


def parse_source(source: str):
    """
    If source is an integer string like "0", return int (webcam).
    Otherwise return path string.
    """
    if source.isdigit():
        return int(source)
    return source


def validate_input_source(source):
    """
    Validate file path sources. Webcam index is allowed.
    """
    if isinstance(source, int):
        return  # webcam index

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Input source not found: {source}")


def ensure_output_parent(output_path: str):
    parent = Path(output_path).parent
    parent.mkdir(parents=True, exist_ok=True)


def cmd_dashboard(args):
    """
    Launch Streamlit dashboard.
    """
    app_path = Path(__file__).parent / "src" / "dashboard" / "app.py"
    if not app_path.exists():
        print(f"Error: dashboard app not found at {app_path}")
        sys.exit(1)

    print("Launching dashboard...")
    print("Press Ctrl+C to stop.\n")

    # Ensure project root is on PYTHONPATH
    project_root = str(Path(__file__).parent)
    env = dict(**{**subprocess.os.environ, "PYTHONPATH": project_root})

    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(app_path)],
            env=env,
            check=True
        )
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    except Exception as e:
        print(f"Failed to launch dashboard: {e}")
        sys.exit(1)


def cmd_predict_image(args):
    source = parse_source(args.source)
    output = args.output

    try:
        validate_input_source(source)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if isinstance(source, int):
        print("Error: predict-image expects an image file path, not a webcam index.")
        sys.exit(1)

    ensure_output_parent(output)

    print(f"Loading model...")
    service = build_service(conf=args.conf, device=args.device)

    frame = cv2.imread(str(source))
    if frame is None:
        print(f"Error: could not read image: {source}")
        sys.exit(1)

    print(f"Processing image: {source}")
    annotated, stats = service.process_frame(frame)

    ok = cv2.imwrite(output, annotated)
    if not ok:
        print(f"Error: failed to save output image: {output}")
        sys.exit(1)

    print(f"Saved annotated image → {output}")
    print(
        f"Workers={stats['active_workers']} | "
        f"Compliant={stats['compliant_count']} | "
        f"Violations={stats['violation_count']} | "
        f"Unknown={stats['unknown_count']}"
    )


def cmd_predict_video(args):
    source = parse_source(args.source)
    output = args.output

    try:
        validate_input_source(source)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    ensure_output_parent(output)

    print("Loading model...")
    service = build_service(conf=args.conf, device=args.device)

    # Probe first frame / properties for writer
    if isinstance(source, int):
        cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(str(source))

    if not cap.isOpened():
        print(f"Error: could not open video source: {source}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output, fourcc, fps, (width, height))
    if not writer.isOpened():
        print(f"Error: could not create output video: {output}")
        sys.exit(1)

    print(f"Processing video source: {source}")
    print(f"Output → {output}")

    frame_id = 0
    try:
        for frame in get_frame_stream(source):
            frame_id += 1
            annotated, stats = service.process_frame(frame)

            # Resize if needed (safety)
            if annotated.shape[1] != width or annotated.shape[0] != height:
                annotated = cv2.resize(annotated, (width, height))

            writer.write(annotated)

            if frame_id % 30 == 0:
                print(
                    f"Frame {frame_id} | "
                    f"Workers={stats['active_workers']} | "
                    f"C={stats['compliant_count']} "
                    f"V={stats['violation_count']} | "
                    f"FPS={stats['fps']}"
                )

            if args.max_frames and frame_id >= args.max_frames:
                print(f"Reached max frames: {args.max_frames}")
                break
    finally:
        writer.release()

    print(f"\nDone. Processed {frame_id} frames.")
    print(f"Saved annotated video → {output}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="PPE Safety Detection System - CLI"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # dashboard
    p_dash = sub.add_parser("dashboard", help="Launch Streamlit dashboard")
    p_dash.set_defaults(func=cmd_dashboard)

    # predict-image
    p_img = sub.add_parser("predict-image", help="Run detection on one image")
    p_img.add_argument("--source", required=True, help="Path to input image")
    p_img.add_argument("--output", required=True, help="Path to save annotated image")
    p_img.add_argument("--conf", type=float, default=0.4, help="Confidence threshold (default: 0.4)")
    p_img.add_argument("--device", default=None, help="Device: 0 for GPU, cpu for CPU (default: auto-detect)")
    p_img.set_defaults(func=cmd_predict_image)

    # predict-video
    p_vid = sub.add_parser("predict-video", help="Run detection on a video or webcam")
    p_vid.add_argument("--source", required=True, help="Video path or webcam index (0)")
    p_vid.add_argument("--output", required=True, help="Path to save annotated video")
    p_vid.add_argument("--conf", type=float, default=0.4, help="Confidence threshold (default: 0.4)")
    p_vid.add_argument("--device", default=None, help="Device: 0 for GPU, cpu for CPU (default: auto-detect)")
    p_vid.add_argument("--max-frames", type=int, default=0, help="Optional max frames (0 = all)")
    p_vid.set_defaults(func=cmd_predict_video)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()