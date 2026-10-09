import streamlit as st
import cv2
import numpy as np
import tempfile
import time
from pathlib import Path
from datetime import datetime

from src.services.detection_service import DetectionService

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="PPE Safety Dashboard",
    page_icon="🦺",
    layout="wide"
)

st.title("🦺 PPE Safety Detection & Compliance Dashboard")
st.caption("Phase 7 — Full version (Image / Video / Webcam / RTSP)")

# -----------------------------
# Base paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"
DEFAULT_SNAPSHOT_DIR = PROJECT_ROOT / "data" / "outputs" / "violations"
DEFAULT_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Load service once
# -----------------------------
@st.cache_resource
def load_service():
    return DetectionService(
        model_path=str(DEFAULT_MODEL_PATH),
        conf_threshold=0.4,
        device=None,  # auto-detects CUDA/CPU
        iob_threshold=0.15,
        snapshot_dir=str(DEFAULT_SNAPSHOT_DIR),
        snapshot_cooldown_sec=5.0
    )

service = load_service()

# -----------------------------
# Session state
# -----------------------------
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "compliance_history" not in st.session_state:
    st.session_state.compliance_history = []  # list of compliance_rate over time
if "last_stats" not in st.session_state:
    st.session_state.last_stats = None

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Controls")

source_type = st.sidebar.selectbox(
    "Source",
    ["Image", "Video File", "Webcam", "RTSP"]
)

conf = st.sidebar.slider("Confidence", 0.10, 1.00, 0.40, 0.05)
service.detector.conf_threshold = conf

max_frames = st.sidebar.slider("Max frames (Video/Stream)", 30, 500, 150, 30)
show_boxes = st.sidebar.checkbox("Show boxes / labels", True)
enable_snapshots = st.sidebar.checkbox("Enable violation snapshots", True)

snapshot_dir = DEFAULT_SNAPSHOT_DIR

# Source inputs
uploaded_image = None
uploaded_video = None
rtsp_url = ""
webcam_index = 0

if source_type == "Image":
    uploaded_image = st.sidebar.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
elif source_type == "Video File":
    uploaded_video = st.sidebar.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"])
elif source_type == "Webcam":
    webcam_index = st.sidebar.number_input("Webcam index", min_value=0, max_value=5, value=0, step=1)
elif source_type == "RTSP":
    rtsp_url = st.sidebar.text_input("RTSP URL", value="rtsp://")

col_btn1, col_btn2 = st.sidebar.columns(2)
start_btn = col_btn1.button("Start", type="primary", use_container_width=True)
stop_btn = col_btn2.button("Stop", use_container_width=True)

if start_btn:
    st.session_state.is_running = True
if stop_btn:
    st.session_state.is_running = False

# -----------------------------
# Layout placeholders
# -----------------------------
metric_box = st.container()
main_left, main_right = st.columns([2, 1])
frame_placeholder = main_left.empty()
gallery_placeholder = main_right.empty()
chart_box = st.container()
status_box = st.empty()

# -----------------------------
# Helpers
# -----------------------------
def show_metrics(stats):
    with metric_box:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Active Workers", stats.get("active_workers", 0))
        m2.metric("Compliant", stats.get("compliant_count", 0))
        m3.metric("Violations", stats.get("violation_count", 0))
        m4.metric("Compliance %", stats.get("compliance_rate", 0.0))


def show_snapshot_gallery(limit=8):
    files = sorted(snapshot_dir.glob("*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
    with gallery_placeholder.container():
        st.subheader("Violation Snapshots")
        if not files:
            st.info("No snapshots yet.")
            return
        for f in files:
            st.image(str(f), caption=f.name, use_container_width=True)


def update_history(stats):
    st.session_state.compliance_history.append(stats.get("compliance_rate", 0.0))
    # keep last 100 points
    if len(st.session_state.compliance_history) > 100:
        st.session_state.compliance_history = st.session_state.compliance_history[-100:]


def show_charts():
    with chart_box:
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Compliance over time")
            if st.session_state.compliance_history:
                st.line_chart(st.session_state.compliance_history)
            else:
                st.info("No history yet.")
        with c2:
            st.subheader("Current breakdown")
            stats = st.session_state.last_stats
            if stats:
                st.bar_chart({
                    "Compliant": stats.get("compliant_count", 0),
                    "Violations": stats.get("violation_count", 0),
                    "Unknown": stats.get("unknown_count", 0),
                })
            else:
                st.info("No data yet.")


def process_frame_and_show(frame):
    annotated, stats = service.process_frame(frame)
    st.session_state.last_stats = stats
    update_history(stats)

    if show_boxes:
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    else:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame_placeholder.image(rgb, use_container_width=True)
    show_metrics(stats)
    if enable_snapshots:
        show_snapshot_gallery()
    show_charts()
    return stats


# -----------------------------
# IMAGE MODE
# -----------------------------
if source_type == "Image":
    if uploaded_image is not None:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if frame is None:
            st.error("Could not read image.")
        else:
            with st.spinner("Processing image..."):
                stats = process_frame_and_show(frame)
            if stats.get("compliance_list"):
                st.subheader("Worker Details")
                for w in stats["compliance_list"]:
                    st.write(
                        f"Worker #{w['worker_id']} → **{w['status']}** | "
                        f"Helmet: {'✓' if w['has_helmet'] else '✗'} | "
                        f"Vest: {'✓' if w['has_vest'] else '✗'} | "
                        f"Missing: {w['missing_items']}"
                    )
    else:
        st.info("Upload an image from the sidebar.")

# -----------------------------
# VIDEO FILE MODE
# -----------------------------
elif source_type == "Video File":
    if uploaded_video is None:
        st.info("Upload a video, then press Start.")
    elif st.session_state.is_running:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)
        if not cap.isOpened():
            st.error("Could not open video.")
            st.session_state.is_running = False
        else:
            frame_id = 0
            status_box.info("Processing video... press Stop to end.")
            while st.session_state.is_running:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_id += 1
                if frame_id > max_frames:
                    break
                process_frame_and_show(frame)
                # tiny yield so Stop can be clicked on next rerun
                time.sleep(0.01)
            cap.release()
            st.session_state.is_running = False
            status_box.success(f"Finished. Processed {frame_id} frames.")
    else:
        st.info("Press **Start** to process the uploaded video.")

# -----------------------------
# WEBCAM MODE
# -----------------------------
elif source_type == "Webcam":
    if st.session_state.is_running:
        cap = cv2.VideoCapture(int(webcam_index))
        if not cap.isOpened():
            st.error(f"Could not open webcam index {webcam_index}")
            st.session_state.is_running = False
        else:
            frame_id = 0
            status_box.info("Webcam running... press Stop to end.")
            while st.session_state.is_running:
                ret, frame = cap.read()
                if not ret:
                    status_box.warning("Webcam frame grab failed.")
                    break
                frame_id += 1
                if frame_id > max_frames:
                    break
                process_frame_and_show(frame)
                time.sleep(0.01)
            cap.release()
            st.session_state.is_running = False
            status_box.success("Webcam stopped.")
    else:
        st.info("Press **Start** to open webcam.")

# -----------------------------
# RTSP MODE
# -----------------------------
elif source_type == "RTSP":
    if not rtsp_url or rtsp_url == "rtsp://":
        st.info("Enter a valid RTSP URL, then press Start.")
    elif st.session_state.is_running:
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            st.error("Could not open RTSP stream.")
            st.session_state.is_running = False
        else:
            frame_id = 0
            status_box.info("RTSP stream running... press Stop to end.")
            while st.session_state.is_running:
                ret, frame = cap.read()
                if not ret:
                    status_box.warning("RTSP frame grab failed / disconnected.")
                    break
                frame_id += 1
                if frame_id > max_frames:
                    break
                process_frame_and_show(frame)
                time.sleep(0.01)
            cap.release()
            st.session_state.is_running = False
            status_box.success("RTSP stopped.")
    else:
        st.info("Press **Start** to connect to RTSP.")

# Always show gallery + charts if we already have data
if st.session_state.last_stats is not None and source_type != "Image":
    show_snapshot_gallery()
    show_charts()