# 🦺 Real-Time PPE Safety Detection & Compliance System
### *Production-Ready Computer Vision & AI Engineering Project (Modular Monolith)*

An end-to-end, enterprise-grade Personal Protective Equipment (PPE) detection, multi-worker tracking, and safety compliance monitoring system built with **YOLOv8 (Ultralytics)**, **Roboflow Supervision**, **OpenCV**, and **Streamlit**.

---

## 📑 Table of Contents
1. [Project Overview](#-project-overview)
2. [Monolith Project Architecture](#-monolith-project-architecture)
3. [Recommended Advanced Features](#-recommended-advanced-features)
4. [Quick Start & Usage](#-quick-start--usage)
5. [🎓 The Mentor Guide: Build This System from Scratch](#-the-mentor-guide-build-this-system-from-scratch)
   - [Phase 1: Environment & Tooling Setup](#phase-1-environment--tooling-setup)
   - [Phase 2: Dataset Acquisition & Annotation Strategy](#phase-2-dataset-acquisition--annotation-strategy)
   - [Phase 3: Model Training, Evaluation & Optimization](#phase-3-model-training-evaluation--optimization)
   - [Phase 4: Core Domain & Violation Business Logic](#phase-4-core-domain--violation-business-logic)
   - [Phase 5: Vision Processing & HUD Annotation Pipeline](#phase-5-vision-processing--hud-annotation-pipeline)
   - [Phase 6: Application Service Orchestration](#phase-6-application-service-orchestration)
   - [Phase 7: Interactive Streamlit Dashboard](#phase-7-interactive-streamlit-dashboard)
   - [Phase 8: CLI Entrypoint](#phase-8-cli-entrypoint)
   - [Phase 9: Unit Testing & Robustness Verification](#phase-9-unit-testing--robustness-verification)
   - [Phase 10: Dockerization & Deployment](#phase-10-dockerization--deployment)
6. [Tech Stack](#-tech-stack)

---

## 🎯 Project Overview

In industrial sites, manufacturing plants, and construction zones, failure to wear **Personal Protective Equipment (PPE)** (Hardhats, Safety Vests, Masks, Gloves, Goggles) is the primary cause of preventable workplace injuries and fatalities.

This system provides:
- **Real-Time Detection:** Accurate multi-class detection of workers and PPE items.
- **Worker-Centric Association:** Spatial association of PPE items to individual worker bounding boxes.
- **Multi-Object Tracking (MOT):** Persistent worker tracking across video frames (Worker Track IDs via ByteTrack).
- **Compliance & Violation Engine:** Real-time business logic determining safety infractions per worker.
- **Interactive Analytics:** Live stream playback, compliance rate calculation, violation snapshot logs, and time-series statistics.

---

## 📁 Monolith Project Architecture

The codebase follows a **Clean Modular Monolith Architecture** with strict layer boundaries:

```plaintext
ppe_detection/
│
├── src/                              # Core Application Monolith
│   ├── core/                         # Configuration, logging & constants
│   │   ├── __init__.py
│   │   ├── config.py                 # Pydantic settings & threshold configurations
│   │   └── logger.py                 # Structured loguru logging setup
│   │
│   ├── detection/                    # Computer Vision Core Logic
│   │   ├── __init__.py
│   │   ├── detector.py               # YOLO model wrapper & batched inference
│   │   ├── tracker.py                # ByteTrack integration for worker IDs
│   │   └── violation.py              # Spatial association & PPE business rules
│   │
│   ├── processing/                   # Visual & Video Pipeline
│   │   ├── __init__.py
│   │   ├── frame_processor.py        # Supervision annotations, labels & HUD overlays
│   │   └── video_processor.py        # Video/Webcam/RTSP stream reader & frame generator
│   │
│   ├── services/                     # Application Service Layer
│   │   ├── __init__.py
│   │   └── detection_service.py      # Pipeline orchestrator & snapshot manager
│   │
│   └── dashboard/                    # Presentation Layer
│       ├── __init__.py
│       └── app.py                    # Streamlit interactive dashboard
│
├── data/
│   ├── raw/                          # Raw datasets & sample videos
│   ├── processed/                    # Cleaned YOLO train/val/test splits
│   └── outputs/                      # Saved violation snapshots & output videos
│
├── models/                           # Model weights (.pt, .onnx, .engine)
│   └── best.pt
│
├── training/                         # Model Training & Benchmarking Pipelines
│   ├── train.py                      # YOLO fine-tuning pipeline
│   └── evaluate.py                   # Model evaluation, mAP & confusion matrix
│
├── notebooks/                        # EDA and research notebooks
├── tests/                            # Pytest unit & integration tests
│   ├── __init__.py
│   └── test_detection.py
│
├── Dockerfile                        # Production containerization
├── main.py                           # CLI Entrypoint (Dashboard, Image/Video processing)
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

---

## 💡 Recommended Advanced Features

To elevate this project into a top-tier industry portfolio showcase, you can implement:

1. **Dynamic Geofencing & Danger Zones (ROI / Polygons):**
   - Allow safety managers to draw custom restricted zones on the camera feed. Workers inside the zone must satisfy stricter PPE rules.
2. **Instant Webhook & Telegram / Slack Alerts:**
   - Automatically dispatch an instant message with the worker's violation snapshot, timestamp, and location tag to safety officers.
3. **Automated PDF Safety Audit Reports:**
   - Generate one-click compliance PDF reports with daily stats, compliance percentages, violation logs, and historical trends.
4. **Audio / Visual Siren Triggers:**
   - In-dashboard audio alarms or external buzzer triggers when a critical violation is sustained for more than $N$ consecutive seconds.
5. **Edge Inference Optimization (ONNX / TensorRT / OpenVINO):**
   - Export trained YOLO models to ONNX/TensorRT for low-latency, high-FPS inference on edge devices (Jetson/Intel NUC).
6. **Multi-Camera & RTSP Stream Switcher:**
   - Support concurrent monitoring of multiple CCTV camera feeds with easy switching in the Streamlit UI.

---

## 🚀 Quick Start & Usage

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/your-username/ppe-detection-system.git
cd ppe-detection-system

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Streamlit Dashboard
```bash
python main.py dashboard
# OR
streamlit run src/dashboard/app.py
```

### 3. Run Inference via CLI

**On a single image:**
```bash
python main.py predict-image --source data/raw/sample.jpg --output data/outputs/result.jpg
```

**On a video file or live webcam:**
```bash
# Process video file:
python main.py predict-video --source data/raw/site_video.mp4 --output data/outputs/output.mp4

# Run on live webcam (Device 0):
python main.py predict-video --source 0
```

---

# 🎓 The Mentor Guide: Build This System from Scratch

> **Mentor Note:** *This guide is designed as an architectural roadmap for you to build this system independently from scratch. Read each phase, understand the engineering rationale behind it, and write the code with your own hands.*

```
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                         PPE DETECTION PIPELINE MAP                          │
 └─────────────────────────────────────────────────────────────────────────────┘
   [ Video / Webcam / RTSP ]
              │
              ▼
   ┌───────────────────────┐
   │    video_processor    │ ───► Read frames (OpenCV)
   └───────────────────────┘
              │
              ▼
   ┌───────────────────────┐
   │       detector        │ ───► Run YOLOv8 inference (Bboxes + Classes + Conf)
   └───────────────────────┘
              │
              ▼
   ┌───────────────────────┐
   │        tracker        │ ───► ByteTrack (Assign persistent Worker IDs)
   └───────────────────────┘
              │
              ▼
   ┌───────────────────────┐
   │       violation       │ ───► Spatial IOU / Containment Association
   │    (Business Logic)   │      Check: Person has Hardhat? Vest? Mask?
   └───────────────────────┘
              │
              ▼
   ┌───────────────────────┐
   │    frame_processor    │ ───► Draw Supervision BBoxes, HUD, Status Badges
   └───────────────────────┘
              │
              ▼
   ┌───────────────────────┐
   │   detection_service   │ ───► Aggregate metrics, save snapshots to disk
   └───────────────────────┘
              │
              ▼
   ┌───────────────────────┐
   │  Streamlit Dashboard  │ ───► Live Visuals, KPI Cards, Violation Gallery
   └───────────────────────┘
```

---

### Phase 1: Environment & Tooling Setup

#### 🎯 Goal:
Set up a clean, reproducible Python environment with pinned dependencies, logging, and configuration management.

#### 🛠️ What to do:
1. **Virtual Environment:** Create an isolated Python 3.10+ environment (`python -m venv venv`).
2. **Dependencies (`requirements.txt`):**
   - Core CV/DL: `ultralytics`, `torch`, `torchvision`, `opencv-python`, `numpy`, `pillow`
   - Tracking & Annotations: `supervision`
   - Presentation: `streamlit`
   - Config & Utilities: `pydantic`, `pyyaml`, `loguru`, `tqdm`
   - Testing: `pytest`
3. **Configuration Module (`src/core/config.py`):**
   - Use `pydantic.BaseModel` or `BaseSettings` to define model paths, confidence thresholds (`conf_threshold = 0.4`), IoU thresholds, required PPE classes, and directory paths.
   - *Why?* Never hardcode magic numbers (like `0.5` or `"models/best.pt"`) across your codebase. Keep them centralized in one configuration class.
4. **Structured Logger (`src/core/logger.py`):**
   - Configure `loguru` to format logs with timestamps, log levels, and write rotation files under `logs/app.log`.

---

### Phase 2: Dataset Acquisition & Annotation Strategy

#### 🎯 Goal:
Obtain high-quality annotated data representing real-world industrial environments (varying lighting, occlusions, distances).

#### 🛠️ What to do:
1. **Data Source:**
   - Download the public **PPE / Construction Site Safety Dataset** from Roboflow Universe or Kaggle.
   - Target Classes: `person`, `hard-hat` (or `helmet`), `safety-vest`, `no-hard-hat`, `no-safety-vest`, `mask`, `gloves`.
2. **YOLO Annotation Format:**
   - Ensure the dataset is formatted in standard YOLO layout:
     ```
     data/
     ├── train/ (images & labels)
     ├── val/   (images & labels)
     └── test/  (images & labels)
     ```
3. **Data Quality Audit:**
   - Write a small exploratory script or notebook (`notebooks/01_data_exploration.ipynb`) to verify bounding box aspect ratios, class distribution, and remove corrupt images.
4. **Data Config (`data/ppe_data.yaml`):**
   ```yaml
   path: ./data
   train: train/images
   val: val/images
   names:
     0: person
     1: hard-hat
     2: safety-vest
     3: no-hard-hat
     4: no-safety-vest
     5: machinery
   ```

---

### Phase 3: Model Training, Evaluation & Optimization

#### 🎯 Goal:
Train and benchmark a high-speed, accurate detector on your custom PPE dataset.

#### 🛠️ What to do:
1. **Model Selection:**
   - Choose **YOLOv8s** (small) or **YOLOv8m** (medium) as the backbone. YOLOv8n (nano) is ultra-fast for edge devices, while YOLOv8s offers an ideal balance of mAP and real-time FPS.
2. **Training Script (`training/train.py`):**
   - Use the Ultralytics Python API:
     ```python
     from ultralytics import YOLO
     model = YOLO("yolov8s.pt")
     results = model.train(
         data="data/ppe_data.yaml",
         epochs=50,
         imgsz=640,
         batch=16,
         device=0, # GPU or 'cpu'
         project="models/runs",
         name="ppe_yolov8s"
     )
     ```
3. **Evaluation Script (`training/evaluate.py`):**
   - Compute **mAP@50**, **mAP@50-95**, Precision, and Recall on the validation set.
   - Plot and inspect the **Confusion Matrix** to detect class confusion (e.g., confusing normal caps with hard-hats).
4. **Save Model Weights:** Copy the resulting `best.pt` into the `models/` directory.

---

### Phase 4: Core Domain & Violation Business Logic

#### 🎯 Goal:
Build the detection wrapper, tracking integration, and the mathematical logic that links PPE items to specific workers.

#### 🛠️ What to do:
1. **The Detector (`src/detection/detector.py`):**
   - Create a `PPEDetector` class wrapping the YOLO model.
   - Implement `detect(frame: np.ndarray) -> sv.Detections` returning standardized `supervision.Detections` objects.
2. **The Tracker (`src/detection/tracker.py`):**
   - Create a `WorkerTracker` class wrapping `sv.ByteTrack()`.
   - Filter detections to only track the `person` class, updating persistent `tracker_id`s across frames.
3. **Spatial Association & Violation Logic (`src/detection/violation.py`):**
   - *How do we know if a helmet belongs to Worker #3?*
   - Implement a bounding box containment / intersection algorithm:
     - For each detected `person` box:
       - Check the upper $30\%$ of the person box (head region) for overlap with `hard-hat` or `no-hard-hat` detections.
       - Check the middle $50\%$ of the person box (torso region) for overlap with `safety-vest` or `no-safety-vest`.
     - Calculate **Intersection over Box (IoB)**: $\text{IoB} = \frac{\text{Area}(\text{Intersection})}{\text{Area}(\text{PPE Item})}$. If $\text{IoB} > 0.5$, the item is assigned to that worker.
   - **Compliance State:** Return a structured dictionary for each worker:
     ```python
     {
         "worker_id": 1,
         "has_helmet": True,
         "has_vest": False,
         "status": "VIOLATION",  # "COMPLIANT" or "VIOLATION"
         "missing_items": ["safety-vest"]
     }
     ```

---

### Phase 5: Vision Processing & HUD Annotation Pipeline

#### 🎯 Goal:
Render clear, professional visual overlays, bounding boxes, labels, and Head-Up Display (HUD) telemetry on video frames.

#### 🛠️ What to do:
1. **Frame Processor (`src/processing/frame_processor.py`):**
   - Use `supervision.BoxAnnotator`, `supervision.LabelAnnotator`, and `supervision.TraceAnnotator`.
   - Color coding:
     - 🟢 **Green:** Fully compliant worker.
     - 🔴 **Red:** Worker with PPE violation.
     - 🟡 **Yellow:** PPE equipment bounding boxes (hard-hat, vest).
   - Draw an **On-Screen HUD bar**: Current FPS, Total Active Workers, Compliant Count, Violation Count.
2. **Video Processor (`src/processing/video_processor.py`):**
   - Create a generator `get_frame_stream(source)` using `cv2.VideoCapture` that yields frames one by one.
   - Handle video end-of-stream, video write-back using `cv2.VideoWriter`, and RTSP reconnection logic.

---

### Phase 6: Application Service Orchestration

#### 🎯 Goal:
Coordinate all domain modules inside a clean Service Layer (`detection_service.py`) that handles end-to-end processing and state.

#### 🛠️ What to do:
1. **Service Class (`src/services/detection_service.py`):**
   - Manage the lifetime of `PPEDetector`, `WorkerTracker`, and `ViolationEngine`.
   - Implement `process_frame(frame)`:
     1. Run detection.
     2. Update worker tracks.
     3. Evaluate violations.
     4. Save high-resolution image snapshots of any detected violation to `data/outputs/violations/`.
     5. Annotate frame and return processed image + frame statistics.
   - Maintain running metrics: Total unique workers seen, total violations recorded, compliance percentage over time.

---

### Phase 7: Interactive Streamlit Dashboard

#### 🎯 Goal:
Provide an intuitive, responsive web UI for safety officers to monitor cameras, upload test videos, and review alerts.

#### 🛠️ What to do (`src/dashboard/app.py`):
1. **Sidebar Controls:**
   - Source selector: **Upload Video File**, **Live Webcam**, or **RTSP Stream**.
   - Model confidence sliders (`0.10` – `1.00`).
   - Toggles: "Show Bounding Boxes", "Show Worker Trails", "Save Violation Snapshots".
2. **Top Metric Cards (`st.metric`):**
   - **Active Workers** | **Compliant Workers** | **Active Violations** | **Overall Compliance Rate (%)**.
3. **Main Viewports:**
   - Left Column: Real-time video player displaying annotated frames.
   - Right Column: **Live Violation Feed** displaying timestamped snapshot cards with missing PPE badges.
4. **Historical Analytics:**
   - Line chart showing compliance rate over time.
   - Bar chart showing breakdown of most common violations (e.g., No Helmet vs. No Vest).

---

### Phase 8: CLI Entrypoint

#### 🎯 Goal:
Provide a developer-friendly command-line interface for running batch operations or starting the dashboard.

#### 🛠️ What to do (`main.py`):
- Use `argparse` to support:
  - `python main.py dashboard` -> Runs `streamlit run src/dashboard/app.py`.
  - `python main.py predict-image --source input.jpg --output out.jpg` -> Processes a single image.
  - `python main.py predict-video --source input.mp4 --output out.mp4` -> Processes a video file.

---

### Phase 9: Unit Testing & Robustness Verification

#### 🎯 Goal:
Ensure your business logic and pipelines do not break when edge cases happen.

#### 🛠️ What to do (`tests/test_detection.py`):
1. **Test Violation Logic:**
   - Create synthetic bounding boxes (e.g., a person box and a helmet box with 0% overlap) and assert that `evaluate_compliance()` flags a violation.
2. **Test Empty Frame Handling:**
   - Pass an all-black image (`np.zeros((640, 640, 3), dtype=np.uint8)`) into `detector.detect()` and verify it returns 0 detections without crashing.
3. **Run Pytest:**
   ```bash
   pytest tests/ -v
   ```

---

### Phase 10: Dockerization & Deployment

#### 🎯 Goal:
Package the entire modular monolith into a portable Docker container for deployment to any cloud VM or on-premise server.

#### 🛠️ What to do (`Dockerfile`):
```dockerfile
FROM python:3.10-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "src/dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**Build and Run:**
```bash
docker build -t ppe-detection-system .
docker run -p 8501:8501 ppe-detection-system
```

---

## 🛠️ Tech Stack

- **Language:** Python 3.10+
- **Deep Learning / Object Detection:** YOLOv8 (Ultralytics), PyTorch
- **Tracking & Annotations:** Roboflow Supervision (ByteTrack)
- **Computer Vision:** OpenCV, Pillow, NumPy
- **Dashboard & Visualization:** Streamlit
- **Configuration & Logging:** Pydantic v2, Loguru
- **Testing & Quality:** Pytest
- **Containerization:** Docker

---

## 📄 License & Attribution
This project is open-source and built for educational and portfolio demonstration purposes. All architectures and code structures adhere to industry-standard modular software engineering practices.
