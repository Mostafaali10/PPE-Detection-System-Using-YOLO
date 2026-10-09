# 🦺 Real-Time PPE Safety Detection & Compliance System
### *Production-Grade Computer Vision & AI Engineering Monolith*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg?logo=yolo)](https://github.com/ultralytics/ultralytics)
[![Supervision](https://img.shields.io/badge/Supervision-ByteTrack-purple.svg)](https://github.com/roboflow/supervision)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Pytest](https://img.shields.io/badge/Tests-16%20Passed-brightgreen.svg?logo=pytest)](https://docs.pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end, real-time Personal Protective Equipment (**PPE**) detection, multi-worker tracking, and safety compliance monitoring platform. Built using **YOLOv8**, **ByteTrack (Supervision)**, **OpenCV**, and an interactive **Streamlit** dashboard.

---

## 📑 Table of Contents
1. [Overview & Problem Statement](#-overview--problem-statement)
2. [Key Features](#-key-features)
3. [Architecture Overview](#-architecture-overview)
4. [Compliance & Association Logic](#-compliance--association-logic)
5. [Installation & Setup](#-installation--setup)
6. [How to Run](#-how-to-run)
   - [Interactive Dashboard](#1-interactive-streamlit-dashboard)
   - [Command-Line Interface (CLI)](#2-command-line-interface-cli)
   - [Automated Testing](#3-running-unit--integration-tests)
   - [Model Training & Evaluation](#4-model-training--evaluation)
7. [Directory Structure](#-directory-structure)
8. [Tech Stack](#-tech-stack)
9. [Portfolio Deployment & Next Steps](#-portfolio-deployment--next-steps)

---

## 🎯 Overview & Problem Statement

In industrial construction zones, manufacturing plants, and oil & gas facilities, failure to wear required Personal Protective Equipment (Hardhats, Safety Vests) is the leading cause of preventable workplace injuries.

Traditional CCTV surveillance requires human monitoring that is error-prone, fatigue-limited, and non-scalable. This system automates compliance auditing in real time:
- Identifies individual workers and tracks them across frames with persistent IDs.
- Determines whether each worker is wearing a **Hardhat** and **Safety Vest** using geometric bounding-box containment.
- Automatically saves evidence snapshots of safety violations with cooldown throttling.
- Delivers live analytics, KPI metrics, and time-series compliance rates via a modern web interface.

---

## ✨ Key Features

- 🔍 **Multi-Class Object Detection:** Fine-tuned YOLOv8 model detecting `Person`, `Hardhat`, and `Safety Vest` with high accuracy and low latency.
- 🆔 **Persistent Worker Tracking:** Integrated ByteTrack tracker assigning stable IDs to workers across camera occlusions and motion.
- 📐 **Geometric Containment Association (IoB):** Robust Intersection-over-Box (IoB) algorithm associating PPE items to worker anatomical regions (head for helmets, torso for safety vests).
- ⚖️ **Three-State Compliance Engine:** Classifies worker states into `COMPLIANT` (green), `VIOLATION` (red), or `UNKNOWN` (yellow when upper body/head is cropped).
- 📸 **Throttled Violation Snapshotting:** Auto-crops and saves timestamped violation snapshots to disk with configurable cooldown intervals (avoids filling storage on persistent violations).
- 📊 **Interactive Streamlit Dashboard:** Live video player, active KPI metrics, violation image feed gallery, and compliance percentage history charts.
- 🖥️ **Cross-Platform CLI:** Comprehensive command-line interface supporting images, local videos, webcams, and RTSP IP camera streams with auto CUDA/CPU detection.
- 🧪 **Full Test Coverage:** 16 automated pytest unit and integration tests covering detector, tracker, violation logic, frame annotator, video streaming, and service orchestration.

---

## 🏗️ Architecture Overview

The system follows a **Clean Modular Monolith Architecture** with decoupled layers:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PPE DETECTION PIPELINE MAP                          │
└─────────────────────────────────────────────────────────────────────────────┘
  [ Video / Image / Webcam / RTSP Stream ]
                     │
                     ▼
  ┌─────────────────────────────────────┐
  │         video_processor.py          │ ───► OpenCV frame generator stream
  └─────────────────────────────────────┘
                     │
                     ▼
  ┌─────────────────────────────────────┐
  │            detector.py              │ ───► YOLOv8 inference (Bboxes + Classes + Conf)
  └─────────────────────────────────────┘
                     │
                     ▼
  ┌─────────────────────────────────────┐
  │            tracker.py               │ ───► ByteTrack (Assign persistent Worker IDs)
  └─────────────────────────────────────┘
                     │
                     ▼
  ┌─────────────────────────────────────┐
  │           violation.py              │ ───► Geometric Containment & IoB Association
  │         (Domain Rules)              │      Check Head (Hardhat) & Torso (Vest)
  └─────────────────────────────────────┘
                     │
                     ▼
  ┌─────────────────────────────────────┐
  │         frame_processor.py          │ ───► Supervision Bounding Boxes, Labels & HUD
  └─────────────────────────────────────┘
                     │
                     ▼
  ┌─────────────────────────────────────┐
  │        detection_service.py         │ ───► Pipeline orchestration, metrics, snapshot saver
  └─────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
┌──────────────────┐   ┌───────────────────────────┐
│   CLI (main.py)  │   │  Streamlit Dashboard      │
│ (Batch / Stream) │   │ (Live UI, KPIs, Gallery)  │
└──────────────────┘   └───────────────────────────┘
```

---

## 📐 Compliance & Association Logic

To prevent false assignments in crowded environments, PPE items are mapped to workers using **Intersection over Box (IoB)** on anatomical sub-regions:

$$\text{IoB} = \frac{\text{Area}(\text{Intersection of PPE and Body Region})}{\text{Area}(\text{PPE Bounding Box})}$$

1. **Head Region:** Upper $40\%$ of the worker bounding box. Tested against detected Hardhats ($\text{IoB} \ge 0.15$).
2. **Torso Region:** Middle $30\% - 80\%$ of the worker bounding box. Tested against detected Safety Vests ($\text{IoB} \ge 0.15$).
3. **One-to-One Assignment:** Candidates are sorted by highest IoB score and assigned greedily to prevent one helmet from satisfying multiple adjacent workers.

---

## 🚀 Installation & Setup

### 1. Prerequisites
- Python 3.10 or 3.11
- NVIDIA GPU with CUDA drivers (optional, CPU inference is fully supported out-of-the-box)

### 2. Clone and Create Virtual Environment
```bash
git clone https://github.com/Mostafaali10/PPE-Detection-System-Using-YOLO.git
cd PPE-Detection-System-Using-YOLO

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 💻 How to Run

### 1. Interactive Streamlit Dashboard
Launch the web interface for real-time monitoring and analytics:
```bash
python main.py dashboard
# OR directly via Streamlit:
streamlit run src/dashboard/app.py
```
Open your browser at `http://localhost:8501`. Features available in the dashboard:
- Upload and test single images or video files.
- Connect live webcam feeds or RTSP IP camera streams.
- Live HUD displaying Active Workers, Compliant Count, Violations, and Compliance Rate.
- Live violation snapshot feed with download options.

### 2. Command-Line Interface (CLI)

**Process a single image:**
```bash
python main.py predict-image --source data/source_sample.jpg --output data/outputs/result.jpg --conf 0.40
```

**Process a video file:**
```bash
python main.py predict-video --source data/test_video1.mp4 --output data/outputs/annotated_output.mp4 --conf 0.40
```

**Run on live webcam (Device 0):**
```bash
python main.py predict-video --source 0 --output data/outputs/webcam_output.mp4
```

### 3. Running Unit & Integration Tests
Run the complete automated test suite with pytest:
```bash
pytest
```
*All 16 tests covering core settings, detector, tracker, violation geometry, frame annotations, video streaming, and detection service will execute.*

### 4. Model Training & Evaluation
To retrain or benchmark the YOLO model on a custom dataset:
```bash
# Fine-tune YOLOv8 on PPE dataset:
python training/train.py

# Evaluate model metrics (Precision, Recall, mAP@50, mAP@50-95):
python training/evaluate.py
```

---

## 📁 Directory Structure

```plaintext
ppe_detection/
├── data/
│   ├── outputs/                    # Output videos & violation snapshots
│   │   └── violations/             # Auto-saved cropped worker violation images
│   ├── ppe_data.yaml               # YOLO dataset configuration
│   ├── remap_labels.py             # Dataset label remapping utility
│   ├── test_video.mp4              # Sample test video
│   └── test_video1.mp4             # Sample test video (compact)
│
├── models/
│   └── best.pt                     # Fine-tuned YOLOv8 PPE detection weights
│
├── notebooks/
│   └── 01_data_exploration.ipynb   # Exploratory data analysis notebook
│
├── src/                            # Modular source code
│   ├── core/
│   │   ├── config.py               # Pydantic v2 configuration settings
│   │   └── logger.py               # Structured loguru logging
│   ├── detection/
│   │   ├── detector.py             # YOLO model inference wrapper
│   │   ├── tracker.py              # Supervision ByteTrack worker tracking
│   │   └── violation.py            # Geometric IoB association & compliance engine
│   ├── processing/
│   │   ├── frame_processor.py      # Supervision annotations, bounding boxes & HUD
│   │   └── video_processor.py      # OpenCV video/stream reader & frame generator
│   ├── services/
│   │   └── detection_service.py    # Pipeline orchestrator & snapshot throttling
│   └── dashboard/
│       └── app.py                  # Interactive Streamlit application
│
├── tests/                          # Automated Pytest Suite
│   ├── conftest.py
│   ├── test_core.py                # Core configuration & logging tests
│   ├── test_detections/            # Detector, tracker, violation & pipeline tests
│   ├── test_processing/            # Frame & video processor tests
│   └── test_services/              # Detection service orchestration tests
│
├── main.py                         # Unified CLI Entrypoint
├── pytest.ini                      # Pytest runner configuration
├── requirements.txt                # Python dependencies
└── README.md                       # Documentation
```

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ | Core application logic |
| **Object Detection** | YOLOv8 (Ultralytics) | Real-time multi-class object detection |
| **Tracking** | Roboflow Supervision (ByteTrack) | Persistent worker ID tracking across frames |
| **Computer Vision** | OpenCV, NumPy, Pillow | Frame manipulation, video I/O, geometry |
| **Dashboard** | Streamlit | Interactive web UI, live metrics & chart analytics |
| **Configuration** | Pydantic v2 & Pydantic-Settings | Type-safe centralized environment configuration |
| **Logging** | Loguru | Structured file and stdout logging |
| **Testing** | Pytest | Unit and integration test suite |

---

## 🌟 Portfolio Deployment & Next Steps

1. **Hugging Face Spaces Deployment:**
   - Host the Streamlit dashboard on [Hugging Face Spaces](https://huggingface.co/spaces) using free CPU/T4 GPU hardware for public live demonstration.
2. **Video Showcase / LinkedIn Demo:**
   - Record a 60-second walkthrough showcasing:
     - Real-time video processing with the HUD telemetry bar.
     - Dynamic threshold adjustment via sidebar sliders.
     - Live violation snapshots appearing in the alert feed.
3. **Edge Optimization:**
   - Export `models/best.pt` to ONNX or TensorRT for deployment on embedded edge devices (NVIDIA Jetson / Raspberry Pi).

---

## 📜 License
This project is open-source and available under the [MIT License](LICENSE).
