import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple

import cv2
import numpy as np
import supervision as sv

from src.detection.detector import PPEDetector
from src.detection.tracker import WorkerTracker
from src.detection.violation import associate_ppe_to_worker, evaluate_compliance
from src.processing.frame_processor import FrameProcessor


class DetectionService:
    """
    Orchestrates the full PPE pipeline for one frame.

    Other parts of the app should call only:
        annotated_frame, stats = service.process_frame(frame)
    """

    def __init__(
        self,
        model_path: str = None,
        conf_threshold: float = 0.4,
        device: str = None,
        iob_threshold: float = 0.15,
        snapshot_dir: str = None,
        snapshot_cooldown_sec: float = 5.0
    ):
        project_root = Path(__file__).resolve().parents[2]
        if model_path is None:
            model_path = str(project_root / "models" / "best.pt")
        if snapshot_dir is None:
            snapshot_dir = str(project_root / "data" / "outputs" / "violations")

        # 1) Create components once
        self.detector = PPEDetector(
            model_path=model_path,
            conf_threshold=conf_threshold,
            device=device
        )
        self.tracker = WorkerTracker()
        self.frame_processor = FrameProcessor()

        # 2) Settings
        self.iob_threshold = iob_threshold
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_cooldown_sec = snapshot_cooldown_sec

        # 3) Running metrics
        self.unique_workers = set()
        self.total_violations_recorded = 0
        self.total_compliant_frames = 0
        self.total_worker_frames = 0

        # 4) Snapshot throttle memory: tracker_id -> last save time
        self._last_snapshot_time: Dict[int, float] = {}

        # 5) FPS helper
        self._prev_time = time.time()

        print("[DetectionService] Ready")

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Full pipeline for one frame.
        Returns:
            annotated_frame, stats
        """
        if frame is None or frame.size == 0:
            empty_stats = self._build_stats(
                active_workers=0,
                compliant_count=0,
                violation_count=0,
                unknown_count=0,
                fps=0.0
            )
            return frame, empty_stats

        # ---------- 1) Detect ----------
        all_detections = self.detector.detect(frame)

        if len(all_detections) > 0:
            person_mask = all_detections.class_id == 1
            ppe_mask = (all_detections.class_id == 0) | (all_detections.class_id == 2)

            person_detections = all_detections[person_mask]
            ppe_detections = all_detections[ppe_mask]
        else:
            person_detections = all_detections
            ppe_detections = all_detections

        # ---------- 2) Track persons only ----------
        tracked_persons = self.tracker.update(person_detections)

        # Update unique workers
        if tracked_persons.tracker_id is not None:
            for tid in tracked_persons.tracker_id:
                self.unique_workers.add(int(tid))

        # ---------- 3) Associate + Compliance ----------
        if len(tracked_persons) > 0:
            assignment = associate_ppe_to_worker(
                person_detections=tracked_persons,
                ppe_detections=ppe_detections,
                iob_threshold=self.iob_threshold
            )
            compliance_list = evaluate_compliance(assignment)
        else:
            compliance_list = []

        # ---------- 4) Counts ----------
        active_workers = len(compliance_list)
        compliant_count = sum(1 for w in compliance_list if w["status"] == "COMPLIANT")
        violation_count = sum(1 for w in compliance_list if w["status"] == "VIOLATION")
        unknown_count = sum(1 for w in compliance_list if w["status"] == "UNKNOWN")

        # Update rolling metrics
        self.total_worker_frames += active_workers
        self.total_compliant_frames += compliant_count

        # ---------- 5) Save violation snapshots (throttled) ----------
        self._save_violation_snapshots(frame, tracked_persons, compliance_list)

        # ---------- 6) FPS ----------
        now = time.time()
        fps = 1.0 / (now - self._prev_time + 1e-6)
        self._prev_time = now

        # ---------- 7) Annotate ----------
        annotated = self.frame_processor.annotate(
            frame=frame,
            tracked_persons=tracked_persons,
            ppe_detections=ppe_detections,
            compliance_list=compliance_list,
            fps=fps,
            total_workers=active_workers,
            compliant_count=compliant_count,
            violation_count=violation_count
        )

        # ---------- 8) Stats ----------
        stats = self._build_stats(
            active_workers=active_workers,
            compliant_count=compliant_count,
            violation_count=violation_count,
            unknown_count=unknown_count,
            fps=fps
        )
        stats["compliance_list"] = compliance_list

        return annotated, stats

    def _save_violation_snapshots(
        self,
        frame: np.ndarray,
        tracked_persons: sv.Detections,
        compliance_list: list
    ) -> None:
        """
        Save a snapshot for a worker only if:
        - status == VIOLATION
        - and no snapshot was saved for this worker in the last N seconds
        """
        if tracked_persons is None or len(tracked_persons) == 0:
            return
        if tracked_persons.tracker_id is None:
            return

        now = time.time()
        status_dict = {int(w["worker_id"]): w["status"] for w in compliance_list}

        for i, tid in enumerate(tracked_persons.tracker_id):
            worker_id = int(tid)
            status = status_dict.get(worker_id, "UNKNOWN")

            if status != "VIOLATION":
                continue

            last_time = self._last_snapshot_time.get(worker_id, 0.0)
            if (now - last_time) < self.snapshot_cooldown_sec:
                continue  # throttle

            # Crop worker box with small padding
            x1, y1, x2, y2 = tracked_persons.xyxy[i].astype(int)
            h, w = frame.shape[:2]
            pad = 20
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"worker_{worker_id}_{timestamp}.jpg"
            save_path = self.snapshot_dir / filename

            cv2.imwrite(str(save_path), crop)
            self._last_snapshot_time[worker_id] = now
            self.total_violations_recorded += 1

            print(f"[Snapshot] Saved: {save_path.name}")

    def _build_stats(
        self,
        active_workers: int,
        compliant_count: int,
        violation_count: int,
        unknown_count: int,
        fps: float
    ) -> Dict[str, Any]:
        if self.total_worker_frames > 0:
            compliance_rate = (self.total_compliant_frames / self.total_worker_frames) * 100.0
        else:
            compliance_rate = 0.0

        return {
            "fps": round(fps, 1),
            "active_workers": active_workers,
            "compliant_count": compliant_count,
            "violation_count": violation_count,
            "unknown_count": unknown_count,
            "unique_workers_seen": len(self.unique_workers),
            "total_violations_recorded": self.total_violations_recorded,
            "compliance_rate": round(compliance_rate, 1),
        }