# src/processing/frame_processor.py

import cv2
import numpy as np
import supervision as sv
from typing import List, Dict


class FrameProcessor:
    """
    Draws results only. Does NOT decide compliance.
    """

    def __init__(self):
        # One annotator per color (most compatible with different supervision versions)
        self.box_compliant = sv.BoxAnnotator(
            thickness=2,
            color=sv.Color.from_hex("#00FF00")  # Green
        )
        self.box_violation = sv.BoxAnnotator(
            thickness=2,
            color=sv.Color.from_hex("#FF0000")  # Red
        )
        self.box_unknown = sv.BoxAnnotator(
            thickness=2,
            color=sv.Color.from_hex("#FFFF00")  # Yellow
        )
        self.box_ppe = sv.BoxAnnotator(
            thickness=2,
            color=sv.Color.from_hex("#00BFFF")  # Blue
        )

        self.label_annotator = sv.LabelAnnotator(
            text_scale=0.5,
            text_thickness=1,
            text_padding=5
        )

    def annotate(
        self,
        frame: np.ndarray,
        tracked_persons: sv.Detections,
        ppe_detections: sv.Detections,
        compliance_list: List[Dict],
        fps: float = 0.0,
        total_workers: int = 0,
        compliant_count: int = 0,
        violation_count: int = 0
    ) -> np.ndarray:
        annotated = frame.copy()

        # 1) Draw PPE
        if ppe_detections is not None and len(ppe_detections) > 0:
            annotated = self.box_ppe.annotate(
                scene=annotated,
                detections=ppe_detections
            )

        # 2) Draw persons by status
        if (
            tracked_persons is not None
            and len(tracked_persons) > 0
            and tracked_persons.tracker_id is not None
        ):
            status_dict = {
                int(w["worker_id"]): w["status"]
                for w in compliance_list
            }

            compliant_idx = []
            violation_idx = []
            unknown_idx = []
            labels_all = []

            for i, tid in enumerate(tracked_persons.tracker_id):
                worker_id = int(tid)
                status = status_dict.get(worker_id, "UNKNOWN")
                labels_all.append(f"#{worker_id} {status}")

                if status == "COMPLIANT":
                    compliant_idx.append(i)
                elif status == "VIOLATION":
                    violation_idx.append(i)
                else:
                    unknown_idx.append(i)

            if len(compliant_idx) > 0:
                annotated = self.box_compliant.annotate(
                    scene=annotated,
                    detections=tracked_persons[compliant_idx]
                )

            if len(violation_idx) > 0:
                annotated = self.box_violation.annotate(
                    scene=annotated,
                    detections=tracked_persons[violation_idx]
                )

            if len(unknown_idx) > 0:
                annotated = self.box_unknown.annotate(
                    scene=annotated,
                    detections=tracked_persons[unknown_idx]
                )

            # Labels
            annotated = self.label_annotator.annotate(
                scene=annotated,
                detections=tracked_persons,
                labels=labels_all
            )

        # 3) HUD
        annotated = self._draw_hud(
            annotated,
            fps=fps,
            total_workers=total_workers,
            compliant_count=compliant_count,
            violation_count=violation_count
        )

        return annotated

    def _draw_hud(
        self,
        frame: np.ndarray,
        fps: float,
        total_workers: int,
        compliant_count: int,
        violation_count: int
    ) -> np.ndarray:
        h, w = frame.shape[:2]

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        text = (
            f"FPS: {fps:.1f}   |   "
            f"Workers: {total_workers}   |   "
            f"Compliant: {compliant_count}   |   "
            f"Violations: {violation_count}"
        )

        cv2.putText(
            frame,
            text,
            (15, 33),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        return frame