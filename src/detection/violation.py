# src/detection/violation.py

from typing import Dict, List
import numpy as np
import supervision as sv


# ============================================================
# 1. Basic Geometry Functions
# ============================================================

def intersection_area(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """
    Calculate the overlapping area between two boxes.
    Each box format: [x1, y1, x2, y2]
    """
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    return float((x2 - x1) * (y2 - y1))


def calculate_iob(ppe_box: np.ndarray, region_box: np.ndarray) -> float:
    """
    IoB = Intersection over Box (of the PPE item)
    Returns value between 0.0 and 1.0
    """
    inter = intersection_area(ppe_box, region_box)
    ppe_area = (ppe_box[2] - ppe_box[0]) * (ppe_box[3] - ppe_box[1])

    if ppe_area <= 0:
        return 0.0

    return inter / ppe_area


# ============================================================
# 2. Body Region Helpers
# ============================================================

def get_head_region(person_box: np.ndarray) -> np.ndarray:
    """
    Top 40% of the person box (where the helmet should be).
    """
    x1, y1, x2, y2 = person_box
    height = y2 - y1
    head_y2 = y1 + height * 0.40
    return np.array([x1, y1, x2, head_y2])


def get_torso_region(person_box: np.ndarray) -> np.ndarray:
    """
    Middle 50% of the person box (where the safety vest should be).
    """
    x1, y1, x2, y2 = person_box
    height = y2 - y1
    torso_y1 = y1 + height * 0.30
    torso_y2 = y1 + height * 0.80
    return np.array([x1, torso_y1, x2, torso_y2])


# ============================================================
# 3. Main Association Logic (FIXED - one PPE to one person)
# ============================================================

def associate_ppe_to_worker(
    person_detections: sv.Detections,
    ppe_detections: sv.Detections,
    iob_threshold: float = 0.3
) -> Dict[int, dict]:
    """
    Assign PPE items to workers using best IoB.
    Important: Each PPE item can be assigned to only ONE worker.
    """

    HARDHAT_ID = 0
    SAFETY_VEST_ID = 2

    result = {}

    if person_detections is None or len(person_detections) == 0:
        return result

    # Separate PPE by type
    hardhats = ppe_detections[ppe_detections.class_id == HARDHAT_ID] if len(ppe_detections) > 0 else sv.Detections.empty()
    vests = ppe_detections[ppe_detections.class_id == SAFETY_VEST_ID] if len(ppe_detections) > 0 else sv.Detections.empty()

    # Initialize all workers as not having PPE
    for i in range(len(person_detections)):
        tracker_id = int(person_detections.tracker_id[i])
        result[tracker_id] = {
            "has_helmet": False,
            "has_vest": False,
            "head_visible": True          # used later for UNKNOWN
        }

    # ----- Assign Helmets (one-to-one) -----
    helmet_candidates = []  # list of (iob, person_idx, helmet_idx)

    for i in range(len(person_detections)):
        head_region = get_head_region(person_detections.xyxy[i])
        for j in range(len(hardhats)):
            iob = calculate_iob(hardhats.xyxy[j], head_region)
            if iob >= iob_threshold:
                helmet_candidates.append((iob, i, j))

    # Sort by highest IoB first
    helmet_candidates.sort(reverse=True, key=lambda x: x[0])

    used_helmets = set()
    used_persons_helmet = set()

    for iob, person_idx, helmet_idx in helmet_candidates:
        if person_idx in used_persons_helmet or helmet_idx in used_helmets:
            continue
        tracker_id = int(person_detections.tracker_id[person_idx])
        result[tracker_id]["has_helmet"] = True
        used_helmets.add(helmet_idx)
        used_persons_helmet.add(person_idx)

    # ----- Assign Vests (one-to-one) -----
    vest_candidates = []

    for i in range(len(person_detections)):
        torso_region = get_torso_region(person_detections.xyxy[i])
        for j in range(len(vests)):
            iob = calculate_iob(vests.xyxy[j], torso_region)
            if iob >= iob_threshold:
                vest_candidates.append((iob, i, j))

    vest_candidates.sort(reverse=True, key=lambda x: x[0])

    used_vests = set()
    used_persons_vest = set()

    for iob, person_idx, vest_idx in vest_candidates:
        if person_idx in used_persons_vest or vest_idx in used_vests:
            continue
        tracker_id = int(person_detections.tracker_id[person_idx])
        result[tracker_id]["has_vest"] = True
        used_vests.add(vest_idx)
        used_persons_vest.add(person_idx)

        # Simple rule for head visibility (for UNKNOWN)
    for i in range(len(person_detections)):
        tracker_id = int(person_detections.tracker_id[i])
        y1 = person_detections.xyxy[i][1]
        # Only mark as not visible if the head is severely cut off
        if y1 < 5:
            result[tracker_id]["head_visible"] = False

    return result


# ============================================================
# 4. Compliance Evaluation (with UNKNOWN)
# ============================================================

def evaluate_compliance(assignment: Dict[int, dict]) -> List[dict]:
    """
    Convert association result into final compliance state.
    Possible status: COMPLIANT | VIOLATION | UNKNOWN
    """
    compliance_list = []

    for worker_id, info in assignment.items():
        has_helmet = info.get("has_helmet", False)
        has_vest = info.get("has_vest", False)
        head_visible = info.get("head_visible", True)

        missing = []
        if not has_helmet:
            missing.append("hardhat")
        if not has_vest:
            missing.append("safety-vest")

        # Decision logic
        if not head_visible:
            # Head is probably outside the frame → we cannot judge helmet reliably
            status = "UNKNOWN"
        elif len(missing) == 0:
            status = "COMPLIANT"
        else:
            status = "VIOLATION"

        compliance_list.append({
            "worker_id": worker_id,
            "has_helmet": has_helmet,
            "has_vest": has_vest,
            "status": status,
            "missing_items": missing
        })

    return compliance_list