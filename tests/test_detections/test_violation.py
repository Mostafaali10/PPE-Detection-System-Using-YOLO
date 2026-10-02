# tests/test_violation.py

import numpy as np
from src.detection.violation import (
    intersection_area,
    calculate_iob,
    get_head_region,
    get_torso_region,
    associate_ppe_to_worker,
    evaluate_compliance
)
import supervision as sv


def test_intersection_area():
    print("\n===== Test 1: intersection_area =====")

    # Box A and Box B fully overlapping
    box_a = np.array([0, 0, 100, 100])
    box_b = np.array([0, 0, 100, 100])
    area = intersection_area(box_a, box_b)
    print(f"Full overlap area: {area} (expected 10000)")

    # No overlap
    box_c = np.array([200, 200, 300, 300])
    area2 = intersection_area(box_a, box_c)
    print(f"No overlap area: {area2} (expected 0)")


def test_calculate_iob():
    print("\n===== Test 2: calculate_iob =====")

    # PPE fully inside region
    ppe = np.array([10, 10, 30, 30])          # area = 400
    region = np.array([0, 0, 100, 100])
    iob = calculate_iob(ppe, region)
    print(f"Full containment IoB: {iob:.2f} (expected 1.00)")

    # No overlap
    ppe2 = np.array([200, 200, 220, 220])
    iob2 = calculate_iob(ppe2, region)
    print(f"No overlap IoB: {iob2:.2f} (expected 0.00)")


def test_regions():
    print("\n===== Test 3: head & torso regions =====")

    person = np.array([0, 0, 100, 200])  # height = 200

    head = get_head_region(person)
    print(f"Head region: {head} (expected y2 ≈ 60)")

    torso = get_torso_region(person)
    print(f"Torso region: {torso} (expected y1 ≈ 60, y2 ≈ 160)")


def test_one_helmet_two_workers():
    print("\n===== Test 4: One helmet → two workers (most important) =====")

    # Two persons
    persons = sv.Detections(
        xyxy=np.array([
            [0, 0, 100, 200],      # Worker A
            [80, 0, 180, 200]      # Worker B (close)
        ]),
        class_id=np.array([1, 1]),
        tracker_id=np.array([1, 2])
    )

    # One helmet closer to Worker B
    ppe = sv.Detections(
        xyxy=np.array([
            [90, 5, 130, 45]       # Helmet closer to Worker B
        ]),
        class_id=np.array([0])
    )

    result = associate_ppe_to_worker(persons, ppe, iob_threshold=0.3)

    print("Result:")
    for worker_id, info in result.items():
        print(f"  Worker #{worker_id} → has_helmet = {info['has_helmet']}")

    # Expected: only one worker should get the helmet (the one with higher IoB)


def test_evaluate_compliance():
    print("\n===== Test 5: evaluate_compliance =====")

    assignment = {
        1: {"has_helmet": True,  "has_vest": True,  "head_visible": True},
        2: {"has_helmet": True,  "has_vest": False, "head_visible": True},
        3: {"has_helmet": False, "has_vest": False, "head_visible": False},  # should be UNKNOWN
    }

    compliance = evaluate_compliance(assignment)

    for worker in compliance:
        print(f"Worker #{worker['worker_id']} → {worker['status']} | missing: {worker['missing_items']}")


if __name__ == "__main__":
    test_intersection_area()
    test_calculate_iob()
    test_regions()
    test_one_helmet_two_workers()
    test_evaluate_compliance()
    print("\n===== All basic tests finished =====")