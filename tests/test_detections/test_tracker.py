# tests/test_detections/test_tracker.py
import numpy as np
import supervision as sv

from src.detection.tracker import WorkerTracker


def make_person_dets(boxes):
    n = len(boxes)
    return sv.Detections(
        xyxy=np.array(boxes, dtype=float),
        class_id=np.array([1] * n),
        confidence=np.array([0.9] * n),
    )


def test_tracker_empty_input():
    tracker = WorkerTracker()
    out = tracker.update(sv.Detections.empty())
    assert len(out) == 0


def test_tracker_single_worker_stable_id():
    tracker = WorkerTracker()

    frames = [
        [[100, 80, 200, 300]],
        [[102, 82, 202, 302]],
        [[104, 84, 204, 304]],
    ]

    ids = []
    for boxes in frames:
        tracked = tracker.update(make_person_dets(boxes))
        assert len(tracked) == 1
        assert tracked.tracker_id is not None
        ids.append(int(tracked.tracker_id[0]))

    assert ids[0] == ids[1] == ids[2]


def test_tracker_two_workers_distinct_ids():
    tracker = WorkerTracker()

    dets = make_person_dets([
        [50, 50, 120, 220],
        [300, 60, 380, 240],
    ])
    tracked = tracker.update(dets)

    assert len(tracked) == 2
    assert tracked.tracker_id is not None
    assert len(set(int(x) for x in tracked.tracker_id)) == 2