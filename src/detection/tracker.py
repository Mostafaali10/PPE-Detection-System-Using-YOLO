# src/detection/tracker.py

import supervision as sv


class WorkerTracker:
    """
    Gives every person a stable ID across video frames.
    Only tracks the Person class.
    """

    def __init__(self):
        # Improved ByteTrack settings for more stable IDs
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.30,   # only start tracking high-confidence detections
            lost_track_buffer=45,              # keep the ID longer when person is temporarily lost
            minimum_matching_threshold=0.7,    # how strictly to match boxes
            frame_rate=30
        )
        print("[WorkerTracker] Tracker is ready")

    def update(self, detections: sv.Detections) -> sv.Detections:
        if detections is None or len(detections) == 0:
            return sv.Detections.empty()

        # Extra safety: keep only Person class
        PERSON_CLASS_ID = 1
        person_mask = detections.class_id == PERSON_CLASS_ID
        person_detections = detections[person_mask]

        if len(person_detections) == 0:
            return sv.Detections.empty()

        tracked = self.tracker.update_with_detections(person_detections)
        return tracked