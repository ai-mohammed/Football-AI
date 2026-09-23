"""Ultralytics tracker adapter with a stable match namespace across resets."""
import inspect
from contextlib import contextmanager
from threading import Lock
from types import SimpleNamespace

import numpy as np
import supervision as sv

_ID_LOCK = Lock()


class FootballTracker:
    def __init__(self, backend="bytetrack", fps=25, buffer_seconds=3.0, device="cpu",
                 reid_model="yolo11n-cls.pt"):
        if backend not in {"bytetrack", "botsort"}:
            raise ValueError("Choose bytetrack or botsort")
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.trackers.bot_sort import BOTSORT
        args = SimpleNamespace(track_high_thresh=0.25, track_low_thresh=0.1,
                               new_track_thresh=0.35, match_thresh=0.8,
                               track_buffer=max(1, round(buffer_seconds * fps)),
                               fuse_score=True, gmc_method="sparseOptFlow",
                               proximity_thresh=0.5, appearance_thresh=0.8,
                               with_reid=backend == "botsort", model=reid_model, device=device)
        cls = BOTSORT if backend == "botsort" else BYTETracker
        # Older 8.x scales the buffer by frame_rate/30; it is already expressed
        # in processed frames here. Newer 8.x accepts only args.
        options = {"frame_rate": 30} if "frame_rate" in inspect.signature(cls).parameters else {}
        self._counter = 0
        with self._isolated_ids():
            self.tracker = cls(args, **options)
        self.namespace = {}
        self.next_id = 1

    @contextmanager
    def _isolated_ids(self):
        # Ultralytics uses a global BaseTrack counter. A second Streamlit
        # session resets it when constructing a tracker. Isolate it per match.
        from ultralytics.trackers.basetrack import BaseTrack
        with _ID_LOCK:
            previous = BaseTrack._count
            BaseTrack._count = self._counter
            try:
                yield
            finally:
                self._counter = BaseTrack._count
                BaseTrack._count = previous

    def reset(self):
        with self._isolated_ids():
            self.tracker.reset()
        self.namespace.clear()

    def update(self, result, frame):
        # Track football people only. The dedicated ball tracker is separate.
        boxes = result.boxes.cpu().numpy()
        boxes = boxes[np.isin(boxes.cls, [1, 2, 3])]
        with self._isolated_ids():
            tracks = self.tracker.update(boxes, frame)
        if not len(tracks):
            empty = sv.Detections.empty()
            empty.tracker_id = np.empty(0, dtype=int)
            return empty
        ids = []
        for raw in tracks[:, 4].astype(int):
            if raw not in self.namespace:
                self.namespace[raw] = self.next_id
                self.next_id += 1
            ids.append(self.namespace[raw])
        return sv.Detections(xyxy=tracks[:, :4], confidence=tracks[:, 5],
                             class_id=tracks[:, 6].astype(int), tracker_id=np.asarray(ids))
