"""Temporal evidence for control changes; ball flight is not lost possession."""
import numpy as np


class ControlFilter:
    """Confirm new owners twice; brief misses never award unobserved control."""
    def __init__(self, minimum_seconds=.08):
        self.minimum_seconds = minimum_seconds
        self.reset()

    def reset(self):
        self.candidate = None
        self.since = None
        self.count = 0
        self.last_claim = None
        self.confirmed = self.confirmed_at = None

    def update(self, candidate, timestamp):
        if candidate is None:
            return None
        if candidate == self.confirmed and timestamp-self.confirmed_at <= 1.5:
            self.confirmed_at = timestamp
            self.candidate, self.since, self.count = candidate, timestamp, 1
            self.last_claim = timestamp
            return candidate
        expired = self.last_claim is not None and timestamp-self.last_claim > .24+1e-6
        self.last_claim = timestamp
        if candidate != self.candidate or expired:
            self.candidate, self.since, self.count = candidate, timestamp, 1
            return None
        self.count += 1
        if self.count >= 2 and timestamp-self.since >= self.minimum_seconds-1e-6:
            self.confirmed, self.confirmed_at = candidate, timestamp
            return candidate
        return None


class BallFlight:
    """Observed pitch-space evidence in centimetres; never synthesize a ball.

    Long transfers require a bounded gap, plausible motion and observed travel.
    The pass is still a hypothesis, not proof of intention or a physical touch.
    """
    def __init__(self, max_seconds=5., max_missing_seconds=.4):
        self.max_seconds = max_seconds
        self.max_missing_seconds = max_missing_seconds
        self.reset()

    def reset(self):
        self.start = self.last = None
        self.samples = 0
        self.broken = False

    def observe(self, point, timestamp):
        if point is None:
            if self.last is not None and timestamp-self.last[0] > self.max_missing_seconds+1e-6:
                self.broken = True
            return
        point = np.asarray(point, dtype=float)
        if point.shape != (2,) or not np.isfinite(point).all():
            self.broken = True
            return
        if self.last is not None:
            dt = timestamp-self.last[0]
            if dt <= 0 or dt > self.max_missing_seconds+1e-6 or np.linalg.norm(point-self.last[1]) > 4500*dt+75:
                self.broken = True
        self.last = (timestamp, point.copy())
        self.samples += 1

    def anchor(self, point, timestamp):
        self.reset()
        if point is not None and np.isfinite(point).all():
            self.start = (timestamp, np.asarray(point, dtype=float).copy())
            self.last = self.start
            self.samples = 1

    def supports_transfer(self, timestamp):
        if self.broken or self.start is None or self.last is None or self.samples < 3:
            return False
        elapsed = timestamp-self.start[0]
        return (0 < elapsed <= self.max_seconds and timestamp-self.last[0] <= self.max_missing_seconds
                and np.linalg.norm(self.last[1]-self.start[1]) >= 200)

    def can_continue(self, timestamp):
        return (not self.broken and self.start is not None and self.last is not None
                and timestamp-self.start[0] <= self.max_seconds
                and timestamp-self.last[0] <= self.max_missing_seconds+1e-6)
