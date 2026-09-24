"""Confidence-filtered pitch calibration; reject unusable geometry."""
import cv2
import numpy as np

from sports.common.view import ViewTransformer


def pitch_transformer(keypoints, vertices, min_confidence=0.5, max_error_px=6.0):
    if keypoints.xy is None or len(keypoints.xy) == 0:
        return None
    xy = np.asarray(keypoints.xy[0], dtype=np.float32)
    target = np.asarray(vertices, dtype=np.float32)
    if xy.shape != target.shape:
        return None  # Human pose (17 landmarks) is not a pitch model (32).
    mask = np.isfinite(xy).all(axis=1) & (xy > 1).all(axis=1)
    confidence = keypoints.keypoint_confidence if hasattr(keypoints, 'keypoint_confidence') else keypoints.confidence
    if confidence is not None:
        mask &= confidence[0] >= min_confidence
    xy, target = xy[mask], target[mask]
    return _fit_pitch(xy, target, max_error_px)


def _fit_pitch(xy, target, max_error_px=6.):
    if len(xy) < 4:
        return None
    if min(cv2.contourArea(cv2.convexHull(xy)), cv2.contourArea(cv2.convexHull(target))) < 100:
        return None
    # Fit pitch -> image so the RANSAC tolerance is expressed in pixels.
    matrix, inliers = cv2.findHomography(target, xy, cv2.RANSAC, max_error_px)
    if matrix is None or inliers is None or inliers.sum() < 4 or inliers.mean() < 0.6:
        return None
    if not np.isfinite(matrix).all() or np.linalg.cond(matrix) > 1e12:
        return None
    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return None
    transformer = ViewTransformer.__new__(ViewTransformer)
    transformer.m = inverse
    transformer.inlier_count = int(inliers.sum())
    transformer.image_points = xy[inliers.ravel() > 0].copy()
    transformer.pitch_points = target[inliers.ravel() > 0].copy()
    projected = cv2.perspectiveTransform(target.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    transformer.error_px = float(np.median(np.linalg.norm(projected - xy, axis=1)[inliers.ravel() > 0]))
    return transformer


class TemporalPitchCalibrator:
    """Track measured landmarks through brief model misses, not stale matrices.

    Direct keypoints take priority. Forward/backward optical flow and the same
    RANSAC geometry checks are required for at most .32 s after a direct fit.
    """
    def __init__(self, vertices, max_gap=.32):
        self.vertices = vertices
        self.max_gap = max_gap
        self.reset()

    def reset(self):
        self.gray = self.transformer = self.direct_at = None
        self.method = None

    def update(self, frame, keypoints, timestamp):
        scale = min(1., 960/frame.shape[1])
        gray = cv2.cvtColor(cv2.resize(frame, None, fx=scale, fy=scale), cv2.COLOR_BGR2GRAY)
        direct = pitch_transformer(keypoints, self.vertices)
        result = direct
        self.method = 'pitch_keypoints' if direct is not None else None
        if direct is not None:
            self.direct_at = timestamp
        elif (self.transformer is not None and self.gray is not None and self.gray.shape == gray.shape
              and self.direct_at is not None and 0 <= timestamp-self.direct_at <= self.max_gap+1e-6):
            points = np.float32(self.transformer.image_points*scale).reshape(-1, 1, 2)
            target = self.transformer.pitch_points
            inside = ((points[:, 0, 0] >= 10) & (points[:, 0, 0] < gray.shape[1]-10)
                      & (points[:, 0, 1] >= 10) & (points[:, 0, 1] < gray.shape[0]-10))
            points, target = points[inside], target[inside]
            if len(points) >= 6:
                moved, status, error = cv2.calcOpticalFlowPyrLK(self.gray, gray, points, None,
                                                               winSize=(31, 31), maxLevel=3)
                if moved is not None:
                    back, reverse_status, _ = cv2.calcOpticalFlowPyrLK(gray, self.gray, moved, None,
                                                                     winSize=(31, 31), maxLevel=3)
                    if back is not None:
                        good = ((status.ravel() > 0) & (reverse_status.ravel() > 0)
                                & (error.ravel() < 20) & (np.linalg.norm(back-points, axis=2).ravel() < 1.))
                        if good.sum() >= 6:
                            result = _fit_pitch(moved[good, 0]/scale, target[good])
                            if result is not None:
                                self.method = 'pitch_optical_flow'
        self.gray, self.transformer = gray, result
        return result


class ShotChangeDetector:
    """Conservative hard-cut heuristic, not a semantic replay detector."""
    def __init__(self, threshold=0.30):
        self.previous = None
        self.threshold = threshold

    def update(self, frame):
        small = cv2.resize(frame, (64, 36)).astype(np.float32) / 255
        changed = self.previous is not None and np.mean(np.abs(small - self.previous)) > self.threshold
        self.previous = small
        return bool(changed)
