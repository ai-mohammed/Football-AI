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
    projected = cv2.perspectiveTransform(target.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    transformer.error_px = float(np.median(np.linalg.norm(projected - xy, axis=1)[inliers.ravel() > 0]))
    return transformer


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
