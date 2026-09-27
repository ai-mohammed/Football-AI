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
    """Cross-check model landmarks against measured camera motion.

    A one-frame landmark jump must not teleport the entire team. Flow is
    independently checked in both directions and expires after .32 seconds.
    The caller resets this history at detected camera cuts.
    """
    def __init__(self, vertices, max_gap=.32):
        self.vertices = vertices
        self.max_gap = max_gap
        self.reset()

    def reset(self):
        self.gray = self.transformer = self.direct_at = None
        self.method = None
        self.rejected_jump = False
        self.smoothed = False
        self.layout_blocked = False

    def update(self, frame, keypoints, timestamp):
        # A heavily letterboxed television montage can contain two simultaneous
        # views. One global homography cannot describe that layout. Abstain until
        # a supported full-frame view returns, rather than map duplicate players.
        preview = cv2.resize(frame,(320,180))
        dark = preview.max(axis=2) < 45
        if dark[:36].mean() > .7 and dark[-27:].mean() > .7 and dark.mean() > .3:
            self.reset()
            self.layout_blocked = True
            return None
        self.layout_blocked = False
        scale = min(1., 960/frame.shape[1])
        gray = cv2.cvtColor(cv2.resize(frame, None, fx=scale, fy=scale), cv2.COLOR_BGR2GRAY)
        direct = pitch_transformer(keypoints, self.vertices)
        flowed = None
        self.rejected_jump = False
        self.smoothed = False
        if (self.transformer is not None and self.gray is not None and self.gray.shape == gray.shape
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
                            flowed = _fit_pitch(moved[good, 0]/scale, target[good])
        if direct is not None and flowed is not None:
            # Compare in centimetres on independently tracked image landmarks.
            # Two metres between adjacent samples is a calibration disagreement,
            # not player motion. It must not refresh the direct-fit expiry.
            displacement = np.linalg.norm(
                direct.transform_points(flowed.image_points)-flowed.pitch_points, axis=1)
            if np.median(displacement) > 200:
                direct = None
                self.rejected_jump = True
            else:
                # Camera motion comes from measured optical flow. Correct its
                # geometry gradually toward new model landmarks, so independent
                # frame-to-frame model jitter cannot move the whole team.
                predicted = cv2.perspectiveTransform(direct.pitch_points[:, None],
                                                    np.linalg.inv(flowed.m))[:, 0]
                stable = _fit_pitch(np.float32(.25*direct.image_points+.75*predicted),
                                    direct.pitch_points)
                if stable is not None:
                    direct = stable
                    self.smoothed = True
        result = direct if direct is not None else flowed
        self.method = 'pitch_keypoints' if direct is not None else 'pitch_optical_flow' if flowed is not None else None
        if direct is not None:
            self.direct_at = timestamp
        self.gray, self.transformer = gray, result
        return result


class ShotChangeDetector:
    """Detect content changes and geometrically unrelated views of green grass.

    Photometric difference alone misses most broadcast cuts. Feature matches
    distinguish a pan/zoom from a new shot when colour remains similar. This
    still does not identify replays or the match clock.
    """
    def __init__(self, threshold=0.30):
        self.previous = None
        self.threshold = threshold
        self.previous_gray = None
        self.previous_hist = None
        self.change_score = 0.

    def update(self, frame):
        small = cv2.resize(frame, (320, 180))[20:170]
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        changed = False
        if self.previous is not None:
            difference = float(np.mean(cv2.absdiff(small, self.previous)))/255
            colour = cv2.compareHist(hist, self.previous_hist, cv2.HISTCMP_BHATTACHARYYA)
            self.change_score = difference
            changed = difference > self.threshold or (difference > .09 and colour > .55)
            if not changed and difference > .075:
                detector = cv2.ORB_create(nfeatures=300, edgeThreshold=12, fastThreshold=12)
                a, da = detector.detectAndCompute(self.previous_gray, None)
                b, db = detector.detectAndCompute(gray, None)
                if da is not None and db is not None and min(len(a), len(b)) >= 20:
                    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
                    matches = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < .7*pair[1].distance]
                    support = 0
                    if len(matches) >= 6:
                        before = np.float32([a[m.queryIdx].pt for m in matches])
                        after = np.float32([b[m.trainIdx].pt for m in matches])
                        _, inliers = cv2.findHomography(before, after, cv2.RANSAC, 3.)
                        support = int(inliers.sum()) if inliers is not None else 0
                    changed = support < 6 and len(matches) < .08*min(len(a), len(b))
                    if changed:
                        # ORB can lose descriptors when a close-up subject turns.
                        # A coherent, reversible flow is evidence of the same shot.
                        points = cv2.goodFeaturesToTrack(self.previous_gray, 100, .02, 8)
                        if points is not None and len(points) >= 12:
                            moved, ok, err = cv2.calcOpticalFlowPyrLK(self.previous_gray, gray, points, None)
                            if moved is not None:
                                back, reverse, _ = cv2.calcOpticalFlowPyrLK(gray, self.previous_gray, moved, None)
                                if back is not None:
                                    valid = ((ok.ravel() > 0) & (reverse.ravel() > 0)
                                             & (err.ravel() < 20)
                                             & (np.linalg.norm(back-points,axis=2).ravel() < 1.))
                                    if valid.sum() >= max(10, .25*len(points)):
                                        _, mask = cv2.estimateAffinePartial2D(points[valid],moved[valid],
                                                                            method=cv2.RANSAC,ransacReprojThreshold=2.)
                                        if mask is not None and mask.sum() >= max(10,.25*len(points)):
                                            changed = False
        self.previous = small
        self.previous_gray, self.previous_hist = gray, hist
        return bool(changed)
