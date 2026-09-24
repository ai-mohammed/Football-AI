"""Opt-in full-pitch aerial analysis. No trained weights or fixed image coordinates.

The white boundary and halfway line must be visible. A missing boundary is never
carried forward: camera movement would turn an old homography into false motion.
"""
import cv2
import numpy as np
import supervision as sv

from sports.common.view import ViewTransformer


def _refine_edges(quad, white):
    """Refine on long white segments; corner arcs/logos are not pitch edges."""
    segments = cv2.HoughLinesP(white, 1, np.pi/1800, threshold=60,
                              minLineLength=min(white.shape)*.15, maxLineGap=15)
    if segments is None:
        return None
    segments = segments.reshape(-1, 2, 2).astype(np.float32)
    vectors = segments[:, 1]-segments[:, 0]
    lengths = np.linalg.norm(vectors, axis=1)
    midpoints = segments.mean(axis=1)
    ys, xs = np.nonzero(white)
    points = np.column_stack((xs, ys)).astype(np.float32)
    lines = []
    for a, b in zip(quad, np.roll(quad, -1, axis=0)):
        edge = b-a
        length = np.linalg.norm(edge)
        tangent = edge/length
        normal = np.float32([-tangent[1], tangent[0]])
        aligned = np.abs(vectors @ tangent)/np.maximum(lengths, 1)
        candidates = np.flatnonzero((aligned > .995) & (lengths > .4*length)
                                     & (np.abs((midpoints-a) @ normal) < .025*length))
        if not len(candidates):
            return None
        seed = segments[candidates[np.argmax(lengths[candidates])]]
        direction = seed[1]-seed[0]
        direction /= np.linalg.norm(direction)
        normal = np.float32([-direction[1], direction[0]])
        along = (points[:, 0]-a[0])*tangent[0] + (points[:, 1]-a[1])*tangent[1]
        across = (points[:, 0]-seed[0, 0])*normal[0] + (points[:, 1]-seed[0, 1])*normal[1]
        selected = points[(along > .1*length) & (along < .9*length) & (np.abs(across) < 4.)]
        if len(selected) < 30:
            return None
        vx, vy, x, y = cv2.fitLine(selected, cv2.DIST_HUBER, 0, .01, .01).ravel()
        lines.append(np.float64([-vy, vx, vy*x-vx*y]))
    corners = []
    for i in range(4):
        crossing = np.cross(lines[i-1], lines[i])
        if abs(crossing[2]) < 1e-6:
            return None
        corners.append(crossing[:2]/crossing[2])
    corners = np.asarray(corners, dtype=np.float32)
    if np.linalg.norm(corners-quad, axis=1).max() > .05*np.linalg.norm(quad[0]-quad[3]):
        return None
    return corners


def full_pitch_boundary(frame):
    """Return TL, TR, BR, BL in source pixels, or abstain on incomplete geometry."""
    height, width = frame.shape[:2]
    scale = min(1., 1280 / width)
    small = cv2.resize(frame, (round(width * scale), round(height * scale)))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, np.array([0, 0, 150]), np.array([179, 65, 255]))
    connected = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    h, w = small.shape[:2]
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:3]:
        area = cv2.contourArea(contour)
        if not .25 * w * h < area < .92 * w * h:
            continue
        polygon = cv2.approxPolyDP(contour, .02 * cv2.arcLength(contour, True), True)
        if len(polygon) != 4 or not cv2.isContourConvex(polygon):
            continue
        quad = polygon.reshape(4, 2).astype(np.float32)
        center = quad.mean(axis=0)
        quad = quad[np.argsort(np.arctan2(quad[:, 1]-center[1], quad[:, 0]-center[0]))]
        quad = np.roll(quad, -np.argmin(quad.sum(axis=1)), axis=0)
        quad = _refine_edges(quad, white)
        if quad is None:
            continue
        if (quad < 3).any() or (quad[:, 0] > w-4).any() or (quad[:, 1] > h-4).any():
            continue
        # This profile expects the long axis roughly horizontal, not a cropped
        # broadcast view or a portrait view requiring a different orientation.
        edges = np.linalg.norm(quad - np.roll(quad, -1, axis=0), axis=1)
        if edges.min() < .3 * h or not 1.15 < edges[[0, 2]].mean()/edges[[1, 3]].mean() < 2.5:
            continue
        inside = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(inside, quad.astype(int), 255)
        green = cv2.inRange(hsv, np.array([25, 30, 35]), np.array([100, 255, 255]))
        if np.mean(green[inside > 0] > 0) < .7:
            continue
        rectangle = np.float32([[0, 0], [599, 0], [599, 399], [0, 399]])
        matrix = cv2.getPerspectiveTransform(quad, rectangle)
        rectified = cv2.warpPerspective(white, matrix, (600, 400))
        # Independent internal marking: prevents mapping an arbitrary green
        # rectangle or the outside fence to the pitch.
        if np.mean((rectified[40:360, 285:315] > 0).any(axis=1)) < .75:
            continue
        return quad / scale
    return None


def boundary_transformer(boundary, length, width):
    if boundary is None:
        return None
    return ViewTransformer(source=np.asarray(boundary, dtype=np.float32),
                           target=np.float32([[0, 0], [length, 0], [length, width], [0, width]]))


def on_pitch_mask(boxes, boundary):
    """Filter box centres, allowing a small margin for touchline detections."""
    if boundary is None:
        return np.ones(len(boxes), dtype=bool)
    polygon = np.asarray(boundary, dtype=np.float32)
    margin = .005 * np.linalg.norm(polygon[0] - polygon[3])
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    return np.asarray([cv2.pointPolygonTest(polygon, tuple(map(float, xy)), True) >= -margin
                       for xy in centers], dtype=bool)


def tile_starts(length, size, overlap=.2):
    if size < 32 or not 0 <= overlap < 1:
        raise ValueError('Invalid tile size or overlap')
    if length <= size:
        return [0]
    step = max(1, round(size * (1-overlap)))
    return sorted(set([*range(0, length-size+1, step), length-size]))


class TiledPlayerDetector:
    """Preserve small players with overlapping tiles and global, class-agnostic NMS."""
    def __init__(self, model, tile_size=1280, imgsz=1280):
        self.model, self.tile_size, self.imgsz = model, tile_size, imgsz

    def __call__(self, frame, boundary=None, conf=.1):
        import torch
        from ultralytics.engine.results import Results
        height, width = frame.shape[:2]
        pieces = []
        for y in tile_starts(height, self.tile_size):
            for x in tile_starts(width, self.tile_size):
                # Sequential calls bound GPU/CPU memory on hosted deployments.
                tile = frame[y:y+self.tile_size, x:x+self.tile_size]
                result = self.model(tile, imgsz=self.imgsz, conf=conf, verbose=False)[0]
                detections = sv.Detections.from_ultralytics(result)
                detections = detections[np.isin(detections.class_id, [1, 2, 3])]
                detections.xyxy += np.float32([x, y, x, y])
                detections = detections[on_pitch_mask(detections.xyxy, boundary)]
                pieces.append(detections)
        detections = sv.Detections.merge(pieces).with_nms(threshold=.4, class_agnostic=True)
        boxes = (np.column_stack((detections.xyxy, detections.confidence, detections.class_id)).astype(np.float32)
                 if len(detections) else np.empty((0, 6), dtype=np.float32))
        return Results(orig_img=frame, path='', names=self.model.names, boxes=torch.from_numpy(boxes))
