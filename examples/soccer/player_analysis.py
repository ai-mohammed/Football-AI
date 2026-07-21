"""
Per-player match analysis: jersey number recognition, ball-touch counting,
pass detection, and pitch cartography (heatmap of where each player played).

Built on top of the existing detection/tracking/team-classification pipeline
(see `main.py`), adding:

- Jersey number reading via OCR (best-effort, majority-voted across frames —
  general-purpose OCR on broadcast footage is inherently noisy, there's no
  dedicated soccer jersey-number model in this repo).
- Ball possession/touch/pass detection via nearest-player-to-ball proximity in
  pitch space (using the pitch homography, same approach as RADAR mode).
- Per-tracker trajectories in pitch coordinates, merged across ID switches
  using the recognized jersey number.
"""
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional

import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import (  # noqa: E402
    ELLIPSE_ANNOTATOR,
    ELLIPSE_LABEL_ANNOTATOR,
    GOALKEEPER_CLASS_ID,
    PLAYER_CLASS_ID,
    REFEREE_CLASS_ID,
    STRIDE,
    get_crops,
    resolve_goalkeepers_team_id,
)
from sports.common.ball import BallAnnotator, BallTracker  # noqa: E402
from sports.common.team import TeamClassifier  # noqa: E402
from sports.common.view import ViewTransformer  # noqa: E402
from sports.configs.soccer import SoccerPitchConfiguration  # noqa: E402

CONFIG = SoccerPitchConfiguration()

# A ball within this distance (pitch cm) of a player's feet is considered "at
# their feet" — soccer pitches are ~7000x12000 cm, so 150cm (~1.5m) is a tight
# but forgiving radius given detection/homography noise.
TOUCH_DISTANCE_CM = 150
# Discard frame-to-frame displacements implying a speed above this (guards
# against homography jitter / tracker-id mix-ups inflating distance covered).
MAX_PLAUSIBLE_SPEED_M_PER_S = 12.0
# Spread OCR reads across frames instead of running it on every tracked player
# every frame (expensive). Each tracker gets sampled roughly this often.
JERSEY_OCR_EVERY_N_FRAMES = 5
# A jersey number is only trusted once it "wins" at least this many OCR reads.
MIN_OCR_READS_FOR_LABEL = 3
# Re-run pitch keypoint detection (and refresh the homography) every N
# processed frames instead of every single one — see the comment in
# PlayerMatchAnalyzer.process for why this is a safe trade-off.
PITCH_DETECTION_INTERVAL = 5


def ocr_available() -> bool:
    try:
        import easyocr  # noqa: F401
        return True
    except ImportError:
        return False


@dataclass
class _TrackerStats:
    team_id: Optional[int] = None
    jersey_votes: Counter = field(default_factory=Counter)
    touches: int = 0
    passes_made: int = 0
    passes_received: int = 0
    distance_cm: float = 0.0
    last_xy: Optional[np.ndarray] = None
    trajectory: List[List[float]] = field(default_factory=list)


def _safe_transformer(keypoints: sv.KeyPoints) -> Optional[ViewTransformer]:
    if keypoints.xy is None or len(keypoints.xy) == 0:
        return None
    xy = keypoints.xy[0]
    mask = (xy[:, 0] > 1) & (xy[:, 1] > 1)
    if mask.sum() < 4:
        return None
    try:
        return ViewTransformer(
            source=xy[mask].astype(np.float32),
            target=np.array(CONFIG.vertices)[mask].astype(np.float32),
        )
    except ValueError:
        return None


class PlayerMatchAnalyzer:
    """Runs the full per-player analysis pipeline over a video."""

    def __init__(
        self,
        player_model_path: str,
        pitch_model_path: str,
        ball_model_path: str,
        device: str,
    ):
        self.player_model = YOLO(player_model_path).to(device=device)
        self.pitch_model = YOLO(pitch_model_path).to(device=device)
        self.ball_model = YOLO(ball_model_path).to(device=device)
        self.device = device

        self._ocr_reader = None
        self.stats: Dict[int, _TrackerStats] = defaultdict(_TrackerStats)
        self.current_possessor: Optional[int] = None

        self.ball_annotator = BallAnnotator(radius=6, buffer_size=10)

    @property
    def ocr_reader(self):
        if self._ocr_reader is None:
            import easyocr
            self._ocr_reader = easyocr.Reader(
                ['en'], gpu=self.device not in ('cpu',), verbose=False)
        return self._ocr_reader

    def process(self, source_video_path: str, stride: int = 1) -> Iterator[np.ndarray]:
        video_info = sv.VideoInfo.from_video_path(source_video_path)
        fps = video_info.fps or 25
        seconds_per_processed_frame = stride / fps

        crop_generator = sv.get_video_frames_generator(
            source_path=source_video_path, stride=STRIDE)
        crops = []
        for frame in tqdm(crop_generator, desc='collecting crops'):
            result = self.player_model(frame, imgsz=1280, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(result)
            crops += get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])

        team_classifier = TeamClassifier(device=self.device)
        if crops:
            team_classifier.fit(crops)

        frame_generator = sv.get_video_frames_generator(
            source_path=source_video_path, stride=stride)
        tracker = sv.ByteTrack(minimum_consecutive_frames=3)
        ball_tracker = BallTracker(buffer_size=20)

        def ball_callback(image_slice: np.ndarray) -> sv.Detections:
            result = self.ball_model(image_slice, imgsz=640, verbose=False)[0]
            return sv.Detections.from_ultralytics(result)

        ball_slicer = sv.InferenceSlicer(
            callback=ball_callback,
            overlap_filter=sv.OverlapFilter.NONE,
            slice_wh=(640, 640),
        )

        frame_idx = 0
        transformer = None
        for frame in frame_generator:
            frame_idx += 1

            # Re-running pitch keypoint detection every single frame is one of
            # three YOLO calls per frame, and the broadcast camera moves little
            # between consecutive frames — so the homography stays a good
            # approximation for a few frames. Refresh it periodically instead.
            if frame_idx % PITCH_DETECTION_INTERVAL == 1:
                pitch_result = self.pitch_model(frame, verbose=False)[0]
                keypoints = sv.KeyPoints.from_ultralytics(pitch_result)
                fresh_transformer = _safe_transformer(keypoints)
                if fresh_transformer is not None:
                    transformer = fresh_transformer

            result = self.player_model(frame, imgsz=1280, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(result)
            detections = tracker.update_with_detections(detections)

            players = detections[detections.class_id == PLAYER_CLASS_ID]
            goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
            referees = detections[detections.class_id == REFEREE_CLASS_ID]

            players_team_id = (
                team_classifier.predict(get_crops(frame, players))
                if len(players) else np.array([])
            )
            goalkeepers_team_id = (
                resolve_goalkeepers_team_id(players, players_team_id, goalkeepers)
                if len(players) and len(goalkeepers) else np.array([])
            )

            pg_detections = sv.Detections.merge([players, goalkeepers])
            pg_color_lookup = np.array(
                players_team_id.tolist() + goalkeepers_team_id.tolist()
            )

            ball_detections = ball_slicer(frame).with_nms(threshold=0.1)
            ball_detections = ball_tracker.update(ball_detections)

            if transformer is not None and len(pg_detections):
                self._update_positions(
                    pg_detections, pg_color_lookup, transformer,
                    seconds_per_processed_frame)
                self._update_possession(
                    pg_detections, pg_color_lookup, ball_detections, transformer)

            self._sample_jersey_numbers(frame, pg_detections, frame_idx)

            all_detections = sv.Detections.merge([players, goalkeepers, referees])
            color_lookup = np.array(
                players_team_id.tolist() + goalkeepers_team_id.tolist() +
                [REFEREE_CLASS_ID] * len(referees)
            )
            yield self._annotate(frame, all_detections, color_lookup, ball_detections)

    def _update_positions(self, pg_detections, pg_color_lookup, transformer, seconds_per_frame):
        xy = transformer.transform_points(
            pg_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER))
        max_delta_cm = MAX_PLAUSIBLE_SPEED_M_PER_S * 100 * seconds_per_frame
        for i, tracker_id in enumerate(pg_detections.tracker_id):
            if tracker_id is None:
                continue
            stat = self.stats[tracker_id]
            stat.team_id = int(pg_color_lookup[i])
            point = xy[i]
            if stat.last_xy is not None:
                delta_cm = float(np.linalg.norm(point - stat.last_xy))
                if delta_cm <= max_delta_cm:
                    stat.distance_cm += delta_cm
            stat.last_xy = point
            stat.trajectory.append([float(point[0]), float(point[1])])

    def _update_possession(self, pg_detections, pg_color_lookup, ball_detections, transformer):
        if len(ball_detections) == 0 or len(pg_detections) == 0:
            return
        ball_xy = transformer.transform_points(
            ball_detections.get_anchors_coordinates(sv.Position.CENTER))
        if ball_xy.size == 0:
            return
        ball_point = ball_xy[0]
        player_xy = transformer.transform_points(
            pg_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER))
        distances = np.linalg.norm(player_xy - ball_point, axis=1)
        nearest_idx = int(np.argmin(distances))
        if distances[nearest_idx] > TOUCH_DISTANCE_CM:
            return

        tracker_id = pg_detections.tracker_id[nearest_idx]
        if tracker_id is None or tracker_id == self.current_possessor:
            return

        team_id = int(pg_color_lookup[nearest_idx])
        stat = self.stats[tracker_id]
        stat.team_id = team_id
        stat.touches += 1

        if self.current_possessor is not None and self.current_possessor in self.stats:
            prev_stat = self.stats[self.current_possessor]
            if prev_stat.team_id == team_id and self.current_possessor != tracker_id:
                prev_stat.passes_made += 1
                stat.passes_received += 1

        self.current_possessor = tracker_id

    def _sample_jersey_numbers(self, frame, pg_detections, frame_idx):
        if not ocr_available() or len(pg_detections) == 0:
            return
        for i, tracker_id in enumerate(pg_detections.tracker_id):
            if tracker_id is None:
                continue
            if frame_idx % JERSEY_OCR_EVERY_N_FRAMES != tracker_id % JERSEY_OCR_EVERY_N_FRAMES:
                continue
            x1, y1, x2, y2 = pg_detections.xyxy[i].astype(int)
            h = y2 - y1
            # Numbers sit on the torso, roughly between the shoulders and the
            # waist — skip the head and cut before the shorts to reduce noise.
            crop = frame[max(y1 + int(h * 0.12), 0):y1 + int(h * 0.55), max(x1, 0):x2]
            if crop.size == 0:
                continue
            number = self._read_jersey_number(crop)
            if number:
                self.stats[tracker_id].jersey_votes[number] += 1

    def _read_jersey_number(self, crop: np.ndarray) -> Optional[str]:
        h, w = crop.shape[:2]
        if h < 8 or w < 5:
            return None
        # Jersey-number crops are tiny in broadcast footage (often well under
        # 100px tall); upscale before OCR so the recognizer has enough pixels
        # to work with.
        scale = max(1.0, 200 / h)
        if scale > 1.0:
            crop = cv2.resize(
                crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        try:
            results = self.ocr_reader.readtext(
                crop, allowlist='0123456789', detail=1, mag_ratio=2)
        except Exception:
            return None
        best, best_conf = None, 0.0
        for _, text, conf in results:
            text = text.strip()
            if text.isdigit() and 0 < len(text) <= 2 and conf > best_conf:
                best, best_conf = text, conf
        return best if best is not None and best_conf >= 0.4 else None

    def _label_for(self, tracker_id: Optional[int]) -> str:
        if tracker_id is None:
            return ""
        stat = self.stats.get(tracker_id)
        if stat and stat.jersey_votes:
            number, votes = stat.jersey_votes.most_common(1)[0]
            if votes >= MIN_OCR_READS_FOR_LABEL:
                return f"#{number}"
        # Jersey number not confirmed yet — clearly mark this as a tracking
        # ID so it's never mistaken for a real jersey number on screen.
        return f"ID{tracker_id}"

    def _annotate(self, frame, detections, color_lookup, ball_detections):
        annotated_frame = frame.copy()
        tracker_ids = detections.tracker_id if detections.tracker_id is not None else []
        labels = [self._label_for(tid) for tid in tracker_ids]
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(
            annotated_frame, detections, custom_color_lookup=color_lookup)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels, custom_color_lookup=color_lookup)
        annotated_frame = self.ball_annotator.annotate(annotated_frame, ball_detections)
        return annotated_frame

    def report(self) -> List[dict]:
        players = []
        for tracker_id, stat in self.stats.items():
            if stat.team_id is None:
                continue
            jersey_number, votes = (
                stat.jersey_votes.most_common(1)[0] if stat.jersey_votes else (None, 0))
            label = jersey_number if votes >= MIN_OCR_READS_FOR_LABEL else None
            players.append({
                'tracker_ids': [tracker_id],
                'jersey_number': label,
                'team_id': stat.team_id,
                'touches': stat.touches,
                'passes_made': stat.passes_made,
                'passes_received': stat.passes_received,
                'distance_m': round(stat.distance_cm / 100, 1),
                'trajectory': np.array(stat.trajectory) if stat.trajectory else np.empty((0, 2)),
            })
        return _merge_by_jersey_number(players)


def _merge_by_jersey_number(players: List[dict]) -> List[dict]:
    merged: Dict[tuple, dict] = {}
    unresolved = []
    for p in players:
        if p['jersey_number'] is None:
            unresolved.append(p)
            continue
        key = (p['team_id'], p['jersey_number'])
        if key not in merged:
            merged[key] = dict(p)
        else:
            m = merged[key]
            m['touches'] += p['touches']
            m['passes_made'] += p['passes_made']
            m['passes_received'] += p['passes_received']
            m['distance_m'] += p['distance_m']
            m['trajectory'] = (
                np.vstack([m['trajectory'], p['trajectory']])
                if m['trajectory'].size and p['trajectory'].size
                else (m['trajectory'] if m['trajectory'].size else p['trajectory'])
            )
            m['tracker_ids'] += p['tracker_ids']

    result = list(merged.values()) + unresolved
    result.sort(key=lambda p: -p['touches'])
    return result
