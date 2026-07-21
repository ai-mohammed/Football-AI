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
from typing import Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO

# See the matching comment in streamlit_app.py: the repo root needs to be on
# sys.path for `import sports...` to resolve, since that package isn't
# pip-installed on cloud deployments — only its own dependencies are (see
# requirements.txt).
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_APP_DIR))
for _path in (_APP_DIR, _REPO_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

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
# Frames to keep a lost track alive before giving up and assigning a new
# tracker id on reappearance (ByteTrack default is 30 — too short: a player
# briefly occluded or stepping out of frame would come back as a "new"
# player, inflating headcounts past 11 per team in the report/pass network).
LOST_TRACK_BUFFER = 90
# Drop tracker fragments shorter than this from reports — almost always
# leftover noise from a track that got lost and re-created with a new id,
# not a genuine extra player.
MIN_TRAJECTORY_FRAMES_FOR_REPORT = 15


def ocr_available() -> bool:
    try:
        import easyocr  # noqa: F401
        return True
    except ImportError:
        return False


@dataclass
class _TrackerStats:
    team_id: Optional[int] = None
    # Every frame's raw team-classification prediction for this tracker;
    # team_id is the majority vote across these, so a single noisy frame
    # can't flip the player's on-screen color or team assignment.
    team_votes: Counter = field(default_factory=Counter)
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
        self.total_frames_processed: int = 0
        # (from_tracker_id, to_tracker_id) -> number of passes exchanged.
        self.pass_edges: Counter = Counter()
        self.seconds_per_processed_frame: float = 0.0

        # A hard camera cut breaks ByteTrack's spatial/motion matching, so a
        # player who was on screen before the cut gets a brand new tracker id
        # after it — even though they never actually left the match. Once a
        # tracker's jersey number is confidently read and it matches a number
        # already confirmed for a *different* tracker on the same team, the
        # new id is folded into the original one: identity_redirect maps the
        # "loser" id to the surviving "winner" id, and every stats lookup
        # resolves through it first.
        self.identity_redirect: Dict[int, int] = {}
        self.confirmed_jersey_owner: Dict[Tuple[int, str], int] = {}

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
        self.seconds_per_processed_frame = seconds_per_processed_frame

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
        tracker = sv.ByteTrack(
            minimum_consecutive_frames=3,
            lost_track_buffer=LOST_TRACK_BUFFER,
            frame_rate=fps,
        )
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
            self.total_frames_processed = frame_idx

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

            if len(pg_detections):
                self._record_team_votes(pg_detections, pg_color_lookup)

            ball_detections = ball_slicer(frame).with_nms(threshold=0.1)
            ball_detections = ball_tracker.update(ball_detections)

            if transformer is not None and len(pg_detections):
                self._update_positions(
                    pg_detections, transformer, seconds_per_processed_frame)
                self._update_possession(pg_detections, ball_detections, transformer)

            self._sample_jersey_numbers(frame, pg_detections, frame_idx)

            all_detections = sv.Detections.merge([players, goalkeepers, referees])
            color_lookup = self._color_lookup_for(all_detections)
            yield self._annotate(frame, all_detections, color_lookup, ball_detections)

    def _resolve(self, tracker_id: Optional[int]) -> Optional[int]:
        """Follows identity_redirect to the canonical tracker id for a
        player who was re-identified across a camera cut."""
        if tracker_id is None:
            return None
        seen = set()
        while tracker_id in self.identity_redirect and tracker_id not in seen:
            seen.add(tracker_id)
            tracker_id = self.identity_redirect[tracker_id]
        return tracker_id

    def _merge_tracker(self, loser: int, winner: int) -> None:
        """Folds `loser`'s accumulated stats into `winner` — used when a
        camera cut gave the same physical player a new tracker id, and we
        recognized them again via a matching confirmed jersey number."""
        if loser == winner:
            return
        loser_stat = self.stats.pop(loser, None)
        self.identity_redirect[loser] = winner
        if loser_stat is None:
            return
        winner_stat = self.stats[winner]
        winner_stat.touches += loser_stat.touches
        winner_stat.passes_made += loser_stat.passes_made
        winner_stat.passes_received += loser_stat.passes_received
        winner_stat.distance_cm += loser_stat.distance_cm
        winner_stat.trajectory.extend(loser_stat.trajectory)
        winner_stat.jersey_votes.update(loser_stat.jersey_votes)
        winner_stat.team_votes.update(loser_stat.team_votes)
        if winner_stat.team_votes:
            winner_stat.team_id = winner_stat.team_votes.most_common(1)[0][0]

        for (a, b), weight in list(self.pass_edges.items()):
            resolved = (winner if a == loser else a, winner if b == loser else b)
            if resolved != (a, b):
                del self.pass_edges[(a, b)]
                if resolved[0] != resolved[1]:
                    self.pass_edges[resolved] += weight

        if self.current_possessor == loser:
            self.current_possessor = winner

    def _record_team_votes(self, pg_detections, pg_color_lookup):
        for i, raw_tracker_id in enumerate(pg_detections.tracker_id):
            tracker_id = self._resolve(raw_tracker_id)
            if tracker_id is None:
                continue
            stat = self.stats[tracker_id]
            stat.team_votes[int(pg_color_lookup[i])] += 1
            stat.team_id = stat.team_votes.most_common(1)[0][0]

    def _color_lookup_for(self, detections) -> np.ndarray:
        lookup = []
        tracker_ids = detections.tracker_id if detections.tracker_id is not None else []
        for raw_tracker_id in tracker_ids:
            tracker_id = self._resolve(raw_tracker_id)
            stat = self.stats.get(tracker_id) if tracker_id is not None else None
            lookup.append(
                stat.team_id if stat is not None and stat.team_id is not None
                else REFEREE_CLASS_ID)
        return np.array(lookup)

    def _update_positions(self, pg_detections, transformer, seconds_per_frame):
        xy = transformer.transform_points(
            pg_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER))
        max_delta_cm = MAX_PLAUSIBLE_SPEED_M_PER_S * 100 * seconds_per_frame
        for i, raw_tracker_id in enumerate(pg_detections.tracker_id):
            tracker_id = self._resolve(raw_tracker_id)
            if tracker_id is None:
                continue
            stat = self.stats[tracker_id]
            point = xy[i]
            # A merge can make the player's position "jump" from wherever
            # the old id last was to wherever the new id picked them up
            # (e.g. across a camera cut) — don't count that jump as
            # covered distance.
            if stat.last_xy is not None and raw_tracker_id == tracker_id:
                delta_cm = float(np.linalg.norm(point - stat.last_xy))
                if delta_cm <= max_delta_cm:
                    stat.distance_cm += delta_cm
            stat.last_xy = point
            stat.trajectory.append([float(point[0]), float(point[1])])

    def _update_possession(self, pg_detections, ball_detections, transformer):
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

        tracker_id = self._resolve(pg_detections.tracker_id[nearest_idx])
        if tracker_id is None or tracker_id == self.current_possessor:
            return

        stat = self.stats[tracker_id]
        team_id = stat.team_id
        stat.touches += 1

        if self.current_possessor is not None and self.current_possessor in self.stats:
            prev_stat = self.stats[self.current_possessor]
            if prev_stat.team_id == team_id and self.current_possessor != tracker_id:
                prev_stat.passes_made += 1
                stat.passes_received += 1
                self.pass_edges[(self.current_possessor, tracker_id)] += 1

        self.current_possessor = tracker_id

    def _sample_jersey_numbers(self, frame, pg_detections, frame_idx):
        if not ocr_available() or len(pg_detections) == 0:
            return
        for i, raw_tracker_id in enumerate(pg_detections.tracker_id):
            tracker_id = self._resolve(raw_tracker_id)
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
                self._register_jersey_read(tracker_id, number)

    def _register_jersey_read(self, tracker_id: int, number: str) -> None:
        stat = self.stats[tracker_id]
        stat.jersey_votes[number] += 1
        top_number, votes = stat.jersey_votes.most_common(1)[0]
        if votes < MIN_OCR_READS_FOR_LABEL or stat.team_id is None:
            return

        key = (stat.team_id, top_number)
        owner = self.confirmed_jersey_owner.get(key)
        if owner is None:
            self.confirmed_jersey_owner[key] = tracker_id
        elif owner != tracker_id:
            # Same team + same confirmed number on a different tracker id —
            # almost certainly the same physical player, re-identified after
            # a camera cut or tracking gap. Fold this one into the original.
            self._merge_tracker(loser=tracker_id, winner=owner)

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
        tracker_id = self._resolve(tracker_id)
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
        merged = _merge_by_jersey_number(players)
        # Scale the cutoff with clip length: a fixed frame count is too
        # lenient on long clips (keeps tiny fragments) and too strict on
        # very short ones.
        min_frames = max(
            MIN_TRAJECTORY_FRAMES_FOR_REPORT,
            int(0.15 * self.total_frames_processed))
        merged = [row for row in merged if len(row['trajectory']) >= min_frames]
        for row in merged:
            seconds = len(row['trajectory']) * self.seconds_per_processed_frame
            row['avg_speed_kmh'] = (
                round((row['distance_m'] / seconds) * 3.6, 1) if seconds > 0 else 0.0)
        return merged

    def team_report(self) -> dict:
        """
        Aggregates `report()` into team-level stats: possession share, a
        combined heatmap of pitch coverage, and a pass network (nodes at each
        player's average position, edges weighted by passes exchanged).
        """
        players = self.report()
        teams: Dict[int, List[dict]] = defaultdict(list)
        for row in players:
            if row['team_id'] in (0, 1):
                teams[row['team_id']].append(row)

        total_touches = sum(row['touches'] for row in players if row['team_id'] in (0, 1))

        possession_pct = {}
        team_heatmaps = {}
        pass_networks = {}
        for team_id in (0, 1):
            rows = teams.get(team_id, [])
            team_touches = sum(row['touches'] for row in rows)
            possession_pct[team_id] = (
                round(team_touches / total_touches * 100, 1) if total_touches else 0.0)

            trajectories = [row['trajectory'] for row in rows if row['trajectory'].size]
            team_heatmaps[team_id] = (
                np.vstack(trajectories) if trajectories else np.empty((0, 2)))

            tracker_to_node: Dict[int, int] = {}
            node_xy, node_labels = [], []
            for idx, row in enumerate(rows):
                for tid in row['tracker_ids']:
                    tracker_to_node[tid] = idx
                node_xy.append(
                    row['trajectory'].mean(axis=0) if row['trajectory'].size
                    else np.array([CONFIG.length / 2, CONFIG.width / 2]))
                node_labels.append(
                    f"#{row['jersey_number']}" if row['jersey_number']
                    else f"ID{row['tracker_ids'][0]}")

            edge_counts: Counter = Counter()
            for (from_id, to_id), weight in self.pass_edges.items():
                i, j = tracker_to_node.get(from_id), tracker_to_node.get(to_id)
                if i is None or j is None or i == j:
                    continue
                edge_counts[(i, j)] += weight

            pass_networks[team_id] = {
                'node_xy': np.array(node_xy) if node_xy else np.empty((0, 2)),
                'node_labels': node_labels,
                'edges': [(i, j, w) for (i, j), w in edge_counts.items()],
            }

        return {
            'possession_pct': possession_pct,
            'team_heatmaps': team_heatmaps,
            'pass_networks': pass_networks,
        }


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
