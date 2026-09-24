"""Football analysis with guarded identities and calibrated, timestamped statistics."""
import os
import sys
from collections import Counter
from typing import Iterator

import numpy as np
import supervision as sv
from ultralytics import YOLO

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_APP_DIR))
for _path in (_APP_DIR, _REPO_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from main import ELLIPSE_ANNOTATOR, ELLIPSE_LABEL_ANNOTATOR, get_crops, resolve_goalkeepers_team_id
from sports.common.ball import BallAnnotator, BallTracker
from sports.common.calibration import ShotChangeDetector, pitch_transformer
from sports.common.identity import MatchState
from sports.common.runtime import resolve_device
from sports.common.team import TeamClassifier
from sports.common.tracking import FootballTracker
from sports.configs.soccer import SoccerPitchConfiguration

CONFIG = SoccerPitchConfiguration()


def ocr_available():
    import importlib.util
    return importlib.util.find_spec("easyocr") is not None


def _safe_transformer(keypoints):
    return pitch_transformer(keypoints, CONFIG.vertices)


def validate_models(player, pitch, ball):
    names = [str(player.names[i]).lower() for i in sorted(player.names)]
    if names != ["ball", "goalkeeper", "player", "referee"]:
        raise ValueError("Le modèle joueurs doit reconnaître ball, goalkeeper, player, referee dans cet ordre. "
                         "Les poids COCO génériques ne remplacent pas un modèle football affiné.")
    shape = getattr(pitch.model, "kpt_shape", None)
    if shape is None:
        shape = getattr(pitch.model.model[-1], "kpt_shape", None)
    if pitch.task != "pose" or list(shape or []) != [32, 3]:
        raise ValueError("Le modèle terrain doit prédire 32 points (x, y, confiance), dans l'ordre du terrain.")
    if len(ball.names) != 1 or str(ball.names[0]).lower() != "ball":
        raise ValueError("Le modèle ballon doit avoir une seule classe : ball.")


class PlayerMatchAnalyzer:
    def __init__(self, player_model_path, pitch_model_path, ball_model_path,
                 device="auto", tracker_backend="bytetrack", enable_ocr=True,
                 reid_model="yolo11n-cls.pt", imgsz=1280):
        self.device = resolve_device(device)
        self.player_model = YOLO(player_model_path).to(self.device)
        self.pitch_model = YOLO(pitch_model_path).to(self.device)
        self.ball_model = YOLO(ball_model_path).to(self.device)
        validate_models(self.player_model, self.pitch_model, self.ball_model)
        self.tracker_backend, self.reid_model = tracker_backend, reid_model
        self.enable_ocr = enable_ocr and ocr_available()
        self.imgsz = imgsz
        self._ocr_reader = None
        self.state = MatchState(CONFIG.length, CONFIG.width)
        self.samples = []
        self.ball_annotator = BallAnnotator(radius=6, buffer_size=10)
        self.seconds_per_processed_frame = 0
        self.model_paths = {"player": str(player_model_path), "pitch": str(pitch_model_path),
                            "ball": str(ball_model_path)}

    @property
    def ocr_reader(self):
        if self._ocr_reader is None:
            import easyocr
            self._ocr_reader = easyocr.Reader(["en"], gpu=self.device.startswith("cuda"), verbose=False)
        return self._ocr_reader

    def process(self, source_video_path, stride=1, max_frames=None) -> Iterator[np.ndarray]:
        if stride < 1:
            raise ValueError("stride must be >= 1")
        info = sv.VideoInfo.from_video_path(source_video_path)
        fps = info.fps or 25
        dt = stride / fps
        self.seconds_per_processed_frame = dt
        # Bound warm-up work even for full-length matches.
        crops = []
        sampling_stride = max(60, info.total_frames // 24)
        for index, frame in enumerate(sv.get_video_frames_generator(source_video_path, stride=sampling_stride)):
            result = self.player_model(frame, imgsz=self.imgsz, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(result)
            crops.extend(c for c in get_crops(frame, detections[detections.class_id == 2]) if c.size)
            if len(crops) >= 320 or index >= 23:
                break
        team_classifier = TeamClassifier(device=self.device) if len(crops) >= 2 else None
        if team_classifier:
            team_classifier.fit(crops[:320])
        tracker = FootballTracker(self.tracker_backend, fps=fps / stride,
                                  device=self.device, reid_model=self.reid_model)
        shots = ShotChangeDetector()
        ball_tracker = BallTracker(buffer_size=20)

        def ball_callback(image):
            result = self.ball_model(image, imgsz=640, verbose=False)[0]
            return sv.Detections.from_ultralytics(result)

        ball_slicer = sv.InferenceSlicer(callback=ball_callback, overlap_filter=sv.OverlapFilter.NONE,
                                         slice_wh=(640, 640), thread_workers=1)
        for index, frame in enumerate(sv.get_video_frames_generator(source_video_path, stride=stride)):
            if max_frames is not None and index >= max_frames:
                break
            timestamp = index * dt
            self.state.processed_frames += 1
            if shots.update(frame):
                tracker.reset()
                ball_tracker = BallTracker(buffer_size=20)
                self.ball_annotator = BallAnnotator(radius=6, buffer_size=10)
                self.state.cut(timestamp)
            result = self.player_model(frame, imgsz=self.imgsz, conf=0.1, verbose=False)[0]
            detections = tracker.update(result, frame)
            players = detections[detections.class_id == 2]
            keepers = detections[detections.class_id == 1]
            teams = np.full(len(players), -1, dtype=int)
            refresh = []
            for i, track in enumerate(players.tracker_id):
                state = self.state.players.get(self.state.resolve(track))
                if state is not None and state.team_id is not None:
                    teams[i] = state.team_id
                if state is None or state.team_votes.total() < 10 or index % 15 == 0:
                    refresh.append(i)
            if team_classifier and refresh:
                teams[refresh] = team_classifier.predict(get_crops(frame, players[refresh]))
            keeper_teams = resolve_goalkeepers_team_id(players, teams, keepers)
            people = sv.Detections.merge([players, keepers])
            for i, track in enumerate(players.tracker_id):
                # Cached assignments must not count as fresh evidence.
                self.state.observe(track, teams[i] if i in refresh else -1, timestamp)
            for track, team in zip(keepers.tracker_id, keeper_teams):
                self.state.observe(track, team, timestamp)

            keypoints = sv.KeyPoints.from_ultralytics(self.pitch_model(frame, verbose=False)[0])
            transformer = _safe_transformer(keypoints)
            # A failed calibration immediately invalidates pitch-space statistics.
            ball = ball_tracker.update(ball_slicer(frame).with_nms(threshold=0.1))
            possessor = None
            sample_players = []
            sample_ball = None
            if transformer is not None:
                self.state.calibrated_frames += 1
                xy = transformer.transform_points(people.get_anchors_coordinates(sv.Position.BOTTOM_CENTER))
                person_ids = people.tracker_id if people.tracker_id is not None else []
                for track, point in zip(person_ids, xy):
                    self.state.position(track, point, timestamp, dt)
                    if np.isfinite(point).all() and 0 <= point[0] <= CONFIG.length and 0 <= point[1] <= CONFIG.width:
                        sample_players.append({'track_id': int(track),
                                               'team_id': self.state.players[self.state.resolve(track)].team_id,
                                               'xy': (point / 100).round(3).tolist()})
                if len(ball) and len(people):
                    ball_xy = transformer.transform_points(ball.get_anchors_coordinates(sv.Position.CENTER))[0]
                    if np.isfinite(ball_xy).all() and 0 <= ball_xy[0] <= CONFIG.length and 0 <= ball_xy[1] <= CONFIG.width:
                        sample_ball = (ball_xy / 100).round(3).tolist()
                        distances = np.linalg.norm(xy - ball_xy, axis=1)
                        distances[~np.isfinite(distances)] = np.inf
                        distances[(xy[:, 0] < 0) | (xy[:, 0] > CONFIG.length)
                                  | (xy[:, 1] < 0) | (xy[:, 1] > CONFIG.width)] = np.inf
                        nearest = int(np.argmin(distances))
                        if distances[nearest] <= 150:
                            possessor = int(people.tracker_id[nearest])
            self.state.possession(possessor, timestamp, dt)
            if not hasattr(self, 'samples'):
                self.samples = []
            self.samples.append({'time_s': round(timestamp, 4), 'dt': round(dt, 4),
                                 'calibrated': transformer is not None, 'players': sample_players,
                                 'ball': sample_ball, 'possessor': int(possessor) if possessor is not None else None})
            if self.enable_ocr and len(people):
                self._sample_jerseys(frame, people, index)
            colors, labels = [], []
            for track, cls in zip(detections.tracker_id, detections.class_id):
                identity = self.state.resolve(track)
                state = self.state.players.get(identity)
                team = state.team_id if state else None
                jersey = state.jersey() if state else None
                if (team, jersey) in self.state.ambiguous_jerseys:
                    jersey = None
                colors.append(team if team is not None else (3 if cls == 3 else 2))
                labels.append("Arbitre" if cls == 3 else f"#{jersey}" if jersey else f"ID{identity}")
            annotated = ELLIPSE_ANNOTATOR.annotate(frame.copy(), detections, custom_color_lookup=np.asarray(colors, dtype=int))
            annotated = ELLIPSE_LABEL_ANNOTATOR.annotate(annotated, detections, labels,
                                                       custom_color_lookup=np.asarray(colors, dtype=int))
            yield self.ball_annotator.annotate(annotated, ball)

    def _sample_jerseys(self, frame, detections, index):
        import cv2
        for track, box in zip(detections.tracker_id, detections.xyxy):
            if index % 5 != int(track) % 5:
                continue
            x1, y1, x2, y2 = box.astype(int)
            h = y2 - y1
            crop = frame[max(0, y1 + int(h * 0.12)):max(0, y1 + int(h * 0.60)),
                         max(0, x1):max(0, x2)]
            if not crop.size or crop.shape[0] < 12 or crop.shape[1] < 8:
                continue
            scale = max(1, 160 / crop.shape[0])
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            reads = self.ocr_reader.readtext(crop, allowlist="0123456789", detail=1)
            reads = [(text, confidence) for _, text, confidence in reads if text.isdigit() and len(text) <= 2]
            if reads:
                text, confidence = max(reads, key=lambda item: item[1])
                self.state.jersey_read(int(track), text, confidence)

    def report(self):
        return self.state.report()

    def export(self, source_video='', source_fps=25, stride=1):
        """Versioned replay data in metres; no interpolated off-screen positions."""
        frames = []
        for sample in self.samples:
            people = []
            for person in sample['players']:
                identity = self.state.resolve(person['track_id'])
                people.append({**person, 'identity_id': identity})
            possessor = sample['possessor']
            frames.append({**sample, 'players': people,
                           'possessor': self.state.resolve(possessor) if possessor is not None else None})
        events = []
        for event in self.state.events:
            value = dict(event)
            if event['type'] in ('probable_pass', 'control_change'):
                value['from'] = self.state.resolve(event['from'])
                value['to'] = self.state.resolve(event['to'])
                if value['from'] == value['to']:
                    continue
            events.append(value)
        return {'schema_version': 3, 'source_video': os.path.basename(source_video),
                'source_fps': float(source_fps), 'stride': int(stride),
                'duration_s': round(len(frames) * stride / source_fps, 4),
                'pitch': {'length': CONFIG.length / 100, 'width': CONFIG.width / 100, 'unit': 'm'},
                'players': self.report(), 'frames': frames, 'events': events,
                'diagnostics': self.diagnostics()}

    def diagnostics(self):
        total = self.state.processed_frames
        return {"tracker": self.tracker_backend, "device": self.device,
                "frames": total, "calibrated_frames": self.state.calibrated_frames,
                "calibration_coverage_pct": round(100 * self.state.calibrated_frames / total, 1) if total else 0,
                "unresolved_identities": sum(p["jersey_number"] is None for p in self.report()),
                "identity_conflicts": len(self.state.ambiguous_jerseys),
                "unknown_possession_seconds": round(self.state.unknown_seconds, 2),
                "pitch_dimensions_m": [CONFIG.length / 100, CONFIG.width / 100],
                "models": self.model_paths, "ocr_enabled": self.enable_ocr,
                "events": self.state.events}

    def team_report(self):
        players = self.report()
        total = sum(p["possession_seconds"] for p in players if p["team_id"] in (0, 1))
        possession, heatmaps, networks = {}, {}, {}
        for team in (0, 1):
            rows = [p for p in players if p["team_id"] == team]
            possession[team] = round(100 * sum(p["possession_seconds"] for p in rows) / total, 1) if total else 0
            trajectories = [p["trajectory"] for p in rows if p["trajectory"].size]
            heatmaps[team] = np.vstack(trajectories) if trajectories else np.empty((0, 2))
            # A track with no usable pitch position cannot be placed on a map.
            mapped = [p for p in rows if p["trajectory"].size]
            nodes = {p["identity_id"]: i for i, p in enumerate(mapped)}
            edges = Counter()
            for (a, b), weight in self.state.pass_edges.items():
                if a in nodes and b in nodes and a != b:
                    edges[nodes[a], nodes[b]] += weight
            networks[team] = {
                "node_xy": np.asarray([p["trajectory"].mean(axis=0) for p in mapped]).reshape(-1, 2),
                "node_labels": [f"#{p['jersey_number']}" if p["jersey_number"] else f"ID{p['identity_id']}" for p in mapped],
                "edges": [(a, b, n) for (a, b), n in edges.items()]}
        return {"possession_pct": possession, "team_heatmaps": heatmaps, "pass_networks": networks,
                "unknown_possession_seconds": self.state.unknown_seconds}
