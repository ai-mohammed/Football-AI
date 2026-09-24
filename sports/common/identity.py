"""Conservative match identities and timestamped, observable statistics.

Track IDs, jersey hypotheses and confirmed identities are deliberately separate.
No rule caps the number of tracks at eleven or merges co-visible players.
"""
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from sports.common.control import BallFlight


@dataclass
class PlayerState:
    team_id: int | None = None
    team_votes: Counter = field(default_factory=Counter)
    jersey_votes: Counter = field(default_factory=Counter)
    track_ids: set = field(default_factory=set)
    intervals: dict = field(default_factory=dict)
    trajectory: list = field(default_factory=list)
    timestamps: list = field(default_factory=list)
    motion_samples: list = field(default_factory=list)
    last_sample: tuple | None = None
    distance_cm: float = 0.0
    measured_seconds: float = 0.0
    possession_seconds: float = 0.0
    touches: int = 0
    passes_made: int = 0
    passes_received: int = 0

    def jersey(self):
        if not self.jersey_votes:
            return None
        number, votes = self.jersey_votes.most_common(1)[0]
        return number if votes >= 3 and votes / self.jersey_votes.total() >= 0.8 else None


class MatchState:
    def __init__(self, length_cm=12000, width_cm=7000):
        self.length_cm, self.width_cm = length_cm, width_cm
        self.players = {}
        self.redirect = {}
        self.pass_edges = Counter()
        self.events = []
        self.ambiguous_jerseys = set()
        self.possessor = None
        self.last_control_time = None
        self.previous_control = None
        self.unknown_seconds = 0.0
        self.calibrated_frames = 0
        self.processed_frames = 0
        self.flight = BallFlight()

    def resolve(self, track):
        track = int(track)
        while track in self.redirect:
            track = self.redirect[track]
        return track

    def observe(self, track, team, timestamp):
        identity = self.resolve(track)
        state = self.players.setdefault(identity, PlayerState())
        state.track_ids.add(int(track))
        interval = state.intervals.setdefault(int(track), [timestamp, timestamp])
        interval[1] = timestamp
        if team in (0, 1):
            state.team_votes[int(team)] += 1
            candidate, count = state.team_votes.most_common(1)[0]
            if state.team_id is None or (count >= 5 and count > 1.5 * state.team_votes[state.team_id]):
                state.team_id = candidate
        return state

    def position(self, track, point, timestamp, sample_period):
        point = np.asarray(point, dtype=float)
        if not np.isfinite(point).all() or not (
            0 <= point[0] <= self.length_cm and 0 <= point[1] <= self.width_cm
        ):
            return
        state = self.players[self.resolve(track)]
        if state.last_sample is not None:
            prev_track, prev_time, prev_point = state.last_sample
            dt = timestamp - prev_time
            # Never bridge lost observations, cuts or identity changes.
            if prev_track == track and 0 < dt <= sample_period * 1.5:
                distance = float(np.linalg.norm(point - prev_point))
                if distance <= 1200 * dt:
                    state.distance_cm += distance
                    state.measured_seconds += dt
                    state.motion_samples.append({'time_s': timestamp, 'dt': dt,
                                                 'distance_m': distance / 100,
                                                 'speed_kmh': distance / dt * 0.036})
        state.last_sample = (track, timestamp, point.copy())
        state.trajectory.append(point.tolist())
        state.timestamps.append(timestamp)

    @staticmethod
    def overlaps(a, b):
        return any(max(x[0], y[0]) <= min(x[1], y[1])
                   for x in a.intervals.values() for y in b.intervals.values())

    def jersey_read(self, track, number, confidence=1.0):
        if confidence < 0.65 or not str(number).isdigit() or not 1 <= int(number) <= 99:
            return
        identity = self.resolve(track)
        state = self.players[identity]
        state.jersey_votes[str(int(number))] += 1
        jersey = state.jersey()
        if jersey is None or state.team_id is None:
            return
        if state.team_votes[state.team_id] < 5 or state.team_votes[state.team_id] / state.team_votes.total() < 0.7:
            return
        matches = [(other_id, other) for other_id, other in self.players.items()
                   if other_id != identity and other.team_id == state.team_id and other.jersey() == jersey
                   and other.team_votes[other.team_id] >= 5
                   and other.team_votes[other.team_id] / other.team_votes.total() >= 0.7]
        key = (state.team_id, jersey)
        if any(self.overlaps(state, other) for _, other in matches):
            if key not in self.ambiguous_jerseys:
                self.events.append({"type": "identity_conflict", "team": key[0], "jersey": jersey})
            self.ambiguous_jerseys.add(key)
            return
        if len(matches) == 1 and key not in self.ambiguous_jerseys:
            other_id, _ = matches[0]
            self.merge(max(identity, other_id), min(identity, other_id))

    def merge(self, loser, winner):
        loser, winner = self.resolve(loser), self.resolve(winner)
        if loser == winner:
            return False
        src, dst = self.players[loser], self.players[winner]
        if src.team_id != dst.team_id or self.overlaps(src, dst):
            return False
        for name in ("touches", "passes_made", "passes_received", "distance_cm", "measured_seconds", "possession_seconds"):
            setattr(dst, name, getattr(dst, name) + getattr(src, name))
        dst.track_ids.update(src.track_ids)
        dst.intervals.update(src.intervals)
        dst.team_votes.update(src.team_votes)
        dst.jersey_votes.update(src.jersey_votes)
        dst.motion_samples = sorted(dst.motion_samples + src.motion_samples, key=lambda s: s['time_s'])
        samples = sorted(zip(dst.timestamps + src.timestamps, dst.trajectory + src.trajectory))
        dst.timestamps = [t for t, _ in samples]
        dst.trajectory = [p for _, p in samples]
        dst.last_sample = None
        self.redirect[loser] = winner
        del self.players[loser]
        edges = Counter()
        for (a, b), weight in self.pass_edges.items():
            a, b = self.resolve(a), self.resolve(b)
            if a != b:
                edges[a, b] += weight
        self.pass_edges = edges
        # Recalculate pass counts so a removed self-edge cannot remain in totals.
        for p in self.players.values():
            p.passes_made = p.passes_received = 0
        for (a, b), weight in edges.items():
            self.players[a].passes_made += weight
            self.players[b].passes_received += weight
        self.possessor = self.resolve(self.possessor) if self.possessor is not None else None
        self.previous_control = self.resolve(self.previous_control) if self.previous_control is not None else None
        self.events.append({"type": "identity_merge", "from": loser, "to": winner,
                            "evidence": "team_and_jersey_consensus_nonoverlapping_tracks"})
        return True

    def cut(self, timestamp):
        for state in self.players.values():
            state.last_sample = None
        self.possessor = self.previous_control = self.last_control_time = None
        self.flight.reset()
        self.events.append({"type": "camera_cut", "time_s": timestamp})

    def possession(self, track, timestamp, sample_period, ball_xy=None):
        self.flight.observe(ball_xy, timestamp)
        identity = self.resolve(track) if track is not None else None
        if identity is None or self.players[identity].team_id not in (0, 1):
            self.unknown_seconds += sample_period
            self.previous_control = None
            if self.last_control_time is not None and timestamp - self.last_control_time > 1.0 and not self.flight.can_continue(timestamp):
                self.possessor = None
            return
        state = self.players[identity]
        state.possession_seconds += sample_period
        # Legacy callers with no ball coordinates keep the conservative one-second
        # rule. Video analysis requires actual ball-flight evidence for a transfer.
        recent = (self.last_control_time is not None and
                  ((ball_xy is None and timestamp-self.last_control_time <= 1.0)
                   or self.flight.can_continue(timestamp)))
        if identity != self.possessor or not recent:
            state.touches += 1  # Control episodes, not every physical touch.
            transfer = ball_xy is None or self.flight.supports_transfer(timestamp)
            if recent and transfer and self.possessor is not None and self.possessor != identity:
                previous = self.players[self.possessor]
                if previous.team_id == state.team_id:
                    self.pass_edges[self.possessor, identity] += 1
                    previous.passes_made += 1
                    state.passes_received += 1
                    self.events.append({"type": "probable_pass", "time_s": timestamp,
                                        "start_s": self.last_control_time,
                                        "evidence": "observed_ball_flight" if ball_xy is not None else "control_transition",
                                        "team_id": state.team_id,
                                        "from": self.possessor, "to": identity})
                else:
                    self.events.append({'type': 'control_change', 'time_s': timestamp,
                                        'start_s': self.last_control_time, 'team_id': state.team_id,
                                        'from': self.possessor, 'to': identity})
        self.possessor = self.previous_control = identity
        self.last_control_time = timestamp
        self.flight.anchor(ball_xy, timestamp)

    def report(self):
        rows = []
        for identity, state in self.players.items():
            jersey = state.jersey()
            if (state.team_id, jersey) in self.ambiguous_jerseys:
                jersey = None
            rows.append({"identity_id": identity, "tracker_ids": sorted(state.track_ids),
                         "jersey_number": jersey, "team_id": state.team_id,
                         "identity_status": "jersey_consensus" if jersey else "unresolved_track",
                         "touches": state.touches, "passes_made": state.passes_made,
                         "passes_received": state.passes_received,
                         "distance_m": round(state.distance_cm / 100, 2),
                         "measured_seconds": round(state.measured_seconds, 2),
                         "possession_seconds": round(state.possession_seconds, 2),
                         "avg_speed_kmh": round(state.distance_cm / 100 / state.measured_seconds * 3.6, 2)
                         if state.measured_seconds else None,
                         "trajectory": np.asarray(state.trajectory).reshape(-1, 2),
                         "timestamps": list(state.timestamps)})
            rows[-1]['motion_samples'] = list(state.motion_samples)
        return sorted(rows, key=lambda r: -r["touches"])
