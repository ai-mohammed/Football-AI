"""Windowed segment statistics. Replay positions are metres, never interpolated.

Frames describe [time_s, time_s + dt); motion describes [time_s - dt, time_s].
This distinction prevents counting movement outside a selected video window.
"""
from collections import Counter, defaultdict
import math


TEAM_NAMES = {0: 'Équipe A', 1: 'Équipe B', None: 'Non attribuée'}
EVENT_NAMES = {'probable_pass': 'Passe probable', 'control_change': 'Changement de contrôle',
               'camera_cut': 'Coupure caméra'}


def overlap(a, b, start, end):
    return max(0., min(b, end) - max(a, start))


def label(player):
    number = player.get('jersey_number')
    return f"N° {number} · ID{player['identity_id']}" if number else f"ID{player['identity_id']}"


def window_frames(data, start, end):
    """Return fresh samples with their observed duration clipped to the window."""
    result = []
    for frame in data.get('frames', []):
        dt = overlap(frame['time_s'], frame['time_s'] + frame['dt'], start, end)
        if dt > 1e-8:
            result.append({**frame, 'weight_s': dt})
    return result


def window_events(data, start, end):
    return [e for e in data.get('events', []) if e.get('type') in EVENT_NAMES
            and start <= e.get('time_s', -1) < end
            and e.get('start_s', e.get('time_s', -1)) >= start]


def summarize(data, start=0., end=None):
    end = min(float(end if end is not None else data.get('duration_s', 0)), data.get('duration_s', 0))
    start = max(0., float(start))
    frames = window_frames(data, start, end)
    events = window_events(data, start, end)
    visible, control, episodes = Counter(), Counter(), Counter()
    possession = Counter({0: 0., 1: 0., None: 0.})
    calibrated = ball = 0.
    positions = defaultdict(list)
    previous_owner = None
    for frame in frames:
        dt = frame['weight_s']
        if not frame.get('calibrated'):
            possession[None] += dt
            previous_owner = None
            continue
        calibrated += dt
        people = {p['identity_id']: p for p in frame.get('players', [])}
        for identity, person in people.items():
            visible[identity] += dt
            positions[identity].append((frame['time_s'], *person['xy'], person.get('team_id'), dt))
        if frame.get('ball') is not None:
            ball += dt
        owner = frame.get('possessor') if frame.get('ball') is not None else None
        team = people.get(owner, {}).get('team_id')
        if team not in (0, 1):
            owner, team = None, None
        possession[team] += dt
        if owner is not None:
            control[owner] += dt
            if owner != previous_owner:
                episodes[owner] += 1
        previous_owner = owner
    # Unsampled tails/gaps are unknown, not possession of the last seen player.
    duration = max(0., end - start)
    possession[None] += max(0., duration - sum(possession.values()))
    passes = [e for e in events if e['type'] == 'probable_pass']
    made, received = Counter(e['from'] for e in passes), Counter(e['to'] for e in passes)
    rows = []
    for player in data.get('players', []):
        identity = player['identity_id']
        if identity not in visible:
            continue
        distance = measured = 0.
        for sample in player.get('motion_samples', []):
            dt = sample['dt']
            weight = overlap(sample['time_s'] - dt, sample['time_s'], start, end)
            if dt > 0 and weight > 0:
                distance += sample['distance_m'] * weight / dt
                measured += weight
        rows.append({'identity_id': identity, 'label': label(player), 'team_id': player.get('team_id'),
                     'visible_s': round(visible[identity], 2), 'distance_m': round(distance, 2),
                     'measured_s': round(measured, 2),
                     'speed_kmh': round(distance / measured * 3.6, 2) if measured else None,
                     'control_s': round(control[identity], 2), 'control_episodes': episodes[identity],
                     'passes_made': made[identity], 'passes_received': received[identity]})
    known = possession[0] + possession[1]
    return {'start_s': start, 'end_s': end, 'duration_s': duration, 'frames': frames,
            'events': events, 'players': rows, 'positions': dict(positions),
            'possession_seconds': dict(possession),
            'possession_pct': {t: 100 * possession[t] / known if known else None for t in (0, 1)},
            'unknown_pct': 100 * possession[None] / duration if duration else 100.,
            'calibration_pct': 100 * calibrated / duration if duration else 0.,
            'ball_pct': 100 * ball / duration if duration else 0., 'passes': len(passes)}


def occupancy(summary, team, length=120., width=70., nx=6, ny=4, identity=None):
    """Player-seconds per zone, normalized over visible, calibrated players only."""
    grid = [[0. for _ in range(nx)] for _ in range(ny)]
    for pid, points in summary['positions'].items():
        if identity is not None and identity != pid:
            continue
        for _, x, y, tid, dt in points:
            if identity is None and tid != team:
                continue
            if math.isfinite(x) and math.isfinite(y) and 0 <= x <= length and 0 <= y <= width:
                grid[min(ny - 1, int(y / width * ny))][min(nx - 1, int(x / length * nx))] += dt
    total = sum(map(sum, grid))
    return [[100 * value / total if total else 0. for value in row] for row in grid]


def structure(summary, team):
    result = []
    for frame in summary['frames']:
        people = [p['xy'] for p in frame.get('players', []) if p.get('team_id') == team]
        if not frame.get('calibrated') or len(people) < 3:
            result.append({'time_s': frame['time_s'], 'width_m': None, 'depth_m': None, 'count': len(people)})
            continue
        xs, ys = zip(*people)
        result.append({'time_s': frame['time_s'], 'width_m': max(ys) - min(ys),
                       'depth_m': max(xs) - min(xs), 'count': len(people)})
    return result


def pass_network(summary, team):
    edges = Counter((e['from'], e['to']) for e in summary['events']
                    if e['type'] == 'probable_pass' and e.get('team_id') == team)
    nodes = {}
    for identity in {pid for edge in edges for pid in edge}:
        points = [p for p in summary['positions'].get(identity, []) if p[3] == team]
        weight = sum(p[4] for p in points)
        if weight:
            nodes[identity] = [sum(p[axis] * p[4] for p in points) / weight for axis in (1, 2)]
    return nodes, {edge: count for edge, count in edges.items() if all(pid in nodes for pid in edge)}
