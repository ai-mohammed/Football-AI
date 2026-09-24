"""Display interpolation is separate from observed statistics and event inference."""
from bisect import bisect_right
from collections import defaultdict
import math


class ReplayInterpolator:
    def __init__(self, data, field='players', max_gap=.32):
        self.field, self.max_gap = field, max_gap
        self.width = (data.get('source_resolution') or [1920])[0]
        self.cuts = [e['time_s'] for e in data.get('events', []) if e['type'] == 'camera_cut']
        self.tracks = defaultdict(list)
        for frame in data.get('frames', []):
            for person in frame.get(field, []):
                self.tracks[person['track_id']].append((frame['time_s'], person))
        self.times = {track: [t for t, _ in points] for track, points in self.tracks.items()}

    def at(self, timestamp):
        result = []
        coordinate = 'xy' if self.field == 'players' else 'xyxy'
        for track, points in self.tracks.items():
            index = bisect_right(self.times[track], timestamp+1e-6)-1
            if index < 0:
                continue
            before, a = points[index]
            if abs(timestamp-before) < 1e-6:
                result.append({**a, 'interpolated': False})
                continue
            if index+1 >= len(points):
                continue  # No forward extrapolation, including off-screen exits.
            after, b = points[index+1]
            dt = after-before
            if (dt > self.max_gap+1e-6 or any(before < cut <= after for cut in self.cuts)
                    or a.get('identity_id') != b.get('identity_id')
                    or a.get('team_id') != b.get('team_id') or a.get('class_id') != b.get('class_id')):
                continue
            if coordinate == 'xy':
                if math.dist(a['xy'], b['xy']) > 12*dt+.1:
                    continue
            elif math.dist(a['xyxy'], b['xyxy']) > self.width*(.03+dt):
                continue
            alpha = (timestamp-before)/dt
            result.append({**a, coordinate: [round(x+(y-x)*alpha, 3) for x, y in zip(a[coordinate], b[coordinate])],
                           'interpolated': True})
        return result


def add_display_positions(data):
    """Only add a display field; never replace raw positions or possession."""
    replay = ReplayInterpolator(data)
    filled = 0
    for frame in data['frames']:
        frame['display_players'] = replay.at(frame['time_s']) if frame['calibrated'] else []
        filled += sum(p['interpolated'] for p in frame['display_players'])
    data.setdefault('diagnostics', {})['display_interpolated_positions'] = filled
    data['diagnostics']['display_max_gap_s'] = replay.max_gap
    return data


def render_replay(source, data, destination, preview=None, max_width=1280):
    """Keep every source frame; inference stride never determines playback FPS."""
    import cv2
    import imageio_ffmpeg
    import subprocess
    from pathlib import Path

    source = str(source)
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or fps <= 0:
        cap.release()
        raise ValueError('Vidéo source illisible pour le rendu.')
    scale = min(1., max_width/cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)*scale)//2*2,
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)*scale)//2*2)
    destination = Path(destination)
    raw = destination.with_suffix('.render.avi')
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*'MJPG'), fps, size)
    if not writer.isOpened():
        cap.release()
        raise RuntimeError('Impossible de créer le rendu vidéo.')
    replay = ReplayInterpolator(data, 'image_detections')
    players = {p['identity_id']: p for p in data['players']}
    times = [f['time_s'] for f in data['frames']]
    count = 0
    try:
        while count/fps < data['duration_s']-1e-6:
            ok, frame = cap.read()
            if not ok:
                break
            t = count/fps
            frame = cv2.resize(frame, size)
            idx = bisect_right(times, t+1e-6)-1
            people = replay.at(t)
            # Retain the last observation for its own sampling interval, including
            # the final source frame. Do not extend it into a missing sample.
            if idx >= 0 and t < times[idx]+data['frames'][idx]['dt']-1e-6:
                shown = {p['track_id'] for p in people}
                people += [p for p in data['frames'][idx]['image_detections'] if p['track_id'] not in shown]
            for person in people:
                identity = person['identity_id']
                player = players.get(identity, {})
                color = {0: (147, 20, 255), 1: (255, 191, 0)}.get(player.get('team_id'), (180, 180, 180))
                x1, y1, x2, y2 = [round(v*scale) for v in person['xyxy']]
                center = ((x1+x2)//2, y2)
                cv2.ellipse(frame, center, (max(3, (x2-x1)//2), 4), 0, 0, 360, color, 2, cv2.LINE_AA)
                label = 'Arbitre' if person.get('class_id') == 3 else f"#{player['jersey_number']}" if player.get('jersey_number') else f'ID{identity}'
                cv2.putText(frame, label, (max(0, x1), max(14, y1-4)), cv2.FONT_HERSHEY_SIMPLEX, .45, (10, 10, 10), 3, cv2.LINE_AA)
                cv2.putText(frame, label, (max(0, x1), max(14, y1-4)), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA)
            if idx >= 0:
                sample = data['frames'][idx]
                # Ball remains observed-only; never draw a made-up trajectory.
                ball = sample.get('ball_image_xy')
                if ball is not None and t-sample['time_s'] < min(sample['dt'], 1/fps)+1e-6:
                    cv2.circle(frame, tuple(round(v*scale) for v in ball), 6, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(frame)
            if count == 0 and preview is not None:
                cv2.imwrite(str(preview), frame)
            count += 1
    finally:
        cap.release()
        writer.release()
    if not count:
        raw.unlink(missing_ok=True)
        raise ValueError('Aucune image disponible pour le rendu.')
    try:
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-nostdin', '-hide_banner', '-loglevel', 'error',
                        '-i', str(raw), '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '25',
                        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(destination)],
                       check=True, capture_output=True)
    finally:
        raw.unlink(missing_ok=True)  # Only this render's own intermediate file.
    data['diagnostics']['replay_fps'] = fps
    data['diagnostics']['replay_frames'] = count
