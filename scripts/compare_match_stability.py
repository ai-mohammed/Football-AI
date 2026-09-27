"""Compare observable coverage and fragmentation; never label them accuracy."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import numpy as np


def metrics(data):
    times = Counter()
    for f in data['frames']:
        for d in f.get('image_detections', []):
            if d.get('class_id') in (1,2):
                times[d['track_id']] += f['dt']
    return {
        'duration_s':data['duration_s'],
        'observations':len(data['frames']),
        'raw_person_tracks':len(times),
        'tracks_visible_under_half_second':sum(t < .5 for t in times.values()),
        'reported_identities':len(data['players']),
        'jersey_consensus_identities':sum(bool(p.get('jersey_number')) for p in data['players']),
        'camera_change_events':sum(e['type']=='camera_cut' for e in data['events']),
        'probable_passes':sum(e['type']=='probable_pass' for e in data['events']),
        'calibrated_seconds':round(sum(f['dt'] for f in data['frames'] if f['calibrated']),2),
        'observed_person_seconds':round(sum(len(f['players'])*f['dt'] for f in data['frames']),2),
        'unknown_control_seconds':data['diagnostics']['unknown_possession_seconds'],
        'calibration_jumps_rejected':data['diagnostics'].get('calibration_jumps_rejected',0),
    }


def position_continuity(data):
    previous, steps = {}, []
    for frame in data['frames']:
        positions = {p['track_id']:np.asarray(p['xy']) for p in frame['players']}
        common = [float(np.linalg.norm(p-previous[k])) for k,p in positions.items() if k in previous]
        previous = positions
        if len(common) >= 3:
            steps.append(float(np.median(common)))
    return {'comparable_frame_pairs':len(steps),
            'median_player_step_m':round(float(np.median(steps)),4) if steps else None,
            'p95_median_player_step_m':round(float(np.percentile(steps,95)),4) if steps else None,
            'frames_median_step_above_1m':sum(t>1 for t in steps)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--before',type=Path,required=True)
    p.add_argument('--after',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    old,new=[json.loads(gzip.decompress(path.read_bytes())) for path in (a.before,a.after)]
    for key in ('source_video','duration_s','source_fps','stride','source_resolution'):
        if old[key]!=new[key]:
            p.error(f'Comparison must use the same source and sampling: {key}')
    report={'before':metrics(old),'after':metrics(new),
            'position_continuity':{'before':position_continuity(old),'after':position_continuity(new)},
            'scope':'Same five-minute source interval. Continuity and coverage diagnostics; no annotated identity, pass or calibration ground truth.',
            'source_video':old['source_video']}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(report,indent=2,ensure_ascii=False))


if __name__=='__main__':
    main()
