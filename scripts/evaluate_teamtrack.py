"""Evaluate exported image detections against a TeamTrack three-header CSV.

This reports both IoU>=0.5 and centre-in-annotation, with one-to-one assignment.
It is not HOTA/IDF1. TeamTrack IDs are annotation identities, not jersey numbers.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def evaluate(data, annotation_path, source_frame_offset=0):
    gt = pd.read_csv(annotation_path, header=[0, 1, 2], index_col=0)
    keys = [k for k in gt.columns.droplevel(2).unique() if k[0] in ('0', '1')]
    totals = Counter()
    track_matches, identity_tracks = defaultdict(Counter), defaultdict(set)
    previous = {}
    switches = 0
    for index, sample in enumerate(data['frames']):
        frame_number = source_frame_offset + index * data['stride'] + 1
        if frame_number not in gt.index:
            raise ValueError(f'Annotation frame {frame_number} is missing')
        row = gt.loc[frame_number]
        truth, identities = [], []
        for key in keys:
            x, y, w, h = (row[key][k] for k in ('bb_left', 'bb_top', 'bb_width', 'bb_height'))
            if np.isfinite([x, y, w, h]).all() and min(w, h) > 0:
                truth.append([x, y, x+w, y+h])
                identities.append(':'.join(key))
        selected = [d for d in sample['image_detections'] if d['class_id'] in (1, 2)]
        pred = np.asarray([d['xyxy'] for d in selected]).reshape(-1, 4)
        truth = np.asarray(truth).reshape(-1, 4)
        totals.update(frames=1, annotated=len(truth), predicted=len(pred))
        current = {}
        if len(pred) and len(truth):
            lower = np.maximum(pred[:, None, :2], truth[None, :, :2])
            upper = np.minimum(pred[:, None, 2:], truth[None, :, 2:])
            intersect = np.maximum(0, upper-lower).prod(2)
            union = ((pred[:, 2:]-pred[:, :2]).prod(1)[:, None]
                     + (truth[:, 2:]-truth[:, :2]).prod(1)[None, :] - intersect)
            iou = intersect/np.maximum(union, 1e-8)
            a, b = linear_sum_assignment(np.where(iou >= .5, 1-iou, 1e6))
            totals['iou_matches'] += int((iou[a, b] >= .5).sum())
            centers = (pred[:, :2]+pred[:, 2:])/2
            valid = ((centers[:, None] >= truth[None, :, :2]) & (centers[:, None] <= truth[None, :, 2:])).all(2)
            distance = np.linalg.norm(centers[:, None]-(truth[:, :2]+truth[:, 2:])[None]/2, axis=2)
            a, b = linear_sum_assignment(np.where(valid, distance, 1e6))
            for p, t in zip(a, b):
                if not valid[p, t]:
                    continue
                totals['center_matches'] += 1
                track = selected[p]['track_id']
                identity = identities[t]
                track_matches[track][identity] += 1
                identity_tracks[identity].add(track)
                current[identity] = track
                if identity in previous and previous[identity] != track:
                    switches += 1
        previous = current
    result = dict(totals)
    for measure in ('iou', 'center'):
        matches = totals[f'{measure}_matches']
        result[f'{measure}_recall_pct'] = round(100*matches/max(1, totals['annotated']), 2)
        result[f'{measure}_precision_pct'] = round(100*matches/max(1, totals['predicted']), 2)
    result['center_matches_per_frame'] = round(totals['center_matches']/max(1, totals['frames']), 2)
    result['mean_annotated_per_frame'] = round(totals['annotated']/max(1, totals['frames']), 2)
    result['consecutive_matched_id_changes'] = switches
    result['annotated_players_with_multiple_tracks'] = sum(len(v) > 1 for v in identity_tracks.values())
    result['matched_track_purity_pct'] = round(100*sum(max(c.values()) for c in track_matches.values())
                                               / max(1, totals['center_matches']), 2)
    result['calibration_pct'] = data['diagnostics']['calibration_coverage_pct']
    result['identities'] = len(data['players'])
    result['source_frame_offset_zero_based'] = source_frame_offset
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', required=True)
    parser.add_argument('--annotations', required=True)
    parser.add_argument('--source-frame-offset', type=int, default=0,
                        help='Zero-based source index of the first analysed frame; coordinates must use original resolution')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    data = json.loads(Path(args.analysis).read_text(encoding='utf-8'))
    result = evaluate(data, args.annotations, args.source_frame_offset)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
