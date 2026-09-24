"""Check geometry using held-out model landmarks, not ground-truth accuracy.

For every confident landmark, fit on the others and measure its pixel residual.
This catches an inconsistent pitch template that an inlier-only error can hide.
"""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import supervision as sv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sports.common.calibration import pitch_transformer
from sports.configs.soccer import SoccerPitchConfiguration


def evaluate(observations, config):
    errors, accepted = [], 0
    for sample in observations:
        points = np.asarray(sample['xy'], np.float32)
        confidence = np.asarray(sample['confidence'], np.float32)
        keypoints = sv.KeyPoints(xy=points, confidence=confidence)
        accepted += pitch_transformer(keypoints, config.vertices) is not None
        xy = points[0]
        target = np.asarray(config.vertices, np.float32)
        mask = (confidence[0] >= .5) & np.isfinite(xy).all(1) & (xy > 1).all(1)
        xy, target = xy[mask], target[mask]
        if len(xy) < 6:
            continue
        for i in range(len(xy)):
            train = np.arange(len(xy)) != i
            matrix, inliers = cv2.findHomography(target[train], xy[train], cv2.RANSAC, 6)
            if matrix is None or inliers is None or inliers.sum() < 4:
                continue
            prediction = cv2.perspectiveTransform(target[i:i+1, None], matrix).reshape(2)
            if np.isfinite(prediction).all():
                errors.append(float(np.linalg.norm(prediction-xy[i])))
    return {'sampled_frames': len(observations), 'accepted_frames': int(accepted),
            'held_out_predictions': len(errors),
            'median_held_out_error_px': float(np.median(errors)) if errors else None,
            'p90_held_out_error_px': float(np.percentile(errors, 90)) if errors else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observations', required=True,
                        help='JSON with observations: [{video, frame, xy: [[[x,y],...]], confidence: [[...]]}]')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    samples = json.loads(Path(args.observations).read_text(encoding='utf-8'))['observations']
    configs = {'legacy_120x70': SoccerPitchConfiguration(length=12000, width=7000,
                                                       penalty_box_length=2015, penalty_box_width=4100),
               'corrected_105x68': SoccerPitchConfiguration()}
    results = {name: {'all': evaluate(samples, config), 'clips': {
        video: evaluate([s for s in samples if s['video'] == video], config)
        for video in sorted(set(s['video'] for s in samples))}} for name, config in configs.items()}
    Path(args.output).write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps({k: v['all'] for k, v in results.items()}, indent=2))


if __name__ == '__main__':
    main()
