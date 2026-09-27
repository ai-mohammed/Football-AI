"""Audit cuts and calibration on the same source frames as a saved analysis.

These are continuity diagnostics, not ground-truth accuracy scores.
"""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'examples/soccer')]
from main import PITCH_DETECTION_MODEL_PATH
from sports.common.calibration import ShotChangeDetector, TemporalPitchCalibrator
from sports.configs.soccer import SoccerPitchConfiguration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = json.loads(gzip.decompress(args.analysis.read_bytes()))
    calibration = TemporalPitchCalibrator(SoccerPitchConfiguration().vertices)
    shots = ShotChangeDetector()
    cuts, records, counts = [], [], Counter()
    cache = args.output/'keypoints.npz'
    cached = np.load(cache) if cache.exists() else None
    signature = json.dumps({'source':str(args.video.resolve()),'bytes':args.video.stat().st_size,
                            'modified':args.video.stat().st_mtime_ns,'stride':data['stride'],
                            'frames':len(data['frames']),'model':str(Path(PITCH_DETECTION_MODEL_PATH).resolve()),
                            'model_modified':Path(PITCH_DETECTION_MODEL_PATH).stat().st_mtime_ns},sort_keys=True)
    if cached is not None and ('signature' not in cached or str(cached['signature']) != signature):
        parser.error('Cache from another source/model; use a new output directory.')
    model = YOLO(PITCH_DETECTION_MODEL_PATH).to(args.device) if cached is None else None
    coords, confidence = [], []
    previous = None
    for i, frame in enumerate(sv.get_video_frames_generator(str(args.video), stride=data['stride'],
                          end=round(data['duration_s']*data['source_fps']))):
        sample = data['frames'][i]
        t = sample['time_s']
        cut = shots.update(frame)
        if cut:
            cuts.append(t)
            calibration.reset()
            previous = None
        if cached is None:
            kp = sv.KeyPoints.from_ultralytics(model(frame, verbose=False)[0])
            xy = kp.xy[0] if len(kp.xy) else np.zeros((32, 2), np.float32)
            scores = kp.keypoint_confidence if hasattr(kp,'keypoint_confidence') else kp.confidence
            conf = scores[0] if scores is not None and len(kp.xy) else np.zeros(32, np.float32)
            coords.append(xy); confidence.append(conf)
        else:
            xy, conf = cached['xy'][i], cached['confidence'][i]
        kp = sv.KeyPoints(xy=np.asarray(xy)[None], confidence=np.asarray(conf)[None])
        tf = calibration.update(frame, kp, t)
        counts[str(calibration.method)] += 1
        counts['rejected_jumps'] += calibration.rejected_jump
        positions = {}
        if tf is not None:
            for d in sample['image_detections']:
                if d['class_id'] == 3:
                    continue
                x1,y1,x2,y2 = d['xyxy']
                point = tf.transform_points(np.float32([[(x1+x2)/2,y2]]))[0]/100
                if 0 <= point[0] <= 105 and 0 <= point[1] <= 68:
                    positions[d['track_id']] = point
        jumps = [] if previous is None else [float(np.linalg.norm(p-previous[k]))
                  for k,p in positions.items() if k in previous]
        records.append({'time_s': t, 'cut': cut, 'calibrated': tf is not None,
                        'rejected_jump': calibration.rejected_jump,
                        'median_step_m': float(np.median(jumps)) if jumps else None,
                        'large_steps': sum(x > 2 for x in jumps)})
        previous = positions
        if i % 250 == 0:
            print(json.dumps({'time_s':t,'cuts':len(cuts), 'rejected_jumps':counts['rejected_jumps']}),flush=True)
    if cached is None:
        np.savez_compressed(cache,xy=np.asarray(coords),confidence=np.asarray(confidence),signature=signature)
    report = {'counts':dict(counts),'cuts':cuts,'samples':records}
    (args.output/'audit.json').write_text(json.dumps(report),encoding='utf-8')
    print(json.dumps({'counts':dict(counts),'cuts':cuts}),flush=True)


if __name__ == '__main__':
    main()
