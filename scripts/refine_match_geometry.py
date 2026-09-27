"""Recompute geometry and all derived statistics from saved observations.

Reuse measured boxes, ball detections and jersey evidence. No inferred position
or event from the previous calibration is carried into the refined result.
"""
import argparse
import copy
import gzip
import json
from pathlib import Path
import sys

import numpy as np
import supervision as sv

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT/'examples/soccer')]
from main import PITCH_DETECTION_MODEL_PATH
from sports.common.calibration import ShotChangeDetector, TemporalPitchCalibrator
from sports.common.control import ControlFilter
from sports.common.identity import MatchState
from sports.common.replay import add_display_positions, render_replay
from sports.configs.soccer import SoccerPitchConfiguration


def refine(data, frames, keypoints, confidence):
    result = copy.deepcopy(data)
    config = SoccerPitchConfiguration(length=round(data['pitch']['length']*100),
                                       width=round(data['pitch']['width']*100))
    calibration, shots, controls = TemporalPitchCalibrator(config.vertices),ShotChangeDetector(),ControlFilter()
    state = MatchState(config.length,config.width)
    metadata = {p['identity_id']:p for p in data['players']}
    for p in data['players']:
        for track in p['tracker_ids']:
            if track != p['identity_id']:
                state.redirect[track] = p['identity_id']
    rejected = smoothed = processed = 0
    previous_layout_blocked = False
    for i, frame in enumerate(frames):
        sample = result['frames'][i]
        t,dt = sample['time_s'],sample['dt']
        shot_changed = shots.update(frame)
        if shot_changed:
            state.cut(t);calibration.reset();controls.reset()
        kp = sv.KeyPoints(xy=keypoints[i:i+1],confidence=confidence[i:i+1])
        transform = calibration.update(frame,kp,t)
        if calibration.layout_blocked != previous_layout_blocked:
            if not shot_changed:
                state.cut(t)
            controls.reset()
        previous_layout_blocked = calibration.layout_blocked
        if calibration.layout_blocked:
            sample['image_detections'] = []
            sample['ball_image_xy'] = None
        rejected += calibration.rejected_jump
        smoothed += calibration.smoothed
        people = []
        for d in sample['image_detections']:
            if d['class_id'] not in (1,2):
                continue
            identity = d['identity_id']
            team = metadata[identity]['team_id']
            state.observe(d['track_id'],team,t)
            if transform is None:
                continue
            x1,y1,x2,y2 = d['xyxy']
            point = transform.transform_points(np.float32([[(x1+x2)/2,y2]]))[0]
            state.position(d['track_id'],point,t,dt)
            if np.isfinite(point).all() and (point>=0).all() and (point<= [config.length,config.width]).all():
                people.append({'track_id':d['track_id'],'identity_id':identity,'team_id':team,
                               'xy':(point/100).round(3).tolist()})
        ball,owner = None,None
        if transform is not None and sample.get('ball_image_xy') is not None:
            b = transform.transform_points(np.float32([sample['ball_image_xy']]))[0]/100
            if np.isfinite(b).all() and (b>=0).all() and (b<= [config.length/100,config.width/100]).all():
                ball = b.round(3).tolist()
                if people:
                    distances = np.linalg.norm(np.asarray([p['xy'] for p in people])-b,axis=1)
                    if distances.min()<=1.5:
                        owner=people[int(distances.argmin())]['identity_id']
        candidate = owner
        owner=controls.update(owner,t)
        state.possession(owner,t,dt,np.asarray(ball)*100 if ball is not None else None)
        sample.update(calibrated=transform is not None,calibration_method=calibration.method,
                      calibration_jump_rejected=calibration.rejected_jump,
                      calibration_smoothed=calibration.smoothed,players=people,
                      unsupported_layout=calibration.layout_blocked,
                      ball=ball,possessor=owner,control_candidate=candidate)
        state.calibrated_frames += transform is not None
        processed += 1
        if i%500==0:
            print(json.dumps({'observations':processed,'time_s':t,'smoothed':smoothed}),flush=True)
    if processed != len(data['frames']):
        raise ValueError('Source ended before the complete analysis.')
    stats={p['identity_id']:p for p in state.report()}
    result['players']=[p for p in result['players'] if p['identity_id'] in stats]
    for p in result['players']:
        measured=stats.get(p['identity_id'],{})
        for field in ('touches','passes_made','passes_received','distance_m','measured_seconds',
                      'possession_seconds','avg_speed_kmh','trajectory','timestamps','motion_samples'):
            p[field]=measured[field]
    result['events']=[e for e in state.events if 'time_s' in e]
    result['diagnostics'].update(calibrated_frames=state.calibrated_frames,
        calibration_coverage_pct=round(100*state.calibrated_frames/processed,1),
        unknown_possession_seconds=round(state.unknown_seconds,2),events=state.events,
        optical_flow_calibrated_frames=sum(f['calibration_method']=='pitch_optical_flow' for f in result['frames']),
        calibration_jumps_rejected=rejected,calibration_smoothed_frames=smoothed,
        unsupported_layout_frames=sum(f.get('unsupported_layout',False) for f in result['frames']),
        unresolved_identities=sum(not p.get('jersey_number') for p in result['players']),
        geometry_refinement='measured_camera_motion_with_landmark_correction',
        statistics_recomputed=True)
    return add_display_positions(result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video',type=Path,required=True)
    p.add_argument('--analysis',type=Path,required=True)
    p.add_argument('--keypoints',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--skip-render',action='store_true')
    a=p.parse_args()
    data=json.loads(gzip.decompress(a.analysis.read_bytes()))
    if data.get('identity_scope')!='continuous' or not 0<data['duration_s']<=300 or data.get('source_start_s')!=0:
        p.error('Expected the bounded continuous five-minute analysis.')
    cache=np.load(a.keypoints)
    signature=json.loads(str(cache['signature']))
    if (signature['source']!=str(a.video.resolve()) or signature['bytes']!=a.video.stat().st_size
            or signature['modified']!=a.video.stat().st_mtime_ns or signature['stride']!=data['stride']
            or signature['frames']!=len(data['frames'])
            or signature['model']!=str(Path(PITCH_DETECTION_MODEL_PATH).resolve())
            or signature['model_modified']!=Path(PITCH_DETECTION_MODEL_PATH).stat().st_mtime_ns):
        p.error('Cached landmarks belong to another source, model or sampling.')
    if a.output.exists():
        p.error('Choose a new output directory.')
    a.output.mkdir(parents=True)
    result=refine(data,sv.get_video_frames_generator(str(a.video),stride=data['stride'],
                  end=round(data['duration_s']*data['source_fps'])),cache['xy'],cache['confidence'])
    (a.output/'analysis.json.gz').write_bytes(gzip.compress(json.dumps(result,ensure_ascii=False,
        separators=(',',':'),default=lambda v:v.tolist()).encode(),mtime=0))
    if not a.skip_render:
        render_replay(a.video,result,a.output/'annotated.mp4',a.output/'preview.jpg')
    print('Refined analysis complete.',flush=True)


if __name__=='__main__':
    main()
