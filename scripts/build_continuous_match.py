"""Analyse a bounded source interval once, without resetting at chapter boundaries."""
import argparse
import gzip
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'examples/soccer')]

from process_full_match import atomic_json, checksum, fit_match_teams


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--render-only', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((args.directory/'manifest.json').read_text(encoding='utf-8'))
    window = manifest['analysis_window']
    if window['start_s'] != 0 or not 0 < window['end_s'] <= 300:
        parser.error('This build is bounded to the approved first five minutes.')
    if checksum(args.video) != manifest['source']['sha256']:
        parser.error('The source does not match the analysed video.')
    args.output.mkdir(parents=True, exist_ok=True)
    analysis_path = args.output/'analysis.json.gz'
    video_path = args.output/'annotated.mp4'
    if not args.render_only and analysis_path.exists():
        parser.error('Analysis already exists; use --render-only or a new output directory.')
    if video_path.exists():
        parser.error('Video already exists; choose a new output directory.')
    from sports.common.replay import render_replay
    if args.render_only:
        data = json.loads(gzip.decompress(analysis_path.read_bytes()))
    else:
        import supervision as sv
        from main import PLAYER_DETECTION_MODEL_PATH, PITCH_DETECTION_MODEL_PATH, BALL_DETECTION_MODEL_PATH
        from player_analysis import PlayerMatchAnalyzer
        info = sv.VideoInfo.from_video_path(str(args.video))
        stride = manifest['processing']['stride']
        end_frame = round(window['end_s']*info.fps)
        analyzer = PlayerMatchAnalyzer(PLAYER_DETECTION_MODEL_PATH, PITCH_DETECTION_MODEL_PATH,
            BALL_DETECTION_MODEL_PATH, tracker_backend='botsort', device=args.device)
        classifier, _ = fit_match_teams(analyzer, args.video, info, manifest['team_prototypes'])
        started = time.perf_counter()
        # One generator, one tracker, one calibration history, one control state.
        # Chapters are navigation markers only; they never restart the analysis.
        for index, _ in enumerate(analyzer.process(str(args.video), stride=stride,
                end_frame=end_frame, team_classifier=classifier)):
            if index % 100 == 0:
                progress = {'observations': index+1, 'source_seconds': round(index*stride/info.fps, 2),
                            'elapsed_s': round(time.perf_counter()-started, 1)}
                atomic_json(args.output/'progress.json', progress)
                print(json.dumps(progress), flush=True)
        data = analyzer.export(args.video.name, info.fps, stride)
        if len(data['frames']) != (end_frame+stride-1)//stride:
            raise RuntimeError('Source ended before the complete interval was analysed.')
        data['duration_s'] = end_frame/info.fps
        data['frames'][-1]['dt'] = round(data['duration_s']-data['frames'][-1]['time_s'], 6)
        data.update(aggregate=True, identity_scope='continuous', source_start_s=0.,
                    clock_basis='source_video', segments=[{'id': s['id'], 'start_s': s['start_s'],
                        'end_s': s['end_s']} for s in manifest['segments']])
        data['events'] = [e for e in data['events'] if 'time_s' in e]
        data['diagnostics'].update(identity_scope='continuous', chapter_resets=0,
            control_episodes_scope='continuous', replay_detection='not_available',
            match_clock_verified=False, models={k: Path(v).name for k,v in analyzer.model_paths.items()})
        raw = json.dumps(data, ensure_ascii=False, separators=(',', ':'),
                         default=lambda v: v.tolist()).encode('utf-8')
        analysis_path.write_bytes(gzip.compress(raw, mtime=0))
        print(f'Analysis saved: {len(data["players"])} tracks, {len(data["events"])} events', flush=True)
    render_replay(args.video, data, video_path, args.output/'preview.jpg')
    if data['diagnostics']['replay_frames'] != round(data['duration_s']*data['source_fps']):
        raise RuntimeError('Incomplete video render; do not publish.')
    print(f'Continuous video ready: {data["diagnostics"]["replay_frames"]} frames', flush=True)


if __name__ == '__main__':
    main()
