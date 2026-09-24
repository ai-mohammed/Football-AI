"""Build portable, genuinely analysed segment demos on the local GPU."""
import argparse
import gc
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'examples/soccer')]

import cv2
import numpy as np
import supervision as sv
import torch

from main import PLAYER_DETECTION_MODEL_PATH, PITCH_DETECTION_MODEL_PATH, BALL_DETECTION_MODEL_PATH
from player_analysis import PlayerMatchAnalyzer, CONFIG
from sports.annotators.soccer import draw_pitch_heatmap, draw_pass_network
from sports.common.replay import render_replay


def encode(value):
    return value.tolist() if isinstance(value, np.ndarray) else value.item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', default='examples/soccer/notebooks')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--seconds', type=float, default=12)
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--clips', nargs='+', default=['08fd33_0', '0bfacc_0', '121364_0', '2e57b9_0', '573e61_0'])
    args = parser.parse_args()
    if args.seconds <= 0 or args.stride < 1:
        parser.error('seconds and stride must be positive')
    for clip in args.clips:
        source = Path(args.source_dir) / f'{clip}.mp4'
        output = Path(args.output_dir) / clip
        if (output / 'analysis.json').exists():
            parser.error(f'Choose a new output directory: {output} already contains an analysis')
        output.mkdir(parents=True, exist_ok=True)
        info = sv.VideoInfo.from_video_path(str(source))
        frames = min(int(args.seconds * info.fps / args.stride), int(np.ceil(info.total_frames / args.stride)))
        analyzer = PlayerMatchAnalyzer(PLAYER_DETECTION_MODEL_PATH, PITCH_DETECTION_MODEL_PATH,
                                       BALL_DETECTION_MODEL_PATH, device=args.device, tracker_backend='botsort')
        for index, _ in enumerate(analyzer.process(str(source), stride=args.stride, max_frames=frames)):
            if index % 50 == 0:
                print(f'{clip}: analysed {index+1}/{frames}', flush=True)
        payload = analyzer.export(str(source), info.fps, args.stride)
        render_replay(source, payload, output / 'annotated.mp4', output / 'preview.jpg')
        payload['diagnostics']['models'] = {k: Path(v).name for k, v in payload['diagnostics']['models'].items()}
        (output / 'analysis.json').write_text(json.dumps(payload, default=encode, separators=(',', ':')), encoding='utf-8')
        team = analyzer.team_report()
        for tid in (0, 1):
            cv2.imwrite(str(output / f'heatmap_team_{tid}.jpg'), draw_pitch_heatmap(CONFIG, xy=team['team_heatmaps'][tid]))
            net = team['pass_networks'][tid]
            cv2.imwrite(str(output / f'pass_network_team_{tid}.jpg'), draw_pass_network(
                CONFIG, node_xy=net['node_xy'], node_labels=net['node_labels'], edges=net['edges'],
                node_color=sv.Color.from_hex('#FF1493' if tid == 0 else '#00BFFF')))
        rows = [{'label': f"#{p['jersey_number']}" if p['jersey_number'] else f"ID{p['identity_id']}",
                 **{k: p[k] for k in ('team_id', 'touches', 'passes_made', 'passes_received', 'distance_m', 'avg_speed_kmh')}}
                for p in analyzer.report()]
        (output / 'stats.json').write_text(json.dumps({'schema_version': 3, 'source_video': source.name,
            'frames': len(payload['frames']), 'stride': args.stride, 'possession_pct': team['possession_pct'],
            'players': rows}, default=encode, indent=2), encoding='utf-8')
        print(f'{clip}: {len(payload["frames"])} frames, calibration {payload["diagnostics"]["calibration_coverage_pct"]}%', flush=True)
        del analyzer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
