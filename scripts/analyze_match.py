"""Reproducible local analysis with exportable trajectories and diagnostics."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples/soccer")]

import cv2
import numpy as np
import supervision as sv

from main import BALL_DETECTION_MODEL_PATH, PITCH_DETECTION_MODEL_PATH, PLAYER_DETECTION_MODEL_PATH
from player_analysis import PlayerMatchAnalyzer


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tracker", choices=["bytetrack", "botsort"], default="bytetrack")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--frames", type=int, help="Limit processed frames for a smoke test")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--no-ocr", action="store_true")
    parser.add_argument("--player-model", default=PLAYER_DETECTION_MODEL_PATH)
    parser.add_argument("--pitch-model", default=PITCH_DETECTION_MODEL_PATH)
    parser.add_argument("--ball-model", default=BALL_DETECTION_MODEL_PATH)
    args = parser.parse_args()
    if args.stride < 1 or (args.frames is not None and args.frames < 1):
        parser.error("stride and frames must be positive")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "analysis.json").exists() or (output / "annotated.mp4").exists():
        parser.error("Output already contains an analysis. Choose a new output directory.")
    analyzer = PlayerMatchAnalyzer(args.player_model, args.pitch_model, args.ball_model,
                                    device=args.device, tracker_backend=args.tracker,
                                    enable_ocr=not args.no_ocr, imgsz=args.imgsz)
    info = sv.VideoInfo.from_video_path(args.video)
    export_info = sv.VideoInfo(width=info.width, height=info.height, fps=info.fps / args.stride)
    started = time.perf_counter()
    with sv.VideoSink(str(output / "annotated.mp4"), export_info) as sink:
        for index, frame in enumerate(analyzer.process(args.video, stride=args.stride, max_frames=args.frames)):
            sink.write_frame(frame)
            if index == 0:
                cv2.imwrite(str(output / "preview.jpg"), frame)
    payload = {"schema_version": 2, "source_video": Path(args.video).name,
               "stride": args.stride, "source_fps": info.fps,
               "elapsed_seconds_including_warmup": round(time.perf_counter() - started, 2),
               "diagnostics": analyzer.diagnostics(), "players": analyzer.report(),
               "teams": analyzer.team_report()}
    (output / "analysis.json").write_text(json.dumps(payload, indent=2, default=json_value), encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k not in ("players", "teams")}, default=json_value, indent=2))


if __name__ == "__main__":
    main()
