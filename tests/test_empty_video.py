import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import supervision as sv
import torch
from ultralytics.engine.results import Results

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples/soccer'))
from player_analysis import PlayerMatchAnalyzer
from sports.common.identity import MatchState
from sports.common.ball import BallAnnotator


class EmptyVideoTests(unittest.TestCase):
    def test_no_people_or_ball_does_not_crash_even_with_valid_pitch(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        empty = Results(frame, path='', names={}, boxes=torch.empty((0, 6)))
        analyzer = PlayerMatchAnalyzer.__new__(PlayerMatchAnalyzer)
        analyzer.device = 'cpu'
        analyzer.imgsz = 640
        analyzer.tracker_backend = 'bytetrack'
        analyzer.reid_model = 'unused.pt'
        analyzer.enable_ocr = True  # No crops => no OCR initialization.
        analyzer.state = MatchState()
        analyzer.player_model = Mock(return_value=[empty])
        analyzer.ball_model = Mock(return_value=[empty])
        analyzer.pitch_model = Mock(return_value=[empty])
        analyzer.ball_annotator = BallAnnotator(radius=6, buffer_size=10)
        transformer = Mock()
        transformer.transform_points.return_value = np.empty((0, 2))
        with patch('player_analysis.sv.VideoInfo.from_video_path', return_value=sv.VideoInfo(100, 100, 25, 2)), \
             patch('player_analysis.sv.get_video_frames_generator', side_effect=lambda *a, **k: iter([frame, frame])), \
             patch('player_analysis.sv.KeyPoints.from_ultralytics', return_value=sv.KeyPoints.empty()), \
             patch('player_analysis._safe_transformer', return_value=transformer):
            frames = list(analyzer.process('unused.mp4'))
        self.assertEqual(len(frames), 2)
        self.assertEqual(analyzer.report(), [])
        self.assertAlmostEqual(analyzer.state.unknown_seconds, 0.08)


if __name__ == '__main__':
    unittest.main()
