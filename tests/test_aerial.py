import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
import supervision as sv
import torch
from ultralytics.engine.results import Results

from sports.common.aerial import (TiledPlayerDetector, boundary_transformer,
                                  full_pitch_boundary, on_pitch_mask, tile_starts)
from sports.configs.soccer import SoccerPitchConfiguration


class AerialTests(unittest.TestCase):
    def test_metric_template_uses_regulation_markings_and_custom_dimensions(self):
        config = SoccerPitchConfiguration(length=11000, width=7200)
        self.assertEqual(config.penalty_box_length, 1650)
        self.assertEqual(config.penalty_box_width, 2*1650+732)
        self.assertEqual(config.vertices[9][0], 1650)
        self.assertEqual(config.vertices[17][0], 11000-1650)
        self.assertEqual(config.vertices[13], (5500, 0))

    def test_aerial_profile_never_claims_jerseys_or_ball_events(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'examples/soccer'))
        from player_analysis import PlayerMatchAnalyzer
        with patch('player_analysis.YOLO'), patch('player_analysis.validate_models'):
            analyzer = PlayerMatchAnalyzer('player', 'pitch', 'ball', device='cpu', profile='aerial',
                                           enable_ocr=True, pitch_length_m=110., pitch_width_m=72.)
        self.assertFalse(analyzer.enable_ocr)
        self.assertFalse(analyzer.diagnostics()['ball_events_available'])
        self.assertEqual(analyzer.export()['pitch'], {'length': 110., 'width': 72., 'unit': 'm'})
        with self.assertRaises(ValueError):
            PlayerMatchAnalyzer('unused', 'unused', 'unused', pitch_length_m=50.)

    def field(self, center_line=True):
        frame = np.full((720, 1280, 3), (65, 125, 70), dtype=np.uint8)
        cv2.rectangle(frame, (150, 90), (1130, 630), (230, 230, 230), 3)
        if center_line:
            cv2.line(frame, (640, 90), (640, 630), (230, 230, 230), 3)
        cv2.circle(frame, (640, 360), 75, (230, 230, 230), 3)
        return frame

    def test_full_pitch_maps_center_and_rejects_off_field_logos(self):
        frame = self.field()
        cv2.putText(frame, 'SPONSOR 11', (200, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 4)
        boundary = full_pitch_boundary(frame)
        self.assertIsNotNone(boundary)
        transform = boundary_transformer(boundary, 12000, 7000)
        np.testing.assert_allclose(transform.transform_points(np.float32([[640, 360]])), [[6000, 3500]], atol=30)
        boxes = np.float32([[400, 20, 500, 65], [610, 320, 645, 360]])
        np.testing.assert_array_equal(on_pitch_mask(boxes, boundary), [False, True])

    def test_incomplete_pitch_or_rectangle_without_halfway_line_is_rejected(self):
        self.assertIsNone(full_pitch_boundary(self.field(False)))
        self.assertIsNone(full_pitch_boundary(self.field()[:500]))
        self.assertIsNone(full_pitch_boundary(np.zeros((720, 1280, 3), dtype=np.uint8)))

    def test_translation_does_not_invent_player_motion(self):
        first = full_pitch_boundary(self.field())
        shifted = cv2.warpAffine(self.field(), np.float32([[1, 0, 12], [0, 1, 8]]), (1280, 720))
        second = full_pitch_boundary(shifted)
        one = boundary_transformer(first, 12000, 7000).transform_points(np.float32([[400, 350]]))
        two = boundary_transformer(second, 12000, 7000).transform_points(np.float32([[412, 358]]))
        np.testing.assert_allclose(one, two, atol=5)

    def test_tiles_cover_odd_sized_frame_without_duplicate_origins(self):
        starts = tile_starts(3841, 1280)
        covered = np.zeros(3841, dtype=bool)
        for start in starts:
            covered[start:start+1280] = True
        self.assertTrue(covered.all())
        self.assertEqual(len(starts), len(set(starts)))
        self.assertEqual(tile_starts(640, 1280), [0])

    def test_tile_overlap_is_deduplicated_before_tracking_and_keeps_source_coordinates(self):
        frame = np.zeros((100, 160, 3), dtype=np.uint8)
        model = Mock()
        model.names = {1: 'goalkeeper', 2: 'player'}
        # Same player in the overlap, classified differently in the second tile.
        model.side_effect = [[Results(frame, path='', names=model.names,
                                      boxes=torch.tensor([[70, 20, 85, 70, .9, 2]]))],
                             [Results(frame, path='', names=model.names,
                                      boxes=torch.tensor([[10, 20, 25, 70, .8, 1]]))]]
        result = TiledPlayerDetector(model, tile_size=100, imgsz=128)(frame)
        detection = sv.Detections.from_ultralytics(result)
        self.assertEqual(len(detection), 1)
        np.testing.assert_allclose(detection.xyxy, [[70, 20, 85, 70]])

    def test_empty_tiles_are_valid_tracker_input(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        model = Mock(names={2: 'player'})
        model.return_value = [Results(frame, path='', names=model.names, boxes=torch.empty((0, 6)))]
        result = TiledPlayerDetector(model)(frame)
        self.assertEqual(len(sv.Detections.from_ultralytics(result)), 0)


if __name__ == '__main__':
    unittest.main()
