import copy
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

from sports.common.ball import MotionBallTracker
from sports.common.calibration import TemporalPitchCalibrator
from sports.common.control import ControlFilter
from sports.common.identity import MatchState
from sports.common.replay import ReplayInterpolator, add_display_positions, render_replay


class ControlTests(unittest.TestCase):
    def make_state(self):
        state = MatchState()
        state.observe(22, 0, 0)
        state.observe(3, 0, 0)
        return state

    def test_long_visible_pass_survives_more_than_one_second_in_flight(self):
        state, controls = self.make_state(), ControlFilter()
        # Regression for demo 1: release .48 s, receiver at 2.24/2.32 s.
        for i in range(30):
            t = round(i*.08, 2)
            candidate = 22 if t <= .48 else 3 if t >= 2.24 else None
            point = [3000+max(0, t-.48)*1000, 4900]
            state.possession(controls.update(candidate, t), t, .08, point)
        self.assertEqual(state.pass_edges[(22, 3)], 1)
        self.assertAlmostEqual(state.events[0]['start_s'], .48)
        self.assertAlmostEqual(state.events[0]['time_s'], 2.32)
        self.assertAlmostEqual(state.players[22].possession_seconds, .48)
        self.assertGreater(state.unknown_seconds, 1.7)

    def test_lost_ball_or_cut_never_bridges_a_long_pass(self):
        for cut in (False, True):
            state = self.make_state()
            state.possession(22, 0, .08, [3000, 4000])
            if cut:
                state.cut(.08)
            for i in range(1, 25):
                state.possession(None, i*.08, .08, [3000+i*80, 4000] if cut else None)
            state.possession(3, 2, .08, [5000, 4000])
            self.assertFalse(state.pass_edges)

    def test_implausible_jump_and_nearby_controls_do_not_create_pass(self):
        for points in ([[1000, 1000], [9000, 1000], [9200, 1000]],
                       [[1000, 1000], [1020, 1000], [1040, 1000]]):
            state = self.make_state()
            state.possession(22, 0, .08, points[0])
            state.possession(None, .08, .08, points[1])
            state.possession(3, .16, .08, points[2])
            self.assertFalse(state.pass_edges)

    def test_single_frame_neighbour_does_not_take_control(self):
        control = ControlFilter()
        claims = [22, 22, 3, 22, None, 3, None, None, None, 3]
        actual = [control.update(who, i*.08) for i, who in enumerate(claims)]
        self.assertEqual(actual, [None, 22, None, 22, None, None, None, None, None, None])
        control.reset()
        self.assertIsNone(control.update(22, 1))

    def test_same_player_reobservation_preserves_actual_launch_time(self):
        control = ControlFilter()
        control.update(22, 0)
        self.assertEqual(control.update(22, .08), 22)
        self.assertIsNone(control.update(None, .8))
        self.assertEqual(control.update(22, .96), 22)


class BallTests(unittest.TestCase):
    def detections(self, xs):
        return sv.Detections(xyxy=np.float32([[x-2, 98, x+2, 102] for x in xs]).reshape(-1, 4),
                             confidence=np.full(len(xs), .9), class_id=np.zeros(len(xs), dtype=int))

    def test_ball_follows_flight_instead_of_stationary_old_candidate(self):
        tracker = MotionBallTracker()
        for i in range(8):
            expected = 100+i*30
            xs = [expected] if i < 2 else [100, expected]
            found = tracker.update(self.detections(xs), i*.08, 1920)
            self.assertAlmostEqual(found.get_anchors_coordinates(sv.Position.CENTER)[0, 0], expected)
        self.assertEqual(len(tracker.update(sv.Detections.empty(), .64, 1920)), 0)
        self.assertEqual(len(tracker.update(self.detections([1700]), .72, 1920)), 0)
        self.assertEqual(len(tracker.update(self.detections([1700]), 1.2, 1920)), 1)


class CalibrationContinuityTests(unittest.TestCase):
    def test_flow_follows_camera_translation_and_expires_or_resets(self):
        frame = np.full((560, 800, 3), 80, dtype=np.uint8)
        points = np.float32([[140, 140], [400, 140], [650, 140], [140, 420], [400, 420], [650, 420], [400, 260]])
        for x, y in points.astype(int):
            cv2.rectangle(frame, (x-12, y-12), (x+12, y+12), (200, 200, 200), -1)
            cv2.rectangle(frame, (x-6, y-6), (x+6, y+6), (30, 30, 30), -1)
        vertices = points*10
        calibrator = TemporalPitchCalibrator(vertices)
        self.assertIsNotNone(calibrator.update(frame, sv.KeyPoints(xy=points[None]), 0))
        shifted = cv2.warpAffine(frame, np.float32([[1, 0, 5], [0, 1, 3]]), (800, 560))
        missing = sv.KeyPoints.empty()
        flowed = calibrator.update(shifted, missing, .08)
        self.assertIsNotNone(flowed)
        self.assertEqual(calibrator.method, 'pitch_optical_flow')
        np.testing.assert_allclose(flowed.transform_points(points+[5, 3]), vertices, atol=1.5)
        self.assertIsNone(calibrator.update(shifted, missing, .4))
        calibrator.update(frame, sv.KeyPoints(xy=points[None]), .5)
        calibrator.reset()
        self.assertIsNone(calibrator.update(shifted, missing, .58))

    def test_feature_loss_does_not_reuse_previous_homography(self):
        points = np.float32([[20, 20], [100, 20], [180, 20], [20, 100], [100, 100], [180, 100]])
        frame = np.random.default_rng(2).integers(0, 256, (140, 220, 3), dtype=np.uint8)
        calibration = TemporalPitchCalibrator(points*10)
        calibration.update(frame, sv.KeyPoints(xy=points[None]), 0)
        self.assertIsNone(calibration.update(np.zeros_like(frame), sv.KeyPoints.empty(), .08))


class ReplayTests(unittest.TestCase):
    def test_render_keeps_source_frames_and_fps_despite_inference_stride(self):
        with tempfile.TemporaryDirectory() as root:
            source, output = Path(root)/'source.avi', Path(root)/'replay.mp4'
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 25, (160, 100))
            self.assertTrue(writer.isOpened())
            for i in range(10):
                writer.write(np.full((100, 160, 3), i*20, dtype=np.uint8))
            writer.release()
            data = {'source_resolution': [160, 100], 'players': [], 'events': [], 'duration_s': .4,
                    'diagnostics': {}, 'frames': [{'time_s': i*.08, 'dt': .08, 'image_detections': []}
                                                 for i in range(5)]}
            render_replay(source, data, output)
            cap = cv2.VideoCapture(str(output))
            self.assertAlmostEqual(cap.get(cv2.CAP_PROP_FPS), 25.)
            levels = []
            while True:
                ok, frame = cap.read()
                if not ok: break
                levels.append(frame.mean())
            cap.release()
            self.assertEqual(len(levels), 10)
            self.assertTrue(all(b-a > 15 for a, b in zip(levels, levels[1:])))

    def data(self):
        person = {'identity_id': 1, 'track_id': 1, 'team_id': 0, 'xy': [10., 20.]}
        return {'events': [], 'frames': [
            {'time_s': 0., 'dt': .08, 'calibrated': True, 'players': [person]},
            {'time_s': .08, 'dt': .08, 'calibrated': True, 'players': []},
            {'time_s': .16, 'dt': .08, 'calibrated': True, 'players': [{**person, 'xy': [11., 20.]}]}]}

    def test_short_display_gap_filled_without_changing_measured_positions(self):
        data = self.data()
        original = copy.deepcopy(data)
        add_display_positions(data)
        self.assertEqual(data['frames'][1]['display_players'][0]['xy'], [10.5, 20.])
        self.assertTrue(data['frames'][1]['display_players'][0]['interpolated'])
        for a, b in zip(data['frames'], original['frames']):
            self.assertEqual(a['players'], b['players'])
        self.assertEqual(ReplayInterpolator(data).at(.24), [])

    def test_no_interpolation_across_cut_id_change_long_gap_or_impossible_speed(self):
        for kind in ('cut', 'identity', 'track', 'team', 'gap', 'speed'):
            data = self.data()
            last = data['frames'][-1]
            if kind == 'cut': data['events'] = [{'type': 'camera_cut', 'time_s': .08}]
            if kind == 'identity': last['players'][0]['identity_id'] = 2
            if kind == 'track': last['players'][0]['track_id'] = 2
            if kind == 'team': last['players'][0]['team_id'] = 1
            if kind == 'gap': last['time_s'] = 1.
            if kind == 'speed': last['players'][0]['xy'] = [50, 50]
            self.assertEqual(ReplayInterpolator(data).at(.08), [], kind)


if __name__ == '__main__':
    unittest.main()
