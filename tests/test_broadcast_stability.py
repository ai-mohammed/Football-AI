"""Regression cases for shot changes, pitch jumps and off-pitch identities."""
import unittest
from types import SimpleNamespace

import cv2
import numpy as np
import supervision as sv
from ultralytics.engine.results import Boxes

from sports.common.calibration import ShotChangeDetector, TemporalPitchCalibrator
from sports.common.tracking import FootballTracker


class ShotTests(unittest.TestCase):
    @staticmethod
    def scene(seed):
        rng = np.random.default_rng(seed)
        scene = np.full((360, 640, 3), (35, 110, 45), np.uint8)
        for x, y in rng.integers([20, 50], [600, 330], size=(55, 2)):
            cv2.rectangle(scene, (x, y), (x+6, y+15), (230, 230, 230), -1)
        cv2.line(scene, (50, 290), (570, 80), (220, 220, 220), 3)
        return scene

    def test_pan_zoom_and_brightness_keep_shot(self):
        scene = self.scene(2)
        detector = ShotChangeDetector()
        self.assertFalse(detector.update(scene))
        for shift in range(1, 7):
            moved = cv2.warpAffine(scene, np.float32([[1.01, 0, shift*4], [0, 1.01, shift]]),
                                   (640, 360), borderMode=cv2.BORDER_REFLECT)
            self.assertFalse(detector.update(moved))
        self.assertFalse(detector.update(np.clip(moved.astype(float)*1.08,0,255).astype(np.uint8)))

    def test_new_view_of_same_green_pitch_detected(self):
        detector = ShotChangeDetector()
        # Same histogram, unrelated detailed texture, photometric change < .30.
        rng = np.random.default_rng(3)
        a = rng.integers(20, 140, (360, 640, 3), dtype=np.uint8)
        b = rng.integers(20, 140, (360, 640, 3), dtype=np.uint8)
        self.assertFalse(detector.update(a))
        self.assertTrue(detector.update(b))
        self.assertFalse(detector.update(b))


class CalibrationTests(unittest.TestCase):
    def test_composite_letterboxed_view_never_uses_global_pitch_geometry(self):
        points=np.float32([[60,50],[160,50],[260,50],[60,120],[160,120],[260,120]])
        frame=np.full((180,320,3),(40,120,40),np.uint8)
        frame[:40]=0;frame[-32:]=0
        calibration=TemporalPitchCalibrator(points*10)
        self.assertIsNone(calibration.update(frame,sv.KeyPoints(xy=points[None]),0))
        self.assertTrue(calibration.layout_blocked)
        self.assertIsNone(calibration.transformer)
        frame[:]=[40,120,40]
        self.assertIsNotNone(calibration.update(frame,sv.KeyPoints(xy=points[None]),.08))
        self.assertFalse(calibration.layout_blocked)

    def test_landmark_noise_is_smoothed_without_lagging_camera_motion(self):
        points = np.float32([[140,140],[400,140],[650,140],[140,420],[400,420],[650,420],[400,260]])
        frame = np.full((560,800,3),80,np.uint8)
        for x,y in points.astype(int):
            cv2.rectangle(frame,(x-12,y-12),(x+12,y+12),(220,220,220),-1)
            cv2.rectangle(frame,(x-6,y-6),(x+6,y+6),(30,30,30),-1)
        calibration = TemporalPitchCalibrator(points*10)
        calibration.update(frame,sv.KeyPoints(xy=points[None]),0)
        moved = cv2.warpAffine(frame,np.float32([[1,0,5],[0,1,3]]),(800,560))
        # Real camera displacement (5,3) plus noisy model landmarks (8,0).
        noisy = sv.KeyPoints(xy=(points+[13,3])[None].astype(np.float32))
        transform = calibration.update(moved,noisy,.08)
        self.assertTrue(calibration.smoothed)
        errors=np.linalg.norm(transform.transform_points(points+[5,3])-points*10,axis=1)
        self.assertLess(float(np.median(errors)),25.)  # Raw model error is 80 cm.

    def test_isolated_model_jump_does_not_teleport_pitch(self):
        points = np.float32([[140,140],[400,140],[650,140],[140,420],[400,420],[650,420],[400,260]])
        frame = np.full((560,800,3),80,np.uint8)
        for x,y in points.astype(int):
            cv2.rectangle(frame,(x-12,y-12),(x+12,y+12),(220,220,220),-1)
            cv2.rectangle(frame,(x-6,y-6),(x+6,y+6),(30,30,30),-1)
        calibration = TemporalPitchCalibrator(points*10)
        calibration.update(frame,sv.KeyPoints(xy=points[None]),0)
        shifted = sv.KeyPoints(xy=(points+[50,0])[None].astype(np.float32))
        transform = calibration.update(frame,shifted,.08)
        self.assertTrue(calibration.rejected_jump)
        self.assertEqual(calibration.method,'pitch_optical_flow')
        np.testing.assert_allclose(transform.transform_points(points),points*10,atol=2)
        # A rejected model observation does not renew the flow lifetime.
        self.assertIsNone(calibration.update(frame,sv.KeyPoints.empty(),.4))
        calibration.reset()
        self.assertIsNone(calibration.update(frame,sv.KeyPoints.empty(),.48))


class TrackTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((240,320,3),np.uint8)

    def boxes(self, values):
        return SimpleNamespace(boxes=Boxes(np.asarray(values,np.float32).reshape(-1,6),self.frame.shape[:2]))

    def test_closeup_does_not_create_identity_but_known_track_can_continue(self):
        tracker = FootballTracker()
        person = self.boxes([[100,40,120,100,.95,2]])
        self.assertEqual(len(tracker.update(person,self.frame,allow_new_tracks=False)),0)
        tracker.update(person,self.frame)
        known = tracker.update(person,self.frame)
        two = self.boxes([[100,40,120,100,.95,2],[200,40,220,100,.95,2]])
        continued = tracker.update(two,self.frame,allow_new_tracks=False)
        self.assertEqual(continued.tracker_id.tolist(),known.tracker_id.tolist())
        tracker.reset()
        self.assertEqual(len(tracker.update(two,self.frame,allow_new_tracks=False)),0)
        self.assertLess(tracker.tracker.args.new_track_thresh,1)

    def test_bench_outside_pitch_excluded_touchline_margin_kept(self):
        tracker = FootballTracker()
        result = self.boxes([[100,40,120,100,.95,2],[200,40,220,100,.95,2],[20,40,40,100,.95,1]])
        class Projection:
            def transform_points(self, points):
                return np.float32([[5000,3000],[5000,8000],[-100,3400]])
        people = tracker.update(result,self.frame,transformer=Projection(),pitch_size=(10500,6800))
        self.assertEqual(len(people),2)
        np.testing.assert_allclose(people.xyxy,result.boxes.xyxy[[0,2]])


if __name__ == '__main__':
    unittest.main()
