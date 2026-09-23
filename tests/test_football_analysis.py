"""Regression tests of identity and measurement guarantees; no weights needed."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import supervision as sv

from sports.common.calibration import ShotChangeDetector, pitch_transformer
from sports.common.identity import MatchState
from sports.common.runtime import resolve_device


class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.state = MatchState()

    def observe(self, track, start, team=0):
        for t in range(5):
            self.state.observe(track, team, start + t * 0.1)

    def jersey(self, track, number="7"):
        for _ in range(3):
            self.state.jersey_read(track, number, 0.95)

    def test_covisible_matching_numbers_never_merge_or_share_confirmed_label(self):
        self.observe(1, 0)
        self.observe(2, 0)
        self.jersey(1)
        self.jersey(2)
        self.assertEqual(len(self.state.report()), 2)
        self.assertTrue(all(row["jersey_number"] is None for row in self.state.report()))
        self.assertFalse(self.state.merge(2, 1))

    def test_reidentification_keeps_original_identity_and_track_provenance(self):
        self.observe(1, 0)
        self.jersey(1)
        self.observe(20, 10)
        self.jersey(20)
        self.assertEqual(self.state.resolve(20), 1)
        self.assertEqual(self.state.report()[0]["tracker_ids"], [1, 20])

    def test_opposing_teams_same_number_remain_distinct(self):
        self.observe(1, 0, 0)
        self.observe(2, 10, 1)
        self.jersey(1)
        self.jersey(2)
        self.assertEqual(len(self.state.report()), 2)

    def test_conflicting_ocr_votes_are_not_confirmed(self):
        self.observe(1, 0)
        for number in ["7", "8", "7", "8", "7"]:
            self.state.jersey_read(1, number)
        self.assertIsNone(self.state.report()[0]["jersey_number"])

    def test_low_confidence_and_zero_are_not_jerseys(self):
        self.observe(1, 0)
        for _ in range(5):
            self.state.jersey_read(1, "7", 0.4)
            self.state.jersey_read(1, "0", 0.9)
        self.assertIsNone(self.state.report()[0]["jersey_number"])

    def test_distance_resumes_after_merging_without_counting_jump(self):
        self.observe(1, 0)
        self.state.position(1, [100, 100], 0.4, 0.1)
        self.observe(2, 10)
        self.state.merge(2, 1)
        self.state.position(2, [9000, 100], 10.4, 0.1)
        self.state.position(2, [9010, 100], 10.5, 0.1)
        self.assertAlmostEqual(self.state.players[1].distance_cm, 10)
        self.assertAlmostEqual(self.state.players[1].measured_seconds, 0.1)

    def test_missing_observations_and_impossible_positions_add_no_distance(self):
        self.observe(1, 0)
        self.state.position(1, [100, 100], 0, 0.04)
        self.state.position(1, [200, 100], 1, 0.04)
        self.state.position(1, [4000, 100], 1.04, 0.04)
        self.state.position(1, [float("nan"), 100], 1.08, 0.04)
        self.state.position(1, [-100, 100], 1.12, 0.04)
        self.assertEqual(self.state.players[1].distance_cm, 0)
        self.assertEqual(len(self.state.players[1].trajectory), 3)

    def test_lost_ball_does_not_create_pass_across_long_gap(self):
        self.observe(1, 0)
        self.observe(2, 0)
        self.state.possession(1, 0, 0.1)
        self.state.possession(None, 2, 0.1)
        self.state.possession(2, 3, 0.1)
        self.assertEqual(sum(self.state.pass_edges.values()), 0)
        self.assertAlmostEqual(self.state.unknown_seconds, 0.1)

    def test_cut_breaks_pass_and_movement_continuity(self):
        self.observe(1, 0)
        self.observe(2, 0)
        self.state.possession(1, 0, 0.1)
        self.state.position(1, [100, 100], 0, 0.1)
        self.state.cut(0.05)
        self.state.possession(2, 0.1, 0.1)
        self.state.position(1, [110, 100], 0.1, 0.1)
        self.assertEqual(sum(self.state.pass_edges.values()), 0)
        self.assertEqual(self.state.players[1].distance_cm, 0)

    def test_possession_measures_duration_not_number_of_controls(self):
        self.observe(1, 0)
        self.observe(2, 0, 1)
        for i in range(9):
            self.state.possession(1, i * 0.1, 0.1)
        self.state.possession(2, 0.9, 0.1)
        self.assertAlmostEqual(self.state.players[1].possession_seconds, 0.9)
        self.assertEqual(self.state.players[1].touches, 1)

    def test_reports_do_not_hide_short_tracks_or_force_eleven_players(self):
        for track in range(12):
            self.state.observe(track, 0, 0)
        self.assertEqual(len(self.state.report()), 12)

    def test_cached_team_and_single_bad_vote_do_not_flip_colour(self):
        self.observe(1, 0)
        self.state.observe(1, 1, 2)
        self.state.observe(1, -1, 3)
        self.assertEqual(self.state.players[1].team_id, 0)
        self.assertEqual(self.state.players[1].team_votes.total(), 6)


class GeometryTests(unittest.TestCase):
    def test_geometry_rejects_insufficient_collinear_or_low_confidence_points(self):
        points = np.asarray([[10, 10], [20, 20], [30, 30], [40, 40]], dtype=np.float32)
        self.assertIsNone(pitch_transformer(sv.KeyPoints(xy=points[None]), points))
        points = np.asarray([[10, 10], [100, 10], [100, 100], [10, 100]], dtype=np.float32)
        keypoints = sv.KeyPoints(xy=points[None], confidence=np.zeros((1, 4)))
        self.assertIsNone(pitch_transformer(keypoints, points))

    def test_ransac_rejects_outlier_and_recovers_pitch_coordinates(self):
        target = np.asarray([[0, 0], [1000, 0], [0, 1000], [1000, 1000],
                             [300, 700], [900, 600], [500, 100]], dtype=np.float32)
        image = target / 2 + 100
        image[-1] = [9000, 8000]
        transformer = pitch_transformer(sv.KeyPoints(xy=image[None]), target)
        self.assertIsNotNone(transformer)
        np.testing.assert_allclose(transformer.transform_points(image[:6]), target[:6], atol=0.1)
        self.assertEqual(transformer.inlier_count, 6)

    def test_human_keypoints_are_not_pitch_keypoints(self):
        self.assertIsNone(pitch_transformer(sv.KeyPoints(xy=np.ones((1, 17, 2))), np.ones((32, 2))))

    def test_cut_detector_does_not_reset_on_identical_frame(self):
        detector = ShotChangeDetector()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        self.assertFalse(detector.update(frame))
        self.assertFalse(detector.update(frame))
        self.assertTrue(detector.update(frame + 255))

    @patch("torch.cuda.is_available", return_value=False)
    def test_cuda_request_falls_back_on_cpu_host(self, _):
        with self.assertWarns(RuntimeWarning):
            self.assertEqual(resolve_device("cuda"), "cpu")


class TrackerTests(unittest.TestCase):
    def test_interleaved_matches_have_independent_track_counters(self):
        from ultralytics.engine.results import Boxes
        from sports.common.tracking import FootballTracker
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        one = SimpleNamespace(boxes=Boxes(np.asarray([[10, 10, 30, 60, .9, 2]], dtype=np.float32), frame.shape[:2]))
        two = SimpleNamespace(boxes=Boxes(np.asarray([[10, 10, 30, 60, .9, 2],
                                                     [120, 10, 140, 60, .9, 2]], dtype=np.float32), frame.shape[:2]))
        match_a = FootballTracker()
        match_a.update(one, frame)
        match_b = FootballTracker()  # Must not reset A's allocation counter.
        match_a.update(two, frame)
        match_b.update(one, frame)
        tracked = match_a.update(two, frame)
        self.assertEqual(len(tracked), 2)
        self.assertEqual(len(set(tracked.tracker_id)), 2)

    def test_tracker_buffer_scales_with_processed_fps_and_ids_survive_reset(self):
        from ultralytics.engine.results import Boxes
        from sports.common.tracking import FootballTracker
        tracker = FootballTracker("bytetrack", fps=5, buffer_seconds=3)
        self.assertEqual(tracker.tracker.args.track_buffer, 15)
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        result = SimpleNamespace(boxes=Boxes(np.asarray([[10, 10, 30, 60, 0.9, 2]], dtype=np.float32), frame.shape[:2]))
        first = tracker.update(result, frame).tracker_id[0]
        tracker.reset()
        second = tracker.update(result, frame).tracker_id[0]
        self.assertNotEqual(first, second)


class TeamTests(unittest.TestCase):
    def test_jersey_colours_remain_separate_in_shadow(self):
        from sports.common.team import TeamClassifier
        green = np.full((80, 30, 3), [70, 220, 100], dtype=np.uint8)
        white = np.full((80, 30, 3), [225, 225, 225], dtype=np.uint8)
        crops = [green, (green * .65).astype(np.uint8), white, (white * .65).astype(np.uint8)]
        classifier = TeamClassifier()
        classifier.fit(crops)
        groups = classifier.predict(crops)
        self.assertEqual(groups[0], groups[1])
        self.assertEqual(groups[2], groups[3])
        self.assertNotEqual(groups[0], groups[2])
        self.assertNotIn(2, groups)

    def test_missing_training_crops_are_unknown(self):
        from sports.common.team import TeamClassifier
        classifier = TeamClassifier()
        classifier.fit([])
        self.assertEqual(classifier.predict([np.zeros((10, 10, 3), dtype=np.uint8)])[0], 2)


if __name__ == "__main__":
    unittest.main()
