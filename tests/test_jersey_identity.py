import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from ultralytics.engine.results import Boxes
from sports.common.identity import MatchState
from sports.common.jersey import crop_quality
from sports.common.roster import apply_numbers, registry, review_version
from sports.common.team import TeamClassifier
from sports.common.tracking import FootballTracker, distinct_people


class JerseyEvidenceTests(unittest.TestCase):
    def state(self):
        state = MatchState()
        for i in range(6): state.observe(1, 0, i*.08)
        return state

    def test_repeated_variants_of_one_image_do_not_confirm_a_number(self):
        state = self.state()
        for t in (1., 1., 1.04, 1.08): state.jersey_read(1, '18', .99, t, 'test')
        self.assertIsNone(state.report()[0]['jersey_number'])
        for t in (1.4, 1.8): state.jersey_read(1, '18', .99, t, 'test')
        row = state.report()[0]
        self.assertEqual(row['jersey_number'], '18')
        self.assertEqual(len(row['jersey_evidence']), 3)

    def test_conflicting_and_weak_reads_remain_candidates(self):
        state = self.state()
        for i in range(4): state.jersey_read(1, '18' if i < 3 else '16', .9, i*.4)
        self.assertIsNone(state.report()[0]['jersey_number'])
        state = self.state()
        for i in range(4): state.jersey_read(1, '18', .7, i*.4)
        self.assertEqual(state.report()[0]['jersey_status'], 'candidate')

    def test_recent_consistent_team_evidence_can_correct_early_mistake(self):
        state = self.state()
        for i in range(100): state.observe(1, 0, i)
        state.observe(1, 1, 101)
        self.assertEqual(state.players[1].team_id, 0)
        for i in range(12): state.observe(1, 1, 102+i, confidence=.9)
        self.assertEqual(state.players[1].team_id, 1)

    def test_small_or_blank_crops_are_not_upscaled_into_evidence(self):
        self.assertEqual(crop_quality(np.zeros((20, 10, 3), dtype=np.uint8)), 0)
        self.assertEqual(crop_quality(np.zeros((100, 50, 3), dtype=np.uint8)), 0)


class DetectorTests(unittest.TestCase):
    def test_tracker_prediction_does_not_export_two_boxes_on_one_person(self):
        tracker = FootballTracker('bytetrack')
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        boxes = Boxes(np.float32([[10, 10, 40, 90, .9, 2], [25, 10, 55, 90, .8, 2]]), frame.shape[:2])
        # Detector observations are distinct but the tracker converges onto one box.
        predictions = np.float32([[10, 10, 40, 90, 1, .9, 2, 0],
                                  [10, 11, 40, 90, 2, .8, 2, 1]])
        with patch.object(tracker.tracker, 'update', return_value=predictions):
            people = tracker.update(SimpleNamespace(boxes=boxes), frame)
        self.assertEqual(len(people), 1)
        self.assertEqual(people.tracker_id.tolist(), [1])

    def test_cross_role_duplicate_removed_but_overlapping_people_kept(self):
        raw = np.float32([[10, 10, 40, 90, .9, 2], [10, 11, 40, 90, .8, 1],
                          [25, 10, 55, 90, .85, 2], [120, 10, 150, 90, .6, 3]])
        boxes = distinct_people(Boxes(raw, (200, 200)))
        self.assertEqual(len(boxes), 3)
        np.testing.assert_allclose(boxes.conf, [.9, .85, .6])

    def test_out_of_distribution_shirt_does_not_get_forced_into_a_team(self):
        crops = [np.full((80, 30, 3), c, np.uint8) for c in
                 [(240, 240, 240), (210, 210, 210), (40, 210, 60), (35, 180, 50)]]
        classifier = TeamClassifier()
        classifier.fit(crops)
        self.assertEqual(classifier.predict([np.full((80, 30, 3), (200, 40, 40), np.uint8)])[0], 2)


class RegistryTests(unittest.TestCase):
    def data(self):
        return {'source_video': 'match.mp4', 'duration_s': 1.,
                'players': [{'identity_id': i, 'team_id': 0, 'jersey_number': None} for i in (1, 2)],
                'frames': [{'players': [{'identity_id': 1}, {'identity_id': 2}]}], 'events': []}

    def test_review_changes_labels_without_mutating_raw_data(self):
        data = self.data(); original = copy.deepcopy(data)
        corrected = apply_numbers(data, {'1': '18'})
        self.assertEqual(registry(corrected)[0]['jersey_number'], '18')
        self.assertEqual(data, original)
        self.assertIs(corrected['frames'], data['frames'])
        self.assertIs(corrected['events'], data['events'])

    def test_same_team_covisible_duplicate_rejected_opponents_allowed(self):
        data = self.data()
        with self.assertRaises(ValueError): apply_numbers(data, {'1': '9', '2': '9'})
        data['players'][1]['team_id'] = 1
        self.assertEqual(len(apply_numbers(data, {'1': '9', '2': '9'})['players']), 2)

    def test_bad_or_unknown_assignments_rejected_and_analysis_change_versions_review(self):
        data = self.data()
        for assignment in ({'1': ''}, {'1': '0'}, {'1': '100'}, {'1': 'xx'}, {'99': '9'}):
            with self.assertRaises(ValueError): apply_numbers(data, assignment)
        before = review_version(data)
        data['players'][0]['jersey_number'] = '18'
        self.assertNotEqual(before, review_version(data))


if __name__ == '__main__': unittest.main()
