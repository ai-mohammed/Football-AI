import unittest

from sports.common.segments import occupancy, pass_network, structure, summarize


def sample(t, owner=1, calibrated=True, team=0):
    return {'time_s': t, 'dt': 1., 'calibrated': calibrated, 'possessor': owner,
            'ball': [10., 20.], 'players': [
                {'identity_id': 1, 'team_id': team, 'xy': [10., 20.]},
                {'identity_id': 2, 'team_id': 0, 'xy': [30., 20.]}]}


class SegmentTests(unittest.TestCase):
    def setUp(self):
        self.data = {'duration_s': 4., 'players': [
            {'identity_id': 1, 'team_id': 0, 'jersey_number': None,
             'motion_samples': [{'time_s': 1., 'dt': 1., 'distance_m': 2., 'speed_kmh': 7.2},
                                {'time_s': 3., 'dt': 1., 'distance_m': 4., 'speed_kmh': 14.4}]},
            {'identity_id': 2, 'team_id': 0, 'jersey_number': '8', 'motion_samples': []}],
            'frames': [sample(0), sample(1, 2), sample(2, calibrated=False), sample(3, team=None)],
            'events': [{'type': 'probable_pass', 'start_s': .8, 'time_s': 1., 'from': 1, 'to': 2, 'team_id': 0}]}

    def test_fractional_window_clips_both_motion_and_possession(self):
        result = summarize(self.data, .5, 1.5)
        self.assertEqual(result['duration_s'], 1.)
        self.assertEqual(result['possession_seconds'][0], 1.)
        self.assertEqual(result['passes'], 1)
        first = result['players'][0]
        self.assertEqual(first['distance_m'], 1.)
        self.assertEqual(first['measured_s'], .5)
        self.assertEqual(first['speed_kmh'], 7.2)

    def test_unknown_teams_and_invalid_geometry_never_become_known_possession(self):
        result = summarize(self.data)
        self.assertEqual(result['unknown_pct'], 50.)
        self.assertEqual(result['calibration_pct'], 75.)
        self.assertEqual(result['possession_seconds'][0], 2.)

    def test_window_excludes_pass_started_before_it_and_half_open_end(self):
        self.assertEqual(summarize(self.data, 1., 2.)['passes'], 0)
        self.assertEqual(summarize(self.data, 0., 1.)['passes'], 0)

    def test_positions_are_metres_and_occupancy_is_normalized(self):
        result = summarize(self.data, 0., 2.)
        grid = occupancy(result, 0)
        self.assertAlmostEqual(sum(map(sum, grid)), 100.)
        self.assertEqual(grid[1][0], 50.)
        nodes, edges = pass_network(result, 0)
        self.assertEqual(nodes[1], [10., 20.])
        self.assertEqual(edges, {(1, 2): 1})

    def test_structure_does_not_claim_complete_team_from_two_players(self):
        self.assertTrue(all(p['width_m'] is None for p in structure(summarize(self.data), 0)))

    def test_missing_samples_and_empty_window_do_not_invent_measurements(self):
        self.data['frames'] = []
        result = summarize(self.data)
        self.assertEqual(result['unknown_pct'], 100.)
        self.assertIsNone(result['possession_pct'][0])
        self.assertEqual(result['players'], [])
        self.assertEqual(summarize(self.data, 2., 2.)['duration_s'], 0.)


if __name__ == '__main__':
    unittest.main()
