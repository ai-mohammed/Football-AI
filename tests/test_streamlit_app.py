"""Render the demo and a report with uncalibrated tracks; no inference."""
import unittest
from pathlib import Path

import numpy as np
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / 'examples/soccer/streamlit_app.py'


class StreamlitTests(unittest.TestCase):
    def test_demo_and_uncalibrated_player_report_render(self):
        app = AppTest.from_file(str(APP), default_timeout=60)
        app.session_state['player_report'] = [{
            'identity_id': 1, 'tracker_ids': [1], 'jersey_number': None,
            'team_id': 0, 'touches': 0, 'passes_made': 0, 'passes_received': 0,
            'distance_m': 0, 'avg_speed_kmh': None, 'trajectory': np.empty((0, 2)),
        }]
        app.session_state['team_report'] = {
            'possession_pct': {0: 0, 1: 0},
            'team_heatmaps': {0: np.empty((0, 2)), 1: np.empty((0, 2))},
            'pass_networks': {t: {'node_xy': np.empty((0, 2)), 'node_labels': [], 'edges': []} for t in (0, 1)},
        }
        app.session_state['analysis_diagnostics'] = {}
        app.run()
        self.assertEqual(list(app.exception), [])
        self.assertTrue(any(metric.value == 'Indéterminée' for metric in app.metric))
        self.assertEqual(len(app.tabs), 2)


if __name__ == '__main__':
    unittest.main()
