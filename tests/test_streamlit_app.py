"""Smoke-test demo exploration and filters without loading inference models."""
import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / 'examples/soccer/streamlit_app.py'

class StreamlitTests(unittest.TestCase):
    def test_demo_window_team_and_player_controls_render(self):
        app = AppTest.from_file(str(APP), default_timeout=30).run()
        self.assertEqual(list(app.exception), [])
        self.assertEqual([t.label for t in app.tabs], ['Tactique', 'Joueurs', 'Événements', 'Fiabilité & exports'])
        app.slider[0].set_value((2., 6.)).run()
        self.assertEqual(list(app.exception), [])
        app.radio(key='team_08fd33_0').set_value(1).run()
        app.selectbox(key='focus_08fd33_0').select(1).run()
        self.assertEqual(list(app.exception), [])
        self.assertTrue(any('2.0–6.0' in c.value for c in app.caption))

    def test_all_five_demos_and_import_empty_state(self):
        app = AppTest.from_file(str(APP), default_timeout=30).run()
        paths = sorted((APP.parent/'demo_data').glob('*/analysis.json'))
        self.assertEqual(len(paths), 5)
        for path in paths:
            app.selectbox(key='demo').select(path.parent).run()
            self.assertEqual(list(app.exception), [], path.parent.name)
        app.sidebar.radio[0].set_value('Importer une vidéo').run()
        self.assertEqual(list(app.exception), [])
        self.assertTrue(any('démonstration immédiate' in message.value for message in app.info))

if __name__ == '__main__':
    unittest.main()
