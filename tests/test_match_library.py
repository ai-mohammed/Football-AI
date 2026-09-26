import hashlib
import copy
import gzip
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'examples/soccer'), str(ROOT/'scripts')]
import match_library
import publish_match
from publish_continuous_match import validate_analysis
from sports.common.matches import plan_segments, combine_segments
from streamlit.testing.v1 import AppTest


class MatchLibraryTests(unittest.TestCase):
    def test_continuous_publication_rejects_holes_and_segment_reset_data(self):
        manifest={'source':{'filename':'test.mp4'},'analysis_window':{'duration_s':24}}
        data={'source_video':'test.mp4','duration_s':24,'identity_scope':'continuous',
              'diagnostics':{'chapter_resets':0},
              'frames':[{'time_s':n/10,'dt':.1} for n in range(240)]}
        validate_analysis(data,manifest)
        missing=copy.deepcopy(data)
        del missing['frames'][120]
        with self.assertRaisesRegex(ValueError,'gap'):
            validate_analysis(missing,manifest)
        data['identity_scope']='segment'
        with self.assertRaisesRegex(ValueError,'does not match'):
            validate_analysis(data,manifest)

    def test_continuous_mode_uses_one_video_and_full_timeline(self):
        data = json.loads(next((ROOT/'examples/soccer/demo_data').glob('*/analysis.json')).read_text(encoding='utf-8'))
        data.update(aggregate=True, identity_scope='continuous', duration_s=24., source_start_s=0.,
                    segments=[{'id':'s0001','start_s':0,'end_s':12}, {'id':'s0002','start_s':12,'end_s':24}])
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'analysis.json.gz').write_bytes(gzip.compress(json.dumps(data).encode()))
            segments=plan_segments(1200,50)
            for s in segments:
                s['status']='ready'
            media_url=match_library.REMOTE_PREFIX+'fixture/continuous-test.mp4'
            manifest={'title':'Continu', 'source':{'duration_s':24}, 'segments':segments,
                'analysis_window':{'start_s':0,'end_s':24},
                'continuous':{'analysis':{'local_path':'analysis.json.gz'},'video':{'url':media_url}}}
            path=root/'manifest.json'
            path.write_text(json.dumps(manifest),encoding='utf-8')
            script=f'''
import sys
sys.path.insert(0, {str(ROOT/'examples/soccer')!r})
import match_library
from pathlib import Path
match_library.catalog=lambda: {{'fixture':Path({str(path)!r})}}
match_library.render_match_library()
'''
            import tactical_dashboard
            with patch.object(match_library,'catalog',match_library.catalog), \
                 patch.object(match_library,'remote_segment') as segment_loader, \
                 patch.object(tactical_dashboard,'PLAYER_COMPONENT') as component:
                app=AppTest.from_string(script,default_timeout=30).run()
                self.assertEqual(list(app.exception),[])
                self.assertFalse(any(b.label=='Segment suivant' for b in app.button))
                self.assertEqual(app.slider[0].value,(0.,24.))
                self.assertEqual(component.call_args.kwargs['video_url'],media_url)
                self.assertIn(media_url,component.call_args.kwargs['media_id'])
                self.assertIsNone(component.call_args.kwargs['video_base64'])
                app.slider[0].set_value((6.,18.)).run()
                self.assertEqual(list(app.exception),[])
                self.assertEqual(component.call_args.kwargs['start'],0.)
                self.assertEqual(component.call_args.kwargs['end'],24.)
                self.assertEqual(len(component.call_args.kwargs['chapters']),2)
                self.assertNotIn('image_detections',component.call_args.kwargs['frames'][0])
                media_id=component.call_args.kwargs['media_id']
                component.reset_mock()
                app.slider[0].set_value((12.,12.)).run()
                self.assertEqual(list(app.exception),[])
                component.assert_called_once()
                self.assertEqual(component.call_args.kwargs['media_id'],media_id)
                self.assertEqual(component.call_args.kwargs['end'],24.)
                self.assertEqual(len(app.metric),0)
                self.assertTrue(any('lecture continue reste disponible' in i.value for i in app.info))
                segment_loader.assert_not_called()

    def test_switching_video_preserves_global_charts_and_filter(self):
        demo = next((ROOT/'examples/soccer/demo_data').glob('*/analysis.json'))
        first = json.loads(demo.read_text(encoding='utf-8'))
        first.update(segment_id='s0001', source_start_s=0.)
        second = copy.deepcopy(first)
        second.update(segment_id='s0002', source_start_s=12.)
        overview = combine_segments([first, second])
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder/'overview.json.gz').write_bytes(gzip.compress(json.dumps(overview).encode()))
            segments = plan_segments(1200, 50)
            for segment, data in zip(segments, [first, second]):
                segment['status'] = 'ready'
                target = folder/'segments'/segment['id']
                target.mkdir(parents=True)
                (target/'analysis.json').write_text(json.dumps(data), encoding='utf-8')
                (target/'annotated.mp4').write_bytes(b'component-video-fixture')
            path = folder/'manifest.json'
            path.write_text(json.dumps({'title':'Test global','source':{'duration_s':24},
                'segments':segments,'overview':{'local_path':'overview.json.gz'}}), encoding='utf-8')
            script = f'''
import sys
sys.path.insert(0, {str(ROOT/'examples/soccer')!r})
import match_library
from pathlib import Path
match_library.catalog = lambda: {{'fixture': Path({str(path)!r})}}
match_library.render_match_library()
'''
            with patch.object(match_library, 'catalog', match_library.catalog):
                app = AppTest.from_string(script, default_timeout=30).run()
                self.assertEqual(list(app.exception), [])
                self.assertEqual(app.slider[0].value, (0.,24.))
                metrics = {m.label:m.value for m in app.metric}
                next(b for b in app.button if b.label=='Segment suivant').click().run()
                self.assertEqual({m.label:m.value for m in app.metric},metrics)
                app.slider[0].set_value((6.,18.)).run()
                next(b for b in app.button if b.label=='Segment précédent').click().run()
                self.assertEqual(app.slider[0].value,(6.,18.))
                self.assertEqual(list(app.exception), [])

    def test_bundle_integrity_and_contents(self):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, 'w') as archive:
            archive.writestr('analysis.json', '{"duration_s": 12}')
            archive.writestr('annotated.mp4', b'test-video')
        content = stream.getvalue()
        match_library.remote_segment.clear()
        with patch.object(match_library, 'download', return_value=content):
            data, video = match_library.remote_segment('test', hashlib.sha256(content).hexdigest())
            self.assertEqual(data['duration_s'], 12)
            self.assertEqual(video, b'test-video')
            with self.assertRaisesRegex(ValueError, 'incomplet'):
                match_library.remote_segment('test', 'invalid-sha')
        with self.assertRaises(ValueError):
            match_library.download('https://unrelated.example/file.zip', 1024)

    def test_pending_segments_navigation_and_failed_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'manifest.json'
            segments = plan_segments(2600, 50)
            segments[1]['status'] = 'running'
            segments[2]['status'] = 'awaiting_publication'
            segments[-1]['status'] = 'failed'
            path.write_text(json.dumps({'title': 'Match de test', 'source': {'duration_s': 52},
                                        'segments': segments}), encoding='utf-8')
            script = f'''
import sys
sys.path.insert(0, {str(ROOT/'examples/soccer')!r})
import match_library
from pathlib import Path
match_library.catalog = lambda: {{'fixture': Path({str(path)!r})}}
match_library.render_match_library()
'''
            # Restore catalog after the fixture script replaces it in-process.
            with patch.object(match_library, 'catalog', match_library.catalog):
                app = AppTest.from_string(script, default_timeout=30).run()
                self.assertEqual(list(app.exception), [])
                self.assertEqual(app.selectbox[1].value, 's0001')
                next(b for b in app.button if b.label == 'Segment suivant').click().run()
                self.assertEqual(app.selectbox[1].value, 's0002')
                self.assertTrue(any('00:00:12 → 00:00:24' in h.value for h in app.subheader))
                self.assertTrue(any('analyse de ce segment est en cours' in i.value for i in app.info))
                app.selectbox[1].select('s0003').run()
                self.assertTrue(any('en cours de publication' in i.value for i in app.info))
                app.radio[0].set_value('À reprendre').run()
                self.assertEqual(list(app.exception), [])
                self.assertEqual(app.selectbox[1].value, 's0005')
                self.assertEqual(len(app.warning), 1)

    def test_public_index_removes_local_paths_and_unpublished_ready_status(self):
        publisher = publish_match.MatchPublisher.__new__(publish_match.MatchPublisher)
        payload = {'segments': [{'id': 's0001', 'status': 'ready',
                                'analysis_path': 'private/path', 'video_path': 'private/video'}],
                   'continuous':{'analysis':{'url':'public','local_path':'private/analysis'},
                                 'video':{'url':'public-video','local_path':'private/video'}}}
        with patch.object(publisher, '_upload') as upload:
            publisher.index(payload)
        data = json.loads(upload.call_args.args[1])
        self.assertEqual(data['segments'][0], {'id': 's0001', 'status': 'awaiting_publication'})
        self.assertEqual(payload['segments'][0]['status'], 'ready')
        self.assertNotIn('local_path',data['continuous']['analysis'])
        self.assertNotIn('local_path',data['continuous']['video'])

    def test_publisher_retries_final_index_when_worker_already_stopped(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder/'manifest.json').write_text(json.dumps({'title': 'Test', 'updated_at': 'done',
                'segments': [{'id': 's0001', 'status': 'ready'}]}))
            (folder/'publication.json').write_text(json.dumps({'s0001': {'bundle_url': 'test'}}))
            with patch.object(sys, 'argv', ['publish_match', '--directory', directory, '--watch']), \
                 patch.object(publish_match, 'MatchPublisher') as factory, \
                 patch.object(publish_match.time, 'sleep') as sleep:
                publisher = factory.return_value
                publisher.index.side_effect = [publish_match.requests.ConnectionError(), None]
                publish_match.main()
                self.assertEqual(publisher.index.call_count, 2)
                sleep.assert_called_once_with(45)
                self.assertFalse((folder/'publisher.lock').exists())


if __name__ == '__main__':
    unittest.main()
