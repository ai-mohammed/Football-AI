import copy
import unittest

from sports.common.matches import crop_analysis, plan_segments, video_time, combine_segments, segment_playback
from sports.common.segments import summarize


class MatchPartitionTests(unittest.TestCase):
    def test_supplied_match_has_537_segments_without_gaps_or_duplicate_frames(self):
        segments = plan_segments(321813, 50.)
        self.assertEqual(len(segments), 537)
        self.assertEqual(segments[0]['start_frame'], 0)
        self.assertEqual(segments[-1]['end_frame'], 321813)
        self.assertAlmostEqual(segments[-1]['duration_s'], 4.26)
        self.assertEqual(sum(s['end_frame']-s['start_frame'] for s in segments), 321813)
        for a, b in zip(segments, segments[1:]):
            self.assertEqual(a['end_frame'], b['start_frame'])
        self.assertTrue(all(0 < s['duration_s'] <= 12 for s in segments))

    def test_fractional_frame_rate_never_rounds_above_twelve_seconds(self):
        self.assertTrue(all(s['duration_s'] <= 12 for s in plan_segments(10000, 29.97)))
        for fps, frames, limit in ((0, 100, 12), (float('nan'), 100, 12), (25, 0, 12), (25, 10, 13)):
            with self.assertRaises(ValueError):
                plan_segments(frames, fps, limit)
        self.assertEqual(video_time(6436.26), '01:47:16')

    def test_context_is_not_counted_twice_and_crossing_pass_belongs_to_arrival(self):
        data = {'duration_s': 4., 'frames': [], 'diagnostics': {}, 'events': [
            {'type': 'probable_pass', 'start_s': 1.5, 'time_s': 2., 'from': 1, 'to': 2, 'team_id': 0}],
            'players': [{'identity_id': i, 'tracker_ids': [i], 'team_id': 0, 'jersey_number': None,
                         'timestamps': [0., 1., 2., 3.], 'trajectory': [[100, 100]]*4,
                         'motion_samples': [{'time_s': 1., 'dt': 1., 'distance_m': 2., 'speed_kmh': 7.2},
                                            {'time_s': 3., 'dt': 1., 'distance_m': 3., 'speed_kmh': 10.8}],
                         'jersey_evidence': [{'time_s': 1.5, 'number': '7', 'track_id': i}],
                         'jersey_previews': []} for i in (1, 2)]}
        for t in range(4):
            data['frames'].append({'time_s': float(t), 'dt': 1., 'calibrated': True, 'ball': [1,1],
                                   'possessor': 1, 'players': [
                                       {'identity_id': i, 'track_id': i, 'team_id': 0, 'xy': [1., 1.]} for i in (1,2)]})
        original = copy.deepcopy(data)
        previous = crop_analysis(data, 0., 2., 0., 's0001')
        current = crop_analysis(data, 2., 2., 2., 's0002')
        self.assertEqual(data, original)
        self.assertEqual(summarize(previous)['passes'], 0)
        self.assertEqual(summarize(current)['passes'], 1)
        self.assertTrue(current['events'][0]['origin_in_context'])
        self.assertEqual(current['events'][0]['start_s'], -.5)
        self.assertEqual(current['frames'][0]['time_s'], 0.)
        self.assertEqual(current['players'][0]['distance_m'], 3.)
        self.assertEqual(current['players'][0]['possession_seconds'], 2.)
        self.assertTrue(current['players'][0]['jersey_evidence'][0]['in_context'])
        self.assertEqual(summarize(current, .1, 2.)['passes'], 0)

    def test_global_view_keeps_all_graph_data_without_merging_segment_ids(self):
        def segment(sid, offset, team):
            return {'schema_version': 3, 'source_video': 'match.mp4', 'source_resolution': [1280,720],
                    'source_fps': 50, 'stride': 4, 'pitch': {'length':105,'width':68,'unit':'m'},
                    'segment_id': sid, 'source_start_s':offset, 'duration_s':12., 'diagnostics':{},
                    'players':[{'identity_id':1,'tracker_ids':[1], 'team_id':team,'jersey_number':'9',
                                'motion_samples':[{'time_s':2.,'dt':1.,'distance_m':3.,'speed_kmh':10.8}],
                                'timestamps':[0.], 'trajectory':[[20.,30.]]}],
                    'frames':[{'time_s':0.,'dt':12.,'calibrated':True,'ball':[20.,30.], 'possessor':1,
                               'image_detections':[{'identity_id':1},{'identity_id':99,'class_id':3}],
                               'players':[{'identity_id':1,'team_id':team,'xy':[20.,30.],'track_id':1}]}],
                    'events':[{'type':'probable_pass','time_s':1.,'start_s':-.2,'origin_in_context':True,
                               'from':1,'to':1,'team_id':team}]}
        a,b=segment('s0001',0.,0),segment('s0002',12.,1)
        original=copy.deepcopy([a,b])
        data=combine_segments([b,a])
        self.assertEqual([a,b],original)
        self.assertEqual(data['duration_s'],24.)
        self.assertEqual(len(data['players']),2)
        self.assertNotEqual(data['players'][0]['identity_id'],data['players'][1]['identity_id'])
        summary=summarize(data)
        self.assertEqual(summary['possession_pct'],{0:50.,1:50.})
        self.assertEqual(summary['passes'],2)
        self.assertEqual(sum(p['distance_m'] for p in summary['players']),6.)
        self.assertEqual(data['events'][1]['start_s'],11.8)
        self.assertEqual(data['players'][1]['motion_samples'][0]['time_s'],14.)
        self.assertIsNone(data['frames'][0]['image_detections'][1]['identity_id'])
        playback=segment_playback(data,'s0002')
        self.assertEqual(playback['frames'][0]['time_s'],0.)
        self.assertEqual(playback['source_start_s'],12.)
        self.assertEqual(playback['events'][0]['start_s'],-.2)
        self.assertEqual(playback['frames'][0]['possessor'],data['players'][1]['identity_id'])
        with self.assertRaises(ValueError): combine_segments([a,a])


if __name__ == '__main__': unittest.main()
