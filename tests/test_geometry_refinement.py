import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from refine_match_geometry import refine


class RefinementTests(unittest.TestCase):
    def data(self):
        return {'pitch':{'length':105,'width':68},'source_resolution':[320,240],
                'players':[{'identity_id':7,'tracker_ids':[7],'team_id':0,
                            'jersey_number':'10','jersey_status':'reviewed'}],
                'frames':[{'time_s':i*.08,'dt':.08,'calibrated':True,
                           'players':[{'track_id':7,'identity_id':7,'team_id':0,'xy':[99,66]}],
                           'image_detections':[{'track_id':7,'identity_id':7,'class_id':2,'xyxy':[90,50,110,100]}],
                           'ball_image_xy':[100,100],'ball':[99,66],'possessor':7} for i in range(4)],
                'events':[{'type':'probable_pass','from':7,'to':9,'time_s':.1}],
                'diagnostics':{}}

    def test_recomputes_all_measurements_keeps_evidence_and_unknown_intervals(self):
        data=self.data();before=copy.deepcopy(data)
        class Transform:
            def transform_points(self,points): return points
        with patch('refine_match_geometry.TemporalPitchCalibrator') as cls, \
             patch('refine_match_geometry.ShotChangeDetector') as shots:
            cal=cls.return_value
            cal.method='pitch_keypoints';cal.rejected_jump=False;cal.smoothed=True
            cal.layout_blocked=False
            cal.update.side_effect=[Transform(),Transform(),None,Transform()]
            shots.return_value.update.side_effect=[False,False,True,False]
            result=refine(data,[np.zeros((240,320,3),np.uint8)]*4,
                          np.zeros((4,32,2),np.float32),np.zeros((4,32),np.float32))
        self.assertEqual(data,before)
        self.assertEqual(result['players'][0]['jersey_status'],'reviewed')
        self.assertEqual(result['players'][0]['jersey_number'],'10')
        self.assertEqual(result['frames'][0]['players'][0]['xy'],[1.,1.])
        self.assertEqual(result['frames'][2]['players'],[])
        self.assertIsNone(result['frames'][2]['ball'])
        self.assertEqual([e['type'] for e in result['events']],['camera_cut'])
        self.assertEqual(result['players'][0]['passes_made'],0)
        self.assertAlmostEqual(result['players'][0]['measured_seconds'],.08)
        self.assertAlmostEqual(result['diagnostics']['unknown_possession_seconds'],.24)
        self.assertTrue(result['diagnostics']['statistics_recomputed'])

    def test_incomplete_source_is_not_exported_as_complete(self):
        data=self.data()
        with self.assertRaisesRegex(ValueError,'Source ended'):
            refine(data,[],np.zeros((4,32,2)),np.zeros((4,32)))


if __name__=='__main__': unittest.main()
