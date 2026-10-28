import unittest
import numpy as np
from baselines.coordinates import inverse_bbox


class InputTransforms(unittest.TestCase):
    def test_inverse_preserves_input_pose_and_scale(self):
        points=np.array([[2.,3.,4.],[7.,8.,9.],[3.,5.,6.]])
        for side in (1.8,2.):
            inverse=inverse_bbox(points,side)
            normalized=(points-np.asarray(inverse["translation"]))/inverse["scale"]
            self.assertAlmostEqual(np.ptp(normalized,axis=0).max(),side)
            recovered=normalized*inverse["scale"]+np.asarray(inverse["translation"])
            np.testing.assert_allclose(recovered,points)

    def test_transform_does_not_hide_prediction_scale_error(self):
        points=np.array([[-1.,-1.,-1.],[1.,1.,1.]])
        inverse=inverse_bbox(points,1.8)
        wrong=np.array([[-.45,-.45,-.45],[.45,.45,.45]])
        mapped=wrong*inverse["scale"]+inverse["translation"]
        self.assertAlmostEqual(np.ptp(mapped,axis=0).max(),1.)

    def test_bad_input_rejected(self):
        for points in ([[0,0,0],[0,0,0]],[[0,0,0],[1,float("nan"),1]]):
            with self.assertRaises(ValueError):inverse_bbox(points,1.8)
