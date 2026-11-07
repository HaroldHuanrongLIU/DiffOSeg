import unittest
import numpy as np
import torch
from torch.nn import functional as F
from reproduction import shell_occupancy as shell


class ShellOccupancyTests(unittest.TestCase):
    def test_pooled_shells_and_duplicate_endpoint(self):
        values=np.array([[2,4,8,10]],np.float32)
        result=shell.segments(values,np.array([0,2,2,4]))
        np.testing.assert_array_equal(result[:,0],[2,3,4,9])

    def test_public_packed_grid_matches_pixel_shuffle(self):
        values=np.arange(8*3*3,dtype=np.float32).reshape(8,3,3)
        numpy_result=shell.unpack_level(values,2)
        torch_result=F.pixel_shuffle(torch.from_numpy(values)[None],2)[0].numpy()
        np.testing.assert_array_equal(numpy_result,torch_result)

    def test_zero_shift_and_expected_dice(self):
        guide=np.full((4,128,128),-1.,np.float32)
        guide[:,60:70,61:69]=1
        packed=np.zeros((1024,32,32),np.float16)
        _,counts,orders,hits=shell.sample_features(np.zeros((128,128)),guide,packed,guide>0)
        masks=shell.apply_sizes(orders[None],counts[None],np.ones((1,4)),0.)
        np.testing.assert_array_equal(masks[0],guide>0)
        self.assertEqual(counts[0,shell.ZERO],80)
        self.assertEqual(hits[0,shell.ZERO],80)
        ct=torch.as_tensor(counts[None],dtype=torch.float32)
        curve,_=shell.expected_curve(torch.zeros_like(ct),torch.zeros(1,4),ct)
        self.assertAlmostEqual(float(curve[0,0,shell.ZERO]),.5,places=5)
