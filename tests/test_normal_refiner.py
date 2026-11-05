import unittest

import numpy as np
import torch

from reproduction import normal_refiner as method
from reproduction import rank_contours as rc


class NormalProfileTests(unittest.TestCase):
    def test_ramp_and_direction(self):
        axis=torch.arange(41,dtype=torch.float32)
        image=axis.expand(1,1,41,41)/40
        guide=(axis-20).expand_as(image)
        z=method.profile_features(image,guide,True)
        expected=4*torch.tensor(method.OFFSETS)/40
        torch.testing.assert_close(z[0,7:,20,20],expected,atol=2e-6,rtol=0)
        reverse=method.profile_features(image,-guide,True)
        torch.testing.assert_close(reverse[0,7:,20,20],-expected,atol=2e-6,rtol=0)

    def test_same_weight_ablation_changes_only_lateral_samples(self):
        torch.manual_seed(31)
        image=torch.rand(2,1,32,32)
        guide=torch.rand_like(image)
        full=method.profile_features(image,guide,True)
        dropped=method.profile_features(image,guide,False)
        torch.testing.assert_close(full[:,:7],dropped[:,:7])
        self.assertEqual(torch.count_nonzero(dropped[:,7:]).item(),0)
        self.assertGreater(torch.count_nonzero(full[:,7:]).item(),0)

    def test_guide_uses_middle_supports_only(self):
        x=np.full((2,4,32,32),-2,dtype=np.float32)
        x[:,1,9:24,9:24]=3
        x[:,2,11:22,11:22]=1
        first=method.image_guide(x)
        x[:,0]=1000; x[:,3]=1000
        np.testing.assert_array_equal(first,method.image_guide(x))
        np.testing.assert_array_equal(first[0,0],(rc.signed(x[0,1]>0)+rc.signed(x[0,2]>0))/2)

    def test_zero_initial_residual_keeps_the_guide(self):
        x=torch.zeros(1,1,32,32)
        guide=torch.randn_like(x)
        result=method.BoundaryRefiner()(x,guide)
        torch.testing.assert_close(result,guide.expand(-1,8,-1,-1))


if __name__=="__main__":
    unittest.main()
