import unittest
import numpy as np
from reproduction import shared_radius as radius


class SharedRadiusTests(unittest.TestCase):
    def test_zero_shift_identity_and_direct_curve(self):
        masks=np.zeros((4,128,128),bool)
        for r in range(4):masks[r,60-r:68+r,59:70]=True
        scores=np.where(masks,1.,-1.).astype(np.float32)
        image=np.arange(128,dtype=np.uint8)[None,:].repeat(128,axis=0)
        target=np.roll(masks,1,axis=-1)
        c=radius.example(image,scores,scores,masks,np.zeros((1024,32,32),np.float16),target)
        np.testing.assert_array_equal(radius.masks_for(c,3.,0.),masks)
        for j in (2,radius.ZERO,20):
            got=radius.masks_for(c,radius.OFFSETS[j],1.)
            count=got.sum((1,2))+target.sum((1,2))
            expected=2*(got&target).sum((1,2))/count
            np.testing.assert_allclose(expected,c.curves[:,j])

    def test_joint_action_and_ties_use_nearest_zero(self):
        values=np.zeros((4,len(radius.OFFSETS)));values[:,radius.ZERO+2]=[1.,1.,.7,.7]
        values[2:,radius.ZERO-3]=1.
        self.assertEqual(radius.best(values.mean(0)),.5)
        np.testing.assert_array_equal(radius.best(values),[.5,.5,-.75,-.75])
        self.assertEqual(radius.best(np.zeros(len(radius.OFFSETS))),0.)
