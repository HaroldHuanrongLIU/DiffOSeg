import unittest

import numpy as np
import torch

from reproduction import aligned_preference as aligned


class AlignmentTests(unittest.TestCase):
    def test_training_rims_recover_known_translation(self):
        yy,xx=np.indices((128,128))
        mask=(xx-65)**2+(yy-64)**2<100
        image=np.roll(mask.astype(np.uint8)*150,(1,2),axis=(0,1))
        shift,_=aligned.estimate_shift(image[None],np.repeat(mask[None,None],4,axis=1),["dummy"])
        self.assertEqual(shift,(-2,-1))
        mapped=aligned.registered_ct(torch.from_numpy(image)[None,None],*shift)[0,0].numpy()
        np.testing.assert_equal(mapped,mask.astype(np.uint8)*150)

    def test_identity_is_exact_and_negative_shift_does_not_wrap(self):
        image=torch.arange(25).reshape(1,1,5,5)
        torch.testing.assert_close(image,aligned.registered_ct(image,0,0))
        moved=aligned.registered_ct(image,-1,0)
        torch.testing.assert_close(moved[:,:,:,-1],image[:,:,:,-1])
        torch.testing.assert_close(moved[:,:,:,:-1],image[:,:,:,1:])
