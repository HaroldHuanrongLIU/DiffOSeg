import unittest
import numpy as np
import torch
from reproduction import texture_preference as tp
from reproduction.texture_study import anchor_logits


class TexturePreferenceTests(unittest.TestCase):
    def test_incumbent_anchor_preserves_discrete_prediction(self):
        logits=np.array([-1.,0.,2.,-.002])
        truth=np.array([True,False,False,True])
        np.testing.assert_array_equal(anchor_logits(logits,truth)>0,truth)

    def test_single_image_and_six_predicted_attributes(self):
        torch.manual_seed(80)
        net=tp.TextureNet().eval()
        x=torch.rand(1,1,64,64)
        with torch.no_grad():
            a,attr=net(torch.cat([x,torch.rand_like(x)]))
            b,attrs=net(torch.cat([x,torch.zeros_like(x)]))
        self.assertEqual(a.shape,(2,8,64,64))
        self.assertEqual(attr.shape,(2,6))
        torch.testing.assert_close(a[0],b[0],rtol=1e-5,atol=1e-5)
        torch.testing.assert_close(attr[0],attrs[0])
