import numpy as np
import unittest
from reproduction import panel_event as event


class EventTest(unittest.TestCase):
    def test_mixed_pixels_and_interior_are_protected(self):
        x=np.zeros((4,24,24),bool);x[:,5:19,5:19]=True;x[0,5,10]=False
        band,_=event.unanimous_band(x)
        self.assertFalse(band[5,10])
        self.assertFalse(band[11,11])
        self.assertFalse(band[0,0])
        self.assertTrue(band[4,10])

    def test_add_and_remove_use_separate_costs(self):
        ref=np.zeros((4,4,8),bool);ref[:,2,1]=True
        case=event.Example(np.packbits(ref,axis=-1),np.array([0,17]),None,None,None,None,None)
        prob=np.full((2,4),.55)
        pred=np.unpackbits(event.flip(case,prob),axis=-1)
        self.assertTrue(pred[:,0,0].all())
        self.assertTrue(pred[:,2,1].all())
        prob[1,0]=.61
        pred=np.unpackbits(event.flip(case,prob),axis=-1)
        self.assertFalse(pred[0,2,1])
        self.assertTrue(pred[1:,2,1].all())
