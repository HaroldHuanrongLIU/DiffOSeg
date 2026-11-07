import unittest
import torch
from reproduction import event_patches as patches


class EventPatchTest(unittest.TestCase):
    def test_neighbor_orientation_center_and_border_replication(self):
        x=torch.arange(128.).expand(128,128)
        y=torch.arange(128.).view(-1,1).expand(128,128)
        bank=torch.stack((x,y),-1)[None]
        i=torch.tensor([0,0]);p=torch.tensor([0,128*40+50])
        real=patches.patches(bank,i,p,True);same=patches.patches(bank,i,p,False)
        torch.testing.assert_close(real[1,:,0,0],torch.tensor([47.,37.]))
        torch.testing.assert_close(real[0,:,0,0],torch.zeros(2))
        torch.testing.assert_close(same[1],torch.tensor([50.,40.])[:,None,None].expand(2,7,7))

    def test_zero_start_preserves_every_rank_and_receives_gradient(self):
        model=patches.ErrorPatch()
        x=torch.randn(5,44,7,7);d=model(x)
        torch.testing.assert_close(d,torch.zeros_like(d))
        d.sum().backward()
        self.assertGreater(model.readout.weight.grad.abs().sum().item(),0)
