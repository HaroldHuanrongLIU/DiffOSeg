import unittest
import numpy as np
import torch
from reproduction import contour_flow as flow


class ContourFlowTest(unittest.TestCase):
    def test_noise_variance_rank_covariance_and_spatial_dependence(self):
        generator=torch.Generator().manual_seed(231)
        draw=torch.randn(24,10,64,64,generator=generator)
        white=flow.noise(draw,"white")
        smooth=flow.noise(draw,"smooth")
        for field in (white,smooth):
            self.assertLess(abs(field.var().item()-1),.055)
            self.assertLess(abs((field[:,0]*field[:,1]).mean().item()-.5),.055)
        self.assertLess(abs((white[:,:,:,1:]*white[:,:,:,:-1]).mean().item()),.035)
        self.assertGreater((smooth[:,:,:,1:]*smooth[:,:,:,:-1]).mean().item(),.6)
        torch.testing.assert_close(flow.noise(-draw,"smooth"),-smooth)

    def test_linear_path_reaches_true_endpoint(self):
        start=torch.randn(2,4,8,8)
        end=torch.randn_like(start)
        t=torch.tensor([.1,.9]).reshape(2,1,1,1)
        current,v=flow.interpolant(start,end,t)
        torch.testing.assert_close(current+(1-t)*v,end)
        mask=np.zeros((2,4,9,9),bool)
        mask[0,:,1:3,4:7]=True
        np.testing.assert_array_equal(flow.distance(mask)>0,mask)

    def test_six_midpoints_integrate_linear_time(self):
        class LinearTime(torch.nn.Module):
            def forward(self,image,guide,state,t):
                return 2*t.expand_as(state)
        guide=torch.zeros(2,4,16,16)
        image=torch.zeros(2,1,16,16)
        zeros=torch.zeros(1,1,10,16,16)
        out=flow.endpoint(LinearTime(),image,guide,"white",zeros)
        torch.testing.assert_close(out,torch.full_like(out,torch.sigmoid(torch.tensor(2.5)).item()))
