import unittest
import numpy as np
import torch
from reproduction import focal_deformation as focal
from reproduction import native_local as native


class FocalDeformationTests(unittest.TestCase):
    def test_inverse_and_positive_jacobian(self):
        center=torch.tensor([[51.,73.],[67.,62.]])
        radius=torch.tensor([8.,14.])
        matrix=torch.tensor([[[.28,0.],[0.,-.28]],[[.08,.19],[.19,-.08]]])
        target=native.coordinates(128,"cpu")[None].expand(2,-1,-1,-1)
        for local in (False,True):
            source,back=focal.grids(center,radius,matrix,local)
            original=(back+1)*(127/2)
            restored=original+focal.displacement(original,center,radius,matrix,local)
            torch.testing.assert_close(restored,target,atol=5e-5,rtol=0)
            source=(source+1)*(127/2)
            dx=(source[:,1:-1,2:]-source[:,1:-1,:-2])/2
            dy=(source[:,2:,1:-1]-source[:,:-2,1:-1])/2
            determinant=dx[...,0]*dy[...,1]-dx[...,1]*dy[...,0]
            self.assertGreater(determinant.min().item(),.50)

    def test_field_changes_near_nodule_and_decays_in_background(self):
        centers=torch.tensor([[64.,64.]])
        radius=torch.tensor([7.]);matrix=torch.tensor([[[.25,0.],[0.,.25]]])
        coords=torch.tensor([[[[74.,64.],[124.,64.]]]])
        local=focal.displacement(coords,centers,radius,matrix,True)
        uniform=focal.displacement(coords,centers,radius,matrix,False)
        self.assertGreater(local[0,0,0,0].item(),1.)
        self.assertLess(local[0,0,1,0].abs().item(),.01)
        self.assertEqual(uniform[0,0,1,0].item(),15.)

    def test_geometry_and_rng_are_sample_local(self):
        mask=np.zeros((2,4,128,128),bool);mask[1,:2,40:44,51:57]=True
        c,r=focal.geometry(mask)
        np.testing.assert_allclose(c,[[63.5,63.5],[53.5,41.5]])
        np.testing.assert_allclose(r,[10.,np.sqrt(24/np.pi)],rtol=1e-6)
        before=torch.random.get_rng_state()
        a=focal.sample_matrix(5,torch.Generator().manual_seed(98))
        b=focal.sample_matrix(5,torch.Generator().manual_seed(98))
        torch.testing.assert_close(before,torch.random.get_rng_state())
        torch.testing.assert_close(a,b)
        self.assertLessEqual(torch.linalg.matrix_norm(a,ord=2).max().item(),.28)
