import unittest
import torch
from reproduction.rank_mixture import assignment_weights,expected_mask_loss
from reproduction.vote_contours import mask_loss


class RankMixtureTests(unittest.TestCase):
    def test_separated_radii_give_original_rank_assignment(self):
        y=torch.zeros(1,4,40,40)
        for j,side in enumerate((4,12,20,28)):
            y[0,j,:side,:side]=1
        torch.testing.assert_close(assignment_weights(y)[0],torch.eye(4))

    def test_tied_radii_share_supervision_and_conserve_columns(self):
        y=torch.zeros(1,4,16,16)
        for rank in range(4):
            y[0,rank,rank:rank+4,rank:rank+4]=1
        p=assignment_weights(y)
        torch.testing.assert_close(p[0],0.5*torch.eye(4)+torch.full((4,4),0.125))
        torch.testing.assert_close(p.sum(1),torch.ones(1,4))
        torch.testing.assert_close(p.sum(2),torch.ones(1,4))

    def test_fractional_pair_losses_and_gradients_match_direct_enumeration(self):
        torch.manual_seed(72)
        z=torch.randn(2,4,6,8,requires_grad=True)
        y=torch.rand_like(z)
        weight=assignment_weights(y>0.5)
        actual=expected_mask_loss(z,y,weight)
        expected=sum(weight[i,r,j]*mask_loss(z[i:i+1,r:r+1],y[i:i+1,j:j+1])
                     for i in range(2) for r in range(4) for j in range(4))/8
        torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-6)
        torch.testing.assert_close(torch.autograd.grad(actual,z)[0],torch.autograd.grad(expected,z)[0],
                                   atol=2e-7,rtol=2e-6)
