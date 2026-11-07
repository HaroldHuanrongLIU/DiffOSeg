import math
import unittest
import torch
from reproduction.joint_prompt import StateDistribution


class JointPromptTests(unittest.TestCase):
    def test_zero_interactions_preserve_independent_logits(self):
        torch.manual_seed(54)
        unary=torch.randn(2,4,5,6)*3
        residual=torch.zeros(2,11,5,6)
        rank,support,law=StateDistribution()(unary,residual)
        torch.testing.assert_close(rank,unary,atol=2e-6,rtol=1e-5)
        torch.testing.assert_close(law.exp().sum(1),torch.ones(2,5,6))
        torch.testing.assert_close(rank.sigmoid().sum(1),support.sigmoid().sum(1))

    def test_non_nested_state_has_the_expected_marginals(self):
        head=StateDistribution()
        unary=torch.zeros(1,4,1,1,requires_grad=True)
        residual=torch.zeros(1,11,1,1)
        residual[0,int(head.index[10])-1,0,0]=math.log(4)
        residual.requires_grad_(True)
        rank,support,law=head(unary,residual)
        torch.testing.assert_close(rank.sigmoid().flatten(),torch.tensor([11,8,11,8])/19)
        torch.testing.assert_close(support.sigmoid().flatten(),torch.tensor([18,14,5,1])/19)
        torch.testing.assert_close(law[0,10,0,0].exp(),torch.tensor(4/19))
        (-law[0,10,0,0]).backward()
        self.assertGreater(unary.grad.abs().sum().item(),0)
        self.assertGreater(residual.grad.abs().sum().item(),0)
