import unittest
import torch
from reproduction.reflection_consistency import boundary_loss, reflect_rotated
from reproduction.prompt_preference import segmentation_loss


class ReflectionConsistencyTests(unittest.TestCase):
    def test_conjugate_reflection_at_every_rotation(self):
        x=torch.arange(35).reshape(1,1,5,7)
        for k in range(4):
            rotated=torch.rot90(x,k,(-2,-1))
            expected=torch.rot90(x.flip([-1]),k,(-2,-1))
            torch.testing.assert_close(reflect_rotated(rotated,k),expected)
            torch.testing.assert_close(reflect_rotated(expected,k),rotated)

    def test_loss_only_affects_the_two_sides_of_the_contour(self):
        y=torch.zeros(1,4,24,24)
        y[:,:3,7:17,7:17]=1
        a=torch.zeros_like(y,requires_grad=True)
        b=torch.ones_like(y,requires_grad=True)
        loss=boundary_loss(a,b,y)
        expected=.75*(torch.sigmoid(torch.tensor(1.))-.5)**2
        torch.testing.assert_close(loss,expected)
        loss.backward()
        self.assertNotEqual(float(a.grad[0,0,7,8]),0.)
        self.assertNotEqual(float(b.grad[0,0,5,8]),0.)
        self.assertEqual(float(a.grad[0,0,0,0]),0.)
        self.assertEqual(float(a.grad[0,0,12,12]),0.)
        self.assertFalse(a.grad[:,3].any())

    def test_shared_original_loss_at_zero_logits(self):
        logits=torch.zeros(1,1,2,2)
        y=torch.tensor([[[[1.,0.],[0.,1.]]]])
        # Both soft and true area are two. The smoothed foreground Dice is 3/5.
        expected=1.4*(1.-.6+3*torch.log(torch.tensor(2.)))
        torch.testing.assert_close(segmentation_loss(logits,logits,y),expected)
