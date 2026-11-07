import unittest
import torch
from reproduction.iterative_prompt import two_pass


class IterativePromptTests(unittest.TestCase):
    def test_two_calls_use_the_same_slice_and_only_final_call_has_gradients(self):
        scale=torch.nn.Parameter(torch.tensor(2.))
        guide=torch.tensor([1.,-1.]).reshape(2,1,1,1)
        image=torch.tensor([.2,.6]).reshape_as(guide)
        embedding=torch.tensor([3.,4.]).reshape_as(guide)
        calls=[]
        def decoder(emb,g,x):
            self.assertIs(emb,embedding)
            self.assertIs(x,image)
            calls.append((torch.is_grad_enabled(),g.clone()))
            return scale*g,g
        final,raw=two_pass(decoder,embedding,guide,image,True)
        self.assertEqual([v for v,_ in calls],[False,True])
        torch.testing.assert_close(calls[0][1],guide)
        torch.testing.assert_close(calls[1][1],1.5*guide)
        torch.testing.assert_close(raw,1.5*guide)
        torch.testing.assert_close(final,3*guide)
        final[0].sum().backward()
        self.assertAlmostEqual(scale.grad.item(),1.5)

    def test_matched_control_ignores_first_output_but_runs_twice(self):
        calls=[]
        def decoder(embedding,guide,image):
            calls.append(guide.clone())
            return guide+4,guide
        g=torch.tensor([0.,2.])
        final,raw=two_pass(decoder,None,g,None,False)
        self.assertEqual(len(calls),2)
        torch.testing.assert_close(calls[0],calls[1])
        torch.testing.assert_close(final,g+4)
        torch.testing.assert_close(raw,g)
