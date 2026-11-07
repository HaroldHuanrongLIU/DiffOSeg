import unittest
import torch
from ddpm.models.diffusion_denoising import DiffusionModel
from reproduction.prior_diffusion import best_prefix, observation_log_ratio, reverse_probability, schedule


class PriorDiffusionTests(unittest.TestCase):
    def test_uniform_posterior_matches_existing_diffusion(self):
        old=DiffusionModel("cosine",32,2)
        clean=torch.tensor([[[[.2,.8]]]])
        xt=torch.tensor([[[[0.,1.]]]])
        reference=old.theta_post_prob(torch.cat((1-xt,xt),1),torch.cat((1-clean,clean),1),torch.tensor([12]))
        actual=reverse_probability(clean,xt,torch.full_like(clean,.5),old.cumalphas[10],old.cumalphas[11])
        torch.testing.assert_close(actual,reference[:,1:],atol=1e-6,rtol=1e-6)

    def test_nonuniform_transition_matches_explicit_bayes_table(self):
        clean=torch.tensor([[[[.65,.65]]]])
        observed=torch.tensor([[[[0.,1.]]]])
        prediction=reverse_probability(clean,observed,torch.full_like(clean,.2),torch.tensor(.7),torch.tensor(.3))
        transition=(.3/.7)*torch.eye(2)+(1-.3/.7)*torch.tensor([[.8,.2],[.8,.2]])
        expected=[]
        for z in (0,1):
            value=0.
            for y,mass in ((0,.35),(1,.65)):
                previous=.7*torch.eye(2)[y]+.3*torch.tensor([.8,.2])
                joint=previous*transition[:,z]
                value+=mass*joint[1]/joint.sum()
            expected.append(value)
        torch.testing.assert_close(prediction.flatten(),torch.stack(expected))
        terminal=reverse_probability(clean,observed,torch.full_like(clean,.2),torch.tensor(1.),torch.tensor(.3))
        torch.testing.assert_close(terminal,clean)
        self.assertEqual(schedule("cpu")[-1].item(),0.)
        torch.testing.assert_close(observation_log_ratio(observed,torch.full_like(clean,.2),torch.tensor(0.)),
                                   torch.zeros_like(clean))

    def test_expected_dice_selects_prefix_and_handles_empty_draws(self):
        probability=torch.tensor([[[[.9,.6,.3,.02]]]])
        samples=torch.tensor([[[[[1,0,0,0]],[[1,1,0,0]],[[1,1,1,0]]]]],dtype=torch.float32)
        mask,field=best_prefix(probability,samples)
        torch.testing.assert_close(mask,torch.tensor([[[[True,True,False,False]]]]))
        torch.testing.assert_close(field>0,mask)
        empty,_=best_prefix(probability,torch.zeros_like(samples))
        self.assertFalse(empty.any())
