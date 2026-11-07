import unittest
import torch
from reproduction import rank_clicks as points


class RankClickTest(unittest.TestCase):
    def test_common_and_personal_share_coordinates_but_not_signs(self):
        g=torch.full((1,4,16,16),-3.)
        g[:,:,6:10,6:10]=3.
        g[0,:,6,9]=torch.tensor([4.,-2.,4.,-2.])
        c0,l0=points.disagreement_points(g,False)
        c1,l1=points.disagreement_points(g,True)
        torch.testing.assert_close(c0,c1,atol=0,rtol=0)
        used=l1[0]>=0
        self.assertEqual(int(used.sum()),1)
        torch.testing.assert_close(c1[0,used][0],torch.tensor([303.5,207.5]))
        self.assertEqual(l0[:,used].flatten().tolist(),[1,1,1,1])
        self.assertEqual(l1[:,used].flatten().tolist(),[1,0,1,0])
        self.assertTrue((l1[:,~used]==-1).all())

    def test_no_disagreement_and_batch_independence(self):
        a=torch.full((1,4,16,16),-1.)
        b=torch.randn(1,4,16,16,generator=torch.Generator().manual_seed(813))
        for mode in (False,True):
            c,l=points.disagreement_points(torch.cat((a,b)),mode)
            x,y=points.disagreement_points(b,mode)
            self.assertTrue((l[:4]==-1).all())
            torch.testing.assert_close(c[4:],x);torch.testing.assert_close(l[4:],y)

    def test_each_rank_receives_its_own_prompt_and_mask_token(self):
        x=torch.arange(2*4*4).reshape(8,4,1,1)
        result=points.select_rank(x).flatten(1)
        self.assertEqual(result.tolist(),[[0,5,10,15],[16,21,26,31]])
