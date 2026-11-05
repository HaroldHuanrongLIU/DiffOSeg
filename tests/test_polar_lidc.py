import unittest

import numpy as np
import torch

from reproduction import polar_contours as p
from reproduction import polar_eval as study


class PolarProtocolTests(unittest.TestCase):
    def test_radius_reconstructs_centered_disc(self):
        truth = p.RADIUS <= 14
        mask = torch.from_numpy(truth[None,None])
        radius = p.mask_radii(mask, p.grid(1,"cpu")).numpy()[0,0]
        self.assertLess(abs(radius.mean()-14), 0.25)
        self.assertGreater((truth & (p.angular_map(radius)>p.RADIUS)).sum() /
                           (truth | (p.angular_map(radius)>p.RADIUS)).sum(), 0.96)

    def test_whole_angular_profile_is_sampled_and_zero_noise_has_no_effect(self):
        circular = np.sin(np.arange(p.ANGLES)*2*np.pi/p.ANGLES).astype("float32")
        radius = np.full((4,p.ANGLES),12,dtype="float32")
        masks = p.angular_map(radius) > p.RADIUS
        fields = study.rank_fields(masks,radius,0.5)
        library = np.broadcast_to(circular, (2,4,p.ANGLES))
        reference, noise = study.draw_fields(fields,np.ones(p.ANGLES),library,1,10)
        self.assertLess(np.abs(noise).max(),1.00001)
        np.testing.assert_array_equal(reference>0,(reference+0*noise)>0)
        self.assertFalse((reference+11>0)[:,0].any())

    def test_selection_patients_are_disjoint_from_library(self):
        class Data:
            series_uid = [f"p{i}_s{j}" for i in range(40) for j in range(2)]
        data = Data()
        outer, train, valid = p.rc.split_for(data,0)
        lib, select = p.calibration_split(data,valid,0)
        names = lambda ids: {data.series_uid[i].split("_")[0] for i in ids}
        partitions = list(map(names,[train,lib,select,outer["test_indices"]]))
        for i, subset in enumerate(partitions):
            self.assertTrue(subset)
            for rest in partitions[i+1:]:
                self.assertTrue(subset.isdisjoint(rest))
        self.assertEqual(set(valid),set(lib+select))

    def test_area_rank_assignment_is_aligned_to_sample_index(self):
        prior = np.zeros((10,1,10),bool)
        for i in range(10):
            prior[i,0,:9-i] = True
        ranks=study.rank_order(10,prior,np.random.default_rng(1))
        self.assertEqual(ranks.tolist(),[3,3,3,2,2,1,1,1,0,0])


if __name__ == "__main__":
    unittest.main()
