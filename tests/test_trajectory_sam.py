import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from reproduction.trajectory_sam import final_identity, select_weight


class TrajectorySAMTests(unittest.TestCase):
    def test_same_last_checkpoint_and_mutation_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            torch.save({"w":torch.tensor([1.,2.])},folder/"ema-5.pt")
            torch.save({"w":torch.tensor([1.,2.])},folder/"final.pt")
            final_identity(folder,5)
            torch.save({"w":torch.tensor([1.,2.01])},folder/"final.pt")
            with self.assertRaisesRegex(ValueError,"snapshot differs"):
                final_identity(folder,5)

    def test_library_selection_prefers_first_optimal_weight(self):
        truth=np.array([[[[1,0]]]],dtype=bool)
        base=np.array([[[[-1,1]]]],dtype=float)
        new=-base
        weight,grid=select_weight(truth,base,new)
        self.assertEqual(weight,.75)
        self.assertEqual(grid,[0.,0.,0.,1.,1.])
        self.assertEqual(select_weight(truth,base,base)[0],0.)
