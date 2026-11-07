import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from reproduction import crossfit_contours as teachers
from reproduction import crossfit_residual as bank
from reproduction import rank_contours as rc


class OuterOOFBankTests(unittest.TestCase):
    def test_full_entry_writes_complete_bank_and_returns(self):
        uids = [f"P{p}_s0" for p in range(20)]
        images = np.zeros((20,128,128),np.uint8)
        truth = np.ones((20,4,128,128),np.uint8)
        eligible, excluded = teachers.training_partition(uids, 1, "full")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/"source/reproduction"
            source.mkdir(parents=True)
            (source/"crossfit_contours.py").write_bytes(Path(teachers.__file__).read_bytes())
            for j,(fit,held) in enumerate(teachers.partitions(eligible,uids,1)):
                folder = root/f"fold-1-sub-{j}"
                folder.mkdir()
                checkpoint = folder/"final.pt"
                checkpoint.write_bytes(bytes([j]))
                rc.write_json(folder/"plan.json",{"fit_indices":fit,"held_indices":held,
                              "excluded_indices":excluded,"scope":"full"})
                rc.write_json(folder/"fit.json",{"source":rc.hash_file(teachers.__file__),
                              "steps":teachers.STEPS,"checkpoint":rc.hash_file(checkpoint),"shift":[-1,-1]})
            args = SimpleNamespace(run=root,fold=1,scope="full",data=root/"synthetic")
            with patch.object(bank.native,"inputs",return_value=(images,truth,None,uids)), patch.object(
                    bank,"teacher_logits",side_effect=lambda checkpoint,im,shift:np.ones((len(im),4,128,128),np.float32)):
                bank.prepare(args)
            with np.load(root/"oof-1.npz",allow_pickle=False) as stored:
                self.assertEqual(stored["ids"].tolist(),sorted(eligible))
            receipt = bank.json.loads((root/"oof-1.json").read_text())
            self.assertEqual(receipt["oof_cases"],len(eligible))
            self.assertEqual(receipt["oof_preference_dice"],1.0)
            self.assertEqual(receipt["payload"],rc.hash_file(root/"oof-1.npz"))
