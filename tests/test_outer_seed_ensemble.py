import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from reproduction import outer_seed_ensemble as study


class FormalSeedTests(unittest.TestCase):
    def test_scoring_waits_for_all_four_prediction_seals(self):
        with patch.object(study,"fixed"), patch.object(study,"verify_seal",side_effect=[{}, {}, FileNotFoundError("fold 2")]), patch.object(study.native,"inputs") as load:
            with self.assertRaises(FileNotFoundError):study.score(SimpleNamespace())
            load.assert_not_called()

    def test_teacher_scope_is_by_patient_and_exhaustive(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);split={"train_indices":list(range(6)),"test_indices":[6,7]}
            args=SimpleNamespace(crossfit=root,fold=0)
            uids=[f"patient{i}_slice" for i in range(8)]
            bank=root/"oof-0.npz"
            np.savez(bank,ids=np.arange(6))
            checks=[]
            for j in range(3):
                folder=root/f"fold-0-sub-{j}";folder.mkdir()
                (folder/"final.pt").write_bytes(bytes([j]))
                sha=study.rc.hash_file(folder/"final.pt");checks.append(sha)
                held=[2*j,2*j+1]
                plan={"scope":"full","fold":0,"sub":j,"fit_indices":[i for i in range(6) if i not in held],
                      "held_indices":held,"excluded_indices":[6,7]}
                (folder/"plan.json").write_text(json.dumps(plan))
                (folder/"fit.json").write_text(json.dumps({"checkpoint":sha}))
            receipt={"scope":"full","fold":0,"payload":study.rc.hash_file(bank),"teachers":checks}
            bank.with_suffix(".json").write_text(json.dumps(receipt))
            self.assertEqual(study.training_bank(args,split,uids),receipt)
            uids[2]=uids[0]
            with self.assertRaisesRegex(ValueError,"exposure or identity"):
                study.training_bank(args,split,uids)
