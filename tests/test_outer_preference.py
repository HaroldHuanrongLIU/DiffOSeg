import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from reproduction import outer_preference as study
from reproduction import rank_contours as rc


class OuterPreferenceTests(unittest.TestCase):
    def test_fold_local_choice_has_the_registered_minimum_gain(self):
        self.assertEqual(study.select_fold(0.9023125331315617, 0.9025658548018157), "adapted")
        self.assertEqual(study.select_fold(0.9030, 0.9034), "ct-adapted")
        self.assertEqual(study.select_fold(0.904, 0.903), "adapted")

    def test_missing_fold_prevents_access_to_evaluation_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/"selection.json").write_text("{}")
            with patch.object(study,"locked",return_value={"folds":[]}), patch.object(study.native,"inputs") as labels:
                with self.assertRaises(FileNotFoundError):
                    study.score(SimpleNamespace(run=root))
                labels.assert_not_called()

    def test_mutated_predictions_fail_the_seal(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            bank = folder/"predictions.npz"
            np.savez(bank, ids=[4], masks=[0])
            rc.write_json(folder/"sealed.json",{"fold":1,"selection":"lock","source":rc.hash_file(study.__file__),
                                                "payload":rc.hash_file(bank)})
            study.verify_bank(folder,1,"lock")
            np.savez(bank,ids=[5],masks=[1])
            with self.assertRaises(ValueError):
                study.verify_bank(folder,1,"lock")
