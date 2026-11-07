import unittest
from types import SimpleNamespace
from unittest.mock import patch
from reproduction import outer_seed_ensemble as base
from reproduction import outer_stable_refinement as stable


class FormalStableTests(unittest.TestCase):
    def test_missing_repeatable_fold_prevents_label_access(self):
        with patch.object(stable,"fixed"), patch.object(stable,"verify_seal",side_effect=[{}, {}, FileNotFoundError("fold2")]), patch.object(base.native,"inputs") as labels:
            with self.assertRaises(FileNotFoundError):
                stable.score(SimpleNamespace())
            labels.assert_not_called()

    def test_shared_scorer_refuses_incomplete_collection(self):
        with patch.object(base.native,"inputs") as labels:
            with self.assertRaisesRegex(ValueError,"four seals"):
                base.score_sealed(SimpleNamespace(),[{}, {}, {}],"source")
            labels.assert_not_called()
