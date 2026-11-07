import unittest

from reproduction.crossfit_contours import partitions, training_partition
from reproduction.data import patient_folds


class OuterTeacherScopeTests(unittest.TestCase):
    def test_full_guides_cover_training_patients_and_exclude_test_patients(self):
        uids = [f"P{p}_slice{s}" for p in range(32) for s in range(3)]
        for fold, outer in enumerate(patient_folds(uids)):
            eligible, excluded = training_partition(uids, fold, "full")
            self.assertEqual(eligible, outer["train_indices"])
            self.assertEqual(excluded, outer["test_indices"])
            held = []
            for fit, predict in partitions(eligible, uids, fold):
                patients = lambda ids: {uids[i].split("_")[0] for i in ids}
                self.assertFalse(patients(fit) & patients(predict))
                self.assertFalse(patients(fit+predict) & patients(excluded))
                held.extend(predict)
            self.assertEqual(sorted(held), sorted(eligible))
            inner, validation = training_partition(uids, fold, "inner")
            self.assertEqual(set(inner+validation), set(eligible))
            self.assertFalse(set(inner) & set(validation))
