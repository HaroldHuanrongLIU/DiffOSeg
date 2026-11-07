import unittest
from types import SimpleNamespace
from unittest.mock import patch
from reproduction.nested_guides import donors_for
from reproduction import prompt_preference as prompt


class NestedGuideTests(unittest.TestCase):
    def test_each_donor_excludes_target_group_and_every_student_hold(self):
        groups=[[0,3,7],[1,4],[2,5,6]]
        all_ids=set(range(8))
        for j in range(3):
            train,donors=donors_for(j,groups)
            self.assertEqual(train,sorted(all_ids-set(groups[j])))
            for i,k in zip(train,donors,strict=True):
                self.assertNotEqual(k,j)
                self.assertNotIn(i,groups[k])
                self.assertFalse(set(groups[j])&set(groups[k]))

    def test_student_subset_cannot_separate_slices_of_a_patient(self):
        args=SimpleNamespace(data=None,fold=0,scope="inner")
        uids=["a_1","a_2","b_1","c_1"]
        with patch.object(prompt.native,"inputs",return_value=(None,None,None,uids)), patch.object(
            prompt,"training_partition",return_value=([0,1,2],[3])):
            with self.assertRaisesRegex(ValueError,"complete patients"):
                prompt.train(args,fit_subset=[0,2])
            with self.assertRaisesRegex(ValueError,"subset of eligible"):
                prompt.train(args,fit_subset=[0,1,3])
