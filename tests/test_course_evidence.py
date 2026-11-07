import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / "scripts/check_course_project.py"
spec = importlib.util.spec_from_file_location("course_check", path)
course = importlib.util.module_from_spec(spec)
spec.loader.exec_module(course)


class CourseEvidenceTests(unittest.TestCase):
    def rows(self):
        return [
            dict(
                condition=c,
                seed=str(s),
                case_index=str(i),
                patient_cluster=str(i // 2),
                GED30="0.1",
                DiceSoft30="0.9",
            )
            for c in ("P", "A")
            for s in course.SEEDS
            for i in range(4)
        ]

    def test_paired_cohort(self):
        arrays, patients = course.validate_cases(self.rows(), slices=4, patients=2)
        self.assertEqual(len(arrays), 6)
        self.assertEqual(patients, [0, 0, 1, 1])

    def test_missing_or_duplicate_case_fails(self):
        rows = self.rows()
        for bad in (rows[:-1], rows + [rows[0]]):
            with self.assertRaises(ValueError):
                course.validate_cases(bad, slices=4, patients=2)

    def test_changed_patient_mapping_fails(self):
        rows = self.rows()
        rows[-1]["patient_cluster"] = "0"
        with self.assertRaisesRegex(ValueError, "mapping"):
            course.validate_cases(rows, slices=4, patients=2)

    def test_nonfinite_score_fails(self):
        rows = self.rows()
        rows[0]["GED30"] = "nan"
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            course.validate_cases(rows, slices=4, patients=2)

    def test_joint_decision_rule(self):
        self.assertEqual(course.classify([[-0.03, -0.01], [0.01, 0.03]]), "supported")
        self.assertEqual(course.classify([[0.01, 0.03], [0.01, 0.03]]), "not_supported")
        self.assertEqual(
            course.classify([[-0.03, -0.01], [-0.03, -0.01]]), "not_supported"
        )
        self.assertEqual(course.classify([[-0.03, 0.01], [0.01, 0.03]]), "inconclusive")
        self.assertEqual(course.classify([[0, 0], [0, 0]]), "inconclusive")


if __name__ == "__main__":
    unittest.main()
