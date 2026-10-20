"""No model dependencies: data isolation and strict evaluation regression tests."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/format_experiment.py"
spec = importlib.util.spec_from_file_location("format_experiment", SCRIPT)
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


class FormatExperimentTests(unittest.TestCase):
    def test_data_is_disjoint_reproducible_and_arithmetically_correct(self):
        with TemporaryDirectory() as first, TemporaryDirectory() as second:
            with contextlib.redirect_stdout(io.StringIO()):
                experiment.prepare(first, 6060)
                experiment.prepare(second, 6060)
            seen = set()
            for split, count in (("train", 64), ("validation", 16), ("test", 32)):
                path, _ = experiment.verify_data(first, split)
                self.assertEqual(path.read_bytes(), (Path(second) / path.name).read_bytes())
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                self.assertEqual(len(rows), count)
                for row in rows:
                    a, b, c = row["operands"]
                    self.assertGreaterEqual(a, 50)
                    self.assertNotIn(row["id"], seen)
                    seen.add(row["id"])
                    self.assertEqual(row["expected"], a * b - c)
                    answer = row["rounds"][0]["assistant"]
                    self.assertTrue(experiment.score_output(answer, a*b-c, "format", "eos")["success"])
            with self.assertRaises(FileExistsError):
                experiment.prepare(first, 1)
            path.write_text(path.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "differs"):
                experiment.verify_data(first, "test")

    def test_never_extracts_intermediate_arithmetic_as_final_answer(self):
        score = experiment.score_output("270 - 7 = 263", 263, "plain", "eos")
        self.assertIsNone(score["prediction"])
        self.assertFalse(score["format_valid"])
        self.assertFalse(score["success"])

    def test_answer_and_structure_are_scored_separately(self):
        raw = "<answer>\\boxed{169}</answer>"
        score = experiment.score_output(raw, 169, "format", "eos")
        self.assertTrue(score["answer_correct"])
        self.assertFalse(score["format_valid"])
        self.assertFalse(score["success"])

    def test_duplicate_blocks_and_empty_latent_fail(self):
        raw = experiment.record(50, 3, 1)["rounds"][0]["assistant"]
        for broken in (raw + "<answer>\\boxed{149}</answer>",
                       raw.replace("50 times 3 is 150. 150 minus 1 is 149.", ""),
                       raw.replace("</latent>", ""), raw.replace("<latent>", "<python>")):
            with self.subTest(broken=broken):
                self.assertFalse(experiment.score_output(broken, 149, "format", "eos")["success"])

    def test_budget_limit_is_not_success_even_with_correct_answer(self):
        score = experiment.score_output("169", 169, "plain", "limit")
        self.assertTrue(score["answer_correct"])
        self.assertTrue(score["truncated"])
        self.assertFalse(score["success"])

    def test_system_prompt_numbers_are_not_part_of_scoring(self):
        raw = experiment.record(50, 3, 1)["rounds"][0]["assistant"]
        wrong = raw.replace("\\boxed{149}", "\\boxed{150}")
        score = experiment.score_output(wrong, 149, "format", "eos")
        self.assertTrue(score["format_valid"])
        self.assertFalse(score["answer_correct"])


if __name__ == "__main__":
    unittest.main()
