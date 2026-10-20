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

    def test_generalization_is_fixed_paired_and_excludes_all_pilot_tuples(self):
        import ast
        import operator
        operations = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul}
        def calculate(node):
            if isinstance(node, ast.Constant):
                return node.value
            return operations[type(node.op)](calculate(node.left), calculate(node.right))

        with TemporaryDirectory() as temp:
            root = Path(temp)
            pilot, first, second = (root / name for name in ("pilot", "first", "second"))
            with contextlib.redirect_stdout(io.StringIO()):
                experiment.prepare(pilot, 6060)
                before = {p.name: p.read_bytes() for p in pilot.iterdir()}
                experiment.prepare_generalization(first, pilot)
                experiment.prepare_generalization(second, pilot)
            self.assertEqual(before, {p.name: p.read_bytes() for p in pilot.iterdir()})
            path, manifest = experiment.verify_data(first, "generalization")
            self.assertEqual(path.read_bytes(), (second / path.name).read_bytes())
            self.assertTrue(manifest["evaluation_only"])
            self.assertEqual(manifest["evaluation"], json.loads(before["experiment.json"])["evaluation"])
            self.assertNotIn("train", manifest["splits"])
            excluded = {tuple(json.loads(line)["operands"])
                        for name, data in before.items() if name.endswith(".jsonl")
                        for line in data.splitlines()}
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len(rows), 32)
            self.assertEqual(len({r["id"] for r in rows}), 32)
            self.assertEqual(len({r["question"] for r in rows}), 32)
            self.assertEqual(sum(r["expected"] < 0 for r in rows), 4)
            for row in rows:
                self.assertNotIn(tuple(row["operands"]), excluded)
                self.assertEqual(row["system"], experiment.SYSTEM)
                self.assertEqual(row["expected"], calculate(ast.parse(row["expression"], mode="eval").body))
                self.assertNotIn("rounds", row)
            for pair in {r["pair_id"] for r in rows}:
                grouped = {r["group"]: r for r in rows if r["pair_id"] == pair}
                self.assertEqual(set(grouped), {"control", "reworded", "parentheses", "new_structure"})
                self.assertEqual(grouped["control"]["expected"], grouped["reworded"]["expected"])
                self.assertNotEqual(grouped["control"]["expected"], grouped["parentheses"]["expected"])
            with self.assertRaises(FileExistsError):
                experiment.prepare_generalization(first, pilot)
            path.write_text(path.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "differs"):
                experiment.verify_data(first, "generalization")

    def test_summary_keeps_format_accuracy_and_truncation_separate(self):
        rows = [
            {"mode": "plain", **experiment.score_output("-12", -12, "plain", "eos")},
            {"mode": "plain", **experiment.score_output("13", -12, "plain", "eos")},
            {"mode": "format", **experiment.score_output("<answer>\\boxed{-12}</answer>", -12, "format", "eos")},
            {"mode": "format", **experiment.score_output("", -12, "format", "limit")},
        ]
        summary = experiment.summarize_results(rows)
        self.assertEqual(summary["plain"], {"count": 2, "answer_correct": 1,
                                          "format_valid": 2, "truncated": 0, "success": 1})
        self.assertEqual(summary["format"], {"count": 2, "answer_correct": 1,
                                           "format_valid": 0, "truncated": 1, "success": 0})

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
