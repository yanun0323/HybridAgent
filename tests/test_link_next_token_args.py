"""Check smoke-only limits without importing model libraries."""
import argparse
import ast
import contextlib
import io
from pathlib import Path
import sys
from unittest.mock import patch
import unittest


class LinkNextTokenArgsTests(unittest.TestCase):
    def test_smoke_limits_and_scale_reuse(self):
        path = Path(__file__).resolve().parents[1] / "hybrid_sft/train.py"
        tree = ast.parse(path.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "parse_args")
        ns = {"argparse": argparse, "FORMAT_ACTIVATION": "format_activation",
              "HYBRIDLINK_ACTIVATION": "hybridlink_activation", "LINK_NEXT_TOKEN": "link_next_token",
              "STAGES": ("format_activation", "hybridlink_activation", "link_next_token"),
              "STAGE_EPOCHS": {"link_next_token": 1},
              "STAGE_HYBRIDLINK_LR": {"link_next_token": 5e-5}, "NUM_WORKERS": 4}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), ns)
        base = ["train", "--stage", "link_next_token", "--model", "source", "--data", "train.jsonl",
                "--output_dir", "new", "--max_steps", "1", "--batch_size", "1", "--grad_accum", "1",
                "--link_calibration", "scale.json"]
        with patch.object(sys, "argv", base):
            args = ns["parse_args"]()
        self.assertEqual(args.hybridlink_lr, 5e-5)
        self.assertEqual(args.link_calibration, "scale.json")
        for extra in (["--max_steps", "64"], ["--batch_size", "2"], ["--grad_accum", "2"],
                      ["--lora"], ["--calibrate_link_scale"]):
            with self.subTest(extra=extra), patch.object(sys, "argv", base + extra), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    ns["parse_args"]()


if __name__ == "__main__":
    unittest.main()
