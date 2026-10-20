"""Run the actual pure tokenization functions without importing GPU dependencies.

AST loading excludes dataset/collation classes and imports only; the tested
functions are compiled unchanged from hybrid_sft/data.py. A character tokenizer
isolates format handling, not real Qwen tokenization (verified separately on GPU).
"""
import ast
import importlib.util
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prompts = load_file("trajectory_prompts", ROOT / "hybrid_agent/prompts.py")
experiment = load_file("trajectory_experiment", ROOT / "scripts/format_experiment.py")
source = ROOT / "hybrid_sft/data.py"
names = {"_first_round_ok", "_tokenize_assistant_text", "tokenize_trajectory"}
tree = ast.parse(source.read_text())
functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
assert len(functions) == len(names)
namespace = {
    "THINK_PREFILL": prompts.THINK_PREFILL, "build_prompt": prompts.build_prompt,
    "ENTRY_ANCHOR": "<latent>", "EXIT_ANCHOR": "</latent>",
    "EXIT_ANCHOR_CE_WEIGHT": 2.0,
    "_LATENT_SPAN_RE": re.compile(r"<latent>(.*?)</latent>", re.S),
}
import __future__
exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec",
             flags=__future__.annotations.compiler_flag), namespace)


class Tokenizer:
    eos_token_id = 999
    def __init__(self, prefill=True):
        self.prefill = prefill
    def apply_chat_template(self, *args, **kwargs):
        return "assistant\n" + ("<think>\n" if self.prefill else "")
    def __call__(self, text, **kwargs):
        special = {"<latent>": 1000, "</latent>": 1001}
        return {"input_ids": [special[text]] if text in special else list(map(ord, text))}


class TrajectoryFormatTests(unittest.TestCase):
    def tokenize(self, record, prefill=True, text_only=True):
        return namespace["tokenize_trajectory"](
            Tokenizer(prefill), record["system"], record["question"],
            record["rounds"], 1024, text_only=text_only,
        )

    def test_generated_record_is_accepted_without_losing_first_character(self):
        record = experiment.record(50, 3, 1)
        sample = self.tokenize(record)
        self.assertIsNotNone(sample, "generated experiment record was discarded")
        suffix = sample["input_ids"][sample["prompt_len"]:]
        self.assertEqual(suffix[:8], list(map(ord, "Multiply")))
        self.assertTrue(all(sample["loss_mask"][sample["prompt_len"]:]))

    def test_optional_newline_and_prompt_prefill(self):
        for newline in ("", "\n"):
            for prefill in (False, True):
                with self.subTest(newline=newline, prefill=prefill):
                    record = experiment.record(50, 3, 1)
                    record["rounds"][0]["assistant"] = record["rounds"][0]["assistant"].replace(
                        "<think>", "<think>" + newline, 1)
                    sample = self.tokenize(record, prefill=prefill, text_only=False)
                    self.assertIsNotNone(sample)
                    suffix = sample["input_ids"][sample["prompt_len"]:]
                    expected = "Multiply" if prefill else "<think>" + newline + "Multiply"
                    self.assertEqual(suffix[:len(expected)], list(map(ord, expected)))
                    entry, exit = suffix.index(1000), suffix.index(1001)
                    mask = sample["latent_mask"][sample["prompt_len"]:]
                    self.assertTrue(all(mask[entry:exit]))
                    self.assertFalse(any(mask[exit:]))

    def test_missing_think_open_is_still_rejected(self):
        record = experiment.record(50, 3, 1)
        record["rounds"][0]["assistant"] = "<answer>149</answer>"
        self.assertIsNone(self.tokenize(record))


if __name__ == "__main__":
    unittest.main()
