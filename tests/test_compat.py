"""Dependency-free checks: python -m unittest discover -s tests -v."""
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load_file(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compat = load_file("compat_under_test", "hybrid_agent/compat.py")
prompts = load_file("prompts_under_test", "hybrid_agent/prompts.py")


class CompatibilityTests(unittest.TestCase):
    def test_legacy_view_hides_peft_collision_without_modifying_checkpoint(self):
        with TemporaryDirectory() as temp:
            directory = Path(temp)
            original = '{"adapter_type": "ln_res_adapter"}'
            (directory / "adapter_config.json").write_text(original)
            (directory / "config.json").write_text('{}')
            (directory / "weights.safetensors").write_bytes(b"weights")
            with compat.backbone_load_path(directory) as view:
                self.assertNotEqual(view, directory)
                self.assertFalse((view / "adapter_config.json").exists())
                self.assertEqual((view / "weights.safetensors").read_bytes(), b"weights")
                self.assertTrue((view / "weights.safetensors").is_symlink())
            self.assertFalse(view.exists())
            self.assertEqual((directory / "adapter_config.json").read_text(), original)

    def test_genuine_peft_config_is_not_hidden(self):
        with TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory / "adapter_config.json").write_text('{"peft_type": "LORA"}')
            with compat.backbone_load_path(directory) as view:
                self.assertEqual(view, directory.resolve())

    def test_config_precedence_and_legacy_default(self):
        with TemporaryDirectory() as temp:
            directory = Path(temp)
            self.assertEqual(compat.read_link_config(directory), {})
            legacy = {"adapter_type": "ln_res_adapter"}
            (directory / compat.LEGACY_LINK_CONFIG).write_text(json.dumps(legacy))
            self.assertEqual(compat.read_link_config(directory), legacy)
            new = {**legacy, "output_scale": 0.012}
            (directory / compat.LINK_CONFIG).write_text(json.dumps(new))
            self.assertEqual(compat.read_link_config(directory), new)

    def test_invalid_scale_is_rejected(self):
        for value in (0, -1, float("inf"), float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                compat.checked_scale(value)
        self.assertEqual(compat.checked_scale(0.012), 0.012)

    def test_qwen_limits(self):
        for model_type in ("qwen3_5", "qwen3_5_text"):
            backbone = SimpleNamespace(config=SimpleNamespace(model_type=model_type))
            compat.check_execution(backbone)
            with self.assertRaisesRegex(ValueError, "batch_size=1"):
                compat.check_execution(backbone, batch_size=2)
            with self.assertRaisesRegex(ValueError, "RL"):
                compat.check_execution(backbone, rl=True)
        compat.check_execution(SimpleNamespace(config=SimpleNamespace(model_type="qwen3")),
                               batch_size=4, rl=True)

    def test_multimodal_text_decoder_selection(self):
        decoder = object()
        backbone = SimpleNamespace(config=SimpleNamespace(model_type="qwen3_5"),
                                   model=SimpleNamespace(language_model=decoder))
        self.assertIs(compat.text_decoder(backbone), decoder)

    def test_text_forward_routes_hidden_and_cache_without_multimodal_wrapper(self):
        class Hidden:
            def __getitem__(self, key):
                return "last-position"
        hidden, cache = Hidden(), object()
        calls = []
        def decoder(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(last_hidden_state=hidden, past_key_values=cache)
        backbone = SimpleNamespace(
            config=SimpleNamespace(model_type="qwen3_5"),
            model=SimpleNamespace(language_model=decoder),
            get_output_embeddings=lambda: lambda value: ("logits", value),
        )
        output = compat.text_forward(
            backbone, inputs_embeds="vectors", cache_position=17,
            output_hidden_states=True, logits_to_keep=1, use_cache=True,
        )
        self.assertEqual(calls, [{"inputs_embeds": "vectors", "use_cache": True}])
        self.assertIs(output.past_key_values, cache)
        self.assertIs(output.hidden_states[0], hidden)
        self.assertEqual(output.logits, ("logits", "last-position"))
        with self.assertRaisesRegex(ValueError, "text inputs only"):
            compat.text_forward(backbone, pixel_values=object())

    def test_prompt_is_explicit_and_rejects_closed_thinking(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                self.messages, self.options = messages, kwargs
                return self.result
        tokenizer = Tokenizer()
        for ending in ("assistant\n", "<think>\n"):
            tokenizer.result = "header\n" + ending
            self.assertEqual(prompts.build_prompt(tokenizer, " sys ", " question "), tokenizer.result)
            self.assertTrue(tokenizer.options["enable_thinking"])
            self.assertEqual(tokenizer.messages[-1]["content"], "question")
        tokenizer.result = "<think>\n\n</think>\n"
        with self.assertRaisesRegex(ValueError, "unsupported chat prefix"):
            prompts.build_prompt(tokenizer, "", "question")


if __name__ == "__main__":
    unittest.main()
