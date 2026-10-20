"""Scripted outputs test engine plumbing, NOT learned latent reasoning."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

AVAILABLE = all(importlib.util.find_spec(n) for n in ("torch", "transformers"))


@unittest.skipUnless(AVAILABLE, "requires torch and transformers")
class LatentEngineTests(unittest.TestCase):
    def test_actual_engine_natural_forced_and_budget_exits_preserve_answer(self):
        self._check_engine(use_peft=False)

    @unittest.skipUnless(importlib.util.find_spec("peft"), "requires peft")
    def test_peft_engine_natural_forced_and_budget_exits_preserve_answer(self):
        self._check_engine(use_peft=True)

    def _check_engine(self, use_peft):
        import torch
        from unittest.mock import patch
        from hybrid_agent.engine import HybridGenerationEngine
        from hybrid_agent.prompts import THINK_PREFILL
        spec = importlib.util.spec_from_file_location("engine_experiment", Path(__file__).resolve().parents[1] / "scripts/latent_experiment.py")
        experiment = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(experiment)

        tokens = ["PAD", "EOS", "<latent>", "</latent>", "<think>", "</think>",
                  "Plan.", "Done.</think><answer>\\boxed{117}</answer>", "latent-readout"]
        class Tokenizer:
            pad_token_id, eos_token_id, unk_token_id = 0, 1, None
            def convert_tokens_to_ids(self, token):
                return tokens.index(token)
            def get_vocab(self):
                return {t: i for i, t in enumerate(tokens)}
            def __len__(self):
                return len(tokens)
            def encode(self, text, **kwargs):
                return [tokens.index(text)]
            def decode(self, ids, **kwargs):
                return "".join(tokens[i] for i in ids)
        class Backbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(len(tokens), 4)
                self.config = SimpleNamespace(model_type="test", eos_token_id=1)
            def get_input_embeddings(self):
                return self.embedding
        class Model:
            def __init__(self):
                self.backbone = Backbone()
            def run_hybridlink(self, hidden, output_dtype=None):
                return torch.full_like(hidden, 0.125, dtype=output_dtype)

        for forced, budget in ((False, 20), (True, 20), (False, 3)):
            with self.subTest(forced=forced, budget=budget):
                model, tokenizer = Model(), Tokenizer()
                if use_peft:
                    from transformers import Qwen3Config, Qwen3ForCausalLM
                    from hybrid_agent.model import HybridAgentModel
                    from hybrid_agent.lora import enable_format_lora
                    config = Qwen3Config(vocab_size=len(tokens), hidden_size=4,
                                         intermediate_size=8, num_hidden_layers=1,
                                         num_attention_heads=2, num_key_value_heads=1,
                                         head_dim=2, tie_word_embeddings=False)
                    wrapped = HybridAgentModel(Qwen3ForCausalLM(config))
                    enable_format_lora(wrapped, tokenizer, rank=2, alpha=4)
                    with torch.no_grad():
                        for name, p in wrapped.backbone.named_parameters():
                            if "trainable_tokens_delta" in name:
                                p.add_(0.25)
                    wrapped.eval()
                    model.backbone = wrapped.backbone
                    embedding = model.backbone.get_input_embeddings()
                    self.assertEqual(type(embedding).__name__, "TrainableTokensWrapper")
                    with torch.no_grad():
                        expected_exit = embedding(torch.tensor([3]))[0].clone()
                engine = HybridGenerationEngine(model, tokenizer, torch.device("cpu"), None,
                    "", "q", batch_size=1, temperature=0, max_response_tokens=budget,
                    min_latent_steps=0, max_latent_steps=1 if forced else 64,
                    max_tool_calls=0, terminate_on_tool_cap=True)
                schedule = iter([6, 2, 8, 8 if forced else 3, 7, 1, 1])
                fed = []
                def advance():
                    engine.next_logits = torch.full((1, len(tokens)), -100.0)
                    engine.next_logits[0, next(schedule)] = 100.0
                    engine.last_hidden = torch.ones((1, 4))
                def init():
                    engine.prompt_ends_in_think = True
                    engine.initial_prompt = THINK_PREFILL
                    engine.rows[0].in_think = True
                    advance()
                    return True
                def forward(inputs, real_rows):
                    fed.append(inputs.detach().clone())
                    advance()
                with patch.object(engine, "_init_generation", side_effect=init), \
                     patch.object(engine, "_lockstep_forward", side_effect=forward):
                    engine.run()
                result = engine.row_result(engine.rows[0])
                text = experiment.scored_engine_text(engine, tokenizer)
                self.assertEqual(result["latent_steps"], 1)
                torch.testing.assert_close(fed[2], torch.full((1, 1, 4), 0.125))
                scored = experiment.score_hybrid(text, 117, result)
                if budget == 3:
                    self.assertEqual(result["end_reason"], "budget")
                    self.assertFalse(scored["success"])
                else:
                    self.assertIn("\\boxed{117}", text)  # never hide the first answer digit
                    self.assertNotIn("EOS", text)
                    self.assertEqual(result["forced_exits"], int(forced))
                    self.assertEqual(result["anchor_exits"], int(not forced))
                    self.assertEqual(scored["success"], not forced)
                    expected = expected_exit if use_peft else model.backbone.embedding.weight[3]
                    torch.testing.assert_close(fed[3][0, 0], expected)
                    if use_peft:
                        with torch.no_grad():
                            expected_entry = embedding(torch.tensor([2]))[0]
                        torch.testing.assert_close(fed[1][0, 0], expected_entry)


if __name__ == "__main__":
    unittest.main()
