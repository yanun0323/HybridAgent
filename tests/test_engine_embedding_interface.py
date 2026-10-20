"""Exercise the real constructor without importing optional model dependencies."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


class EngineEmbeddingInterfaceTests(unittest.TestCase):
    def test_constructor_accepts_embedding_weight_without_num_embeddings(self):
        path = Path(__file__).resolve().parents[1] / "hybrid_agent/engine.py"
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "HybridGenerationEngine")
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        import __future__
        ns = {"check_execution": lambda *a, **kw: None,
              "special_token_id": lambda tok, tag: {"<latent>": 2, "</latent>": 3, "<think>": 4, "</think>": 5}[tag],
              "ENTRY_ANCHOR": "<latent>", "EXIT_ANCHOR": "</latent>",
              "THINK_OPEN": "<think>", "THINK_CLOSE": "</think>",
              "TOOL_STOP_TAGS": (), "RowState": lambda **kw: kw}
        exec(compile(ast.Module(body=[init], type_ignores=[]), str(path), "exec",
                     flags=__future__.annotations.compiler_flag), ns)
        sizes = []
        class Engine:
            _collect_eos_ids = lambda self: {1}
            def _bool_mask(self, ids, vocab):
                sizes.append(vocab)
        embedding = SimpleNamespace(weight=SimpleNamespace(shape=(34, 16)))
        model = SimpleNamespace(backbone=SimpleNamespace(get_input_embeddings=lambda: embedding))
        tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=1)
        engine = Engine()
        ns["__init__"](engine, model, tokenizer, "cpu", None, "", "q", batch_size=1)
        self.assertEqual(sizes, [34, 34])
        self.assertIs(engine.embed_layer, embedding)


if __name__ == "__main__":
    unittest.main()
