"""Fixed, no-tools latent pilot. Preparation/scoring require only stdlib.

Run from the repository root. Text control reuses format_experiment.evaluate;
hybrid evaluation uses the production single-request engine without forced entry.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import re
import subprocess
import sys

# Also supports import-by-path in dependency-free tests.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from format_experiment import ROOT, SYSTEM, sha256, write_json, verify_data, score_output, evaluate


def prepare(directory, pilot, generalization, calibration):
    excluded, sources = set(), {}
    for folder, splits in ((pilot, ("train", "validation", "test")),
                           (generalization, ("generalization",))):
        for split in splits:
            path, _ = verify_data(folder, split)
            sources[split] = sha256(path)
            excluded.update(tuple(json.loads(line)["operands"])
                            for line in path.read_text().splitlines() if line.strip())
    scale_info = json.loads(Path(calibration).read_text())
    import math
    scale = float(scale_info["output_scale"])
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("invalid training-calibrated scale")
    if scale_info["data_sha256"] != sources["train"]:
        raise ValueError("calibration must come from the verified training data")
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError("use a new directory; no overwrite or regeneration")
    pool = [(a, b, c) for a in range(50, 90) for b in range(2, 10) for c in range(1, 20)
            if (a, b, c) not in excluded]
    random.Random(6065).shuffle(pool)
    if len(pool) < 24:
        raise ValueError("insufficient unused questions")
    directory.mkdir(parents=True)
    splits = {}
    for split, keys in (("latent-dev", pool[:8]), ("latent-test", pool[8:24])):
        path = directory / f"{split}.jsonl"
        with path.open("x", encoding="utf-8") as stream:
            for a, b, c in keys:
                stream.write(json.dumps({"id": f"latent-{a}-{b}-{c}", "operands": [a, b, c],
                    "question": f"Calculate {a} * {b} - {c}.", "system": SYSTEM,
                    "expected": a*b-c}) + "\n")
        splits[split] = {"count": len(keys), "sha256": sha256(path)}
    write_json(directory / "experiment.json", {
        "seed": 6065, "splits": splits, "excluded_sha256": sources,
        "output_scale": scale, "calibration_sha256": sha256(calibration),
        "evaluation": {"sampling": "greedy", "plain_max_tokens": 32, "format_max_tokens": 256},
        "hybrid": {"max_response_tokens": 256, "min_latent_steps": 0, "max_latent_steps": 64},
        "warning": "Development is diagnostic; test stays reserved until a checkpoint is fixed. "
                   "Same format instruction including text-in-latent wording for all conditions. "
                   "Engine masks differ from ordinary text decoding. No speed or reasoning claim.",
    })
    print(f"Prepared 8 development and 16 reserved test questions in {directory}")


def score_hybrid(text, expected, result):
    # Vector inputs are not text tokens: a genuine hybrid span is visibly empty.
    structure = re.fullmatch(
        r"\s*<think>([^<>]+)<latent></latent>([^<>]+)</think>"
        r"\s*<answer>\s*\\boxed\{[+-]?[0-9]+\}\s*</answer>\s*", text, re.S)
    ended = result["end_reason"] == "eos"
    correct = score_output(text, expected, "format", result["end_reason"])["answer_correct"]
    valid = bool(ended and structure and all(s.strip() for s in structure.groups()))
    entered = result["latent_spans"] == 1 and result["latent_steps"] > 0
    natural = (entered and result["anchor_exits"] == 1 and result["forced_exits"] == 0)
    return {"answer_correct": correct, "format_valid": valid,
            "used_vectors": result["latent_steps"] > 0,
            "natural_exit": natural, "forced_exit": result["forced_exits"] > 0,
            "truncated": result["end_reason"] == "budget", "terminated": ended,
            "success": bool(correct and valid and natural and result["tool_requests"] == 0)}


def scored_engine_text(engine, tokenizer):
    row = engine.rows[0]
    # Preserve structural special tokens; remove only a terminal EOS for scoring.
    ids = list(row.generated_ids)
    if row.end_reason == "eos" and ids and ids[-1] in engine.eos_ids:
        ids.pop()
    text = tokenizer.decode(ids, skip_special_tokens=False)
    from hybrid_agent.prompts import THINK_PREFILL
    return (THINK_PREFILL if engine.prompt_ends_in_think else "") + text


def check_reference_config(model_path, ref):
    # Check tokenizer/base identity. Callers separately verify adapter weights.
    for name in ("hybrid_lora.json", "tokenizer.json", "tokenizer_config.json"):
        actual = json.loads((model_path / name).read_text())
        expected = json.loads((ref / name).read_text())
        if name == "tokenizer_config.json":
            # Loading provenance, not tokenization behavior. Offline loading may
            # persist local_files_only=True. Reject malformed values, and never
            # edit source files or ignore any vocabulary/template settings.
            for config in (actual, expected):
                for key in ("is_local", "local_files_only"):
                    if key in config and type(config[key]) is not bool:
                        raise ValueError(f"checkpoint/reference mismatch: {name}: invalid {key}")
                    config.pop(key, None)
        if actual != expected:
            raise ValueError(f"checkpoint/reference mismatch: {name}")


def evaluate_hybrid(directory, model_spec, reference, split, output):
    import torch
    from safetensors.torch import load_file
    from hybrid_agent.model import load_hybrid_agent
    from hybrid_agent.engine import HybridGenerationEngine
    from hybrid_agent.tools import ToolExecutor

    path, manifest = verify_data(directory, split)
    output = Path(output)
    if output.exists() or output.with_suffix(".summary.json").exists():
        raise FileExistsError("evaluation output already exists")
    model_path, ref = Path(model_spec), Path(reference)
    check_reference_config(model_path, ref)
    before = load_file(str(ref / "lora/adapter_model.safetensors"))
    after = load_file(str(model_path / "lora/adapter_model.safetensors"))
    if before.keys() != after.keys() or any(
        before[k].dtype != after[k].dtype or not torch.equal(before[k], after[k]) for k in before
    ):
        raise ValueError("LoRA or anchor weights differ from the fixed text reference")
    del before, after
    if not torch.cuda.is_available():
        raise RuntimeError("run this large-model pilot on the target GPU")
    model, tokenizer = load_hybrid_agent(model_spec, device="cuda", dtype=torch.bfloat16,
                                       require_hybridlink=True)
    # Identical precision and training-only magnitude for untrained/trained links.
    model.hybridlink.float()
    stored_scale = model.output_scale
    model.output_scale = manifest["output_scale"]
    rows = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with output.open("x", encoding="utf-8") as stream:
        for row in rows:
            engine = HybridGenerationEngine(
                model, tokenizer, torch.device("cuda"), ToolExecutor(), row["system"], row["question"],
                batch_size=1, temperature=0.0, top_p=1.0, top_k=0,
                max_tool_calls=0, terminate_on_tool_cap=True, tool_workers=1,
                **manifest["hybrid"],
            )
            with torch.inference_mode():
                engine.run()
            result = engine.row_result(engine.rows[0])
            text = scored_engine_text(engine, tokenizer)
            item = {"id": row["id"], "question": row["question"], "expected": row["expected"],
                    **result, "scored_text": text, **score_hybrid(text, row["expected"], result)}
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
            stream.flush()
            results.append(item)
            print(f"Evaluated {len(results)}/{len(rows)}", flush=True)
            del engine
    summary = {
        "model": str(model_spec), "reference": str(reference), "split": split,
        "base_revision": model.base_model_revision, "data_sha256": sha256(path),
        "link_sha256": sha256(model_path / "adapter.pt"),
        "lora_sha256": sha256(model_path / "lora/adapter_model.safetensors"),
        "stored_scale": stored_scale, "evaluation_scale": model.output_scale,
        "limits": manifest["hybrid"], "torch": torch.__version__, "count": len(results),
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "counts": {key: sum(r[key] for r in results) for key in (
            "answer_correct", "format_valid", "used_vectors", "natural_exit", "forced_exit",
            "truncated", "terminated", "success")},
        "total_latent_steps": sum(r["latent_steps"] for r in results),
        "note": "Engine has structural token masks; compare learned/untrained links under identical masks. "
                "A correct answer alone does not demonstrate useful latent computation.",
    }
    write_json(output.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--directory", required=True)
    p.add_argument("--pilot", required=True)
    p.add_argument("--generalization", required=True)
    p.add_argument("--calibration", required=True)
    p = commands.add_parser("evaluate")
    p.add_argument("--directory", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--reference", required=True, help="unchanged format checkpoint")
    p.add_argument("--mode", choices=("text", "hybrid"), required=True)
    p.add_argument("--split", choices=("latent-dev", "latent-test"), default="latent-dev")
    p.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.directory, args.pilot, args.generalization, args.calibration)
    elif args.mode == "text":
        if Path(args.model).resolve() != Path(args.reference).resolve():
            parser.error("text control must use the fixed format reference")
        evaluate(args.directory, args.model, None, args.split, args.output)
    else:
        evaluate_hybrid(args.directory, args.model, args.reference, args.split, args.output)


if __name__ == "__main__":
    main()
