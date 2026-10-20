"""Reproducible, text-only LoRA format experiment. No latent generation.

Run from the repository root. Preparation/scoring use only the standard library.
Model evaluation imports the installed project and its ML dependencies lazily.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SYSTEM = (
    "Solve the arithmetic question. Do not use tools. Follow this format exactly: "
    "<think>brief plan<latent>calculation steps</latent>brief conclusion</think>"
    "<answer>\\boxed{integer}</answer>. "
    "Keep the answer outside the latent block. Write text inside the latent block."
)
PLAIN_INSTRUCTION = (
    "Follow the standard order of operations. Reply with only the final integer, "
    "without explanation, equations, or punctuation."
)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(a, b, c):
    product, answer = a * b, a * b - c
    return {
        "id": f"mul-sub-{a}-{b}-{c}", "operands": [a, b, c], "expected": answer,
        "system": SYSTEM, "question": f"Calculate {a} * {b} - {c}.",
        "rounds": [{
            "assistant": (
                "<think>Multiply first, then subtract."
                f"<latent>{a} times {b} is {product}. "
                f"{product} minus {c} is {answer}.</latent>"
                f"The result is {answer}.</think><answer>\\boxed{{{answer}}}</answer>"
            ), "tool": None, "tool_result": None,
        }],
    }


def prepare(directory, seed):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = [directory / f"{split}.jsonl" for split in ("train", "validation", "test")]
    if any(p.exists() for p in paths) or (directory / "experiment.json").exists():
        raise FileExistsError("experiment data already exists; use it unchanged or choose a new directory")
    # Prior notebook mul/sub experiments used a<=49. Keep this experiment disjoint.
    pool = [(a, b, c) for a in range(50, 90) for b in range(2, 10) for c in range(1, 20)]
    random.Random(seed).shuffle(pool)
    splits = {"train": pool[:64], "validation": pool[64:80], "test": pool[80:112]}
    assert len({key for values in splits.values() for key in values}) == 112
    for split, keys in splits.items():
        path = directory / f"{split}.jsonl"
        with path.open("x", encoding="utf-8") as stream:
            for key in keys:
                stream.write(json.dumps(record(*key), ensure_ascii=False) + "\n")
        print(f"{split}: {len(keys)} questions", flush=True)
    write_json(directory / "experiment.json", {
        "seed": seed, "task": "a*b-c", "a_range": [50, 89],
        "splits": {name: {"count": len(keys), "sha256": sha256(directory / f"{name}.jsonl")}
                   for name, keys in splits.items()},
        "evaluation": {"sampling": "greedy", "plain_max_tokens": 32, "format_max_tokens": 256},
        "warning": "Synthetic single-template format pilot, not a general reasoning benchmark.",
    })


def prepare_generalization(directory, pilot):
    """Fixed evaluation-only probe; exclude all operand tuples in the pilot."""
    excluded, sources = set(), {}
    for split in ("train", "validation", "test"):
        path, manifest = verify_data(pilot, split)
        sources[split] = sha256(path)
        for line in path.read_text().splitlines():
            if line.strip():
                excluded.add(tuple(json.loads(line)["operands"]))
    directory = Path(directory)
    # A separate, new directory prevents accidental reuse as training data.
    if directory.exists():
        raise FileExistsError("choose a new directory; existing experiments are never overwritten")
    pool = [(a, b, c) for a in range(50, 90) for b in range(2, 10)
            for c in range(1, 20) if (a, b, c) not in excluded]
    random.Random(6063).shuffle(pool)
    if len(pool) < 8:
        raise ValueError("not enough unused operand tuples")
    rows = []
    for i, (a, b, c) in enumerate(pool[:8]):
        variants = [
            ("control", f"Calculate {a} * {b} - {c}.", f"{a}*{b}-{c}", a*b-c),
            ("reworded", (f"What remains if you subtract {c} from the product of {a} and {b}?"
                           if i % 2 == 0 else
                           f"There are {b} boxes with {a} beads each. After removing {c} beads "
                           "in total, how many beads remain?"), f"{a}*{b}-{c}", a*b-c),
            ("parentheses", f"Calculate ({a} - {c}) * {b}.", f"({a}-{c})*{b}", (a-c)*b),
            ("new_structure", f"Calculate {a} + {b} * {c}." if i % 2 == 0 else
                              f"Calculate {c} - {a} * {b}.",
             f"{a}+{b}*{c}" if i % 2 == 0 else f"{c}-{a}*{b}",
             a+b*c if i % 2 == 0 else c-a*b),
        ]
        for group, question, expression, answer in variants:
            rows.append({"id": f"probe-{i}-{group}", "pair_id": f"probe-{i}",
                         "group": group, "operands": [a, b, c], "expression": expression,
                         "expected": answer, "system": SYSTEM, "question": question})
    directory.mkdir(parents=True)
    path = directory / "generalization.jsonl"
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(directory / "experiment.json", {
        "seed": 6063, "evaluation_only": True, "pilot_sha256": sources,
        "splits": {"generalization": {"count": len(rows), "sha256": sha256(path)}},
        "evaluation": manifest["evaluation"],
        "warning": "Eight paired problems in four groups, not 32 independent problems. "
                   "Keep the existing checkpoint fixed; do not train on or tune against this probe. "
                   "The format instruction is unchanged. Intermediate reasoning is not scored.",
    })
    print(f"Prepared {len(rows)} evaluation questions (8 per group) in {directory}", flush=True)


def summarize_results(results):
    return {mode: {"count": sum(r["mode"] == mode for r in results), **{
        key: sum(r[key] for r in results if r["mode"] == mode)
        for key in ("answer_correct", "format_valid", "truncated", "success")
    }} for mode in ("plain", "format")}


def verify_data(directory, split):
    directory = Path(directory)
    manifest = json.loads((directory / "experiment.json").read_text())
    path = directory / f"{split}.jsonl"
    if sha256(path) != manifest["splits"][split]["sha256"]:
        raise ValueError(f"{split} data differs from the prepared experiment")
    return path, manifest


def score_output(raw, expected, mode, end_reason):
    """Score explicit answers only; never search intermediate numbers for a match."""
    if mode == "plain":
        match = re.fullmatch(r"\s*([+-]?[0-9]+)\s*", raw)
        answer = int(match.group(1)) if match else None
        structure = match is not None
    else:
        blocks = re.findall(r"<answer>(.*?)</answer>", raw, flags=re.S)
        boxed = re.fullmatch(r"\s*\\boxed\{([+-]?[0-9]+)\}\s*", blocks[0]) if len(blocks) == 1 else None
        answer = int(boxed.group(1)) if boxed else None
        # This experiment teaches exactly one latent span with plain text on both sides.
        full = re.fullmatch(
            r"\s*<think>([^<>]+)<latent>([^<>]+)</latent>([^<>]+)</think>"
            r"\s*<answer>\s*\\boxed\{[+-]?[0-9]+\}\s*</answer>\s*", raw, flags=re.S,
        )
        structure = full is not None and all(part.strip() for part in full.groups())
    ended = end_reason == "eos"
    correct = answer == expected if answer is not None else False
    return {
        "prediction": answer, "answer_correct": correct, "format_valid": bool(structure and ended),
        "truncated": not ended, "success": bool(ended and structure and correct),
    }


def validate(directory, model, revision):
    """Check every training record with the real tokenizer before loading weights."""
    from transformers import AutoTokenizer
    from hybrid_agent.model import add_anchor_tokens
    from hybrid_sft.data import tokenize_trajectory

    path, manifest = verify_data(directory, "train")
    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
    add_anchor_tokens(tokenizer)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rejected, lengths = [], []
    for row in rows:
        sample = tokenize_trajectory(tokenizer, row["system"], row["question"],
                                     row["rounds"], max_length=1024, text_only=True)
        if sample is None:
            rejected.append(row["id"])
        else:
            lengths.append(sample["n_tokens"])
    print(f"Training records accepted: {len(lengths)}/{len(rows)}", flush=True)
    if rejected or len(rows) != manifest["splits"]["train"]["count"]:
        raise ValueError(f"training data validation failed; rejected IDs: {rejected}")
    print(f"Maximum sequence length: {max(lengths)} / 1024", flush=True)
    print("Validation passed; no model weights loaded and no data modified.", flush=True)


def train(directory, model, revision):
    directory = Path(directory).resolve()
    data, _ = verify_data(directory, "train")
    for split in ("validation", "test"):
        verify_data(directory, split)
    output = directory / "checkpoint"
    log = directory / "train.log"
    if output.exists() or log.exists():
        raise FileExistsError("training output/log exists; refusing to overwrite or silently resume")
    command = [
        sys.executable, "-u", "-m", "hybrid_sft.train", "--stage", "format_activation",
        "--model", model, "--data", str(data), "--output_dir", str(output), "--lora",
        "--lora_rank", "8", "--lora_alpha", "16", "--lr", "1e-4", "--epochs", "2",
        "--batch_size", "1", "--grad_accum", "1", "--max_length", "1024",
        "--num_workers", "0", "--save_steps", "64", "--seed", "6062",
    ]
    if revision:
        command += ["--revision", revision]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    write_json(directory / "training-run.json", {
        "command": command, "code_commit": commit, "train_sha256": sha256(data),
        "warning": "No automatic validation selection or optimizer-state resume.",
    })
    print("Do not run while a model is loaded in the notebook GPU.", flush=True)
    started = time.time()
    with log.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps({"command": command}) + "\n")
        stream.flush()
        with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, bufsize=1, env={**os.environ, "PYTHONUNBUFFERED": "1"}) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                stream.write(line)
                stream.flush()
            code = process.wait()
        stream.write(f"\nreturncode={code} elapsed_seconds={time.time() - started:.1f}\n")
    if code:
        raise subprocess.CalledProcessError(code, command)
    print(f"Saved training log: {log}", flush=True)


def evaluate(directory, model_spec, revision, split, output):
    import torch
    from hybrid_agent.model import load_hybrid_agent
    from hybrid_agent.compat import text_forward
    from hybrid_agent.prompts import build_prompt, THINK_PREFILL

    path, manifest = verify_data(directory, split)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"evaluation output exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model, tokenizer = load_hybrid_agent(model_spec, revision=revision, device=device,
                                       dtype=dtype, require_hybridlink=False)
    backbone = model.backbone
    eos = backbone.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    eos_ids.add(tokenizer.eos_token_id)
    results = []
    with output.open("x", encoding="utf-8") as stream, torch.inference_mode():
        for row in rows:
            for mode in ("plain", "format"):
                if mode == "format":
                    prompt = build_prompt(tokenizer, row["system"], row["question"])
                else:
                    prompt = tokenizer.apply_chat_template(
                        [{"role": "user", "content": row["question"] + " " + PLAIN_INSTRUCTION}],
                        tokenize=False, add_generation_prompt=True, enable_thinking=False,
                    )
                ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
                result = text_forward(backbone, input_ids=ids, use_cache=True, return_dict=True)
                position = ids.shape[1]
                cache = result.past_key_values
                generated, end_reason = [], "limit"
                budget = manifest["evaluation"][f"{mode}_max_tokens"]
                for step in range(budget):
                    logits = result.logits[:, -1].float()
                    if not torch.isfinite(logits).all():
                        raise RuntimeError("nonfinite generation logits")
                    token = logits.argmax().item()
                    if token in eos_ids:
                        end_reason = "eos"
                        break
                    generated.append(token)
                    if step + 1 < budget:
                        result = text_forward(
                            backbone, input_ids=torch.tensor([[token]], device=device),
                            attention_mask=torch.ones((1, position + 1), dtype=torch.long, device=device),
                            position_ids=torch.tensor([[position]], device=device),
                            past_key_values=cache, use_cache=True, return_dict=True,
                        )
                        cache = result.past_key_values
                        position += 1
                        if cache.get_seq_length() != position:
                            raise RuntimeError("cache position mismatch")
                raw = tokenizer.decode(generated, skip_special_tokens=False)
                # The opening think token belongs to the assistant even if supplied by the template.
                scored_text = (THINK_PREFILL if mode == "format" and prompt.endswith(THINK_PREFILL) else "") + raw
                item = {
                    "id": row["id"], "question": row["question"], "expected": row["expected"],
                    **{key: row[key] for key in ("group", "pair_id") if key in row},
                    "mode": mode, "raw": raw, "scored_text": scored_text, "end_reason": end_reason,
                    "output_tokens": len(generated),
                    **score_output(scored_text, row["expected"], mode, end_reason),
                }
                results.append(item)
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
                stream.flush()
                del result, cache
            print(f"Evaluated {len(results) // 2}/{len(rows)}", flush=True)
    summary = {
        "model": model_spec, "base_revision": model.base_model_revision, "split": split,
        "data_sha256": sha256(path), "limits": manifest["evaluation"],
        "torch": torch.__version__, "device": str(device),
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "modes": {},
    }
    summary["modes"] = summarize_results(results)
    if any("group" in r for r in results):
        summary["groups"] = {
            group: summarize_results([r for r in results if r.get("group") == group])
            for group in sorted({r["group"] for r in results if "group" in r})
        }
    write_json(output.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--directory", required=True)
    p.add_argument("--seed", type=int, default=6060)
    p = commands.add_parser("prepare-generalization")
    p.add_argument("--directory", required=True)
    p.add_argument("--pilot", required=True)
    p = commands.add_parser("validate")
    p.add_argument("--directory", required=True)
    p.add_argument("--model", default="Qwen/Qwen3.8-27B")
    p.add_argument("--revision")
    p = commands.add_parser("train")
    p.add_argument("--directory", required=True)
    p.add_argument("--model", default="Qwen/Qwen3.8-27B")
    p.add_argument("--revision")
    p = commands.add_parser("evaluate")
    p.add_argument("--directory", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--revision")
    p.add_argument("--split", choices=("validation", "test", "generalization"), default="validation")
    p.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.directory, args.seed)
    elif args.action == "prepare-generalization":
        prepare_generalization(args.directory, args.pilot)
    elif args.action == "validate":
        validate(args.directory, args.model, args.revision)
    elif args.action == "train":
        train(args.directory, args.model, args.revision)
    else:
        evaluate(args.directory, args.model, args.revision, args.split, args.output)


if __name__ == "__main__":
    main()
