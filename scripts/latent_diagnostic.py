"""Controlled vector substitution, not autonomous generation or evaluation.

Only the first two latent-dev questions; no training and no reserved test access.
The question's known reasoning trajectory supplies prefix, length and token targets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from format_experiment import ROOT, record, verify_data, sha256, write_json
from latent_experiment import check_reference_config


def span_bounds(sample, entry, exit):
    # System instructions also contain anchors: search assistant positions only.
    ids = sample["input_ids"]
    start = ids.index(entry, sample["prompt_len"])
    end = ids.index(exit, start + 1)
    if end <= start + 1 or entry in ids[start + 1:end]:
        raise ValueError("expected one nonempty reference latent span")
    return start, end


def replacement_count(condition, length):
    if condition == "teacher":
        return 0
    if condition == "first-1":
        return min(1, length)
    if condition == "first-4":
        return min(4, length)
    if condition == "all":
        return length
    raise ValueError("unknown substitution condition")


def run_condition(model, tokenizer, sample, entry, exit, condition):
    import torch
    import torch.nn.functional as F
    from hybrid_agent.compat import text_forward

    device = next(model.backbone.parameters()).device
    embed = model.backbone.get_input_embeddings()
    start, end = span_bounds(sample, entry, exit)
    ids = sample["input_ids"]
    length = end - start - 1
    replace = replacement_count(condition, length)
    # Rebuild a fresh cache for every condition, always using identical segmentation:
    # one prompt prefill, then one embedding per assistant token, including entry.
    prompt_len = sample["prompt_len"]
    out = text_forward(model.backbone,
                       input_ids=torch.tensor([ids[:prompt_len]], device=device),
                       use_cache=True, return_dict=True)
    position = prompt_len

    def feed(vector):
        nonlocal out, position
        out = text_forward(model.backbone, inputs_embeds=vector.reshape(1, 1, -1),
            past_key_values=out.past_key_values,
            attention_mask=torch.ones((1, position + 1), dtype=torch.long, device=device),
            position_ids=torch.tensor([[position]], device=device),
            use_cache=True, return_dict=True)
        position += 1
        if out.past_key_values.get_seq_length() != position:
            raise RuntimeError("cache position mismatch")

    def embedding(token):
        return embed(torch.tensor([[token]], device=device))

    for token in ids[prompt_len:start + 1]:
        feed(embedding(token))
    steps = []
    for step in range(length + 1):
        target = ids[start + 1 + step]  # final target is the exit anchor
        logits = out.logits[0, -1].float()
        if not torch.isfinite(logits).all():
            raise RuntimeError("nonfinite logits")
        probs = logits.softmax(-1)
        predicted_id = logits.argmax().item()
        target_embed = embedding(target)
        prediction = model.run_hybridlink(out.hidden_states[-1][:, -1:, :],
                                         output_dtype=target_embed.dtype)
        if not torch.isfinite(prediction).all():
            raise RuntimeError("nonfinite latent vector")
        p, t = prediction.float().flatten(), target_embed.float().flatten()
        step_result = {
            "step": step, "target_id": target, "target_text": tokenizer.decode([target], skip_special_tokens=False),
            "predicted_id": predicted_id, "predicted_text": tokenizer.decode([predicted_id], skip_special_tokens=False),
            "target_probability": probs[target].item(), "exit_probability": probs[exit].item(),
            "exit_rank": int((logits > logits[exit]).sum().item()) + 1,
            "exit_argmax": predicted_id == exit,
            "cosine": F.cosine_similarity(p, t, dim=0).item(),
            "prediction_rms": p.square().mean().sqrt().item(),
            "target_rms": t.square().mean().sqrt().item(),
            "fed": "stop-before-exit" if step == length else ("predicted" if step < replace else "teacher"),
        }
        steps.append(step_result)
        if step < length:
            feed(prediction if step < replace else target_embed)

    # Force the real exit anchor at reference length in ALL conditions. This
    # continuation tests recovery, not whether the model learned when to exit.
    feed(embedding(exit))
    eos = model.backbone.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    eos_ids.add(tokenizer.eos_token_id)
    continuation, reason = [], "limit"
    for i in range(96):
        logits = out.logits[0, -1].float()
        if not torch.isfinite(logits).all():
            raise RuntimeError("nonfinite continuation logits")
        token = logits.argmax().item()
        if token in eos_ids:
            reason = "eos"
            break
        continuation.append(token)
        if i < 95:
            feed(embedding(token))
    return {"condition": condition, "reference_latent_tokens": length,
            "replaced_tokens": replace, "steps": steps,
            "continuation_after_forced_exit": tokenizer.decode(continuation, skip_special_tokens=False),
            "continuation_end_reason": reason,
            "expected_text_continuation": tokenizer.decode(ids[end + 1:-1], skip_special_tokens=False)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path, manifest = verify_data(args.directory, "latent-dev")
    output = Path(args.output)
    if output.exists() or output.with_suffix(".summary.json").exists():
        raise FileExistsError("diagnostic output exists; keep it unchanged")
    check_reference_config(Path(args.model), Path(args.reference))
    import torch
    from safetensors.torch import load_file
    from hybrid_agent.model import load_hybrid_agent
    from hybrid_sft.data import tokenize_trajectory

    a = load_file(str(Path(args.model) / "lora/adapter_model.safetensors"))
    b = load_file(str(Path(args.reference) / "lora/adapter_model.safetensors"))
    if a.keys() != b.keys() or any(a[k].dtype != b[k].dtype or not torch.equal(a[k], b[k]) for k in a):
        raise ValueError("LoRA/anchor weights differ from reference")
    del a, b
    if not torch.cuda.is_available():
        raise RuntimeError("run the 27B diagnostic on the target GPU")
    model, tokenizer = load_hybrid_agent(args.model, device="cuda", dtype=torch.bfloat16, require_hybridlink=True)
    model.hybridlink.float()
    model.output_scale = manifest["output_scale"]
    entry, exit = (tokenizer.convert_tokens_to_ids(t) for t in ("<latent>", "</latent>"))
    rows = [json.loads(s) for s in path.read_text().splitlines() if s.strip()][:2]
    output.parent.mkdir(parents=True, exist_ok=True)
    summaries = []
    with output.open("x", encoding="utf-8") as stream, torch.inference_mode():
        for row in rows:
            reference = record(*row["operands"])
            if reference["question"] != row["question"] or reference["expected"] != row["expected"]:
                raise ValueError("development question is not supported by this reference trajectory")
            sample = tokenize_trajectory(tokenizer, row["system"], row["question"], reference["rounds"], 1024)
            if sample is None:
                raise ValueError("reference trajectory rejected")
            for condition in ("teacher", "first-1", "first-4", "all"):
                result = run_condition(model, tokenizer, sample, entry, exit, condition)
                item = {"id": row["id"], "question": row["question"], "expected": row["expected"], **result}
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
                stream.flush()
                summaries.append({k: v for k, v in item.items() if k != "steps"} | {
                    "first_step": result["steps"][0], "exit_step": result["steps"][-1],
                    "early_exit_steps": [s["step"] for s in result["steps"][:-1] if s["exit_argmax"]],
                })
                print(f"{row['id']} {condition}: exit_rank={result['steps'][-1]['exit_rank']}", flush=True)
    summary = {"model": args.model, "base_revision": model.base_model_revision,
        "data_sha256": sha256(path), "output_scale": model.output_scale,
        "link_sha256": sha256(Path(args.model) / "adapter.pt"),
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "torch": torch.__version__, "results": summaries,
        "warning": "Known prefix, reference length and real exit anchor are forced. "
                   "All conditions use the same sequential cache segmentation. "
                   "Continuation is ordinary text, including any emitted latent tags; NOT autonomous latent reasoning."}
    write_json(output.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
