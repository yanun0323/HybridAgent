"""One averaged-gradient update on eight fixed training records, not a benchmark.

The next eight training records are observation-only for THIS update. All sixteen
were available to earlier training. No development/test data is opened.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from format_experiment import ROOT, verify_data, write_json, sha256


def select_rows(rows):
    if len(rows) < 16:
        raise ValueError("need at least 16 training rows")
    selected = rows[:16]
    ids = [r["id"] for r in selected]
    if len(set(ids)) != 16:
        raise ValueError("training record IDs must be unique")
    return selected[:8], selected[8:]


def average_backward(model, batches, stage, backward):
    """One graph per record; no optimizer step here, no graph held across records."""
    import torch
    if not batches:
        raise ValueError("no update batches")
    metrics = []
    for batch in batches:
        out = model(**batch, stage=stage)
        if not torch.isfinite(out["loss"]):
            raise RuntimeError("nonfinite training loss")
        metrics.append({"ce": out["loss"].detach().float().item(),
                        "target_id": out["target_id"].item(),
                        "target_position": out["target_position"],
                        "substitution_position": out["substitution_position"]})
        backward(out["loss"] / len(batches))
        del out
    return metrics


def comparison(rows, before, after):
    if len(rows) != len(before) or len(rows) != len(after):
        raise ValueError("comparison row counts differ")
    result = []
    for row, old, new in zip(rows, before, after):
        for key in ("target_id", "target_position", "substitution_position"):
            if old[key] != new[key]:
                raise ValueError("objective positions changed")
        result.append({"id": row["id"], "question": row["question"],
                       "target_id": old["target_id"], "target_position": old["target_position"],
                       "substitution_position": old["substitution_position"],
                       "before_ce": old["ce"], "after_ce": new["ce"],
                       "delta_ce": new["ce"] - old["ce"]})
    return {"count": len(rows), "mean_before_ce": sum(r["before_ce"] for r in result) / len(rows),
            "mean_after_ce": sum(r["after_ce"] for r in result) / len(rows),
            "lower_ce": sum(r["delta_ce"] < -1e-6 for r in result),
            "higher_ce": sum(r["delta_ce"] > 1e-6 for r in result),
            "unchanged_within_1e-6": sum(abs(r["delta_ce"]) <= 1e-6 for r in result),
            "rows": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", required=True)
    parser.add_argument("--model", required=True, help="original 64-step link checkpoint, not the single-example update")
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    source, output = Path(args.model).resolve(), Path(args.output_dir).resolve()
    if output.exists() or output == source or source in output.parents or output in source.parents:
        raise ValueError("use a new output directory separate from the source")
    data, _ = verify_data(args.pilot, "train")
    rows = [json.loads(line) for line in data.read_text().splitlines() if line.strip()]
    update_rows, observe_rows = select_rows(rows)
    selected = update_rows + observe_rows
    # Fail before loading 27B if the selected source is not the intended baseline.
    source_args = json.loads((source / "train_args.json").read_text())
    if source_args["stage"] != "hybridlink_activation" or source_args.get("max_steps") != 64:
        raise ValueError("start from the original 64-step hybridlink_activation checkpoint")

    import torch
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    from hybrid_agent.model import add_anchor_tokens
    from hybrid_agent.lora import is_lora_model
    from hybrid_sft.calibration import read_training_calibration
    from hybrid_sft.data import tokenize_trajectory, collate_batch
    from hybrid_sft.model import HybridSFTModel, LINK_NEXT_TOKEN
    from hybrid_sft.train import ADAM_BETAS, WEIGHT_DECAY, MAX_GRAD_NORM

    calibration = read_training_calibration(args.calibration, data)
    source_config = json.loads((source / "hybridlink_config.json").read_text())
    if source_config["output_scale"] != calibration["output_scale"]:
        raise ValueError("source scale differs from the fixed training calibration")
    accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=1)
    if (accelerator.device.type != "cuda" or accelerator.num_processes != 1
            or str(accelerator.distributed_type).endswith("DEEPSPEED")):
        raise ValueError("run with one GPU process, no DeepSpeed")
    set_seed(6064)
    tokenizer = AutoTokenizer.from_pretrained(str(source))
    add_anchor_tokens(tokenizer)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    batches = []
    for row in selected:
        sample = tokenize_trajectory(tokenizer, row["system"], row["question"], row["rounds"], 1024)
        if sample is None:
            raise ValueError(f"record rejected: {row['id']}")
        positions = [i for i, value in enumerate(sample["latent_mask"]) if value]
        if len(positions) < 2:
            raise ValueError(f"empty latent span: {row['id']}")
        batches.append(collate_batch(tokenizer, [sample]))
    output.mkdir(parents=True)
    write_json(output / "run.json", {
        "source": str(source), "data_sha256": sha256(data),
        "source_link_sha256": sha256(source / "adapter.pt"),
        "source_lora_sha256": sha256(source / "lora/adapter_model.safetensors"),
        "calibration": calibration, "update_ids": [r["id"] for r in update_rows],
        "observation_ids": [r["id"] for r in observe_rows],
        "learning_rate": 5e-5, "optimizer_steps": 1, "seed": 6064,
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "torch": torch.__version__,
        "warning": "Observation rows are from training, not held-out generalization. No recursive rollout.",
    })
    print("Update IDs:", [r["id"] for r in update_rows], flush=True)
    print("Observation-only IDs:", [r["id"] for r in observe_rows], flush=True)
    model = HybridSFTModel.from_pretrained(source, dtype=torch.bfloat16, require_hybridlink=True)
    if not is_lora_model(model):
        raise ValueError("expected a saved LoRA checkpoint")
    actual = [tokenizer.convert_tokens_to_ids(t) for t in ("<latent>", "</latent>")]
    if actual != model.lora_anchor_ids:
        raise ValueError("anchor IDs differ from checkpoint")
    model.resize_to_tokenizer(tokenizer)
    model.set_trainable(LINK_NEXT_TOKEN)
    model.output_scale = calibration["output_scale"]
    optimizer = torch.optim.AdamW(model.hybridlink.parameters(), lr=5e-5,
                                 betas=ADAM_BETAS, weight_decay=WEIGHT_DECAY)
    model, optimizer = accelerator.prepare(model, optimizer)
    model.train()
    batches = [{k: v.to(accelerator.device) for k, v in b.items()} for b in batches]
    raw = accelerator.unwrap_model(model)
    link_before = {k: v.detach().cpu().clone() for k, v in raw.hybridlink.state_dict().items()}
    print("Trainable parameters: backbone=", sum(p.numel() for p in raw.backbone.parameters() if p.requires_grad),
          "hybridlink=", sum(p.numel() for p in raw.hybridlink.parameters() if p.requires_grad), flush=True)
    torch.cuda.reset_peak_memory_stats()

    def measure(batch):
        # Keep the same grad-enabled forward/checkpoint/autocast path as training;
        # discard its graph without backward. No no_grad/inference-mode switch.
        out = model(**batch, stage=LINK_NEXT_TOKEN)
        if not torch.isfinite(out["loss"]):
            raise RuntimeError("nonfinite measurement")
        result = {"ce": out["loss"].detach().float().item(), "target_id": out["target_id"].item(),
                  "target_position": out["target_position"], "substitution_position": out["substitution_position"]}
        del out
        return result

    observed_before = [measure(b) for b in batches[8:]]
    optimizer.zero_grad(set_to_none=True)
    update_before = average_backward(model, batches[:8], LINK_NEXT_TOKEN, accelerator.backward)
    write_json(output / "before.json", {"update": update_before, "observation": observed_before})
    if any(p.requires_grad or p.grad is not None for p in raw.backbone.parameters()):
        raise RuntimeError("backbone/LoRA/anchors are not frozen")
    grads = [p.grad for p in raw.hybridlink.parameters()]
    if any(g is None or not torch.isfinite(g).all() for g in grads) or not any(g.abs().sum() > 0 for g in grads):
        raise RuntimeError("missing, nonfinite or zero HybridLink gradients")
    del grads
    accelerator.clip_grad_norm_(raw.hybridlink.parameters(), MAX_GRAD_NORM)
    optimizer.step()  # exactly one step on the mean of eight losses
    optimizer.zero_grad(set_to_none=True)
    updated = [k for k, v in raw.hybridlink.state_dict().items() if not torch.equal(v.detach().cpu(), link_before[k])]
    if not updated:
        raise RuntimeError("HybridLink weights did not change")
    del link_before
    after = [measure(b) for b in batches]
    checkpoint = output / "checkpoint"
    raw.save_pretrained(checkpoint, tokenizer)
    a = load_file(str(source / "lora/adapter_model.safetensors"))
    b = load_file(str(checkpoint / "lora/adapter_model.safetensors"))
    if a.keys() != b.keys() or any(a[k].dtype != b[k].dtype or not torch.equal(a[k], b[k]) for k in a):
        raise RuntimeError("saved LoRA/anchor weights changed")
    del a, b
    summary = {"optimizer_steps": 1, "gradient_average_count": 8, "learning_rate": 5e-5,
               "scale": raw.output_scale, "updated_link_tensors": updated, "lora_anchors_unchanged": True,
               "checkpoint": str(checkpoint), "peak_gpu_gib": torch.cuda.max_memory_allocated() / 1024**3,
               "update": comparison(update_rows, update_before, after[:8]),
               "observation": comparison(observe_rows, observed_before, after[8:]),
               "warning": "Training-set first-position CE only, not answer accuracy or generalization."}
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
