"""Probe fixed fractions of one saved joint update; never train or save weights."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from format_experiment import ROOT, verify_data, write_json, sha256
from link_joint_step import select_rows, comparison
from latent_experiment import check_reference_config

# Endpoints first: do not interpret fractions unless the saved experiment replays.
FRACTIONS = (0.0, 1.0, 0.01, 0.1)


def interpolate_state(before, after, fraction):
    import torch
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("fraction must be between zero and one")
    if before.keys() != after.keys():
        raise ValueError("HybridLink tensor names differ")
    result = {}
    for key in before:
        a, b = before[key], after[key]
        if a.shape != b.shape or a.dtype != torch.float32 or b.dtype != torch.float32:
            raise ValueError(f"expected matching FP32 link tensors: {key}")
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError(f"nonfinite link tensor: {key}")
        # Copy endpoints exactly, avoiding rounding from a + (b - a).
        result[key] = a.clone() if fraction == 0 else b.clone() if fraction == 1 else a + fraction * (b - a)
    return result


def verify_endpoint(rows, measurements, saved_rows, field):
    if len(rows) != len(measurements) or len(rows) != len(saved_rows):
        raise ValueError("endpoint record counts differ")
    largest_error = 0.0
    for row, measured, saved in zip(rows, measurements, saved_rows):
        if row["id"] != saved["id"] or any(measured[k] != saved[k] for k in
            ("target_id", "target_position", "substitution_position")):
            raise ValueError("endpoint IDs or target positions differ")
        actual, expected = measured["ce"], saved[field]
        if not (math.isfinite(actual) and math.isfinite(expected)):
            raise ValueError("nonfinite endpoint CE")
        # Fixed numeric tolerance, not a tuning criterion.
        if not math.isclose(actual, expected, rel_tol=1e-4, abs_tol=1e-5):
            raise ValueError(f"endpoint CE mismatch for {row['id']}: {actual} vs {expected}; stop the probe")
        largest_error = max(largest_error, abs(actual - expected))
    return largest_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", required=True)
    parser.add_argument("--model", required=True, help="original 64-step checkpoint")
    parser.add_argument("--joint-run", required=True, help="joint run containing run.json, summary.json and checkpoint/")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    source, joint, output = (Path(v).resolve() for v in (args.model, args.joint_run, args.output_dir))
    for protected in (source, joint, Path(args.pilot).resolve()):
        if output == protected or output in protected.parents or protected in output.parents:
            raise ValueError("output must be separate from checkpoints and data")
    if output.exists():
        raise FileExistsError("output exists; preserve the prior diagnostic")
    updated = joint / "checkpoint"
    run = json.loads((joint / "run.json").read_text())
    previous = json.loads((joint / "summary.json").read_text())
    data, _ = verify_data(args.pilot, "train")
    if sha256(data) != run["data_sha256"]:
        raise ValueError("training data differs from joint experiment")
    if sha256(source / "adapter.pt") != run["source_link_sha256"]:
        raise ValueError("source HybridLink changed")
    if sha256(source / "lora/adapter_model.safetensors") != run["source_lora_sha256"]:
        raise ValueError("source LoRA changed")
    check_reference_config(updated, source)
    source_config = json.loads((source / "hybridlink_config.json").read_text())
    if source_config != json.loads((updated / "hybridlink_config.json").read_text()):
        raise ValueError("checkpoint scales/configs differ")
    scale = run["calibration"]["output_scale"]
    if scale != source_config["output_scale"] or scale != previous["scale"]:
        raise ValueError("recorded scales differ")
    rows = [json.loads(s) for s in data.read_text().splitlines() if s.strip()]
    update_rows, observe_rows = select_rows(rows)
    for selected, key in ((update_rows, "update_ids"), (observe_rows, "observation_ids")):
        if [r["id"] for r in selected] != run[key]:
            raise ValueError("joint selection changed")
    selected = update_rows + observe_rows

    import torch
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    from hybrid_agent.model import add_anchor_tokens
    from hybrid_sft.data import tokenize_trajectory, collate_batch
    from hybrid_sft.model import HybridSFTModel, LINK_NEXT_TOKEN

    a = load_file(str(source / "lora/adapter_model.safetensors"))
    b = load_file(str(updated / "lora/adapter_model.safetensors"))
    if a.keys() != b.keys() or any(a[k].dtype != b[k].dtype or not torch.equal(a[k], b[k]) for k in a):
        raise ValueError("LoRA/anchor parameters differ")
    del a, b
    before = torch.load(source / "adapter.pt", map_location="cpu", weights_only=True)
    after = torch.load(updated / "adapter.pt", map_location="cpu", weights_only=True)
    del_check = interpolate_state(before, after, 0)  # validate before GPU loading
    del del_check
    accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=1)
    if (accelerator.device.type != "cuda" or accelerator.num_processes != 1
            or str(accelerator.distributed_type).endswith("DEEPSPEED")):
        raise ValueError("requires one GPU process without DeepSpeed")
    set_seed(run["seed"])
    tokenizer = AutoTokenizer.from_pretrained(str(source))
    add_anchor_tokens(tokenizer)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    batches = []
    for row in selected:
        sample = tokenize_trajectory(tokenizer, row["system"], row["question"], row["rounds"], 1024)
        if sample is None:
            raise ValueError(f"rejected record {row['id']}")
        batches.append(collate_batch(tokenizer, [sample]))
    output.mkdir(parents=True)
    write_json(output / "run.json", {
        "source": str(source), "joint_run": str(joint), "fractions_in_execution_order": FRACTIONS,
        "source_link_sha256": sha256(source / "adapter.pt"), "updated_link_sha256": sha256(updated / "adapter.pt"),
        "joint_run_sha256": sha256(joint / "run.json"), "joint_summary_sha256": sha256(joint / "summary.json"),
        "data_sha256": sha256(data), "scale": scale, "torch": torch.__version__,
        "update_ids": run["update_ids"], "observation_ids": run["observation_ids"],
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "endpoint_tolerance": {"relative": 1e-4, "absolute": 1e-5},
    })
    model = HybridSFTModel.from_pretrained(source, dtype=torch.bfloat16, require_hybridlink=True)
    if [tokenizer.convert_tokens_to_ids(t) for t in ("<latent>", "</latent>")] != model.lora_anchor_ids:
        raise ValueError("anchor IDs differ")
    model.resize_to_tokenizer(tokenizer)
    model.set_trainable(LINK_NEXT_TOKEN)
    model.output_scale = scale
    model = accelerator.prepare(model)
    model.train()
    raw = accelerator.unwrap_model(model)
    batches = [{k: v.to(accelerator.device) for k, v in batch.items()} for batch in batches]
    results, endpoints = [], {}
    baseline = None
    with (output / "measurements.jsonl").open("x", encoding="utf-8") as stream:
        for fraction in FRACTIONS:
            state = interpolate_state(before, after, fraction)
            raw.hybridlink.load_state_dict(state, strict=True)
            del state
            measured = []
            for row, batch in zip(selected, batches):
                # Match joint-run precision and grad-enabled checkpoint forward.
                # Discard graph per record; NEVER backward or optimizer.step.
                out = model(**batch, stage=LINK_NEXT_TOKEN)
                item = {"ce": out["loss"].detach().float().item(), "target_id": out["target_id"].item(),
                        "target_position": out["target_position"], "substitution_position": out["substitution_position"]}
                del out
                if not math.isfinite(item["ce"]):
                    raise RuntimeError("nonfinite measured CE")
                measured.append(item)
                stream.write(json.dumps({"fraction": fraction, "id": row["id"], **item}) + "\n")
                stream.flush()
            if fraction in (0, 1):
                field = "before_ce" if fraction == 0 else "after_ce"
                saved = previous["update"]["rows"] + previous["observation"]["rows"]
                endpoints[str(fraction)] = verify_endpoint(selected, measured, saved, field)
                print(f"Endpoint {fraction} verified; max CE difference={endpoints[str(fraction)]:.8g}", flush=True)
            if fraction == 0:
                baseline = measured
            result = {"fraction": fraction,
                      "update": comparison(update_rows, baseline[:8], measured[:8]),
                      "observation": comparison(observe_rows, baseline[8:], measured[8:])}
            results.append(result)
            print(f"fraction={fraction} update_mean={result['update']['mean_after_ce']:.8g} "
                  f"observation_mean={result['observation']['mean_after_ce']:.8g}", flush=True)
    summary = {"endpoint_max_ce_difference": endpoints, "results": results,
               "warning": "Fixed fractions of saved AdamW displacement, not a gradient check or new training. "
                          "Training-set first-position CE only. No checkpoint selected or saved; "
                          "BF16 quantization can hide very small changes."}
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
