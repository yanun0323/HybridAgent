"""HybridSFT trainer: runs one stage, format activation, HybridLink
activation or self-conditioned HybridSFT, with the recipe's hyperparameters as defaults.

Launch on 8 GPUs with DeepSpeed ZeRO-2 (see scripts/run_hybrid_sft.sh):
  accelerate launch --config_file hybrid_sft/configs/accelerate_zero2.yaml -m hybrid_sft.train \
      --stage format_activation --model Qwen/Qwen3-8B-Base --data <jsonl or HF dataset id> \
      --output_dir outputs/hybrid_sft/format_activation
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from hybrid_agent.model import add_anchor_tokens
from hybrid_agent.lora import enable_format_lora, is_lora_model
from hybrid_agent.compat import check_execution
from hybrid_sft.calibration import read_training_calibration
from hybrid_sft.data import HybridSFTDataset, collate_batch, resolve_data_file
from hybrid_sft.model import FORMAT_ACTIVATION, HYBRIDLINK_ACTIVATION, SELF_CONDITIONED, LINK_NEXT_TOKEN, STAGES, HybridSFTModel

# Epochs per stage. 1/1/1 and 1/1/2 are good practice as well; evaluate the saved HybridSFT
# checkpoints (e.g. on AIME and MATH-500) and take the best one as the initialization for HybridRL.
STAGE_EPOCHS = {FORMAT_ACTIVATION: 1, HYBRIDLINK_ACTIVATION: 2, SELF_CONDITIONED: 1, LINK_NEXT_TOKEN: 1}
STAGE_HYBRIDLINK_LR = {HYBRIDLINK_ACTIVATION: 2e-4, SELF_CONDITIONED: 5e-5, LINK_NEXT_TOKEN: 5e-5}

WEIGHT_DECAY = 0.0
ADAM_BETAS = (0.9, 0.95)
WARMUP_RATIO = 0.1
MAX_GRAD_NORM = 1.0
NUM_WORKERS = 4
LOG_EVERY = 10
WANDB_PROJECT = "HybridAgent"


def parse_args():
    p = argparse.ArgumentParser(description="HybridSFT: one training stage")
    p.add_argument("--stage", required=True, choices=STAGES)
    p.add_argument("--model", required=True,
                   help="base model for format activation, otherwise the previous stage's output (dir or HF id)")
    p.add_argument("--data", required=True, help="hybrid trajectories: local JSONL path or HF dataset repo id")
    p.add_argument("--data_file", default="train.jsonl", help="file name inside the HF dataset repo")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--epochs", type=int, default=None, help="default per stage: 1 / 2 / 1")
    p.add_argument("--lr", type=float, default=7e-6, help="backbone learning rate")
    p.add_argument("--hybridlink_lr", type=float, default=None,
                   help="HybridLink learning rate (default 2e-4 for activation, 5e-5 for self-conditioning)")
    p.add_argument("--batch_size", type=int, default=1, help="per device")
    p.add_argument("--grad_accum", type=int, default=2, help="must match the accelerate/DeepSpeed config")
    p.add_argument("--max_length", type=int, default=16888, help="longer trajectories are dropped")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_steps", type=int, default=2000)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--lora", action="store_true", help="single-process format activation only")
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--max_steps", type=int, default=None, help="cap optimizer updates (e.g. 1 for a smoke test)")
    p.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    p.add_argument("--revision", default=None, help="pin the base Hugging Face revision")
    scale_options = p.add_mutually_exclusive_group()
    scale_options.add_argument("--calibrate_link_scale", action="store_true",
                               help="match HybridLink magnitude on training data before frozen activation")
    scale_options.add_argument("--link_calibration", help="reuse a calibration JSON matching the training file")
    args = p.parse_args()
    if args.epochs is None:
        args.epochs = STAGE_EPOCHS[args.stage]
    if args.hybridlink_lr is None:
        args.hybridlink_lr = STAGE_HYBRIDLINK_LR.get(args.stage, args.lr)
    if args.lora and args.stage != FORMAT_ACTIVATION:
        p.error("--lora currently supports format_activation only")
    if args.calibrate_link_scale and args.stage != HYBRIDLINK_ACTIVATION:
        p.error("fresh scale calibration requires hybridlink_activation")
    if args.link_calibration and args.stage not in (HYBRIDLINK_ACTIVATION, LINK_NEXT_TOKEN):
        p.error("scale reuse requires a frozen link training stage")
    if args.stage == LINK_NEXT_TOKEN and (args.max_steps != 1 or args.batch_size != 1 or args.grad_accum != 1):
        p.error("link_next_token is a smoke test: require --max_steps 1 --batch_size 1 --grad_accum 1")
    if args.max_steps is not None and args.max_steps < 1:
        p.error("--max_steps must be positive")
    if args.num_workers < 0:
        p.error("--num_workers must be nonnegative")
    return args


def cosine_schedule_with_warmup(optimizer, num_warmup_steps: int, num_training_steps: int) -> LambdaLR:
    """Linear warmup, then cosine decay clamped at the end of training."""
    if num_training_steps <= 0:
        raise ValueError("num_training_steps must be > 0")

    def lr_lambda(current_step: int) -> float:
        if num_warmup_steps > 0 and current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        if num_training_steps <= num_warmup_steps:
            return 0.0
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def main():
    args = parse_args()
    accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum, mixed_precision="bf16")
    if args.lora and (accelerator.num_processes != 1 or str(accelerator.distributed_type).endswith("DEEPSPEED")):
        raise ValueError("LoRA format training requires one process without DeepSpeed")
    link_only = args.stage in (HYBRIDLINK_ACTIVATION, LINK_NEXT_TOKEN)
    if link_only:
        if accelerator.num_processes != 1 or str(accelerator.distributed_type).endswith("DEEPSPEED"):
            raise ValueError("frozen HybridLink activation requires one process without DeepSpeed")
        output = Path(args.output_dir).resolve()
        source = Path(args.model).resolve()
        if output == source or output in source.parents or source in output.parents or output.exists():
            raise ValueError("use a new output directory separate from the source checkpoint")
    set_seed(args.seed)
    is_main = accelerator.is_main_process

    # Fetch tokenizer files only before data validation, not a full model snapshot.
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    add_anchor_tokens(tokenizer)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    data_file = resolve_data_file(args.data, args.data_file)
    calibration = read_training_calibration(args.link_calibration, data_file) if args.link_calibration else None
    dataset = HybridSFTDataset(
        tokenizer, data_file, max_length=args.max_length, text_only=(args.stage == FORMAT_ACTIVATION),
    )
    if is_main:
        print(f"Dataset: {len(dataset)} trajectories from {data_file}", flush=True)
    if len(dataset) == 0:
        raise RuntimeError("no usable trajectories in the data file")
    if link_only and any(not any(sample["latent_mask"][:-1]) for sample in dataset.samples):
        raise ValueError("every HybridLink training trajectory must contain a latent span")

    if is_main:
        print(f"Loading model from {args.model} ...", flush=True)
    model = HybridSFTModel.from_pretrained(
        args.model, revision=args.revision, dtype=torch.bfloat16,
        require_hybridlink=(args.stage != FORMAT_ACTIVATION),
    )
    model.resize_to_tokenizer(tokenizer)
    if hasattr(model, "lora_anchor_ids"):
        actual = [tokenizer.convert_tokens_to_ids(t) for t in ("<latent>", "</latent>")]
        if actual != model.lora_anchor_ids:
            raise ValueError("checkpoint tokenizer anchor IDs do not match the saved model")
    if link_only:
        check_execution(model.backbone, batch_size=args.batch_size)
    if args.lora:
        enable_format_lora(model, tokenizer, rank=args.lora_rank, alpha=args.lora_alpha)
    elif is_lora_model(model) and not link_only:
        raise ValueError("a loaded LoRA checkpoint requires a frozen link training stage")
    lora_checkpoint = is_lora_model(model)
    model.set_trainable(args.stage)
    if not link_only:
        model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if link_only:
        model.to(accelerator.device)
        if args.link_calibration:
            model.output_scale = calibration["output_scale"]
            print("Reused training-only scale calibration: " + json.dumps(calibration), flush=True)
        elif args.calibrate_link_scale:
            def calibration_batches():
                for sample in dataset.samples:
                    batch = collate_batch(tokenizer, [sample])
                    yield {k: v.to(accelerator.device) for k, v in batch.items()}
            calibration = model.calibrate_link_scale(calibration_batches())
            print("Training-only scale calibration: " + json.dumps(calibration), flush=True)
        else:
            calibration = {"output_scale": model.output_scale, "calibrated": False}
        # Record the source of the scale, never consult evaluation data.
        import hashlib
        calibration["data_sha256"] = hashlib.sha256(Path(data_file).read_bytes()).hexdigest()

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
        collate_fn=lambda batch: collate_batch(tokenizer, batch),
        num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0,
    )

    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    link_params = [p for p in model.hybridlink.parameters() if p.requires_grad]
    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": args.lr})
    if link_params:
        param_groups.append({"params": link_params, "lr": args.hybridlink_lr})
    if is_main:
        print(f"Trainable parameters: backbone={sum(p.numel() for p in backbone_params):,} "
              f"hybridlink={sum(p.numel() for p in link_params):,}", flush=True)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY, betas=ADAM_BETAS)

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    if accelerator.gradient_accumulation_steps != args.grad_accum:
        raise RuntimeError(
            f"gradient accumulation mismatch: --grad_accum {args.grad_accum} vs "
            f"{accelerator.gradient_accumulation_steps} in the accelerate/DeepSpeed config"
        )

    steps_per_epoch = math.ceil(len(dataloader) / args.grad_accum)
    max_steps = args.epochs * steps_per_epoch
    if max_steps <= 0:
        raise ValueError("no optimizer steps: reduce grad_accum or batch_size, or use more data")
    if args.max_steps is not None:
        max_steps = min(max_steps, args.max_steps)
    warmup_steps = min(max_steps - 1, max(1, int(max_steps * WARMUP_RATIO)))
    scheduler = cosine_schedule_with_warmup(optimizer, warmup_steps, max_steps)

    os.makedirs(args.output_dir, exist_ok=True)
    if link_only and is_main:
        with open(Path(args.output_dir) / "link_calibration.json", "w") as f:
            json.dump(calibration, f, indent=2)
    wandb_run = None
    if is_main:
        with open(os.path.join(args.output_dir, "train_args.json"), "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2)
        print(f"Training {args.stage}: {max_steps} steps "
              f"(effective batch {args.batch_size * args.grad_accum * accelerator.num_processes})", flush=True)
        if args.wandb:
            import wandb

            wandb_run = wandb.init(
                project=WANDB_PROJECT, name=f"hybrid_sft-{args.stage}-{Path(args.output_dir).name}",
                config=vars(args), dir=args.output_dir,
            )

    def save(ckpt_dir: str) -> None:
        accelerator.wait_for_everyone()
        if lora_checkpoint:
            if is_main:
                accelerator.unwrap_model(model).save_pretrained(ckpt_dir, tokenizer)
        else:
            full_sd = accelerator.get_state_dict(model)  # collective
            if is_main:
                accelerator.unwrap_model(model).save_pretrained(ckpt_dir, tokenizer, full_sd)
            del full_sd
        accelerator.wait_for_everyone()

    model.train()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    global_step = 0
    pbar = tqdm(total=max_steps, desc=args.stage, disable=not is_main)
    accum = {"loss": 0.0, "backbone_loss": 0.0, "hybridlink_loss": 0.0, "n": 0}

    for epoch in range(args.epochs):
        for batch in dataloader:
            with accelerator.accumulate(model):
                out = model(
                    input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                    loss_mask=batch["loss_mask"], latent_mask=batch["latent_mask"], stage=args.stage,
                )
                if not torch.isfinite(out["loss"]):
                    raise RuntimeError("nonfinite training loss")
                if args.stage == LINK_NEXT_TOKEN and is_main:
                    print(f"Next-token CE={out['next_token_ce'].item():.8g} "
                          f"substitution_position={out['substitution_position']} "
                          f"target_position={out['target_position']} target_id={out['target_id'].item()}", flush=True)
                accelerator.backward(out["loss"])
                for key in ("loss", "backbone_loss", "hybridlink_loss"):
                    accum[key] += out[key].item()
                accum["n"] += 1

                if accelerator.sync_gradients:
                    if link_only and global_step == 0:
                        raw_model = accelerator.unwrap_model(model)
                        if any(p.requires_grad or p.grad is not None for p in raw_model.backbone.parameters()):
                            raise RuntimeError("frozen backbone/LoRA/anchor parameters received gradients")
                        gradients = [p.grad for p in raw_model.hybridlink.parameters()]
                        if (any(g is None or not torch.isfinite(g).all() for g in gradients)
                                or not any(g.abs().sum() > 0 for g in gradients)):
                            raise RuntimeError("HybridLink did not receive finite nonzero gradients")
                        print("HybridLink gradients verified; backbone, LoRA and anchors frozen", flush=True)
                    if args.lora and global_step == 0:
                        raw_model = accelerator.unwrap_model(model)
                        anchor_parameters = [p for name, p in raw_model.backbone.named_parameters()
                                             if "trainable_tokens_delta" in name and p.requires_grad]
                        if not anchor_parameters or not all(
                            p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                            for p in anchor_parameters
                        ):
                            raise RuntimeError("anchor rows did not receive finite nonzero gradients")
                        if any(p.grad is not None for p in raw_model.backbone.parameters() if not p.requires_grad):
                            raise RuntimeError("a frozen backbone parameter received gradients")
                        print("Anchor gradients verified; frozen base has no parameter gradients", flush=True)
                    accelerator.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    n = max(accum["n"], 1)
                    stats = {k: accum[k] / n for k in ("loss", "backbone_loss", "hybridlink_loss")}
                    stats["lr"] = scheduler.get_last_lr()[0]
                    accum = {"loss": 0.0, "backbone_loss": 0.0, "hybridlink_loss": 0.0, "n": 0}
                    pbar.set_postfix({k: f"{v:.4f}" for k, v in stats.items() if k != "lr"})
                    pbar.update(1)
                    if wandb_run is not None:
                        wandb_run.log({f"train/{k}": v for k, v in stats.items()} | {"train/step": global_step})
                    if is_main and (global_step == 1 or global_step % LOG_EVERY == 0 or global_step == max_steps):
                        print(f"step={global_step} loss={stats['loss']:.8g} lr={stats['lr']:.2e} "
                              f"cos_loss={out['cos_loss'].item():.8g} mse_loss={out['mse_loss'].item():.8g}",
                              flush=True)
                    if args.save_steps > 0 and global_step % args.save_steps == 0:
                        save(os.path.join(args.output_dir, f"checkpoint-{global_step}"))
                    if global_step >= max_steps:
                        break
        if global_step >= max_steps:
            break

    save(args.output_dir)
    if is_main:
        if wandb_run is not None:
            wandb_run.finish()
        print(f"Saved to {args.output_dir}", flush=True)
        if torch.cuda.is_available():
            print(f"Peak GPU allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB", flush=True)


if __name__ == "__main__":
    main()
