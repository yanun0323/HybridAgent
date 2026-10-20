# HybridSFT

Hybrid supervised fine-tuning turns a pre-trained base model into a HybridAgent that
coordinates its backbone with HybridLink for stable tool-integrated hybrid generation.

<p align="center"><img src="../assets/hybrid_sft.png" width="100%" alt="Self-conditioned HybridSFT"></p>

## Stages

| Stage | `--stage` | Trainable | Objective |
|---|---|---|---|
| Format activation | `format_activation` | backbone | cross-entropy on the hybrid trajectories: the backbone learns when to emit the entry and exit anchors `<latent>` / `</latent>` |
| HybridLink activation | `hybridlink_activation` | HybridLink | regression of the HybridLink outputs at latent positions onto their target input embeddings |
| Self-conditioned HybridSFT | `self_conditioned` | backbone + HybridLink | joint training of backbone and HybridLink on the model's own latent inputs, closing the train–inference gap |

## Data

One JSON object per line. Latent spans are enclosed by the anchors inside the assistant text; tool observations are kept in `tool_result` and serve as context only.

```json
{"id": 0,
 "system": "<system prompt>",
 "question": "...",
 "rounds": [
   {"assistant": "<think>\nKey step.<latent>routine reasoning</latent>Decision.\n</think><python>...</python>",
    "tool": "python", "tool_result": "<result>...</result>"},
   {"assistant": "<think>\n...<latent>...</latent>...\n</think><answer>\\boxed{...}</answer>",
    "tool": null, "tool_result": null}]}
```

`--data` accepts a local JSONL file or a Hugging Face dataset repo id (with `--data_file` naming the file in the repo, default `train.jsonl`). The released data are `togethercomputer/HybridAgent-HybridSFT-v1` and `togethercomputer/HybridAgent-HybridSFT-v2`; v2 is the upgraded version, produced by the same annotation pipeline with a stronger labeling model. Each record's `system` field is used as the system prompt; when it is absent, the default prompt in `hybrid_agent/prompts.py` is used.

## Training

```bash
MODEL=Qwen/Qwen3-8B-Base DATA=<jsonl or HF dataset id> OUT=outputs/hybrid_sft bash scripts/run_hybrid_sft.sh all
```

The launcher runs the three stages in order, each initialized from the previous output. Recommended hardware: 8 or 16 GPUs. The provided accelerate configs assume one node with 8 GPUs (batch size 1 × gradient accumulation 2 × 8 = 16); with a different GPU count adjust `--grad_accum` and the matching `gradient_accumulation_steps` in the config so the effective batch stays the same. For the 14B model use the CPU-offload config for the last stage: `CONFIG_SELF_CONDITIONED=hybrid_sft/configs/accelerate_zero2_offload.yaml`. A single stage can be run directly:

```bash
accelerate launch --config_file hybrid_sft/configs/accelerate_zero2.yaml -m hybrid_sft.train \
    --stage hybridlink_activation --model outputs/hybrid_sft/format_activation \
    --data <jsonl or HF dataset id> --output_dir outputs/hybrid_sft/hybridlink_activation
```

Epochs default to 1 / 2 / 1 for the three stages (`--epochs`; 1 / 1 / 1 and 1 / 1 / 2 are good practice as well); the other settings are the trainer defaults, listed by `python -m hybrid_sft.train --help`. Evaluate the saved HybridSFT checkpoints, e.g. on AIME and MATH-500, and take the best one as the initialization for HybridRL.

## Output

Every checkpoint directory (`checkpoint-N/` and the final output) is a complete HybridAgent checkpoint: the backbone in Hugging Face format, the tokenizer with the anchor tokens, and the HybridLink weights (`adapter.pt`, `hybridlink_config.json`). It loads with `hybrid_agent.load_hybrid_agent` and is the initialization for HybridRL.


## Experimental single-GPU LoRA format activation

This is a resource-saving adaptation, not a reproduction of full-parameter
HybridSFT. Only `format_activation` is supported; HybridLink stays frozen.
PEFT LoRA targets text attention/DeltaNet/MLP projections, not vision layers.
Only the `<latent>` and `</latent>` rows of the input embedding and untied output
head are trainable; the complete vocabulary matrices are not optimizer parameters.
PEFT may still materialize full embedding matrices during forward/backward, so
GPU memory must be measured on the real model.

Install the optional dependency in the existing environment:

```bash
pip install 'peft==0.21.2'
pip install -e . --no-deps
python -m unittest discover -s tests -v
```

Use plain Python, one process, without a DeepSpeed configuration. Start with one
update and a short real JSONL trajectory using the data format above:

```bash
python -m hybrid_sft.train --stage format_activation \
  --model Qwen/Qwen3.8-27B --data /content/format-smoke.jsonl \
  --output_dir /content/qwen-format-smoke --lora \
  --lora_rank 8 --lora_alpha 16 --lr 1e-4 \
  --epochs 1 --batch_size 1 --grad_accum 1 --max_steps 1 \
  --max_length 1024 --num_workers 0 --save_steps 0
```

A one-step smoke test has a nonzero initial learning rate. For longer runs the
existing warmup/cosine schedule remains in use. The trainer does not evaluate a
validation set or resume optimizer state; saved directories are model checkpoints,
not resumable training-state snapshots. `--revision` can pin the base model.
Do not launch this in a second process while a 27B model already occupies the GPU.

LoRA saves contain the tokenizer, HybridLink weights/config, `lora/` PEFT files,
and `hybrid_lora.json` recording the base reference/revision, vocabulary size and
anchor IDs. They do not duplicate the base weights. A local base path must remain
available; a Hub base is downloaded/cached on the destination machine. Reload with
`load_hybrid_agent(checkpoint_directory)`. Base vocabulary resizing happens before
adapter loading; both input and output anchor rows are restored by PEFT. The
saved model retains vision weights from the base but does not train/use them.

Do not use the hybrid engine to judge format-only checkpoints yet: HybridLink is
still untrained. Evaluate format learning with ordinary text generation so the
model writes the text inside `<latent>...</latent>`; later activation and
self-conditioning are separate work.
