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
HybridSFT. During `format_activation`, HybridLink stays frozen. A saved LoRA
checkpoint can also initialize frozen `hybridlink_activation` (see below);
LoRA self-conditioning and RL remain unsupported.
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

### Frozen HybridLink activation

Use an existing format checkpoint **without** `--lora`. The backbone, existing
LoRA and both anchor rows remain frozen and in evaluation mode; only HybridLink
is optimized in FP32. This activation path currently requires one process without
DeepSpeed; Qwen also requires batch size one. Do not overwrite or nest the output
inside the source checkpoint. Start with one update:

```bash
python -u -m hybrid_sft.train --stage hybridlink_activation \
  --model /content/format-pilot/checkpoint --data /content/format-pilot/train.jsonl \
  --output_dir /content/qwen-link-smoke --calibrate_link_scale \
  --hybridlink_lr 2e-4 --epochs 1 --batch_size 1 --grad_accum 1 --max_steps 1 \
  --max_length 1024 --num_workers 0 --save_steps 0 --seed 6064
```

`--calibrate_link_scale` makes one pass over the supplied training dataset before
optimization. It matches the RMS of unscaled HybridLink predictions to the RMS of
next-token embeddings at latent sources, including exit-anchor targets. The scalar
is then fixed; this is magnitude calibration, not learned reasoning or directional
alignment. `link_calibration.json` records both RMS values, counts, scale and input
file hash. Without the flag the checkpoint's existing scale is preserved. Never
supply validation/test data for calibration. Its full pass is additional work even
when `--max_steps 1` limits optimizer updates.

To run a fixed 64-update pilot from the original format checkpoint, reuse the
smoke calibration rather than recalibrating or continuing its updated link:

```bash
python -u -m hybrid_sft.train --stage hybridlink_activation \
  --model /content/format-pilot/checkpoint --data /content/format-pilot/train.jsonl \
  --output_dir /content/qwen-link-64 \
  --link_calibration /content/qwen-link-smoke/link_calibration.json \
  --hybridlink_lr 2e-4 --epochs 1 --batch_size 1 --grad_accum 1 --max_steps 64 \
  --max_length 1024 --num_workers 0 --save_steps 0 --seed 6064
```

`--link_calibration` and `--calibrate_link_scale` are mutually exclusive. Reuse
verifies the training file hash and positive finite scale before loading weights;
it records the source calibration hash in the new output. It does not modify the
source checkpoint or calibration. This uses the existing warmup/cosine schedule
and saves only the final checkpoint; `max_steps` counts optimizer-loop iterations
(the initial warmup iteration uses zero learning rate). Evaluate on `latent-dev`
with unchanged limits; do not open `latent-test` for tuning.

Activation uses frozen teacher states with the real preceding tokens; it predicts
only latent next-token embeddings and the exit-anchor embedding. Text CE is not
computed on this path (`backbone_loss=0` means omitted, not perfect text accuracy).
The first update checks finite nonzero HybridLink gradients and the absence of any
backbone gradients. Logs include cosine and MSE losses. FP32 link weights remain
FP32 when reloaded alongside a BF16 backbone. LoRA checkpoints save only the existing
PEFT files, tokenizer, HybridLink and metadata, not a copy of the 27B backbone.

The regression test `test_frozen_link_activation_calibration_and_checkpoint_roundtrip`
uses a tiny local model to check target positions, frozen parameters, unchanged text
logits, nonzero link updates, magnitude calibration, and checkpoint round-trip. Run
it in the environment with torch/Transformers/PEFT before the real smoke test.
Neither a passed smoke test nor declining regression loss demonstrates successful
recursive latent generation. That requires a separate generation evaluation with
natural versus forced exits counted separately. No optimizer resume, LoRA
self-conditioning or speed claim is added by this training path.

### Reproducible format pilot

`scripts/format_experiment.py` provides `prepare`, `prepare-generalization`,
`validate`, `train` and `evaluate` commands.
It uses synthetic `a*b-c` trajectories with one latent span, no tools, and a
matching explicit system prompt at training and evaluation time. This is a
single-template format-learning check, not evidence of improved reasoning.

```bash
python scripts/format_experiment.py prepare --directory /content/format-pilot
python scripts/format_experiment.py validate --directory /content/format-pilot
python scripts/format_experiment.py evaluate --directory /content/format-pilot \
  --model Qwen/Qwen3.8-27B --split validation \
  --output /content/format-pilot/base-validation.jsonl
python scripts/format_experiment.py train --directory /content/format-pilot
python scripts/format_experiment.py evaluate --directory /content/format-pilot \
  --model /content/format-pilot/checkpoint --split validation \
  --output /content/format-pilot/trained-validation.jsonl
```

Preparation fixes 64 training, 16 validation and 32 reserved test questions and
records their hashes. The operand range differs from the earlier notebook pilot;
only exact operand tuples, not answer values or linguistic templates, are disjoint.
Existing data/logs/evaluation files are not overwritten. Commands verify data
hashes. Use `--revision` on base evaluation and training to pin the same Hub commit.

Training runs two epochs, saving model checkpoints after 64 and 128 updates, and
streams combined stdout/stderr to both the console and `train.log`. It records the
command and code commit in `training-run.json`. On failure, keep the log; do not
interpret a partial output directory as a completed experiment. The wrapper does
not resume optimizer state. Run with no model already occupying the notebook GPU.

Evaluation uses ordinary greedy **text** generation, never HybridLink or forced
mode transitions. Every question is tested twice: plain integer-answer prompting
and the trained format prompt. The opening `<think>` supplied by the chat template
is included for structural scoring but is not credited as a generated token.
Only an entire integer is accepted in plain mode. Format mode extracts only the
single final `<answer>\\boxed{integer}</answer>` and separately checks exactly one
nonempty latent span inside a think block, with plain text on both sides. Output
budgets are fixed at 32/256 tokens. Truncated answers cannot count as success even
when the parsed number happens to be correct. Raw outputs and separate answer,
format, truncation and joint-success counts are saved as JSONL and summary JSON.

Compare the base and trained validation results first. If selecting between saved
checkpoints, use validation only, record the choice, then explicitly run
`--split test` once for the selected checkpoint and its matching base. Test is never
evaluated automatically. No automatic claim of improvement or model selection is
made by the script. Large-model training/evaluation must be checked on the target
GPU; dependency-free tests cover dataset isolation and scoring edge cases only.


The first assistant round must start with `<think>`; a following newline is
optional. If the chat template already supplies `<think>\n`, the loader removes
only the actual opening tag and one optional newline from the assistant text.
This accepts the generated pilot records without changing their content or hashes.
`validate` checks all training records with the real tokenizer without loading
model weights. The trainer also validates its dataset before loading weights.

If a run failed before creating `checkpoint/`, preserve `train.log` and
`training-run.json` under new names before retrying. Do not regenerate the data
or delete existing baseline evaluation. If checkpoints exist, inspect them first:
this wrapper does not resume interrupted training.

For a fixed-checkpoint generalization check, prepare a **separate** evaluation-only
suite. This reads and verifies the existing pilot splits without changing them:

```bash
python scripts/format_experiment.py prepare-generalization \
  --pilot /content/format-pilot --directory /content/format-generalization
python scripts/format_experiment.py evaluate --directory /content/format-generalization \
  --model Qwen/Qwen3.8-27B --revision <same-base-commit-as-training> \
  --split generalization --output /content/format-generalization/base.jsonl
python scripts/format_experiment.py evaluate --directory /content/format-generalization \
  --model /content/format-pilot/checkpoint --split generalization \
  --output /content/format-generalization/trained.jsonl
```

The suite fixes eight new operand tuples, excluding all pilot splits, within the
same number ranges. Each tuple produces four questions: original-template control,
reworded equivalent (direct wording or a bead-counting story), parentheses changing
operation order, and a new structure (addition with multiplication or subtraction
with a negative answer). These are **eight paired problems, not 32 independent
problems**. The format system instruction, scoring and token budgets stay unchanged.
Reports include both overall and per-group counts; raw outputs retain `group` and
`pair_id` for paired inspection. Correct final answers do not verify intermediate
calculations. This remains a small synthetic arithmetic probe, not a broad
capability or latent-reasoning benchmark. Keep the checkpoint fixed and do not use
these results for further tuning or checkpoint selection.

### Fixed latent-generation pilot

`scripts/latent_experiment.py` reuses the ordinary text evaluator for the control
and the production `HybridGenerationEngine` for real recursive vector inputs.
Preparation excludes operand tuples from all original splits and the generalization
probe. It creates 8 development questions and 16 reserved test questions. It does
not create training data or launch training. Keep the reserved test unopened until
training choices and the final checkpoint are fixed.

```bash
python scripts/latent_experiment.py prepare --directory /content/latent-pilot \
  --pilot /content/format-pilot --generalization /content/format-generalization \
  --calibration /content/qwen-link-smoke/link_calibration.json
python scripts/latent_experiment.py evaluate --directory /content/latent-pilot \
  --model /content/format-pilot/checkpoint --reference /content/format-pilot/checkpoint \
  --mode text --output /content/latent-pilot/text-dev.jsonl
python scripts/latent_experiment.py evaluate --directory /content/latent-pilot \
  --model /content/format-pilot/checkpoint --reference /content/format-pilot/checkpoint \
  --mode hybrid --output /content/latent-pilot/untrained-dev.jsonl
python scripts/latent_experiment.py evaluate --directory /content/latent-pilot \
  --model /content/qwen-link-smoke --reference /content/format-pilot/checkpoint \
  --mode hybrid --output /content/latent-pilot/one-update-dev.jsonl
```

Run these sequentially without another model occupying the GPU. The one-update
checkpoint is a plumbing diagnostic, not a trained-reasoning result. A future
trained checkpoint can use the same evaluator without changing data or limits.
`--split latent-test` explicitly selects the reserved split; the default is dev.

All conditions use the existing format instruction unchanged, including its
text-inside-latent wording, so this probe measures that prompt rather than a newly
optimized vector prompt. Hybrid conditions verify that LoRA/anchor tensors,
base-reference metadata and tokenizer match the fixed format reference. Both use
FP32 HybridLink and the **same** scale from the training-only calibration manifest,
overriding the checkpoint scale in memory only. This avoids comparing an unscaled
random link against a scaled updated one. Scale is magnitude calibration, not link
training. Original checkpoints are never modified.

Generation is greedy, single-request, with tools disabled, a total budget of 256
text/vector steps and at most 64 vector steps per span. Minimum vector steps is
zero: the evaluator never forces the model to stay in latent mode. Entry is not
forced. Existing engine structural token masks remain enabled and differ from
ordinary text decoding; only the two hybrid conditions share exactly the same
engine constraints. This is not an unrestricted-generation or speed comparison.

Raw outputs, scored text and engine counters are saved per question. Only terminal
EOS is removed from structural scoring; an opening think prefix supplied by the
chat template is restored. Real vector spans appear as `<latent></latent>` because
vectors are not decoded as text. Strict success requires the right boxed answer,
one well-formed span with actual vector steps, an anchor exit, no forced exits or
tools, and EOS termination. Empty immediate exits, no entry, budget truncation and
forced exits never count as success, even with a correct final answer. Counts for
answer correctness, structure, vector use, exits and termination are reported
separately. Correct answers still do not establish that latent computation was
necessary; the model can perform work in visible text before or after the span.

Tests cover disjoint reproducible splits, rejection of non-training calibration,
and scoring edge cases. A scripted-output test drives the real engine to verify
vector feeds, natural/forced exits, budget stops and intact answer digits; it is
an engine test, not evidence of learned reasoning. Large-model runs remain a
separate target-GPU check.

### Controlled substitution diagnostic

When recursive generation fails despite lower activation loss, use
`scripts/latent_diagnostic.py` on the fixed trained checkpoint. It uses only the
first two `latent-dev` questions and never reads the reserved split:

```bash
python -u scripts/latent_diagnostic.py --directory /content/latent-pilot \
  --model /content/qwen-link-64 --reference /content/format-pilot/checkpoint \
  --output /content/latent-pilot/diagnostic-64.jsonl
```

Each question has four fresh-cache conditions: all correct reference embeddings,
first one predicted vector, first four predicted vectors, and all predicted vectors.
The prefix through `<latent>` is supplied from the known trajectory. Every condition
uses identical segmentation (prompt prefill, then one assistant embedding per step).
Predicted vectors are cast to the actual embedding dtype before both feeding and
measurement. Only the first N latent-content inputs are substituted; remaining
inputs use the correct reference tokens, allowing recovery to be observed.

For every reference position and the expected exit position, JSONL records the
next-token argmax, target probability, exit probability/rank, and cosine/RMS of the
predicted vector versus the reference next-token embedding. At the reference span
length, the real exit-anchor embedding is **forced** in all conditions; ordinary
unmasked text generation then runs for at most 96 tokens. The continuation does
not activate latent mode even if it emits more latent tags. It is a recovery probe,
not the production hybrid engine or a speed test. Exit rank 1 can include ties;
`exit_argmax` separately records whether argmax actually selected the exit.

The summary includes initial and exit-position measurements, early exit argmax
positions and full continuations. Exact trajectories and answer lengths are known
to this diagnostic, so none of its outcomes count as autonomous reasoning success.
It compares sequential execution, not full-sequence training execution; further
segmentation comparisons may be needed if the all-reference control fails. No
training, checkpoint modification, retuning, or change to evaluation rules occurs.

For a direction-versus-magnitude diagnostic, use a new output file and
`--suite direction-size` on the same checkpoint and two development questions:

```bash
python -u scripts/latent_diagnostic.py --directory /content/latent-pilot \
  --model /content/qwen-link-64 --reference /content/format-pilot/checkpoint \
  --suite direction-size --output /content/latent-pilot/direction-size-64.jsonl
```

This runs four fresh-cache conditions: `teacher` (reference embeddings), `all`
(unmodified predictions), `pred-direction` (prediction multiplied by reference RMS
/ prediction RMS), and `target-direction` (reference embedding multiplied by
prediction RMS / reference RMS). Scaling uses FP32 arithmetic and the result is
cast to the actual input dtype before feeding. Each condition recomputes its own
predictions from its current state; magnitudes are not replayed from another run.
This therefore compares recursive input policies, not identical hidden states at
all later steps. The teacher/all controls are rerun under the same code.

All latent-content positions are substituted in the two mixed conditions, but the
exit remains the same forced **real** anchor in every condition. Zero/nonfinite
vectors are rejected. JSONL records actual fed RMS, target RMS ratio and cosine
after casting, in addition to the unmodified prediction measurements. Summaries
list reference-token mismatch positions. Reference direction or magnitude uses
known correct tokens: these conditions are diagnostic oracles, not deployable
inference methods. Improvement supports a role for that intervention in these two
trajectories; it does not establish general reasoning gains or an exclusive cause.
The default substitution suite and all autonomous evaluation rules are unchanged.
