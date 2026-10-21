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


## Experimental LoRA + HybridLink two-pass backward probe

`scripts/lora_self_conditioned_probe.py` tests one full trajectory using the existing
self-conditioned two-pass objective: text/anchor CE plus latent regression and
extra exit regression. First-pass predictions replace the shifted latent inputs
(including the exit input) in pass two; gradients flow through both passes. These
predictions come from teacher-forced history, not a free-running recursive rollout.

The script explicitly enables only the saved LoRA A/B tensors and FP32 HybridLink.
Original text/vision weights and both anchor deltas remain frozen. This is a
resource-saving adaptation, not official full-parameter training. The normal
training CLI and default `set_trainable` behavior still reject LoRA
self-conditioning. No optimizer, parameter update or model export is provided.

After restoring and verifying the original Colab artifacts, and passing the full
test suite without skips, the separately authorized GPU probe is:

```bash
python -u scripts/lora_self_conditioned_probe.py \
  --pilot /content/format-pilot --model /content/qwen-link-64 \
  --calibration /content/qwen-link-smoke/link_calibration.json \
  --output-dir /content/qwen-lora-self-conditioned-probe
```

It requires one CUDA process without DeepSpeed and verifies the original 64-step
link and saved LoRA hashes before model loading. It selects only the first training
record (`mul-sub-54-5-6`) in fixed file order, preserves the full text/latent masks,
uses no padding, keeps the recorded output scale, and enables non-reentrant decoder
checkpointing. No development or reserved test data is read. Do not run alongside
another loaded 27B model. Existing/overlapping output paths are rejected; preserve
failed output and logs, with no automatic retries.

`run.json` records source-file hashes, data/tokenization hashes, versions, GPU and
settings; `forward.json` is written before backward; `summary.json` records losses,
LoRA/Link gradient norms and peak allocated/reserved GPU memory. Every trainable
tensor must have finite gradients and each group a nonzero gradient. Individual
zero gradients are allowed. All parameter versions and frozen-parameter absence
of gradients are checked; LoRA/Link/anchor values are compared exactly on CPU, and
source-file hashes are rechecked. Version checks are not a full bytewise comparison
of the 27B backbone. A successful probe proves neither learning nor natural exits,
and does not establish memory feasibility once optimizer state is allocated.

## Experimental single-GPU LoRA format activation

This is a resource-saving adaptation, not a reproduction of full-parameter
HybridSFT. During `format_activation`, HybridLink stays frozen. A saved LoRA
checkpoint can also initialize frozen `hybridlink_activation` (see below).
The normal trainer still rejects LoRA self-conditioning; a standalone backward-only
probe is available above. LoRA RL remains unsupported.
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

To compare the original 64-step link with **10% of the saved joint update** on
these same two questions, use paired mode after the fixed-fraction probe:

```bash
python -u scripts/latent_diagnostic.py --directory /content/latent-pilot \
  --model /content/qwen-link-64 --reference /content/format-pilot/checkpoint \
  --joint-run /content/qwen-link-joint-8 \
  --step-size-run /content/qwen-link-step-size \
  --output /content/latent-pilot/diagnostic-joint-10pct.jsonl
```

This loads the backbone once, runs all four substitution conditions with the
original link, then repeats with FP32 `before + 0.1 * (after - before)` in memory.
Every condition starts with a fresh cache. Fractions are fixed at `0` and `0.1`;
there is no training, fraction search, checkpoint selection, or weight export.
The source/updated link and joint-run hashes are checked against the completed
step-size probe; tokenizer, LoRA/anchor tensors and scales must also match.

Each result includes `fraction`. JSONL retains every step; the summary includes
mismatch positions, exit rank/probability/argmax and forced-exit continuation.
Teacher-input predictions and continuation must match exactly between fractions,
otherwise the script stops. Link-vector cosine/RMS may differ even in teacher
conditions because the link is still measured but not fed. The source link hash
in the summary names the original artifact; `paired_update` records both endpoint
hashes and the fixed interpolation rule's fractions. This remains a known-prefix,
known-length, forced-exit diagnostic, not autonomous generation. Its sequential
cached execution differs from the full-prefix next-token CE probe.

To isolate the third content input at the **fixed 10% update**, use the
`third-input` suite with the same source arguments:

```bash
python -u scripts/latent_diagnostic.py --directory /content/latent-pilot \
  --model /content/qwen-link-64 --reference /content/format-pilot/checkpoint \
  --joint-run /content/qwen-link-joint-8 \
  --step-size-run /content/qwen-link-step-size --suite third-input \
  --output /content/latent-pilot/diagnostic-third-input-10pct.jsonl
```

This suite runs only fraction `0.1`, on the same first two development questions.
Its four fresh-cache conditions are `teacher` (no replacements), `first-2`
(replace positions 0 and 1), `third-only` (replace position 2), and `first-3`
(replace positions 0, 1 and 2). All later content inputs are correct reference
embeddings. Step 3 measures the prediction **after** those three inputs, before
feeding the fourth. The summary includes the first six steps and replacement
positions, as well as exit metrics and the forced-exit continuation. Comparing
`first-2` to `first-3`, and `teacher` to `third-only`, tests the third input under
predicted versus correct preceding inputs. Each predicted vector is recomputed
from its own condition's hidden state; it is not a fixed vector transplanted
across contexts. Results cannot alone prove a context-independent defect in the
third vector or establish autonomous reasoning. No baseline fraction is rerun,
so `teacher_controls_identical` is null (no cross-fraction check), not true.
No training or model export occurs; checkpoint/scale checks are unchanged.

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

### One-input next-token gradient smoke test

`link_next_token` is an experimental **one-update-only** path, separate from both
activation regression and full self-conditioning. It keeps the backbone, LoRA and
anchor parameters frozen in eval mode and updates only FP32 HybridLink. For the
first latent span it supplies the correct prefix through the entry anchor, predicts
the first content embedding, feeds that prediction, and computes cross-entropy
for the following token. No target or future token enters the decoder inputs.

The teacher prefix pass has no gradients. The second decoder pass has input
gradients enabled despite frozen weights; non-reentrant whole-pass checkpointing
recomputes that pass on backward. It uses no cache, so no mutable recurrent cache
is reused during recomputation. It does **not** yet reproduce sequential cached
inference or train a multi-step recursive rollout. Whole-pass checkpointing does
not guarantee low peak memory during backward; measure on the target A100.

```bash
python -u -m hybrid_sft.train --stage link_next_token \
  --model /content/qwen-link-64 --data /content/format-pilot/train.jsonl \
  --output_dir /content/qwen-link-next-token-smoke \
  --link_calibration /content/qwen-link-smoke/link_calibration.json \
  --hybridlink_lr 5e-5 --epochs 1 --batch_size 1 --grad_accum 1 --max_steps 1 \
  --max_length 1024 --num_workers 0 --save_steps 0 --seed 6064
```

The CLI rejects larger step counts, batches or accumulation for this smoke path.
Start from the fixed 64-step link checkpoint and reuse its training-only scale.
The printed target ID/position makes the one-step alignment inspectable. The loss
is **only next-token CE**, not cosine/MSE; their logged zeros mean omitted terms,
not perfect alignment. `backbone_loss=0` does not describe text accuracy. Finite
nonzero link gradients and no backbone gradients are checked before the update.
Saved output is separate from its source and retains the compact LoRA format.

The tiny-model regression test checks actual CE gradients through the frozen
PEFT decoder, exact input/target positions, checkpoint recomputation, exclusion of
future tokens, changed link weights, unchanged backbone weights/text logits and
save/load equality. Pass this in Colab before the real 27B smoke. Passing one step
establishes only differentiability and resource feasibility, not improved latent
reasoning. No long training or reserved-test evaluation is authorized by this path.

### Eight-record joint-update probe

`scripts/link_joint_step.py` starts from the original 64-step activation checkpoint,
not the single-example next-token update. It selects the first eight verified
training records for one update and the next eight for observation only. The latter
are **not held-out examples**: they were available to earlier training. No development
or reserved-test records are loaded. Selection is fixed by file order, not losses.

```bash
python -u scripts/link_joint_step.py --pilot /content/format-pilot \
  --model /content/qwen-link-64 \
  --calibration /content/qwen-link-smoke/link_calibration.json \
  --output-dir /content/qwen-link-joint-8
```

The objective is the same first-input next-token CE as the smoke path. Each loss
is divided by eight and backpropagated separately, freeing its graph before the
next record. After all eight, gradients are clipped once and AdamW takes exactly
one step at `5e-5` with the previous optimizer settings. There is no warmup or
scheduler in this one-step probe. Backbone, LoRA and anchors remain frozen; scale
is reused with training-data hash validation. Peak memory still needs target-GPU
measurement; graphs are not held for eight records simultaneously.

`run.json` records IDs, hashes, source, settings and code version before model
loading. `before.json` preserves measurements before the update. `summary.json`
reports each record's before/after CE, objective positions and target ID, mean CE,
and counts of decreases/increases (tolerance `1e-6`) separately for the update and
observation groups. Measurements use the same grad-enabled, mixed-precision forward
path as training, without backward for observation or post-update measurements.
Only HybridLink is updated; saved LoRA/anchor tensors are checked against the source.
The compact result is saved to `checkpoint/` within the new output directory.
Existing directories are rejected; this script does not resume failed runs.

This changes both the selected records and their joint objective relative to the
prior single-record probe. It is not a controlled comparison of accumulation alone,
and lower average CE can hide individual regressions. No decrease is guaranteed,
and these first-position training-set measurements are not answer accuracy,
recursive-generation success, or generalization evidence.

### Saved-update step-size probe

If the joint update raises even its own mean CE, `scripts/link_step_size.py` probes
fixed fractions of the saved parameter displacement without computing gradients,
creating an optimizer, or saving new model weights:

```bash
python -u scripts/link_step_size.py --pilot /content/format-pilot \
  --model /content/qwen-link-64 --joint-run /content/qwen-link-joint-8 \
  --output-dir /content/qwen-link-step-size
```

The probe loads 27B once and swaps only HybridLink tensors in memory. Fractions
are fixed at `0, 1, 0.01, 0.1` in execution order: first replay both endpoints on
the same sixteen records and objective positions. Per-record endpoint CE must
match the saved joint-run summary within `rtol=1e-4, atol=1e-5`, otherwise the
script stops before measuring fractions. This is a fixed numerical check, not a
criterion to tune after seeing results. Original endpoint tensors are copied
exactly; intermediate tensors use FP32 `before + fraction * (after - before)`.

The input file, source weights, row IDs, tokenizer, LoRA/anchor tensors and scale
are checked against the joint experiment. Grad-enabled mixed-precision forward
matches that experiment; graphs are discarded per record without backward.
`run.json` records source hashes and settings, `measurements.jsonl` streams all CE
measurements, and `summary.json` reports per-group means and individual changes.
Existing outputs are rejected. There is no automatic model selection or checkpoint
export, and no development or test split is read.

This examines the actual saved AdamW update direction, not a raw-gradient finite
difference test. Smaller fractions lowering training CE would support overshooting
on this local update, not prove generalization or recursive latent success. BF16
input quantization can hide small parameter changes. Observation rows remain
training-set observations, not held-out evaluation.

After completing that probe, the same script can check the remaining 48 training
records at the **fixed** fraction `0.1` versus `0`, with no further fraction search:

```bash
python -u scripts/link_step_size.py --pilot /content/format-pilot \
  --model /content/qwen-link-64 --joint-run /content/qwen-link-joint-8 \
  --remaining-train --step-size-run /content/qwen-link-step-size \
  --output-dir /content/qwen-link-remaining-48
```

This mode requires exactly 64 unique training IDs and selects records 17–64 in
file order. It verifies data, source/updated link, and joint-run hashes against
the completed step-size probe. For each fraction it first replays the original
16 control records against that probe's saved per-record CE; a mismatch stops
execution before measuring the remaining 48 at that fraction. Both fractions
use the same grad-enabled forward path, with no backward or optimizer.

`summary.json` keeps `update`, `observation`, and `remaining_train` separate for
each fraction, including each record's CE and counts of improvements/regressions.
All measurements stream to `measurements.jsonl`. No model is selected or saved.
These 48 records did not participate in the eight-record joint update, but they
were used in earlier training: they are not a held-out test or evidence of
recursive-generation success. The reserved test remains unopened.

### Three- or six-input backward-only probe

Before attempting recursive training, `scripts/link_three_step_probe.py` checks
one backward pass through three predicted inputs on the **first training record**:

```bash
python -u scripts/link_three_step_probe.py --pilot /content/format-pilot \
  --model /content/qwen-link-64 \
  --calibration /content/qwen-link-smoke/link_calibration.json \
  --output-dir /content/qwen-link-three-step-probe
```

This deliberately starts from the original 64-step activation checkpoint, not the
10% interpolated diagnostic weights. It tests gradient feasibility, not whether
that diagnostic improvement is preserved. No optimizer is created, no weights are
updated or exported, and no development/test records are read. The standalone
probe uses `LINK_NEXT_TOKEN` with `link_steps=3` by default; the training CLI remains limited
to its existing single-input smoke path.

The known prefix ends at the entry anchor. Each prediction is fed back through
the frozen decoder to produce the state for the next prediction, without detach.
Loss is the mean of three next-token CEs after feeding one, two, and three vectors
(target positions `entry+2`, `entry+3`, `entry+4`). The third target may be the exit
if the reference contains exactly three content tokens. No reference content is
fed to the decoder, and target IDs are used only for CE. FP32 link outputs are cast
to the embedding dtype for input; their FP32 tensors retain gradients for reporting.
The decoder recomputes the full prefix without a cache, checkpointing each pass
with its own explicit attention mask. This matches the single-input training
probe's execution style, **not** the sequential cached development diagnostic.
A successful backward pass alone does not establish agreement between those paths.

`run.json` saves the fixed sample ID, record/input hashes, data/checkpoint hashes,
scale, seed and code version before model loading. `forward.json` saves per-step
CE and target/substitution positions before backward. `summary.json` reports link
and per-input gradient norms, peak allocated GPU memory and elapsed time. All
selected input gradients must be finite and nonzero. Backbone/LoRA/anchors must remain
frozen, gradient-free and have unchanged tensor versions; link tensors are compared
exactly to their pre-backward copies, and source link/LoRA file hashes are rechecked.
The backbone check avoids duplicating 27B weights and is not a full bytewise
comparison. Outputs must use a new directory; failures are never resumed.

Tests on a tiny LoRA model compare the checkpointed three- and six-step objectives and link
gradients with an independent uncheckpointed rollout, verify that the final loss
reaches every input vector, and check that future/target tokens do not leak
into the decoder. Actual A100 memory and runtime remain to be measured. This probe
makes no claim of improved CE, generation, natural exits, or reasoning ability.

For the six-input feasibility check, use the same first training record and original
64-step source, changing only the rollout length and output directory:

```bash
python -u scripts/link_three_step_probe.py --link-steps 6 \
  --pilot /content/format-pilot --model /content/qwen-link-64 \
  --calibration /content/qwen-link-smoke/link_calibration.json \
  --output-dir /content/qwen-link-six-step-probe
```

Six predicted inputs are fed without detach. The objective averages six CEs at
`entry+2` through `entry+7`; no correct content embeddings are fed. The forward
report includes target and predicted text for all six positions. The probe checks
finite nonzero gradients for all six inputs and the link, with the same frozen
weight and source-file checks. It does not start from the eight-update checkpoint,
does not read development/test questions, and does not update/export weights.
The tiny-model tests also check final-loss reachability, independent uncheckpointed
gradient equivalence, no target leakage, and short/discontiguous span rejection at
both lengths. This does not establish six-step cached equivalence or A100 memory
feasibility; those cannot be inferred from the three-step run. Preserve any failed
output and log rather than retrying with changed settings in place.

### Compare full-prefix and cached execution

Before interpreting three- or six-input CE as evidence about cached inference, compare
both paths on the **first eight training records**, original 64-step weights:

```bash
python -u scripts/link_execution_probe.py --pilot /content/format-pilot \
  --model /content/qwen-link-64 \
  --calibration /content/qwen-link-smoke/link_calibration.json \
  --output-dir /content/qwen-link-execution-8
```

The backbone is loaded once. Each record runs `full_grad` through the actual
selected-length objective (graph discarded, no backward), then `full_no_grad` through
the same objective, then `cached_no_grad`. The cached path prefills only the prompt,
feeds assistant prefix tokens one by one through the entry, then feeds the selected
number of recursive link predictions with explicit positions and a fresh cache. No known
latent-content embedding is fed. All paths use BF16 autocast and FP32 HybridLink
with outputs cast to embedding dtype. This holds precision and decoder/head access
constant; the cached segmentation matches the development diagnostic, but this is
not an exact replay of its older inference-mode/no-autocast environment.

`grad_mode_comparison` isolates enabling gradients/checkpointing;
`cache_segmentation_comparison` compares the two no-grad paths. Each path generates
its own vectors, so later differences can include feedback from earlier differences,
not just a single decoder-call difference. JSONL and summary retain each record's
three or six CE values, probabilities, argmax IDs/text, target positions, and cross-path
vector cosine/max-absolute differences. No threshold chooses a winning path or
updates weights. The source files, parameter versions and absence of gradients are
checked after measurement. No development or test data is opened. An observed
mismatch does not by itself identify a kernel bug or establish reasoning ability.

For the six-input comparison, keep the original source and first eight records:

```bash
python -u scripts/link_execution_probe.py --link-steps 6 \
  --pilot /content/format-pilot --model /content/qwen-link-64 \
  --calibration /content/qwen-link-smoke/link_calibration.json \
  --output-dir /content/qwen-link-six-step-execution-8
```

The default remains three inputs, including callers from existing joint-update and
remaining-record scripts. Six-input mode measures all 48 positions on all three
paths, without backward, an optimizer or model export. `run.json` records length,
IDs, input/source hashes and execution settings before loading weights. Measurements
stream to `measurements.jsonl`. The summary includes each path's per-position mean
CE and correct counts, all-position-correct count, and each record's first wrong
position (`null` if none). These positions count from 1: position k is measured
after k predicted inputs. The elapsed time excludes model loading; peak memory
covers the three execution paths. No path is required to match or chosen as a
winner; every discrepancy is retained. Tiny-model tests cover both lengths, fresh
cache segmentation, target alignment, no reference-content leakage, mean CE over
the actual number of positions, and full/cached agreement on an attention-only
control. That control does not establish equality for the 27B Qwen model.

### One eight-record three-input update

After the gradient and execution probes, `scripts/link_three_step_joint.py` takes
exactly one AdamW step from the original 64-step activation checkpoint:

```bash
python -u scripts/link_three_step_joint.py --pilot /content/format-pilot \
  --model /content/qwen-link-64 \
  --calibration /content/qwen-link-smoke/link_calibration.json \
  --output-dir /content/qwen-link-three-step-joint-8
```

Selection is fixed: first eight training records update, next eight observe only.
Each record's three recursive next-token CEs are averaged, divided by eight and
backpropagated separately. The resulting objective is the mean over 24 positions.
Graphs are freed between records; gradients are clipped once and AdamW steps once
at `5e-6`, existing betas/weight decay, with no scheduler or optimizer resume.
Backbone, LoRA and anchors remain frozen. No development/test split is read.

Before and after the step, all sixteen records are measured on the grad-enabled
full-prefix objective and on fresh-cache sequential execution with no gradients,
using common BF16 autocast and FP32 HybridLink. This checks whether improvements
transfer to cached execution, not just the training objective. Both are three-input
rollouts, not complete answers or natural exits. The learning rate is fixed in
advance; improvement is not guaranteed.

`run.json` saves settings, source/data/input hashes and exact IDs. `before.json`,
`gradients.json` and `after.json` preserve measurements around the sole update.
Per-record measurements also stream to phase JSONL files; `updates.jsonl` records
completed updates and gradient reports.
`summary.json` separates update/observation groups and full/cached paths, including
per-position CE, individual changes, counts of regressions, and counts with all
three argmax predictions correct. Lower average CE does not hide the per-position
reports. All observation records were available to earlier training, so this is
not held-out generalization.

The compact checkpoint is written under the new output directory. Saved LoRA and
anchor tensors must equal the source exactly; tokenizer/manifest/scale are checked.
Saved HybridLink tensors must match the in-memory FP32 tensors. Those link tensors
are then reloaded into the existing model, and both execution paths are replayed
on all sixteen records with exact metric equality required. This is a saved-link
replay, not a fresh reload of the 27B backbone. Source hashes are rechecked, and
backbone parameter versions and absence of gradients are checked around the step.
Existing outputs are rejected. Failed runs are not resumed or overwritten.

### One eight-record six-input update

Use the same joint-update script with `--link-steps 6`, starting from the original
64-step activation checkpoint, not either three-input updated checkpoint:

```bash
python -u scripts/link_three_step_joint.py --link-steps 6 \
  --pilot /content/format-pilot --model /content/qwen-link-64 \
  --calibration /content/qwen-link-smoke/link_calibration.json \
  --output-dir /content/qwen-link-six-step-joint-8
```

The first eight records update and the next eight observe only, in fixed file
order. Each record averages six recursive CEs, then divides by eight before
backward: one objective over 48 positions. A fresh AdamW optimizer steps exactly
once at `5e-6` after one gradient clip. No scheduler, resume or loss-based selection
is used. Six-input mode rejects `--all-train`; three-input behavior remains the
default. No development or reserved test data is read.

All sixteen records are measured before/after on the full-prefix grad-enabled path
and cached no-grad path with common BF16 autocast and FP32 link weights. The summary
preserves six per-position reports, full per-record before/after metrics, and
`all_six_correct_before`/`all_six_correct_after` counts for update and observation
groups separately. Terminal output includes any position's CE increase over `1e-6`
or correct-to-wrong prediction, even if that record's average improves. Three-input
reports retain their existing `all_three_correct_*` names.

The existing source hashes, frozen backbone checks, exact saved LoRA/anchor equality,
configuration checks, and saved-link replay on all sixteen records and both paths
remain required. No second 27B backbone is loaded for replay. Settings and streamed
measurements use the same files as the three-input joint update. Only the final
checkpoint is saved. GPU memory for eight accumulated backward passes and optimizer
state must be measured; a successful one-record backward probe does not guarantee
this run fits. Preserve failed output/logs, with no automatic retries or overwrites.
This measures local training-set prediction, not complete reasoning or natural exits.

### Fixed eight-update pass over all 64 training records

The same script accepts `--all-train` for one fixed-order pass starting from the
original 64-step activation checkpoint, not the prior three-input update:

```bash
python -u scripts/link_three_step_joint.py --all-train \
  --pilot /content/format-pilot --model /content/qwen-link-64 \
  --calibration /content/qwen-link-smoke/link_calibration.json \
  --output-dir /content/qwen-link-three-step-pass-64
```

All 64 unique records are used in file order, in eight consecutive groups of eight.
A single fresh AdamW optimizer retains its moments across the eight updates at
fixed `5e-6`. Each update averages 24 position losses, clips once, steps once, and
clears gradients. There is no shuffle, scheduler, resume, intermediate checkpoint,
loss-based selection, or early success stop. Invalid/nonfinite gradients or changed
frozen weights stop the run. Only the final link is exported after eight updates.

All 64 records are measured before any update and after the final update, on both
execution paths, then replayed exactly using saved link tensors. `run.json` records
the eight ordered ID groups. `updates.jsonl` and `gradients-step-N.json` retain each
completed update's gradient report and pre-update metrics. Those group metrics are
measured at different weights and must not be treated as a common baseline.
`before.jsonl`, `after.jsonl`, and `reloaded.jsonl` stream per-record measurements.
The full paired report is in `summary.json` under `update`; `observation` is empty.
Terminal output includes aggregates and any per-position CE regression over `1e-6`
or correct-to-wrong prediction. Every record is training data; no development or
reserved test data is read. Three-position improvement does not establish complete
reasoning or natural exit behavior. Failed outputs and logs must be preserved.

### Remaining 48 records after the three-input update

Use the saved three-input joint checkpoint unchanged, comparing it with the
original 64-step source on records 17–64 in the same training file:

```bash
python -u scripts/link_three_step_remaining.py --pilot /content/format-pilot \
  --model /content/qwen-link-64 --joint-run /content/qwen-link-three-step-joint-8 \
  --output-dir /content/qwen-link-three-step-remaining-48
```

This loads the backbone once and swaps only the two saved FP32 link states in
memory. Both checkpoints first replay all sixteen original controls on both
paths against the joint summary, requiring exact metric equality before any
remaining record is measured. A mismatch stops the run; no tolerances are tuned.
Data hashes, fixed IDs, control tokenization hashes, checkpoint hashes, scale,
tokenizer configuration and LoRA/anchor equality are checked. The full path stays
grad-enabled (graphs discarded without backward), while cached execution uses
no-grad with the same BF16 autocast as the joint experiment.

`run.json` records provenance and selection; `measurements.jsonl` streams controls
and remaining measurements with state/path labels. `summary.json` contains all
48 paired per-record results, mean and per-position CE changes and three-position
correct counts separately for full/cached paths. Terminal output prints totals
and all records with any position's CE increase above `1e-6` or a correct-to-wrong
argmax change, rather than dumping every unchanged record. No backward, optimizer,
model export, fraction search or checkpoint selection occurs. Existing outputs
are rejected, and source files remain unchanged. These records were used in earlier
training; this is not held-out generalization or complete latent reasoning.
