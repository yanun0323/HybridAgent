"""Exact Rollout Replay: rebuild the full hybrid
trajectory in embedding space in one differentiable forward pass with the recorded
latent inputs fixed, and recompute log pi_theta at the sampled text positions only
(Hybrid Policy Factorization). Latent slots and tool observations are conditioning
context, never likelihood targets.
"""
from __future__ import annotations

import math

import torch
import torch.utils.checkpoint

from hybrid_agent.compat import check_execution
from hybrid_rl.config import LOGIT_CHUNK_SIZE, TEMPERATURE
from hybrid_rl.trajectory import HybridTrajectory, TextAction


def _check_trajectory(traj: HybridTrajectory, vocab_size: int) -> tuple[list[TextAction], int]:
    """Structural checks of a recorded trajectory; returns its actions and replay length."""
    if not traj.valid:
        raise ValueError(f"cannot replay an invalid trajectory: {traj.invalid_reason}")
    seq_len = len(traj.slot_tokens)
    prompt_len = int(traj.prompt_len)
    if prompt_len <= 0 or prompt_len > seq_len:
        raise ValueError(f"invalid prompt_len={prompt_len} for sequence length {seq_len}")
    if traj.max_sequence_tokens > 0 and seq_len > traj.max_sequence_tokens:
        raise ValueError(f"trajectory length {seq_len} exceeds the recorded cap {traj.max_sequence_tokens}")

    for pos in traj.observation_slots:
        if not prompt_len <= int(pos) < seq_len or traj.slot_tokens[int(pos)] is None:
            raise ValueError(f"invalid observation slot {pos!r}")
    latent_positions: set[int] = set()
    for pos, vector in traj.latent_inputs.items():
        pos = int(pos)
        if not prompt_len <= pos < seq_len or traj.slot_tokens[pos] is not None:
            raise ValueError(f"invalid latent position {pos}")
        if not isinstance(vector, torch.Tensor) or vector.ndim != 1 or not bool(torch.isfinite(vector).all()):
            raise ValueError(f"invalid latent input at {pos}")
        latent_positions.add(pos)
    for pos, token_id in enumerate(traj.slot_tokens):
        if token_id is None:
            if pos not in latent_positions:
                raise ValueError(f"latent slot {pos} has no recorded input")
        elif pos in latent_positions or not 0 <= int(token_id) < vocab_size:
            raise ValueError(f"invalid token slot {pos}")

    actions = list(traj.text_actions)
    if int(traj.n_text_tokens) != len(actions):
        raise ValueError(f"n_text_tokens={traj.n_text_tokens} does not match {len(actions)} recorded actions")
    if not actions:
        return [], 0
    positions = [int(a.pos) for a in actions]
    if positions != sorted(positions) or len(positions) != len(set(positions)):
        raise ValueError("text action positions must be strictly increasing")
    if positions[0] < max(prompt_len, 1) or positions[-1] >= seq_len:
        raise ValueError("text action positions must lie inside the response")
    for action, pos in zip(actions, positions):
        if traj.slot_tokens[pos] is None or int(traj.slot_tokens[pos]) != int(action.token_id):
            raise ValueError(f"action/slot mismatch at {pos}")
        if pos in traj.observation_slots:
            raise ValueError(f"observation slot {pos} cannot be a policy action")
        blocked = tuple(int(t) for t in action.blocked_ids)
        if not blocked or tuple(sorted(set(blocked))) != blocked:
            raise ValueError(f"action at {pos} has no canonical support mask")
        if int(action.token_id) in set(blocked) or not 0 <= int(action.token_id) < vocab_size:
            raise ValueError(f"action token at {pos} is outside the rollout support")
        if not math.isfinite(float(action.logp)):
            raise ValueError(f"non-finite rollout log probability at {pos}")
    # scoring the action at p needs hidden[p-1]: the replayed prefix ends at p-1
    return actions, positions[-1]


def _apply_support_masks(logits: torch.Tensor, actions: list[TextAction]) -> torch.Tensor:
    masked = logits.clone()
    for row, action in enumerate(actions):
        if action.blocked_ids:
            idx = torch.tensor(action.blocked_ids, device=masked.device, dtype=torch.long)
            masked[row].index_fill_(0, idx, float("-inf"))
    return masked


def exact_rollout_replay(
    model, traj: HybridTrajectory, device, *, with_grad: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (replayed log-probs, rollout log-probs) of the trajectory's text actions."""
    device = torch.device(device)
    lm = model.backbone
    check_execution(lm, rl=True)
    embedding = lm.get_input_embeddings()
    output_head = lm.get_output_embeddings()
    actions, replay_length = _check_trajectory(traj, int(embedding.weight.shape[0]))
    empty = torch.zeros(0, device=device, dtype=torch.float32)
    if not actions:
        return empty, empty

    decoder = lm.get_decoder()
    token_ids = [int(t) if t is not None else 0 for t in traj.slot_tokens[:replay_length]]
    input_ids = torch.tensor([token_ids], device=device, dtype=torch.long)
    latent_positions = sorted(p for p in traj.latent_inputs if p < replay_length)

    def autocast():
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")

    grad_context = torch.enable_grad() if with_grad else torch.no_grad()
    with grad_context, autocast():
        inputs_embeds = embedding(input_ids)
        if latent_positions:
            # recorded latent inputs are constants: they condition the backbone, no gradient flows into them
            latents = torch.stack([traj.latent_inputs[p] for p in latent_positions])
            latents = latents.to(device=device, dtype=inputs_embeds.dtype).detach()
            inputs_embeds = inputs_embeds.index_copy(
                1, torch.tensor(latent_positions, device=device, dtype=torch.long), latents.unsqueeze(0)
            )
        hidden = decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=torch.ones((1, replay_length), device=device, dtype=torch.long),
            use_cache=False,
            return_dict=True,
        ).last_hidden_state[0]

    rows = torch.tensor([int(a.pos) - 1 for a in actions], device=device, dtype=torch.long)
    targets = torch.tensor([int(a.token_id) for a in actions], device=device, dtype=torch.long)
    old_text_logp = torch.tensor([float(a.logp) for a in actions], device=device, dtype=torch.float32)

    def score_chunk(chunk_hidden: torch.Tensor, offset: int) -> torch.Tensor:
        head_input = chunk_hidden
        if device.type != "cuda":
            head_dtype = next(output_head.parameters()).dtype
            if head_input.dtype != head_dtype:
                head_input = head_input.to(head_dtype)
        with autocast():
            logits = output_head(head_input).float()
        logits = _apply_support_masks(logits, actions[offset : offset + logits.shape[0]])
        chunk_targets = targets[offset : offset + logits.shape[0]]
        return torch.log_softmax(logits / max(TEMPERATURE, 1e-5), dim=-1).gather(1, chunk_targets[:, None]).squeeze(1)

    scored: list[torch.Tensor] = []
    with grad_context:
        for start in range(0, rows.numel(), LOGIT_CHUNK_SIZE):
            selected = hidden[rows[start : start + LOGIT_CHUNK_SIZE]]
            if with_grad and selected.requires_grad:
                out = torch.utils.checkpoint.checkpoint(
                    lambda value, offset=start: score_chunk(value, offset), selected, use_reentrant=False,
                )
            else:
                out = score_chunk(selected, start)
            scored.append(out)
    text_logp = torch.cat(scored) if scored else empty
    if text_logp.shape != old_text_logp.shape:
        raise RuntimeError(f"replay/rollout action count mismatch: {text_logp.shape} vs {old_text_logp.shape}")
    return text_logp, old_text_logp


def replay_agreement(text_logp: torch.Tensor, old_text_logp: torch.Tensor) -> tuple[float, int]:
    """Sum of |log pi_theta - log pi_theta_old| over the actions and their count."""
    with torch.no_grad():
        return float((text_logp.detach() - old_text_logp).abs().sum()), int(text_logp.numel())
