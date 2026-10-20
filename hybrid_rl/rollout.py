"""Hybrid Policy Rollout: sample a group of G hybrid
trajectories for one prompt with the hybrid generation engine, recording the latent
inputs z = R(h), the sampled text tokens y with their rollout log-probs and sampling
support, and the tool observations o.
"""
from __future__ import annotations

import torch

from hybrid_agent.compat import check_execution
from hybrid_agent.engine import DONE, HybridGenerationEngine, RowState
from hybrid_rl.config import MAX_SEQUENCE_TOKENS, MAX_TURNS, TEMPERATURE, TOP_K, TOP_P
from hybrid_rl.trajectory import HybridTrajectory, TextAction


class HybridPolicyRollout(HybridGenerationEngine):
    """The generation engine plus trajectory recording, one HybridTrajectory per row."""

    def __init__(self, *args, max_sequence_tokens: int = MAX_SEQUENCE_TOKENS, **kwargs):
        model = args[0] if args else kwargs["model"]
        check_execution(model.backbone, rl=True)
        super().__init__(*args, **kwargs)
        self.max_sequence_tokens = int(max_sequence_tokens)
        if self.max_sequence_tokens < 2:
            raise ValueError("max_sequence_tokens must be at least 2")
        self.trajs = [HybridTrajectory(max_sequence_tokens=self.max_sequence_tokens) for _ in range(self.B)]
        self._blocked_tuples = {
            key: tuple(sorted(int(t) for t in ids)) for key, ids in self.blocked_token_sets.items()
        }

    # ---- engine hooks ------------------------------------------------------------

    def _prompt_length_ok(self, n: int) -> bool:
        if n >= self.max_sequence_tokens:
            for row, traj in zip(self.rows, self.trajs):
                traj.valid = False
                traj.invalid_reason = f"prompt length {n} reaches the sequence cap {self.max_sequence_tokens}"
                row.mode = DONE
            return False
        return True

    def _sequence_cap_ok(self, row: RowState) -> bool:
        if row.true_len >= self.max_sequence_tokens:
            row.extra["budget_truncated"] = True
            row.end_reason = row.end_reason or "context"
            row.mode = DONE
            return False
        return True

    def _env_tokens_fit(self, row: RowState, n_new: int) -> bool:
        if row.true_len + n_new > self.max_sequence_tokens:
            traj = self.trajs[row.idx]
            traj.valid = False
            traj.invalid_reason = (
                f"tool observation would exceed the sequence cap: "
                f"{row.true_len}+{n_new}>{self.max_sequence_tokens}"
            )
            row.end_reason = "invalid:observation_overflow"
            row.mode = DONE
            return False
        return True

    def _record_text_action(self, row: RowState, token_id: int,
                            log_probs_row: torch.Tensor, blocked_key: str) -> None:
        traj = self.trajs[row.idx]
        traj.text_actions.append(TextAction(
            pos=row.true_len - 1,
            token_id=int(token_id),
            logp=log_probs_row[token_id].detach().clone(),
            blocked_ids=self._blocked_tuples[blocked_key],
        ))
        traj.n_text_tokens += 1

    def _record_latent_vec(self, row: RowState, injected_row: torch.Tensor) -> None:
        self.trajs[row.idx].latent_inputs[row.true_len - 1] = injected_row.detach().reshape(-1).contiguous().clone()

    def _on_latent_span_opened(self, row: RowState) -> None:
        self.trajs[row.idx].n_latent_spans += 1

    def _on_env_token_fed(self, row: RowState) -> None:
        self.trajs[row.idx].observation_slots.add(row.true_len - 1)

    def _row_invalidate(self, row: RowState, reason: str) -> None:
        traj = self.trajs[row.idx]
        if traj.valid:
            traj.valid = False
            traj.invalid_reason = str(reason)
        super()._row_invalidate(row, reason)

    # ---- finish ------------------------------------------------------------------

    def finish_row(self, row: RowState) -> HybridTrajectory:
        """Pack the recorded trajectory, check its structure, and apply the budget crop."""
        traj = self.trajs[row.idx]
        result = self.row_result(row)
        traj.prompt_len = self.prompt_len
        traj.prompt_ends_in_think = self.prompt_ends_in_think
        traj.n_latent_steps = row.latent_steps_total
        traj.slot_tokens = list(row.kv_ledger)

        # one device-to-host transfer per trajectory
        if traj.latent_inputs:
            positions = sorted(traj.latent_inputs)
            packed = torch.stack([traj.latent_inputs[p] for p in positions]).detach().to(device="cpu").contiguous()
            traj.latent_inputs = {p: packed[i] for i, p in enumerate(positions)}
        if traj.text_actions:
            tensor_logps = [a.logp.reshape(()) for a in traj.text_actions if isinstance(a.logp, torch.Tensor)]
            if tensor_logps:
                if len(tensor_logps) != len(traj.text_actions):
                    traj.valid = False
                    traj.invalid_reason = "mixed action-logp storage"
                else:
                    packed_logps = torch.stack(tensor_logps).detach().to("cpu")
                    for action, logp in zip(traj.text_actions, packed_logps):
                        action.logp = float(logp)
                    if not bool(torch.isfinite(packed_logps).all()):
                        traj.valid = False
                        traj.invalid_reason = "nonfinite rollout action logp"

        traj.output = row.full_output
        traj.n_tool_calls = int(result["python_calls"]) + int(result["search_calls"])
        traj.turns = int(result["turns"])

        if len(traj.slot_tokens) != row.true_len:
            traj.valid = False
            traj.invalid_reason = "slot/kv desync"
        elif any(pos >= len(traj.slot_tokens) or traj.slot_tokens[pos] is not None for pos in traj.latent_inputs):
            traj.valid = False
            traj.invalid_reason = "latent/slot desync"
        elif len(traj.slot_tokens) > self.max_sequence_tokens:
            traj.valid = False
            traj.invalid_reason = "trajectory exceeds the sequence cap"
        elif any(a.pos in traj.observation_slots for a in traj.text_actions):
            traj.valid = False
            traj.invalid_reason = "observation slot recorded as a policy action"
        elif any(not a.blocked_ids for a in traj.text_actions):
            traj.valid = False
            traj.invalid_reason = "text action missing its support mask"
        elif any(tuple(sorted(set(a.blocked_ids))) != a.blocked_ids for a in traj.text_actions):
            traj.valid = False
            traj.invalid_reason = "text action support mask is not canonical"
        elif any(int(a.token_id) in set(a.blocked_ids) for a in traj.text_actions):
            traj.valid = False
            traj.invalid_reason = "sampled action is outside the rollout support"

        traj.response_len = len(traj.slot_tokens) - traj.prompt_len

        # a tool observation may overshoot the budget: crop back to the cap unless a
        # scored action would be removed
        cap = self.max_response_tokens if self.count_tool_tokens else 0
        if cap > 0 and traj.response_len > cap:
            keep = traj.prompt_len + cap
            beyond = [a.pos for a in traj.text_actions if a.pos >= keep]
            if beyond:
                traj.valid = False
                traj.invalid_reason = f"crop would remove {len(beyond)} scored actions"
            else:
                traj.cropped_slots = traj.response_len - cap
                traj.slot_tokens = traj.slot_tokens[:keep]
                traj.observation_slots = {p for p in traj.observation_slots if p < keep}
                traj.latent_inputs = {p: v for p, v in traj.latent_inputs.items() if p < keep}
                traj.response_len = cap
        return traj


def rollout_group(*, model, tokenizer, device, tool_executor, system_prompt, prompt, cfg, rng) -> list[HybridTrajectory]:
    """One group of cfg.group_size trajectories for `prompt` under the rollout settings."""
    processor = HybridPolicyRollout(
        model=model,
        tokenizer=tokenizer,
        device=device,
        tool_executor=tool_executor,
        system_prompt=system_prompt,
        question=prompt.question,
        batch_size=int(cfg.group_size),
        max_response_tokens=cfg.max_response_tokens,
        count_tool_tokens=True,
        max_tool_calls=cfg.max_tool_calls,
        max_search_calls=cfg.max_tool_calls,
        max_python_calls=cfg.max_tool_calls,
        terminate_on_tool_cap=True,
        max_turns=MAX_TURNS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        min_latent_steps=cfg.min_latent_steps,
        max_latent_steps=cfg.max_latent_steps,
        think_guards=False,
        reject_fabricated_result=False,
        max_sequence_tokens=MAX_SEQUENCE_TOKENS,
    )
    device_type = torch.device(device).type
    with torch.no_grad(), torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=device_type == "cuda"):
        processor.run(rng)
    return [processor.finish_row(row) for row in processor.rows]
