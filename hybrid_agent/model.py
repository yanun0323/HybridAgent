"""HybridAgentModel: a causal-LM backbone paired with HybridLink.

Provides the anchor-token registration, HybridLink application, checkpoint save/load
and a loader that accepts either a local directory or a Hugging Face repo id.
Shared by HybridSFT, HybridRL and inference.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .compat import LINK_CONFIG, backbone_load_path, checked_scale, read_link_config
from .constants import ANCHOR_TOKENS
from .hybridlink import HYBRIDLINK_CONFIG, HybridLink
from .lora import LORA_MANIFEST, LORA_DIRECTORY, is_lora_model

HYBRIDLINK_WEIGHTS = "adapter.pt"
HYBRIDLINK_CONFIG_FILE = LINK_CONFIG


def resolve_model_dir(path_or_repo: str | os.PathLike, revision: str | None = None) -> Path:
    """Local directory as is; otherwise download the Hugging Face repo snapshot."""
    path = Path(path_or_repo)
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=str(path_or_repo), revision=revision))


def add_anchor_tokens(tokenizer) -> int:
    """Register `<latent>` / `</latent>` as special tokens; returns how many were added."""
    known = set(getattr(tokenizer, "all_special_tokens", None) or [])
    missing = [tok for tok in ANCHOR_TOKENS if tok not in known]
    if missing:
        tokenizer.add_special_tokens({"additional_special_tokens": missing})
    return len(missing)


class HybridAgentModel(nn.Module):
    def __init__(self, backbone: nn.Module, hybridlink: HybridLink | None = None) -> None:
        super().__init__()
        self.backbone = backbone
        self.output_scale = 1.0
        self.base_model_reference = None
        self.base_model_revision = None
        hidden_size = backbone.get_input_embeddings().weight.shape[-1]
        self.hybridlink = hybridlink if hybridlink is not None else HybridLink(hidden_size)
        self.hybridlink.to(dtype=next(backbone.parameters()).dtype)

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo: str | os.PathLike,
        *,
        dtype: torch.dtype = torch.bfloat16,
        trust_remote_code: bool = False,
        require_hybridlink: bool = False,
        revision: str | None = None,
    ) -> "HybridAgentModel":
        """Load the backbone and, when present, the HybridLink weights of a checkpoint."""
        local = resolve_model_dir(path_or_repo, revision)
        manifest_path = local / LORA_MANIFEST
        manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
        base_spec = manifest["base_model"] if manifest else str(local)
        base_revision = manifest.get("base_revision") if manifest else revision
        base_local = resolve_model_dir(base_spec, base_revision) if manifest else local
        with backbone_load_path(base_local) as load_path:
            config = AutoConfig.from_pretrained(str(load_path), trust_remote_code=trust_remote_code)
            loader = AutoModelForCausalLM
            if config.model_type == "qwen3_5":
                from transformers import Qwen3_5ForConditionalGeneration
                loader = Qwen3_5ForConditionalGeneration
            backbone = loader.from_pretrained(
                str(load_path), config=config, dtype=dtype, trust_remote_code=trust_remote_code
            )
        if manifest:
            from peft import PeftModel
            backbone.resize_token_embeddings(manifest["vocab_size"])
            backbone = PeftModel.from_pretrained(
                backbone, str(local / LORA_DIRECTORY), is_trainable=False,
            )
        model = cls(backbone)
        model.base_model_reference = manifest["base_model"] if manifest else str(path_or_repo)
        model.base_model_revision = base_revision or getattr(config, "_commit_hash", None)
        if manifest:
            model.lora_anchor_ids = manifest["anchor_ids"]
        weights = local / HYBRIDLINK_WEIGHTS
        if weights.is_file():
            link_config = read_link_config(local)
            if link_config:
                tag = link_config.get("adapter_type")
                if tag != HYBRIDLINK_CONFIG["adapter_type"]:
                    raise ValueError(f"unsupported HybridLink config {tag!r}")
            model.output_scale = checked_scale(link_config.get("output_scale", 1.0))
            state = torch.load(str(weights), map_location="cpu", weights_only=True)
            model.hybridlink.load_state_dict(state, strict=True)
        elif require_hybridlink:
            raise FileNotFoundError(f"{HYBRIDLINK_WEIGHTS} not found in {local}")
        return model

    def run_hybridlink(
        self, hidden_states: torch.Tensor, output_dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        """R(h): next-step latent inputs from last-layer hidden states."""
        link_dtype = next(self.hybridlink.parameters()).dtype
        x = hidden_states if hidden_states.dtype == link_dtype else hidden_states.to(link_dtype)
        out = self.hybridlink(x) * checked_scale(self.output_scale)
        if output_dtype is not None and out.dtype != output_dtype:
            out = out.to(output_dtype)
        return out

    def resize_to_tokenizer(self, tokenizer) -> None:
        """Match the embedding table to the tokenizer (needed once for a fresh base model)."""
        if self.backbone.get_input_embeddings().weight.shape[0] != len(tokenizer):
            if is_lora_model(self):
                raise ValueError("resize the base vocabulary before attaching LoRA")
            self.backbone.resize_token_embeddings(len(tokenizer))

    def save_pretrained(self, output_dir: str | os.PathLike, tokenizer=None, state_dict=None) -> None:
        """Write the backbone (HF format), tokenizer and HybridLink weights into one directory."""
        os.makedirs(output_dir, exist_ok=True)
        if is_lora_model(self):
            if state_dict is not None:
                raise ValueError("LoRA save currently supports single-process training only")
            if not self.base_model_reference:
                raise ValueError("LoRA checkpoints require a base_model_reference")
            if tokenizer is None:
                raise ValueError("LoRA checkpoints require the matching tokenizer")
            self.backbone.save_pretrained(
                os.path.join(output_dir, LORA_DIRECTORY), save_embedding_layers=False,
            )
            tokenizer.save_pretrained(output_dir)
            torch.save(self.hybridlink.state_dict(), os.path.join(output_dir, HYBRIDLINK_WEIGHTS))
            with open(os.path.join(output_dir, HYBRIDLINK_CONFIG_FILE), "w") as f:
                json.dump({**HYBRIDLINK_CONFIG, "output_scale": checked_scale(self.output_scale)}, f, indent=2)
            with open(os.path.join(output_dir, LORA_MANIFEST), "w") as f:
                json.dump({
                    "base_model": self.base_model_reference,
                    "base_revision": self.base_model_revision,
                    "vocab_size": len(tokenizer),
                    "anchor_ids": self.lora_anchor_ids,
                }, f, indent=2)
            return
        if state_dict is None:
            state_dict = self.state_dict()
        backbone_sd = {k[len("backbone."):]: v for k, v in state_dict.items() if k.startswith("backbone.")}
        link_sd = {k[len("hybridlink."):]: v for k, v in state_dict.items() if k.startswith("hybridlink.")}
        self.backbone.save_pretrained(output_dir, state_dict=backbone_sd)
        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir)
        torch.save(link_sd, os.path.join(output_dir, HYBRIDLINK_WEIGHTS))
        with open(os.path.join(output_dir, HYBRIDLINK_CONFIG_FILE), "w", encoding="utf-8") as f:
            json.dump({**HYBRIDLINK_CONFIG, "output_scale": checked_scale(self.output_scale)}, f, indent=2)


def load_hybrid_agent(
    path_or_repo: str | os.PathLike,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device | None = None,
    trust_remote_code: bool = False,
    require_hybridlink: bool = True,
    revision: str | None = None,
):
    """Return `(model, tokenizer)` ready for generation from a local dir or HF repo id."""
    local = resolve_model_dir(path_or_repo, revision)
    tokenizer = AutoTokenizer.from_pretrained(str(local), trust_remote_code=trust_remote_code)
    add_anchor_tokens(tokenizer)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = HybridAgentModel.from_pretrained(
        path_or_repo, dtype=dtype, trust_remote_code=trust_remote_code,
        require_hybridlink=require_hybridlink, revision=revision,
    )
    if hasattr(model, "lora_anchor_ids"):
        actual_ids = [tokenizer.convert_tokens_to_ids(token) for token in ANCHOR_TOKENS]
        if actual_ids != model.lora_anchor_ids:
            raise ValueError("LoRA checkpoint tokenizer anchor IDs do not match the saved model")
    model.resize_to_tokenizer(tokenizer)
    if device is not None:
        model.to(device)
    model.eval()
    return model, tokenizer
