"""Text-only model access and checkpoint compatibility.

Qwen3.5-style multimodal checkpoints (including Qwen3.8-27B) keep their
vision weights, but hybrid training and generation only use the text decoder.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

LINK_CONFIG = "hybridlink_config.json"
LEGACY_LINK_CONFIG = "adapter_config.json"


def is_qwen_hybrid(backbone) -> bool:
    return getattr(backbone.config, "model_type", None) in {"qwen3_5", "qwen3_5_text"}


def check_execution(backbone, *, batch_size=1, rl=False) -> None:
    if is_qwen_hybrid(backbone):
        if rl:
            raise ValueError("Qwen hybrid RL is not supported: rollout/replay agreement is unverified")
        if batch_size != 1:
            raise ValueError("Qwen hybrid inference currently requires batch_size=1 (num_samples=1)")


def text_decoder(backbone):
    if getattr(backbone.config, "model_type", None) == "qwen3_5":
        return backbone.model.language_model
    return backbone.get_decoder()


def text_forward(backbone, **kwargs):
    if not is_qwen_hybrid(backbone):
        return backbone(**kwargs)
    for name in ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"):
        if kwargs.get(name) is not None:
            raise ValueError("Qwen hybrid generation currently supports text inputs only")
        kwargs.pop(name, None)
    # Do not pass the attention-only cache slot index to the mixed-state decoder.
    kwargs.pop("cache_position", None)
    kwargs.pop("output_hidden_states", None)
    kwargs.pop("logits_to_keep", None)
    outputs = text_decoder(backbone)(**kwargs)
    hidden = outputs.last_hidden_state
    return SimpleNamespace(
        past_key_values=outputs.past_key_values,
        logits=backbone.get_output_embeddings()(hidden[:, -1:, :]),
        hidden_states=(hidden,),
    )


def read_link_config(directory) -> dict:
    directory = Path(directory)
    path = directory / LINK_CONFIG
    if not path.is_file():
        path = directory / LEGACY_LINK_CONFIG
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def checked_scale(value) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("HybridLink output_scale must be finite and positive")
    return value


@contextmanager
def backbone_load_path(directory):
    """Hide only a legacy HybridLink config from HF's PEFT auto-detection.

    The original checkpoint/cache is never renamed or edited. Symlinks avoid
    copying model weights. Genuine PEFT configs remain visible.
    """
    directory = Path(directory).resolve()
    legacy = directory / LEGACY_LINK_CONFIG
    config = json.loads(legacy.read_text()) if legacy.is_file() else {}
    if config.get("adapter_type") != "ln_res_adapter" or "peft_type" in config:
        yield directory
        return
    with TemporaryDirectory(prefix="hybrid-backbone-") as temporary:
        view = Path(temporary)
        for source in directory.iterdir():
            if source.name not in {LEGACY_LINK_CONFIG, "adapter.pt"}:
                (view / source.name).symlink_to(source, target_is_directory=source.is_dir())
        yield view
