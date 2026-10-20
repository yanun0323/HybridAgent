"""Optional PEFT format training. Only LoRA and the two anchor rows are trained."""
from __future__ import annotations

from .compat import text_decoder
from .constants import ANCHOR_TOKENS

LORA_MANIFEST = "hybrid_lora.json"
LORA_DIRECTORY = "lora"


def enable_format_lora(model, tokenizer, *, rank=8, alpha=16):
    from peft import LoraConfig, get_peft_model

    if rank <= 0 or alpha <= 0:
        raise ValueError("LoRA rank and alpha must be positive")
    backbone = model.backbone
    if hasattr(backbone, "peft_config"):
        raise ValueError("LoRA is already attached; start format training from the base checkpoint")
    ids = []
    for token in ANCHOR_TOKENS:
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if len(encoded) != 1 or encoded[0] == tokenizer.unk_token_id:
            raise ValueError(f"anchor {token!r} must be registered as one token")
        ids.append(encoded[0])
    if len(set(ids)) != 2:
        raise ValueError("entry and exit anchors must have distinct token IDs")

    model.resize_to_tokenizer(tokenizer)
    decoder = text_decoder(backbone)
    decoder_names = {id(module) for module in decoder.modules()}
    # Full module paths exclude the vision encoder. Cover full attention,
    # DeltaNet projections and MLPs without touching convolution/state parameters.
    suffixes = {
        "q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z",
        "in_proj_a", "in_proj_b", "out_proj", "gate_proj", "up_proj", "down_proj",
    }
    targets = [
        name for name, module in backbone.named_modules()
        if id(module) in decoder_names and name.rsplit(".", 1)[-1] in suffixes
    ]
    if not targets:
        raise ValueError("no supported text projection layers found for LoRA")

    embedding = backbone.get_input_embeddings()
    output = backbone.get_output_embeddings()
    names = {id(module): name for name, module in backbone.named_modules()}
    token_layers = {names[id(embedding)]: ids}
    # Qwen3.8 has an untied output head: input-only token training is insufficient.
    if output.weight is not embedding.weight:
        token_layers[names[id(output)]] = ids

    config = LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=0.0, bias="none",
        target_modules=targets, trainable_token_indices=token_layers,
    )
    model.backbone = get_peft_model(backbone, config)
    model.hybridlink.requires_grad_(False)
    # Keep adapter updates in FP32 even if the original backbone is BF16.
    for parameter in model.backbone.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    model.lora_anchor_ids = ids
    return model


def is_lora_model(model) -> bool:
    return hasattr(model.backbone, "peft_config") and hasattr(model.backbone, "get_base_model")
