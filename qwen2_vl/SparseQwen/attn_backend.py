"""Attention backend selection and prefill attention weights for token pruning."""

import torch
import torch.nn as nn
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    apply_multimodal_rotary_pos_emb,
    repeat_kv,
)
from transformers.utils import logging

logger = logging.get_logger(__name__)

ATTN_BACKENDS = ("sdpa", "flash_attention_2", "eager")
ATTN_BACKEND_ALIASES = {"fa": "flash_attention_2", "fa2": "flash_attention_2"}
DEFAULT_ATTN_BACKEND = "sdpa"


def resolve_attn_backend(attn_backend: str) -> str:
    attn_backend = ATTN_BACKEND_ALIASES.get(attn_backend, attn_backend)
    if attn_backend not in ATTN_BACKENDS:
        raise ValueError(f"attn_backend must be one of {ATTN_BACKENDS}, got {attn_backend!r}")
    if attn_backend == "flash_attention_2":

        try:
            # Import the CUDA extension to check that FlashAttention can load.
            import flash_attn_2_cuda
            from flash_attn import flash_attn_func
        except Exception as e:
            logger.warning(f"flash_attention_2 不可用（{type(e).__name__}: {e}），回退到 sdpa")
            return "sdpa"
    return attn_backend


def eager_attn_weights(decoder_layer, hidden_states, position_embeddings, attention_mask):
    """Compute uncached prefill attention weights with shape [batch, heads, sequence, sequence]."""
    attn = decoder_layer.self_attn
    x = decoder_layer.input_layernorm(hidden_states)
    bsz, q_len, _ = x.shape

    query_states = attn.q_proj(x).view(bsz, q_len, -1, attn.head_dim).transpose(1, 2)
    key_states = attn.k_proj(x).view(bsz, q_len, -1, attn.head_dim).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, attn.rope_scaling["mrope_section"]
    )
    key_states = repeat_kv(key_states, attn.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * attn.scaling
    if attention_mask is not None and attention_mask.dim() == 4 and attention_mask.is_floating_point():
        attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]
    else:
        causal = torch.ones(q_len, q_len, dtype=torch.bool, device=x.device).triu(1)
        attn_weights = attn_weights.masked_fill(causal, torch.finfo(attn_weights.dtype).min)

    return nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)


def prune_attention_mask(mask, keep_idx):
    """Select retained token positions from a 2D or 4D attention mask."""
    if mask is None:
        return None
    if mask.dim() == 4:
        return mask[:, :, keep_idx, :][:, :, :, keep_idx]
    if mask.dim() == 2:
        return mask[:, keep_idx]
    raise ValueError(f"unexpected attention mask shape {tuple(mask.shape)}")
