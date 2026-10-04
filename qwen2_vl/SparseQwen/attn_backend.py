"""
注意力后端选择 + 剪枝层打分用的注意力计算。

- 模型主干（视觉塔 + LLM 所有层）的注意力后端由 attn_backend 决定：
  "sdpa"（默认）/ "flash_attention_2" / "eager"。
- 剪枝层的打分注意力始终按 transformers eager_attention_forward 的公式单独计算，
  所以 attn_backend="eager" 时与 baseline_eager/ 的推理结果逐位一致；
  sdpa / FA2 只会因主干注意力的数值差异带来微小偏差。
"""

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
        # transformers 只检查包元数据；flash_attn 与 torch ABI 不匹配时要真正 import 才会暴露
        try:
            import flash_attn_2_cuda  # noqa: F401
            from flash_attn import flash_attn_func  # noqa: F401
        except Exception as e:
            logger.warning(f"flash_attention_2 不可用（{type(e).__name__}: {e}），回退到 sdpa")
            return "sdpa"
    return attn_backend


def eager_attn_weights(decoder_layer, hidden_states, position_embeddings, attention_mask):
    """
    复现 decoder_layer 内部 eager 注意力的权重 [B, H, L, L]（不经过 KV cache，仅用于 prefill 打分）。
    attention_mask: eager 后端下的 4D 加性 mask；其它后端（None / bool / 2D）时按纯因果 mask 构造。
    """
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
    """按保留下标裁剪 mask；兼容 eager/sdpa 的 4D mask、FA2 的 2D padding mask 和 None。"""
    if mask is None:
        return None
    if mask.dim() == 4:
        return mask[:, :, keep_idx, :][:, :, :, keep_idx]
    if mask.dim() == 2:
        return mask[:, keep_idx]
    raise ValueError(f"unexpected attention mask shape {tuple(mask.shape)}")
