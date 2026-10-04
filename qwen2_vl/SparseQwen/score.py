import torch
import torch.nn as nn
import torch.nn.functional as F
import einops as ein

RETAINED_TOKEN_INDICES = {}

pruning_loc = [0, 9, 19]
layer_dict = {k: i for i, k in enumerate(pruning_loc)}


# Retention presets map to stage-wise visual-token keep ratios.
sparse_token_dict = {
    10: [0.17, 0.6, 0.3],
    30: [0.5, 0.6, 0.3],
    50: [0.7, 0.9, 0.3],
}


def attn_postprocess_topk_with_t2t_weighting2(
    self_attn_weights,
    v_token_start,
    v_token_num,
    text_token_start,
    layer_idx,
    retained_tokens,
    lb=1,
    disable_t2t_prior=True,
):
    """Return a visual-token keep mask with optional uniform text guidance."""
    B = self_attn_weights.size(0)


    t2v = self_attn_weights[
        :,
        :,
        text_token_start:,
        v_token_start:v_token_start + v_token_num,
    ]

    if disable_t2t_prior:


        per_head_scores = t2v.mean(dim=2)


        per_head_scores = (
            per_head_scores
            / per_head_scores.sum(dim=-1, keepdim=True)
        )


        visual_scores = per_head_scores.sum(dim=1)

    else:


        t2t = self_attn_weights[
            :,
            :,
            text_token_start:,
            text_token_start:,
        ]


        # Symmetrize text-to-text attention to compute text importance.
        t2t_transpose = t2t.transpose(-1, -2)
        diag = torch.diagonal(
            t2t,
            dim1=-2,
            dim2=-1,
        )

        t2t_2 = (
            t2t
            + t2t_transpose
            - torch.diag_embed(diag)
        )


        full_t2t = (
            (1.0 - lb)
            * t2t_2.mean()
            * torch.ones_like(t2t_2)
            + lb
            * t2t_2
        )


        text_importance = full_t2t.sum(dim=2)


        visual_scores = (
            text_importance.unsqueeze(-2)
            @ t2v
        )


        visual_scores = (
            visual_scores
            .squeeze(-2)
            .sum(dim=1)
        )


    sparse_list = sparse_token_dict[retained_tokens]

    k = min(
        int(
            sparse_list[layer_dict[layer_idx]]
            * v_token_num
        ),
        v_token_num - 1,
    )


    mask = torch.zeros_like(
        visual_scores,
        dtype=torch.bool,
    )

    if k > 0:
        topk_idx = torch.topk(
            visual_scores,
            k=k,
            dim=1,
        ).indices


        mask.scatter_(
            dim=1,
            index=topk_idx,
            value=True,
        )


        if B == 1:
            sorted_idx = torch.sort(
                topk_idx[0]
            ).values

            global RETAINED_TOKEN_INDICES
            RETAINED_TOKEN_INDICES[layer_idx] = (
                sorted_idx.detach().cpu()
            )

    return mask

def attn_postprocess_topk_with_t2t_weighting(
    self_attn_weights,
    v_token_start,
    v_token_num,
    text_token_start,
    layer_idx,
    retained_tokens,
    lb = 1,
):
    """Return a visual-token keep mask using text-to-text-weighted text-to-vision attention."""
    B = self_attn_weights.size(0)


    t2t = self_attn_weights[
        :, :,
         text_token_start:,
        text_token_start:
    ]


    # Symmetrize text-to-text attention to compute text importance.
    t2t_transpose = t2t.transpose(-1, -2)
    diag = torch.diagonal(t2t, dim1=-2, dim2=-1)

    t2t_2 = t2t + t2t_transpose - torch.diag_embed(diag)

    full_t2t = (1-lb)*torch.mean(t2t_2).item()*torch.ones_like(t2t) + lb* t2t_2


    text_importance = full_t2t.sum(dim=2)


    t2v = self_attn_weights[
        :, :,
        text_token_start:,
        v_token_start:v_token_start + v_token_num
    ]


    visual_scores = text_importance.unsqueeze(-2) @ t2v
    visual_scores = visual_scores.squeeze(-2).sum(dim=1)

    sparse_list = sparse_token_dict[retained_tokens]
    k = min(int(sparse_list[layer_dict[layer_idx]]*v_token_num), v_token_num - 1)


    mask = torch.zeros_like(visual_scores, dtype=torch.bool)
    if k>0:
        _, topk_idx = torch.topk(visual_scores, k=k, dim=1)
        mask[0][topk_idx.squeeze(0) ] = 1

    if k > 0 and visual_scores.size(0) == 1:
        sorted_idx, _ = torch.sort(topk_idx[0])
        global RETAINED_TOKEN_INDICES
        RETAINED_TOKEN_INDICES[layer_idx] = (
            sorted_idx.detach().cpu()
        )

    return mask


def attn_postprocess_topk_sparsevlm(self_attn_weights, v_token_start, v_token_num, t_token_start, t_token_idx, layer_idx, retained_tokens):
    """Return a visual-token keep mask using unweighted text-to-vision attention."""
    self_attn_weights = self_attn_weights.mean(1)

    t_token_idx = t_token_idx[1] + t_token_start
    relation_vis_text = self_attn_weights[:, t_token_idx , v_token_start: v_token_start+v_token_num]

    relation_vis_text = relation_vis_text.mean(1)

    relation_vis = relation_vis_text
    s_flag = True

    sparse_token_list = sparse_token_dict[retained_tokens]

    if v_token_num != 0:
        mask = torch.zeros_like(relation_vis, dtype=bool)
        _, indices = torch.topk(relation_vis, min(int(sparse_token_list[layer_dict[layer_idx]]*v_token_num), v_token_num - 1), dim=1)
        mask[0][indices] = 1
    else:
        mask = torch.ones_like(relation_vis_text, dtype=bool)
        s_flag = False
    s_flag = 0
    return mask, relation_vis_text, s_flag

