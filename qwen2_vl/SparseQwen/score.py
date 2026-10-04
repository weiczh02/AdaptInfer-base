import torch
import torch.nn as nn
import torch.nn.functional as F
import einops as ein

RETAINED_TOKEN_INDICES = {}

pruning_loc = [0, 9, 19]
layer_dict = {k: i for i, k in enumerate(pruning_loc)}   # {layer_idx: 0‥28}

# Qwen2-VL-2B：key 为视觉 token 整体保留率(%)，value 为 pruning_loc 各剪枝层的保留比例
sparse_token_dict = {
    10: [0.17, 0.6, 0.3],
    30: [0.5, 0.6, 0.3],
    50: [0.7, 0.9, 0.3],
}


def attn_postprocess_topk_with_t2t_weighting2(#这是加入uniform的代码
    self_attn_weights,    # [B, H, L, L]
    v_token_start,
    v_token_num,
    text_token_start,
    layer_idx,
    retained_tokens,
    lb=1,
    disable_t2t_prior=True,
):
    """
    计算视觉 token 分数并执行 top-k 剪枝。

    disable_t2t_prior=False:
        使用 T2T importance 加权 T2V。

    disable_t2t_prior=True:
        完全不使用 T2T；
        对每个 head 的 T2V 在文本维度做均匀聚合，
        然后在视觉 token 维度独立归一化，
        最后累加所有 head。

    返回：
        mask: [B, V]，True 表示保留
    """
    B = self_attn_weights.size(0)

    # Text query -> visual key
    # [B, H, T, V]
    t2v = self_attn_weights[
        :,
        :,
        text_token_start:,
        v_token_start:v_token_start + v_token_num,
    ]

    if disable_t2t_prior:
        # 对所有 text query 均匀聚合
        # [B, H, V]
        per_head_scores = t2v.mean(dim=2)

        # 每个 head 在 vision-token 维度独立归一化
        # 每个 [B, H] 对应的视觉分数之和变为 1
        per_head_scores = (
            per_head_scores
            / per_head_scores.sum(dim=-1, keepdim=True)
        )

        # 累加所有 attention heads
        # [B, V]
        visual_scores = per_head_scores.sum(dim=1)

    else:
        # Text query -> text key
        # [B, H, T, T]
        t2t = self_attn_weights[
            :,
            :,
            text_token_start:,
            text_token_start:,
        ]

        # 将 causal 下三角 attention 对称补全
        # [B, H, T, T]
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

        # lb=1：完整动态 T2T
        # lb=0：uniform T2T matrix，但总尺度来自 T2T mean
        full_t2t = (
            (1.0 - lb)
            * t2t_2.mean()
            * torch.ones_like(t2t_2)
            + lb
            * t2t_2
        )

        # 每个文本 token 的被关注度：列和
        # [B, H, T]
        text_importance = full_t2t.sum(dim=2)

        # T2T importance 加权 T2V
        # [B, H, 1, T] @ [B, H, T, V]
        # -> [B, H, 1, V]
        visual_scores = (
            text_importance.unsqueeze(-2)
            @ t2v
        )

        # [B, H, 1, V] -> [B, V]
        visual_scores = (
            visual_scores
            .squeeze(-2)
            .sum(dim=1)
        )

    # 决定本层保留多少视觉 token
    sparse_list = sparse_token_dict[retained_tokens]

    k = min(
        int(
            sparse_list[layer_dict[layer_idx]]
            * v_token_num
        ),
        v_token_num - 1,
    )

    # 生成 top-k mask
    # [B, V]
    mask = torch.zeros_like(
        visual_scores,
        dtype=torch.bool,
    )

    if k > 0:
        topk_idx = torch.topk(
            visual_scores,
            k=k,
            dim=1,
        ).indices  # [B, k]

        # 支持整个 batch，而不只是 mask[0]
        mask.scatter_(
            dim=1,
            index=topk_idx,
            value=True,
        )

        # 仅记录单样本推理时的 token index
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
    self_attn_weights,    # [B, H, L, L] 该层的自注意力权重
    v_token_start, 
    v_token_num,
    text_token_start, 
    layer_idx, 
    retained_tokens,
    lb = 1,
):
    """
    融合 T2T 和 T2V 注意力的视觉 token 剪枝决策函数。

    参数：
        self_attn_weights: Tensor，[B, H, L, L]
        v_token_start: 视觉 token 在序列中的起始 index
        v_token_num: 视觉 token 数量 V
        text_token_start: 文本 token 在序列中的起始 index
        text_token_num: 文本 token 数量 T #不用 直接到结尾
        layer_idx: 当前层编号
        retained_tokens: 剪枝配置（192/128/64）

    返回：
        mask: [B, V]，布尔类型，True 表示保留
        s_flag: bool，是否后续还做 merge
        visual_scores: [B, V]，每个视觉 token 的加权得分
    """
    B = self_attn_weights.size(0)

    # 1. 先从 self_attn_weights 中 slice 出 text->text 的部分
    t2t = self_attn_weights[
        :, :, 
         text_token_start:, #-1
        text_token_start: 
    ]  # [B, H, T, T]

    # 3. 把三角阵补全成方阵：
    #    full_t2t = lower + upper = t2t + t2t^T - diag(t2t)
    t2t_transpose = t2t.transpose(-1, -2)          # [B, H, T, T]
    diag = torch.diagonal(t2t, dim1=-2, dim2=-1)   # [B, H, T]
    
    t2t_2 = t2t + t2t_transpose - torch.diag_embed(diag)
    #t2t_2 = t2t
    full_t2t = (1-lb)*torch.mean(t2t_2).item()*torch.ones_like(t2t) + lb* t2t_2  # [B, H, T, T]
    #full_t2t = torch.ones_like(t2t)
    
    # 4. 计算每个文本 token 的“被关注度”（列和）
    text_importance = full_t2t.sum(dim=2)  # [B,H,T]  #记得改回来
 
    # 5. slice 出 T2V attention：text query -> visual key
    t2v = self_attn_weights[
        :, :, 
        text_token_start:,
        v_token_start:v_token_start + v_token_num
    ]  # [B, H, T, V]


    visual_scores = text_importance.unsqueeze(-2) @ t2v #[B,H,1,V]
    visual_scores = visual_scores.squeeze(-2).sum(dim=1) #[B,V]
    # 7. 决定本层保留多少视觉 token
    sparse_list = sparse_token_dict[retained_tokens]
    k = min(int(sparse_list[layer_dict[layer_idx]]*v_token_num), v_token_num - 1)
    
    #k = max(k,1)

    # 8. 取 top-k，生成 mask
    mask = torch.zeros_like(visual_scores, dtype=torch.bool)  # [B, V]
    if k>0:
        _, topk_idx = torch.topk(visual_scores, k=k, dim=1)
        mask[0][topk_idx.squeeze(0) ] = 1

    if k > 0 and visual_scores.size(0) == 1:  
        sorted_idx, _ = torch.sort(topk_idx[0]) # 只处理单样本推理
        global RETAINED_TOKEN_INDICES
        RETAINED_TOKEN_INDICES[layer_idx] = (
            sorted_idx.detach().cpu()                # shape[k]
        )

    return mask




def attn_postprocess_topk_sparsevlm(self_attn_weights, v_token_start, v_token_num, t_token_start, t_token_idx, layer_idx, retained_tokens):
    '''
    self_attn_weights: [B, H, L, L]
    '''
    self_attn_weights = self_attn_weights.mean(1) # B, L[Q], L[K]

    t_token_idx = t_token_idx[1] + t_token_start
    relation_vis_text = self_attn_weights[:, t_token_idx , v_token_start: v_token_start+v_token_num] # B, L2, L1

    relation_vis_text = relation_vis_text.mean(1) # B, L1

    relation_vis = relation_vis_text
    s_flag = True       # s_flag controls whether token merge is needed.

    sparse_token_list = sparse_token_dict[retained_tokens]

    if v_token_num != 0:
        mask = torch.zeros_like(relation_vis, dtype=bool)
        _, indices = torch.topk(relation_vis, min(int(sparse_token_list[layer_dict[layer_idx]]*v_token_num), v_token_num - 1), dim=1)
        mask[0][indices] = 1
    else:
        mask = torch.ones_like(relation_vis_text, dtype=bool)
        s_flag = False
    s_flag = 0 # use_merge 7.11 zwc
    return mask, relation_vis_text, s_flag

