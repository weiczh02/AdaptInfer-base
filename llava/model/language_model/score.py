#This code is adapted from SparseVLM on https://github.com/Gumpest/SparseVLMs and LLava on https://github.com/haotian-liu/LLaVA.

import torch
import torch.nn as nn
import torch.nn.functional as F

RETAINED_TOKEN_INDICES = {}

init_vis_tokens   = 576 #

pruning_loc       = [1,10,20]         

layer_dict = {k: i for i, k in enumerate(pruning_loc)}  

    
sparse_token_dict = {
    # ours 7b
    128: [290,72,18],
    64: [133,27,3],
    48: [80,24,2],
    32: [40,12,0],
    #ours 13b
    12813: [300,94,30],
    6413: [148,40,5],
}

def attn_postprocess_topk_with_t2t_weighting(
    self_attn_weights,    # [B, H, L, L]
    v_token_start, v_token_num,
    text_token_start, 
    layer_idx, retained_tokens,
    lb = 1,
    s_flag = 0
):
    """
    input:
        self_attn_weights: Tensor:[B, H, L, L]
        v_token_start: vision token start index
        v_token_num: number of vision tokens
        text_token_start: text token start index
        text_token_num: number of text tokens
        layer_idx: 
        retained_tokens: prunning settings

    returns:
        mask: [B, V] bool
        s_flag: bool 
        visual_scores: [B, V]
    """
    B = self_attn_weights.size(0)

    t2t = self_attn_weights[
        :, :, 
         text_token_start:, #-1
        text_token_start: 
    ]  # [B, H, T, T]

    t2t_transpose = t2t.transpose(-1, -2)          # [B, H, T, T]
    diag = torch.diagonal(t2t, dim1=-2, dim2=-1)   # [B, H, T]
    
    t2t_2 = t2t + t2t_transpose - torch.diag_embed(diag)
    full_t2t = (1-lb)*torch.mean(t2t_2).item()*torch.ones_like(t2t) + lb* t2t_2  # [B, H, T, T]
    
    text_importance = full_t2t.sum(dim=2)  # [B,H,T]  

    t2v = self_attn_weights[
        :, :, 
        text_token_start:,
        v_token_start:v_token_start + v_token_num
    ]  # [B, H, T, V]

    visual_scores = text_importance.unsqueeze(-2) @ t2v #[B,H,1,V]
    visual_scores = visual_scores.squeeze(-2).sum(dim=1) #[B,V]
    sparse_list = sparse_token_dict[retained_tokens]
    k = min(sparse_list[layer_dict[layer_idx]], v_token_num - 1)
    
    #k = max(k,1)

    mask = torch.zeros_like(visual_scores, dtype=torch.bool)  # [B, V]
    if k>0:
        _, topk_idx = torch.topk(visual_scores, k=k, dim=1)
        mask[0][topk_idx.squeeze(0) ] = 1

    if k > 0 and visual_scores.size(0) == 1:  
        sorted_idx, _ = torch.sort(topk_idx[0]) 
        global RETAINED_TOKEN_INDICES
        RETAINED_TOKEN_INDICES[layer_idx] = (
            sorted_idx.detach().cpu()                # shape[k]
        )
        
    s_flag = s_flag
    #s_flag = (v_token_num > 0)
    
    return mask, s_flag, visual_scores
