# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified for AdaptInfer visual token pruning.

import torch
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLModel, Qwen2VLTextModel

from .pruning import select_visual_tokens


class AdaptInferQwen2VLTextModel(Qwen2VLTextModel):
    def forward(
        self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
        inputs_embeds=None, use_cache=None, output_attentions=None, output_hidden_states=None,
        return_dict=None, cache_position=None, visual_token_start=None, visual_token_count=0, **kwargs,
    ):
        arguments = dict(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
            past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict, cache_position=cache_position, **kwargs,
        )
        past_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        if not self.pruning_config.layers or not visual_token_count or past_length:
            if self.pruning_config.layers and past_length:
                if not isinstance(past_key_values, DynamicCache):
                    raise ValueError("AdaptInfer requires a dynamic KV cache.")
                sequence_length = inputs_embeds.shape[1] if inputs_embeds is not None else input_ids.shape[1]
                if sequence_length != 1:
                    raise ValueError("Cached generation supports one new token per step.")
            return super().forward(**arguments)

        if self.training:
            raise ValueError("This implementation supports inference. Call model.eval().")
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds.")
        if past_key_values is not None and not isinstance(past_key_values, DynamicCache):
            raise ValueError("AdaptInfer requires a dynamic KV cache.")
        if self.has_sliding_layers:
            raise ValueError("Sliding-window attention is not supported by this pruning implementation.")
        output_attentions = self.config.output_attentions if output_attentions is None else output_attentions
        output_hidden_states = self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        use_cache = self.config.use_cache if use_cache is None else use_cache
        return_dict = self.config.use_return_dict if return_dict is None else return_dict
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if cache_position is None:
            cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = None
        if isinstance(attention_mask, dict):
            causal_mask = attention_mask["full_attention"]
        else:
            causal_mask = create_causal_mask(
                config=self.config, input_embeds=inputs_embeds, attention_mask=attention_mask,
                cache_position=cache_position, past_key_values=past_key_values,
                position_ids=text_position_ids,
            )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        ratios = dict(zip(self.pruning_config.layers, self.pruning_config.keep_ratios))
        self.last_prefill_token_counts = []
        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            self.last_prefill_token_counts.append(visual_token_count)
            prune = layer_idx in ratios
            layer_outputs = decoder_layer(
                hidden_states, attention_mask=causal_mask, position_ids=text_position_ids,
                past_key_values=past_key_values, output_attentions=output_attentions or prune,
                use_cache=use_cache, cache_position=cache_position,
                position_embeddings=position_embeddings, **kwargs,
            )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions += (layer_outputs[1],)
            if prune:
                visual_keep = select_visual_tokens(
                    layer_outputs[1], visual_token_start, visual_token_count, ratios[layer_idx],
                )[0]
                keep = torch.ones(hidden_states.shape[1], dtype=torch.bool, device=hidden_states.device)
                keep[visual_token_start:visual_token_start + visual_token_count] = visual_keep
                indices = keep.nonzero(as_tuple=True)[0]
                hidden_states = hidden_states.index_select(1, indices)
                # Keep the original multimodal positions after removing visual tokens.
                position_ids = position_ids.index_select(2, indices)
                cache_position = cache_position.index_select(0, indices)
                if text_position_ids is not None:
                    text_position_ids = text_position_ids.index_select(1, indices)
                position_embeddings = tuple(embedding.index_select(2, indices) for embedding in position_embeddings)
                if causal_mask is not None:
                    causal_mask = causal_mask.index_select(2, indices).index_select(3, indices)
                visual_token_count = int(visual_keep.sum().item())

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states, past_key_values=past_key_values,
            hidden_states=all_hidden_states, attentions=all_attentions,
        )
        return output if return_dict else output.to_tuple()


class AdaptInferQwen2VLModel(Qwen2VLModel):
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        visual_start = None
        visual_count = 0
        if self.language_model.pruning_config.layers and input_ids is not None:
            if input_ids.shape[0] != 1:
                raise ValueError("AdaptInfer Qwen2-VL supports batch size 1 and num_beams=1.")
            if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 2:
                if not bool(attention_mask.all()):
                    raise ValueError("Use an unpadded single-example prompt.")
            visual_mask = (input_ids == self.config.image_token_id) | (input_ids == self.config.video_token_id)
            visual_indices = visual_mask[0].nonzero(as_tuple=True)[0]
            visual_count = visual_indices.numel()
            if visual_count:
                visual_start = int(visual_indices[0].item())
                if int(visual_indices[-1].item()) - visual_start + 1 != visual_count:
                    raise ValueError("Visual tokens must form one contiguous block; use one image or video per prompt.")
        elif self.language_model.pruning_config.layers and (
            kwargs.get("pixel_values") is not None or kwargs.get("pixel_values_videos") is not None
        ):
            raise ValueError("Visual pruning requires input_ids to locate the visual tokens.")
        return super().forward(
            input_ids=input_ids, attention_mask=attention_mask,
            visual_token_start=visual_start, visual_token_count=visual_count, **kwargs,
        )
