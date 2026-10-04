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
# AdaptInfer modifications to Qwen2-VL inference.

# qwen2_5vl_builder.py

from typing import Optional, Union, Tuple
import time

import torch
import torch.nn as nn

from transformers import (
    AutoModelForVision2Seq,
)
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VLTextModel,
    Qwen2VLModel,
    Qwen2VLForConditionalGeneration,
    Qwen2VLConfig,
    Qwen2VLModelOutputWithPast,
    Qwen2VisionTransformerPretrainedModel,

)
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    create_causal_mask,
    create_sliding_window_causal_mask,
    Qwen2VLRotaryEmbedding,
)
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.utils import logging
from transformers.processing_utils import Unpack

from score import attn_postprocess_topk_with_t2t_weighting
from attn_backend import (
    DEFAULT_ATTN_BACKEND,
    eager_attn_weights,
    prune_attention_mask,
    resolve_attn_backend,
)

logger = logging.get_logger(__name__)


def  batch_index_select(x, idx):

    if len(x.size()) == 4:
        B, H, N, C = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N, H, C)[idx.reshape(-1)].reshape(B, H, N_new, C)
        return out
    elif len(x.size()) == 3:

        B, N, C = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N, C)[idx.reshape(-1)].reshape(B, N_new, C)
        return out
    elif len(x.size()) == 2:
        B, N = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N)[idx.reshape(-1)].reshape(B, N_new)
        return out
    else:
        raise NotImplementedError


class Qwen2VLTextModelWithPruning(Qwen2VLTextModel):
    """Qwen2-VL text decoder with visual token pruning."""

    def __init__(self, config,pruning_loc=[],retained_tokens=10):
        super().__init__(config)

        self.pruning_loc = pruning_loc
        self.retained_tokens = retained_tokens
        self.num_forward = 0
        self.num_token_pool = 0

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        v_token_num=None,
        v_token_start=None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """Decode text and prune visual tokens during prefill."""


        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False


        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache(config=self.config)


        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)


        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )


        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = None


        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": text_position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)


        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        B, L, _ = hidden_states.shape
        idx_sprase_layer = 0


        t_token_start = v_token_start + v_token_num
        num_token = []


        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        num_layers = len(self.layers)
        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if layer_idx in self.pruning_loc and hidden_states.shape[1] !=1:
                layer_mask = causal_mask_mapping[decoder_layer.attention_type]

                # Pruning attention weights are computed independently of the decoding backend.
                self_attn = eager_attn_weights(decoder_layer, hidden_states, position_embeddings, layer_mask)
                layer_outputs = decoder_layer(
                    hidden_states=hidden_states,
                    attention_mask=layer_mask,
                    position_ids=text_position_ids,
                    past_key_values=past_key_values,
                    output_attentions=False,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                    )
                layer_outputs = (layer_outputs[0], self_attn)


                pred_score_vis = attn_postprocess_topk_with_t2t_weighting(self_attn, v_token_start[0], v_token_num[0], t_token_start[0], layer_idx, self.retained_tokens)
                policy = torch.ones(B, hidden_states.shape[1], dtype=torch.bool, device=hidden_states.device)

                policy[:, v_token_start[0]:t_token_start[0]] = pred_score_vis.bool()

                for batch in range(len(v_token_start)):

                    prompt_length = v_token_start[batch] -1
                    policy[batch,:prompt_length,] = 1

                    t_token = t_token_start[batch]
                    policy[batch, t_token:,] = 1


                select_token_idx = torch.where(policy == 1)[1].unsqueeze(0)


                layer_outputs = (batch_index_select(layer_outputs[0], select_token_idx), layer_outputs[1])


                # Preserve each retained token's original multimodal position.
                position_ids = position_ids[:, :, select_token_idx[0]]


                if text_position_ids is not None:

                    text_position_ids = text_position_ids[:, select_token_idx[0]]


                if position_embeddings is not None:

                    cos_emb, sin_emb = position_embeddings


                    cos_emb = cos_emb[:, :, select_token_idx[0], :]
                    sin_emb = sin_emb[:, :, select_token_idx[0], :]
                    position_embeddings = (cos_emb, sin_emb)


                for mask_type in causal_mask_mapping:
                    causal_mask_mapping[mask_type] = prune_attention_mask(
                        causal_mask_mapping[mask_type], select_token_idx[0]
                    )


                v_token_num = pred_score_vis.sum(dim=1)

                t_token_start = v_token_start + v_token_num

                num_token.append(v_token_num[0])
                idx_sprase_layer = idx_sprase_layer + 1

            elif hidden_states.shape[1] !=1:
                layer_outputs = decoder_layer(
                    hidden_states=hidden_states,
                    attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                    position_ids=text_position_ids,
                    past_key_values=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                    )
                num_token.append(v_token_num[0])

            else:
                layer_outputs = decoder_layer(
                    hidden_states=hidden_states,
                    attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                    position_ids=text_position_ids,
                    past_key_values=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                    )

            hidden_states = layer_outputs[0]
            self_attn = layer_outputs[1] if len(layer_outputs) > 1 else None

            if output_attentions:
                all_self_attns += (self_attn,)


        if hidden_states.shape[1] !=1:
            self.num_forward += 1
            self.num_token_pool += (sum(num_token) / num_layers)
            print(f"equal token num until now: {self.num_token_pool / self.num_forward}")
            num_token = []


        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, past_key_values, all_hidden_states, all_self_attns]
                if v is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=all_self_attns,
        )


class Qwen2VLModelWithPruning(Qwen2VLModel):
    """Qwen2-VL backbone with an AdaptInfer text decoder."""

    def __init__(self, config: Qwen2VLConfig):
        super().__init__(config)


        self.language_model = Qwen2VLTextModelWithPruning._from_config(config.text_config)

        self.rope_deltas = None
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[tuple, Qwen2VLModelOutputWithPast]:
        """Encode image or video inputs and pass visual-token positions to the text decoder."""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if position_ids is None:
            if self.rope_deltas is None or cache_position is None or cache_position[0] == 0:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids, image_grid_thw, video_grid_thw, attention_mask
                )
                self.rope_deltas = rope_deltas

            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
                if cache_position is not None:
                    delta = (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                else:
                    delta = torch.zeros((batch_size, seq_length), device=inputs_embeds.device)
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids + delta.to(position_ids.device)

        vision_token_mask = None
        vision_token_start = None
        vision_token_num = None

        if input_ids is not None:

            # Pruning treats visual tokens as one contiguous block.
            vision_token_mask = (input_ids == self.config.image_token_id) | \
                        (input_ids == self.config.video_token_id)

            if attention_mask is not None:
                vision_token_mask = vision_token_mask & attention_mask.bool()


            vision_token_num = vision_token_mask.sum(dim=-1)


            bsz, seqlen = vision_token_mask.shape
            vision_token_start = torch.full((bsz,), -1, device=input_ids.device)
            any_vision = vision_token_mask.any(dim=-1)
            if any_vision.any():
                first_pos = vision_token_mask.float().argmax(dim=-1)
                vision_token_start[any_vision] = first_pos[any_vision]

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=False,
            return_dict=True,
            cache_position=cache_position,
            v_token_num = vision_token_num,
            v_token_start = vision_token_start,
            **kwargs,
        )

        output = Qwen2VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )
        return output if return_dict else output.to_tuple()


class Qwen2VLForConditionalGenerationWithPruning(Qwen2VLForConditionalGeneration):
    """Qwen2-VL conditional generation with AdaptInfer pruning."""

    def __init__(self, config: Qwen2VLConfig):
        super().__init__(config)


        self.model = Qwen2VLModelWithPruning(config)

        self.post_init()


from transformers import AutoProcessor
import argparse

def build(
    pretrained_model_name_or_path: str,
    torch_dtype: torch.dtype,
    device_map: str,
    attn_backend: str = DEFAULT_ATTN_BACKEND,
    **kwargs,
) -> Qwen2VLForConditionalGenerationWithPruning:
    """Load a Qwen2-VL checkpoint and return the model and processor."""
    attn_backend = resolve_attn_backend(attn_backend)
    print(f"✓ 注意力后端: {attn_backend}")
    start_time = time.time()

    start_time = time.time()
    pruned_model = Qwen2VLForConditionalGenerationWithPruning.from_pretrained(
        pretrained_model_name_or_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        attn_implementation=attn_backend,

        **kwargs,
    )

    print(f"✓ 直接加载pruned model完成 - 用时: {time.time() - start_time:.2f}秒")
    start_time = time.time()
    processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path,
                min_pixels = 256 * 28 * 28,
                max_pixels = 1280 * 28 * 28,
                trust_remote_code=True)
    print(f"✓ 加载processor完成 - 用时: {time.time() - start_time:.2f}秒")

    return pruned_model, processor

if __name__ == '__main__':
    print(f"\n{'='*80}")
    print("开始模型构建和推理测试")
    print(f"{'='*80}\n")

    total_start = time.time()
    model, processor = build("Qwen/Qwen2-VL-2B-Instruct", torch.bfloat16, "cuda")


    print(f"\n✓ 总构建时间: {time.time() - total_start:.2f}秒\n")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        type=str,
        default='playground/demo/duck.jpg',
        help="测试图片路径",
    )
    args = parser.parse_args()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": args.image},
                {"type": "text", "text": "这张图片里有什么动物？"},
            ],
        }
    ]

    start_time = time.time()
    inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)


    print(f"✓ 输入处理完成 - 用时: {time.time() - start_time:.2f}秒")

    start_time = time.time()
    generated_ids = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            use_cache=True,
        )

    print(f"✓ 生成完成 - 用时: {time.time() - start_time:.2f}秒")


    start_time = time.time()
    output = processor.batch_decode(
            [generated_ids[0][len(inputs.input_ids[0]):]],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
    print(f"✓ 解码完成 - 用时: {time.time() - start_time:.2f}秒")

    print(f"\n{'='*80}")
    print("生成结果:")
    print(f"{'='*80}")
    print(output)
    print(f"{'='*80}")
    print(f"\n✓ 总运行时间: {time.time() - total_start:.2f}秒\n")
