"""Load Qwen2-VL and generate answers with AdaptInfer."""

import argparse

import torch
from transformers import AutoConfig, AutoProcessor, Qwen2VLForConditionalGeneration

from .modeling_qwen2_vl import AdaptInferQwen2VLModel, AdaptInferQwen2VLTextModel
from .pruning import PruningConfig


def build_model(model_path, pruning_config=None, device="cuda", dtype="bfloat16", min_visual_tokens=256, max_visual_tokens=1280):
    config = AutoConfig.from_pretrained(model_path)
    if config.model_type != "qwen2_vl":
        raise ValueError("Use a Qwen2-VL checkpoint (model_type=qwen2_vl).")
    pruning_config = pruning_config if pruning_config is not None else PruningConfig()
    pruning_config.validate_layers(config.text_config.num_hidden_layers)
    if config.text_config.use_sliding_window:
        raise ValueError("Use a checkpoint with full attention.")
    if not 0 < min_visual_tokens <= max_visual_tokens:
        raise ValueError("Visual token bounds must satisfy 0 < min <= max.")
    dtypes = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    if dtype not in dtypes:
        raise ValueError(f"Choose a dtype from {tuple(dtypes)}.")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, config=config, dtype=dtypes[dtype], device_map=device, attn_implementation="eager",
    )
    # Reuse the loaded modules and weights while replacing both forward methods.
    model.model.__class__ = AdaptInferQwen2VLModel
    model.model.language_model.__class__ = AdaptInferQwen2VLTextModel
    model.model.language_model.pruning_config = pruning_config
    model.model.language_model.last_prefill_token_counts = []
    model.eval()
    processor = AutoProcessor.from_pretrained(
        model_path, min_pixels=min_visual_tokens * 28 * 28, max_pixels=max_visual_tokens * 28 * 28,
    )
    return model, processor


@torch.inference_mode()
def generate_answer(model, processor, question, image=None, video=None, fps=4.0, max_new_tokens=128, temperature=0.0):
    if (image is None) == (video is None):
        raise ValueError("Provide exactly one image or video.")
    if max_new_tokens < 1 or temperature < 0 or fps <= 0:
        raise ValueError("Use positive max_new_tokens/fps and a nonnegative temperature.")
    media = {"type": "image", "image": image} if image is not None else {"type": "video", "video": video}
    messages = [{"role": "user", "content": [media, {"type": "text", "text": question}]}]
    processor_kwargs = {"fps": fps} if video is not None else {}
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt", **processor_kwargs,
    ).to(model.device)
    generation_kwargs = dict(max_new_tokens=max_new_tokens, num_beams=1, do_sample=temperature > 0, use_cache=True)
    if temperature > 0:
        generation_kwargs["temperature"] = temperature
    generated = model.generate(**inputs, **generation_kwargs)
    answer = generated[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(answer, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def add_model_arguments(parser):
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--pruning-layers", nargs="+", type=int, default=[0, 9, 19])
    parser.add_argument("--keep-ratios", nargs="+", type=float, default=[0.17, 0.6, 0.3])
    parser.add_argument("--min-visual-tokens", type=int, default=256)
    parser.add_argument("--max-visual-tokens", type=int, default=1280)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)


def build_from_args(args):
    return build_model(
        args.model_path, PruningConfig(tuple(args.pruning_layers), tuple(args.keep_ratios)),
        args.device, args.dtype, args.min_visual_tokens, args.max_visual_tokens,
    )


def main():
    parser = argparse.ArgumentParser(description="AdaptInfer inference with Qwen2-VL.")
    add_model_arguments(parser)
    media = parser.add_mutually_exclusive_group(required=True)
    media.add_argument("--image")
    media.add_argument("--video")
    parser.add_argument("--question", required=True)
    parser.add_argument("--fps", type=float, default=4.0)
    args = parser.parse_args()
    model, processor = build_from_args(args)
    print(generate_answer(
        model, processor, args.question, image=args.image, video=args.video, fps=args.fps,
        max_new_tokens=args.max_new_tokens, temperature=args.temperature,
    ))


if __name__ == "__main__":
    main()
