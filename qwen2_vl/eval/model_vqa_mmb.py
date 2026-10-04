
"""Qwen2-VL sparse inference for MMBench TSV files."""

import argparse
import torch
import os
import json
import sys
import math
import base64
import io

import pandas as pd
from tqdm import tqdm
import shortuuid
from PIL import Image


sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'SparseQwen'))
torch.set_num_threads(8)
torch.set_num_interop_threads(4)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

from builder import build


all_options = ['A', 'B', 'C', 'D']


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def is_none(value):
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.lower() in ['nan', 'none']:
        return True
    return False


def get_options(row, options):
    parsed_options = []
    for option in options:
        option_value = row[option]
        if is_none(option_value):
            break
        parsed_options.append(option_value)
    return parsed_options


def load_image_from_base64(img_b64: str) -> Image.Image:
    """Decode a base64 image as a PIL RGB image."""
    if ',' in img_b64:
        img_b64 = img_b64.split(',')[-1]
    img_bytes = base64.b64decode(img_b64)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return img


def get_model_name_from_path(model_path: str) -> str:
    """Extract the checkpoint name for prediction metadata."""
    return os.path.basename(os.path.normpath(model_path))


def eval_model(args):

    model_path = os.path.expanduser(args.model_path)
    print(f"正在加载 Qwen2-VL 模型: {model_path}")

    model, processor = build(
        pretrained_model_name_or_path=model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_backend=args.attn_backend,
    )


    try:
        if hasattr(model, "model") and hasattr(model.model, "language_model"):
            lm = model.model.language_model
            setattr(lm, "pruning_loc", args.pruning_loc)
            setattr(lm, "retained_tokens", args.retained_tokens)
            print(f"已设置 retained_tokens = {args.retained_tokens} 到 language_model 上")
    except Exception as e:
        print(f"警告：设置 retained_tokens 失败（不会影响正常推理）: {e}")

    model.eval()


    print(f"正在加载问题文件: {args.question_file}")
    questions = pd.read_table(os.path.expanduser(args.question_file))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)


    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w", encoding="utf-8")

    model_name = get_model_name_from_path(model_path)

    print(f"开始评估，共 {len(questions)} 个问题")
    print(f"结果将保存到: {answers_file}")


    for index, row in tqdm(questions.iterrows(), total=len(questions), desc="评估进度"):
        options = get_options(row, all_options)
        cur_option_char = all_options[:len(options)]

        if args.all_rounds:
            num_rounds = len(options)
        else:
            num_rounds = 1

        for round_idx in range(num_rounds):
            idx = row['index']
            question = row['question']
            hint = row['hint']
            img_b64 = row['image']

            image = load_image_from_base64(img_b64)


            if not is_none(hint):
                question_full = str(hint) + '\n' + str(question)
            else:
                question_full = str(question)


            for option_char, option in zip(all_options[:len(options)], options):
                question_full = question_full + '\n' + f"{option_char}. {option}"


            if args.single_pred_prompt:
                if args.lang == 'cn':
                    question_full = question_full + '\n' + "请直接回答选项字母。"
                else:
                    question_full = question_full + '\n' + "Answer with the option's letter from the given choices directly."

            cur_prompt = question_full


            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": cur_prompt},
                    ],
                }
            ]

            try:
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(model.device)

                with torch.inference_mode():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=32,
                        do_sample=True if args.temperature > 0 else False,
                        temperature=args.temperature if args.temperature > 0 else None,
                        top_p=args.top_p if args.top_p is not None else None,
                        num_beams=args.num_beams,
                        use_cache=True,
                    )


                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
                ]
                outputs = processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0].strip()

            except Exception as e:
                print(f"\n处理问题 {idx}（round {round_idx}）时出错: {e}")
                outputs = "error"


            ans_id = shortuuid.uuid()
            ans_record = {
                "question_id": idx,
                "round_id": round_idx,
                "prompt": cur_prompt,
                "text": outputs,
                "options": options,
                "option_char": cur_option_char,
                "answer_id": ans_id,
                "model_id": model_name,
                "metadata": {},
            }
            ans_file.write(json.dumps(ans_record, ensure_ascii=False) + "\n")
            ans_file.flush()


            options = options[1:] + options[:1]
            cur_option_char = cur_option_char[1:] + cur_option_char[:1]

    ans_file.close()
    print(f"\n评估完成！结果已保存到: {answers_file}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")  
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")  
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--all-rounds", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--retained_tokens", type=int, default=10)
    parser.add_argument("--attn-backend", type=str, default="sdpa",
                        choices=["sdpa", "fa", "flash_attention_2", "eager"],
                        help='Attention backend: sdpa, fa (FlashAttention 2), or eager.')
    parser.add_argument("--pruning-loc", type=int, nargs="+", default=[0, 9, 19])
    args = parser.parse_args()

    eval_model(args)
