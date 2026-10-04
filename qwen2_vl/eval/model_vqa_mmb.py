# model_vqa_qwen_mmbench.py
"""
使用带剪枝能力的 Qwen2-VL 在 MMBench 上评估的脚本

- 命令行参数、输入/输出格式与 llava 的 model_vqa_mmbench.py 保持一致
- 模型加载与数据预处理方式与 model_vqa_qwen.py 中的 Qwen2-VL 方案一致
"""

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

# ====== Qwen2-VL & SparseQwen 相关设置（来自 model_vqa_qwen） ======
# 添加 SparseQwen 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'SparseQwen'))
torch.set_num_threads(8)
torch.set_num_interop_threads(4)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

from builder import build  # Qwen2-VL 构建函数

# ====== 与原 mmbench 脚本保持一致的一些常量和工具函数 ======
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
    """将 TSV 中的 base64 图像字段转为 PIL.Image，与 llava.mm_utils 行为兼容"""
    if ',' in img_b64:
        img_b64 = img_b64.split(',')[-1]
    img_bytes = base64.b64decode(img_b64)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return img


def get_model_name_from_path(model_path: str) -> str:
    """简化版模型名解析（用于写入答案中的 model_id 字段）"""
    return os.path.basename(os.path.normpath(model_path))


def eval_model(args):
    # ====== 加载 Qwen2-VL 模型（沿用 model_vqa_qwen 的方式） ======
    model_path = os.path.expanduser(args.model_path)
    print(f"正在加载 Qwen2-VL 模型: {model_path}")

    model, processor = build(
        pretrained_model_name_or_path=model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_backend=args.attn_backend,
    )

    # 设置剪枝参数：保留视觉 tokens 数量
    # 注意：这里假设你的 SparseQwen 中语言模型挂在 model.model.language_model 上，
    # 与 model_vqa_qwen.py 中的写法一致。
    try:
        if hasattr(model, "model") and hasattr(model.model, "language_model"):
            lm = model.model.language_model
            setattr(lm, "pruning_loc", args.pruning_loc)
            setattr(lm, "retained_tokens", args.retained_tokens)
            print(f"已设置 retained_tokens = {args.retained_tokens} 到 language_model 上")
    except Exception as e:
        print(f"警告：设置 retained_tokens 失败（不会影响正常推理）: {e}")

    model.eval()

    # ====== 加载 MMBench 问题 TSV（与原 mmbench 脚本一致） ======
    print(f"正在加载问题文件: {args.question_file}")
    questions = pd.read_table(os.path.expanduser(args.question_file))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    # ====== 创建输出文件，格式与 model_vqa_mmbench.py 保持一致 ======
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w", encoding="utf-8")

    model_name = get_model_name_from_path(model_path)

    print(f"开始评估，共 {len(questions)} 个问题")
    print(f"结果将保存到: {answers_file}")

    # ====== 主循环：按照原 mmbench 的多轮和旋转选项逻辑 ======
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

            # 将 hint 拼在前面（与原 mmbench 行为相同）
            if not is_none(hint):
                question_full = str(hint) + '\n' + str(question)
            else:
                question_full = str(question)

            # 拼接选项文本
            for option_char, option in zip(all_options[:len(options)], options):
                question_full = question_full + '\n' + f"{option_char}. {option}"

            # 单选字母提示（与原脚本的 single_pred_prompt 逻辑一致）
            if args.single_pred_prompt:
                if args.lang == 'cn':
                    question_full = question_full + '\n' + "请直接回答选项字母。"
                else:
                    question_full = question_full + '\n' + "Answer with the option's letter from the given choices directly."

            cur_prompt = question_full  # 与原 mmbench 中的字段命名保持一致

            # ====== 使用 Qwen2-VL 的 chat 模板构建输入 ======
            # 注意：这里不再使用 llava 的 IMAGE_TOKEN，而是用 Qwen2-VL 的 messages 格式
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

                # 只解码新生成的部分（与 model_vqa_qwen 写法一致）
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

            # ====== 写出结果，字段与原 model_vqa_mmbench.py 完全一致 ======
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

            # ====== 旋转选项（与 all_rounds 配合，保持原评估逻辑） ======
            options = options[1:] + options[:1]
            cur_option_char = cur_option_char[1:] + cur_option_char[:1]

    ans_file.close()
    print(f"\n评估完成！结果已保存到: {answers_file}")


if __name__ == "__main__":
    # 为了保持“输入格式”完全一致，这里沿用 model_vqa_mmbench.py 的参数定义
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")  # 保留但在本脚本中不使用
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")  # 为兼容保留，但不再使用
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
                        help="主干注意力后端；剪枝打分始终按 eager 公式计算，选 eager 与论文/baseline 逐位一致")
    parser.add_argument("--pruning-loc", type=int, nargs="+", default=[0, 9, 19])
    args = parser.parse_args()

    eval_model(args)
