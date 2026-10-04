
"""Qwen2-VL video QA inference with id, question, answer, and pred output fields."""

import os
import sys
import math
import json
import argparse

import torch
from tqdm import tqdm


sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'SparseQwen'))
from builder import build


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Qwen2-VL VideoQA 推理（输出格式与 run_inference_video_qa.py 一致）"
    )


    parser.add_argument("--model_path", type=str, required=True,
                        help='Qwen2-VL checkpoint path or Hugging Face model ID.')
    parser.add_argument("--video_dir", type=str, required=True,
                        help='Directory containing the input videos.')
    parser.add_argument("--gt_file_question", type=str, required=True,
                        help='JSON file containing video QA questions.')
    parser.add_argument("--gt_file_answers", type=str, required=True,
                        help='JSON file containing reference answers.')
    parser.add_argument("--output_dir", type=str, required=True,
                        help='Directory for prediction files.')
    parser.add_argument("--output_name", type=str, required=True,
                        help='Output filename without the .json suffix.')


    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)


    parser.add_argument(
        "--pruning_loc",
        type=int,
        nargs="+",
        default=[0,9,19],
        help='Zero-based pruning layers (default: 0 9 19).'
    )
    parser.add_argument(
        "--retained_tokens",
        type=int,
        default=10,
        help='Visual-token retention preset: 10, 30, or 50.'
    )


    parser.add_argument("--temperature", type=float, default=0.0,
                        help='Sampling temperature; 0 uses greedy decoding.')
    parser.add_argument("--top_p", type=float, default=None,
                        help='Top-p sampling threshold.')
    parser.add_argument("--num_beams", type=int, default=1,
                        help='Number of beams.')
    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help='Maximum number of generated tokens.')

    parser.add_argument("--attn-backend", type=str, default="sdpa",
                        choices=["sdpa", "fa", "flash_attention_2", "eager"],
                        help='Attention backend: sdpa, fa (FlashAttention 2), or eager.')
    return parser.parse_args()


def load_qwen_model_and_processor(args):
    """Load the Qwen2-VL model and configure visual token pruning."""
    print(f"正在加载 Qwen2-VL 模型: {args.model_path}")
    model, processor = build(
        pretrained_model_name_or_path=args.model_path,
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
    torch.set_num_threads(8)
    torch.set_num_interop_threads(4)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    return model, processor


def get_model_output(model, processor, video_path, question_text, args):
    """Generate an answer for a video and question."""


    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": question_text},
            ],
        }
    ]


    inputs = processor.apply_chat_template(
        messages,
        fps=4,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature if args.temperature > 0 else None,
            top_p=args.top_p if args.top_p is not None else None,
            num_beams=args.num_beams,
            use_cache=True,
        )


    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


    return output_text


def run_inference(args):
    """Generate video QA predictions and save one record per line."""

    model, processor = load_qwen_model_and_processor(args)


    gt_questions = json.load(open(args.gt_file_question, "r"))
    gt_questions = gt_questions[:1000]
    gt_questions = get_chunk(gt_questions, args.num_chunks, args.chunk_idx)

    gt_answers = json.load(open(args.gt_file_answers, "r"))
    gt_answers = gt_answers[:1000]


    os.makedirs(args.output_dir, exist_ok=True)
    answers_file = os.path.join(args.output_dir, f"{args.output_name}.json")
    ans_file = open(answers_file, "w", encoding="utf-8")

    print(f"开始推理，共 {len(gt_questions)} 个问题")
    print(f"输出写入: {answers_file}")

    video_formats = [".mp4", ".avi", ".mov", ".mkv"]

    index = 0
    for sample in tqdm(gt_questions, desc="VideoQA", mininterval=10):
        video_name = sample["video_name"]
        question = sample["question"]
        qid = sample["question_id"]


        # Read reference answers in file order.
        answer = gt_answers[index]["answer"]
        index += 1

        sample_set = {
            "id": qid,
            "question": question,
            "answer": answer,
        }


        video_path = None
        for fmt in video_formats:
            temp_path = os.path.join(args.video_dir, f"{video_name}{fmt}")
            if os.path.exists(temp_path):
                video_path = temp_path
                break

        if video_path is None:

            print(f"[Warning] 未找到视频文件: {video_name} (尝试后缀: {video_formats})")
            sample_set["pred"] = ""
            ans_file.write(json.dumps(sample_set, ensure_ascii=False) + "\n")
            ans_file.flush()
            continue


        try:
            pred = get_model_output(model, processor, video_path, question, args)
        except Exception as e:
            print(f"[Error] 处理视频 {video_path} 时出错: {e}")
            pred = "error"

        sample_set["pred"] = pred


        ans_file.write(json.dumps(sample_set, ensure_ascii=False) + "\n")
        ans_file.flush()

    ans_file.close()
    print("推理完成！")


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
