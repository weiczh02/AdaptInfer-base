# qwen_inference_video_qa.py
"""
使用 Qwen2-VL 在视频上做 VideoQA 推理（带可选剪枝）
- 加载方式参考: model_vqa_qwen.py (SparseQwen/build)
- I/O 格式参考: run_inference_video_qa.py
  每行输出: {"id": ..., "question": ..., "answer": ..., "pred": ...}
"""

import os
import sys
import math
import json
import argparse

import torch
from tqdm import tqdm

# ============= Qwen2-VL / SparseQwen 相关 =============
# 将 SparseQwen 加到路径中（与 model_vqa_qwen.py 保持一致）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'SparseQwen'))
from builder import build  # 来自 SparseQwen 的构建函数


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

    # 保持与原 run_inference_video_qa.py 类似的参数命名
    parser.add_argument("--model_path", type=str, required=True,
                        help="Qwen2-VL 模型路径（传给 SparseQwen/builder 的 pretrained_model_name_or_path）")
    parser.add_argument("--video_dir", type=str, required=True,
                        help="视频文件目录")
    parser.add_argument("--gt_file_question", type=str, required=True,
                        help="问题 GT 文件（JSON 列表，包含 video_name, question, question_id）")
    parser.add_argument("--gt_file_answers", type=str, required=True,
                        help="答案 GT 文件（JSON 列表，与问题按 index 对齐）")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出目录")
    parser.add_argument("--output_name", type=str, required=True,
                        help="输出文件名（不含后缀，最终为 output_dir/output_name.json）")

    # 分块相关（保持与原脚本一致）
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)

    # 剪枝相关（与 model_vqa_qwen 的接口保持一致）
    parser.add_argument(
        "--pruning_loc",
        type=int,
        nargs="+",
        default=[0,9,19],
        help="剪枝层位置，例如: 1 10"
    )
    parser.add_argument(
        "--retained_tokens",
        type=int,
        default=10,
        help="保留的视觉 token 数量 (192/128/64/48/32 等)"
    )

    # 生成相关参数（参考 model_vqa_qwen.py）
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="采样温度")
    parser.add_argument("--top_p", type=float, default=None,
                        help="nucleus sampling")
    parser.add_argument("--num_beams", type=int, default=1,
                        help="beam search 数量")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help="最大生成 token 数")

    parser.add_argument("--attn-backend", type=str, default="sdpa",
                        choices=["sdpa", "fa", "flash_attention_2", "eager"],
                        help="主干注意力后端；剪枝打分始终按 eager 公式计算，选 eager 与论文/baseline 逐位一致")
    return parser.parse_args()


def load_qwen_model_and_processor(args):
    """
    使用 SparseQwen 的 build() 加载带剪枝功能的 Qwen2-VL 模型
    """
    print(f"正在加载 Qwen2-VL 模型: {args.model_path}")
    model, processor = build(
        pretrained_model_name_or_path=args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_backend=args.attn_backend,
    )

    # 如果你的 SparseQwen 中已经实现 pruning_loc 和 retained_tokens，
    # 可以像下面这样把参数写回模型（与 model_vqa_qwen 里注释的用法一致）：
    #
    # model.model.language_model.pruning_loc = args.pruning_loc
    # model.model.language_model.retained_tokens = args.retained_tokens
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
    """
    调用 Qwen2-VL 对单个 video + question 生成答案
    - 输出为纯文本字符串，与原 run_inference_video_qa.py 中的 outputs 格式一致（不做额外包装）
    """
    # Qwen2-VL 的多模态输入格式：messages -> processor.apply_chat_template(...)
    # 这里假设 processor 支持 {"type": "video", "video": <path>} 形式；
    # 如果你本地的视频接口不同，只需要改这一块 messages 的构造即可。
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": question_text},
            ],
        }
    ]

    # 使用 Qwen2-VL 的 processor 构建模型输入
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

    # 只解码新生成的部分（与 model_vqa_qwen 一致）
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    #print(output_text)
    return output_text


def run_inference(args):
    """
    整体流程：
    - 加载 Qwen2-VL 模型 & processor（SparseQwen/build）
    - 读取问题 + 答案 GT
    - 遍历每条样本，找到对应视频，调用 Qwen2-VL 推理
    - 写出 JSON 行，字段与 run_inference_video_qa.py 完全一致:
        {
            "id": <question_id>,
            "question": <question>,
            "answer": <gt_answer>,
            "pred": <model_output>
        }
    """
    # 1. 加载模型
    model, processor = load_qwen_model_and_processor(args)

    # 2. 加载 GT 问题和答案（格式与原始脚本保持一致）
    gt_questions = json.load(open(args.gt_file_question, "r"))
    gt_questions = gt_questions[:1000]
    gt_questions = get_chunk(gt_questions, args.num_chunks, args.chunk_idx)

    gt_answers = json.load(open(args.gt_file_answers, "r"))
    gt_answers = gt_answers[:1000]
    
    # 原脚本按 index 对齐，这里保持一致
    # gt_answers = get_chunk(gt_answers, args.num_chunks, args.chunk_idx)  # 如需分块答案，可取消注释

    # 3. 准备输出文件
    os.makedirs(args.output_dir, exist_ok=True)
    answers_file = os.path.join(args.output_dir, f"{args.output_name}.json")
    ans_file = open(answers_file, "w", encoding="utf-8")

    print(f"开始推理，共 {len(gt_questions)} 个问题")
    print(f"输出写入: {answers_file}")

    video_formats = [".mp4", ".avi", ".mov", ".mkv"]

    index = 0  # 用于与 gt_answers 对齐
    for sample in tqdm(gt_questions, desc="VideoQA", mininterval=10):
        video_name = sample["video_name"]
        question = sample["question"]
        qid = sample["question_id"]

        # 与原脚本一样按顺序取答案
        answer = gt_answers[index]["answer"]
        index += 1

        sample_set = {
            "id": qid,
            "question": question,
            "answer": answer,
        }

        # 4. 找到真实的视频文件
        video_path = None
        for fmt in video_formats:
            temp_path = os.path.join(args.video_dir, f"{video_name}{fmt}")
            if os.path.exists(temp_path):
                video_path = temp_path
                break

        if video_path is None:
            # 找不到视频时，行为与原脚本基本保持：这里给个提示，并跳过
            print(f"[Warning] 未找到视频文件: {video_name} (尝试后缀: {video_formats})")
            sample_set["pred"] = ""
            ans_file.write(json.dumps(sample_set, ensure_ascii=False) + "\n")
            ans_file.flush()
            continue

        # 5. 调用 Qwen2-VL 做推理
        try:
            pred = get_model_output(model, processor, video_path, question, args)
        except Exception as e:
            print(f"[Error] 处理视频 {video_path} 时出错: {e}")
            pred = "error"

        sample_set["pred"] = pred

        # 6. 按行写出，与原 run_inference_video_qa.py 保持一致
        ans_file.write(json.dumps(sample_set, ensure_ascii=False) + "\n")
        ans_file.flush()

    ans_file.close()
    print("推理完成！")


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
