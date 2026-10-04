"""
Qwen2-VL TextVQA 评估脚本
支持带剪枝推理的 Qwen2-VL 模型评估
"""

import argparse
import torch
import os
import json
import sys
from tqdm import tqdm
import shortuuid
from PIL import Image
import math

# 添加 SparseQwen 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'SparseQwen'))
torch.set_num_threads(8)
torch.set_num_interop_threads(4)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

from builder import build


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    """
    使用 Qwen2-VL 模型在 TextVQA 上进行评估
    """
    # 加载带剪枝的 Qwen2-VL 模型
    print(f"正在加载模型: {args.model_path}")
    #print(f"剪枝配置 - 层位置: {args.pruning_loc}, 保留tokens: {args.retained_tokens}")
    
    model, processor = build(
        pretrained_model_name_or_path=args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_backend=args.attn_backend,
    )
    
    # 设置剪枝参数
    model.model.language_model.pruning_loc = args.pruning_loc
    model.model.language_model.retained_tokens = args.retained_tokens
    
    model.eval()
    
    # 加载问题数据
    print(f"正在加载问题文件: {args.question_file}")
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    
    # 创建输出文件
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    
    print(f"开始评估，共 {len(questions)} 个问题")
    print(f"结果将保存到: {answers_file}")
    
    # 逐个处理问题
    for idx, line in enumerate(tqdm(questions, desc="评估进度", mininterval=10)):
        question_id = line["question_id"]
        image_file = line["image"]
        question_text = line["text"]
        
        # 加载图像
        image_path = os.path.join(args.image_folder, image_file)
        
        try:
            # 构建 Qwen2-VL 的消息格式
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": question_text},
                    ],
                }
            ]
            
            # 使用 processor 处理输入
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)
            
            # 将输入移到 GPU
            #inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # 生成答案
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
            
            # 解码输出（只解码新生成的部分）
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0].strip()
            
        except Exception as e:
            print(f"\n处理问题 {question_id} 时出错: {str(e)}")
            output_text = "error"
        
        # 保存结果
        ans_id = shortuuid.uuid()
        result = {
            "question_id": question_id,
            "prompt": question_text,
            "text": output_text,
            "answer_id": ans_id,
            "model_id": args.model_path,
            "metadata": {
                "pruning_loc": args.pruning_loc,
                "retained_tokens": args.retained_tokens,
            }
        }
        ans_file.write(json.dumps(result) + "\n")
        ans_file.flush()
        
        # 可选：每 100 个问题打印一次进度
        if (idx + 1) % 100 == 0:
            print(f"\n已处理 {idx + 1}/{len(questions)} 个问题")

        #del inputs, generated_ids, generated_ids_trimmed
        #torch.cuda.empty_cache()

    ans_file.close()
    print(f"\n评估完成！结果已保存到: {answers_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen2-VL TextVQA 评估")
    
    # 模型参数
    parser.add_argument("--model-path", type=str, required=True, help="Qwen2-VL 模型路径")
    
    # 数据参数
    parser.add_argument("--question-file", type=str, required=True, help="问题文件路径 (jsonl)")
    parser.add_argument("--image-folder", type=str, required=True, help="图像文件夹路径")
    parser.add_argument("--answers-file", type=str, required=True, help="输出答案文件路径")
    
    # 剪枝参数
    parser.add_argument("--pruning-loc", type=int, nargs="+", default=[0, 9, 19], 
                        help="剪枝层位置，例如: 1 10")
    parser.add_argument("--retained-tokens", type=int, default=10, 
                        help="保留的视觉token数量 (192/128/64/48/32)")
    
    # 生成参数
    parser.add_argument("--temperature", type=float, default=0.0, help="采样温度")
    parser.add_argument("--top_p", type=float, default=None, help="nucleus sampling")
    parser.add_argument("--num_beams", type=int, default=1, help="beam search 数量")
    parser.add_argument("--max_new_tokens", type=int, default=32, help="最大生成token数")
    
    # 分块处理参数
    parser.add_argument("--num-chunks", type=int, default=1, help="数据分块数量")
    parser.add_argument("--chunk-idx", type=int, default=0, help="当前处理的分块索引")
    
    parser.add_argument("--attn-backend", type=str, default="sdpa",
                        choices=["sdpa", "fa", "flash_attention_2", "eager"],
                        help="主干注意力后端；剪枝打分始终按 eager 公式计算，选 eager 与论文/baseline 逐位一致")
    args = parser.parse_args()
    
    eval_model(args)

