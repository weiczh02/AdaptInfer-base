#!/bin/bash
# Usage: bash scripts/v1_5/eval/textvqa.sh [MODEL_PATH] [RETAINED_TOKENS]

MODEL_PATH=${1:-"liuhaotian/llava-v1.5-7b"}
RETAINED_TOKENS=${2:-64}
MODEL_NAME="${MODEL_PATH%/}"
EXPERIMENT="${MODEL_NAME##*/}"

python -m llava.eval.model_vqa_loader \
    --model-path "$MODEL_PATH" \
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ./playground/data/eval/textvqa/train_images \
    --answers-file "./playground/data/eval/textvqa/answers/${EXPERIMENT}.jsonl" \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --retained_tokens "$RETAINED_TOKENS"

python -m llava.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file "./playground/data/eval/textvqa/answers/${EXPERIMENT}.jsonl"