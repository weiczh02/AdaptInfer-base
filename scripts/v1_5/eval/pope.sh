#!/bin/bash
# Usage: bash scripts/v1_5/eval/pope.sh [MODEL_PATH] [RETAINED_TOKENS]

MODEL_PATH=${1:-"liuhaotian/llava-v1.5-7b"}
RETAINED_TOKENS=${2:-64}
MODEL_NAME="${MODEL_PATH%/}"
EXPERIMENT="${MODEL_NAME##*/}"

python -m llava.eval.model_vqa_loader \
    --model-path "$MODEL_PATH" \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder ./playground/data/eval/pope/val2014 \
    --answers-file "./playground/data/eval/pope/answers/${EXPERIMENT}.jsonl" \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --retained_tokens "$RETAINED_TOKENS"

python llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file "./playground/data/eval/pope/answers/${EXPERIMENT}.jsonl"