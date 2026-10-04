#!/bin/bash
# Usage: bash scripts/v1_5/eval/mme.sh [MODEL_PATH] [RETAINED_TOKENS]

MODEL_PATH=${1:-"liuhaotian/llava-v1.5-7b"}
RETAINED_TOKENS=${2:-64}
MODEL_NAME="${MODEL_PATH%/}"
EXPERIMENT="${MODEL_NAME##*/}"

python -m llava.eval.model_vqa_loader \
    --model-path "$MODEL_PATH" \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file "./playground/data/eval/MME/answers/${EXPERIMENT}.jsonl" \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --retained_tokens "$RETAINED_TOKENS"

cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment "$EXPERIMENT"

cd eval_tool

python calculation.py --results_dir "answers/$EXPERIMENT"