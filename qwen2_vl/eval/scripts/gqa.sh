#!/usr/bin/env bash
# Qwen2-VL-2B inference and evaluation on GQA.
# Usage: bash qwen2_vl/eval/scripts/gqa.sh [MODEL_PATH]
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT" || exit 1
DATA_ROOT="${DATA_ROOT:-playground/data/eval}"


gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="qwen2vl-2b"
SPLIT="llava_gqa_testdev_balanced"
GQADIR="${DATA_ROOT}/gqa/"

MODEL_PATH=${1:-"Qwen/Qwen2-VL-2B-Instruct"}
RETAINED_TOKENS=${2:-10}
PRUNING_LOC=${3:-"0 9 19"}
ATTN_BACKEND=${4:-${ATTN_BACKEND:-sdpa}}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python qwen2_vl/eval/model_vqa_qwen.py \
        --attn-backend $ATTN_BACKEND \
        --model-path $MODEL_PATH \
        --question-file ${DATA_ROOT}/gqa/$SPLIT.jsonl \
        --image-folder ${DATA_ROOT}/gqa/images \
        --answers-file ${DATA_ROOT}/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --retained-tokens $RETAINED_TOKENS \
        --pruning-loc $PRUNING_LOC &
done

wait

output_file=${DATA_ROOT}/gqa/answers/$SPLIT/$CKPT/merge.jsonl

> "$output_file"

for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${DATA_ROOT}/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python qwen2_vl/eval/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/testdev_balanced_predictions.json

cd $GQADIR
python qwen2_vl/eval/1_eval.py --tier testdev_balanced