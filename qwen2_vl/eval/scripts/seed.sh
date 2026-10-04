#!/usr/bin/env bash
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
ATTN_BACKEND=${4:-${ATTN_BACKEND:-sdpa}}  # sdpa / fa / eager（eager 与论文结果一致）

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python qwen2_vl/eval/model_vqa_qwen.py \
        --attn-backend $ATTN_BACKEND \
        --model-path $MODEL_PATH \
        --question-file ${DATA_ROOT}/seed_bench/llava-seed-bench.jsonl \
        --image-folder ${DATA_ROOT}/seed_bench \
        --answers-file ${DATA_ROOT}/seed_bench/answers/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 &
done

wait

output_file=${DATA_ROOT}/seed_bench/answers/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${DATA_ROOT}/seed_bench/answers/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

# Evaluate
python qwen2_vl/eval/convert_seed_for_submission.py \
    --annotation-file ${DATA_ROOT}/seed_bench/SEED-Bench/SEED-Bench.json \
    --result-file $output_file \
    --result-upload-file ${DATA_ROOT}/seed_bench/answers_upload/qwen2vl-2b.jsonl

