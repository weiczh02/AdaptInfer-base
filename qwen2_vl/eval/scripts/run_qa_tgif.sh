#!/usr/bin/env bash
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT" || exit 1
DATA_ROOT="${DATA_ROOT:-playground/data/eval}"


ATTN_BACKEND=${ATTN_BACKEND:-sdpa}  # sdpa / fa / eager（eager 与论文结果一致）

CKPT_NAME="Qwen2-VL-2B-Instruct"
model_path="Qwen/${CKPT_NAME}"
cache_dir="./cache_dir"
GPT_Zero_Shot_QA="${DATA_ROOT}/GPT_Zero_Shot_QA"
video_dir="${GPT_Zero_Shot_QA}/TGIF_Zero_Shot_QA/mp4"
gt_file_question="${GPT_Zero_Shot_QA}/TGIF_Zero_Shot_QA/test_q.json"
gt_file_answers="${GPT_Zero_Shot_QA}/TGIF_Zero_Shot_QA/test_a.json"
output_dir="${GPT_Zero_Shot_QA}/TGIF_Zero_Shot_QA/${CKPT_NAME}"


gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}


for IDX in $(seq 0 $((CHUNKS-1))); do
  CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python qwen2_vl/eval/run_inference_video_qa.py \
      --attn-backend $ATTN_BACKEND \
      --model_path ${model_path} \
      --video_dir ${video_dir} \
      --gt_file_question ${gt_file_question} \
      --gt_file_answers ${gt_file_answers} \
      --output_dir ${output_dir} \
      --output_name ${CHUNKS}_${IDX} \
      --num_chunks $CHUNKS \
      --chunk_idx $IDX &
done

wait

output_file=${output_dir}/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${output_dir}/${CHUNKS}_${IDX}.json >> "$output_file"
done