#!/usr/bin/env bash
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT" || exit 1
DATA_ROOT="${DATA_ROOT:-playground/data/eval}"

SPLIT="mmbench_dev_20230712"


MODEL_PATH=${1:-"Qwen/Qwen2-VL-2B-Instruct"}
RETAINED_TOKENS=${2:-10}
PRUNING_LOC=${3:-"0 9 19"}
ATTN_BACKEND=${4:-${ATTN_BACKEND:-sdpa}}  # sdpa / fa / eager（eager 与论文结果一致）

# 数据路径
QUESTION_FILE="${DATA_ROOT}/mmbench/$SPLIT.tsv"
ANNOTATION_DIR="${DATA_ROOT}/pope/coco"

# 输出文件
ANSWERS_FILE="${DATA_ROOT}/mmbench/answers/$SPLIT/qwen2vl-2b.jsonl"

python qwen2_vl/eval/model_vqa_mmb.py \
    --attn-backend $ATTN_BACKEND \
    --pruning-loc $PRUNING_LOC \
    --model-path $MODEL_PATH \
    --question-file $QUESTION_FILE \
    --answers-file $ANSWERS_FILE \
    --single-pred-prompt \
    --temperature 0 \

mkdir -p ${DATA_ROOT}/mmbench/answers_upload/$SPLIT

python qwen2_vl/eval/convert_mmbench_for_submission.py \
    --annotation-file ${DATA_ROOT}/mmbench/$SPLIT.tsv \
    --result-dir ${DATA_ROOT}/mmbench/answers/$SPLIT \
    --upload-dir ${DATA_ROOT}/mmbench/answers_upload/$SPLIT \
    --experiment qwen2vl-2b