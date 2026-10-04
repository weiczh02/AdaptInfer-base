#!/usr/bin/env bash
# Qwen2-VL-2B inference and evaluation on MME.
# Usage: bash qwen2_vl/eval/scripts/mme.sh [MODEL_PATH]
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT" || exit 1
DATA_ROOT="${DATA_ROOT:-playground/data/eval}"



MODEL_PATH=${1:-"Qwen/Qwen2-VL-2B-Instruct"}
RETAINED_TOKENS=${2:-10}
PRUNING_LOC=${3:-"0 9 19"}
ATTN_BACKEND=${4:-${ATTN_BACKEND:-sdpa}}

QUESTION_FILE="${DATA_ROOT}/MME/llava_mme.jsonl"
IMAGE_FOLDER="${DATA_ROOT}/MME/MME_Benchmark_release_version"

ANSWERS_FILE="${DATA_ROOT}/MME/answers/qwen2-vl-2b.jsonl"

echo "================================================"
echo "Qwen2-VL MME evaluation"
echo "================================================"
echo "模型路径: $MODEL_PATH"
echo "Retention preset: $RETAINED_TOKENS"
echo "剪枝层位置: $PRUNING_LOC"
echo "问题文件: $QUESTION_FILE"
echo "图像文件夹: $IMAGE_FOLDER"
echo "输出文件: $ANSWERS_FILE"
echo "================================================"

python qwen2_vl/eval/model_vqa_qwen.py \
    --attn-backend $ATTN_BACKEND \
    --model-path $MODEL_PATH \
    --question-file $QUESTION_FILE \
    --image-folder $IMAGE_FOLDER \
    --answers-file $ANSWERS_FILE \
    --temperature 0 \
    --num_beams 1 \
    --max_new_tokens 64 \
    --retained-tokens $RETAINED_TOKENS \
    --pruning-loc $PRUNING_LOC

echo ""
echo "================================================"
echo "开始计算准确率..."
echo "================================================"

cd ${DATA_ROOT}/MME

python "$REPO_ROOT/qwen2_vl/eval/mme/convert_answer_to_mme.py" --experiment qwen2-vl-2b

cd eval_tool

python "$REPO_ROOT/qwen2_vl/eval/mme/calculation.py" --results_dir answers/qwen2-vl-2b

echo ""
echo "================================================"
echo "评估完成！"
echo "================================================"