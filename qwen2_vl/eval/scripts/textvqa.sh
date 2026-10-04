#!/usr/bin/env bash
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT" || exit 1
DATA_ROOT="${DATA_ROOT:-playground/data/eval}"


# Qwen2-VL TextVQA 评估脚本
# 用法: bash textvqa.sh [MODEL_PATH] [RETAINED_TOKENS] [PRUNING_LOC...]

# 默认参数
MODEL_PATH=${1:-"Qwen/Qwen2-VL-2B-Instruct"}
RETAINED_TOKENS=${2:-10}
PRUNING_LOC=${3:-"0 9 19"}
ATTN_BACKEND=${4:-${ATTN_BACKEND:-sdpa}}  # sdpa / fa / eager（eager 与论文结果一致）

# 数据路径
QUESTION_FILE="${DATA_ROOT}/textvqa/llava_textvqa_val_v051_ocr.jsonl"
IMAGE_FOLDER="${DATA_ROOT}/textvqa/train_images"
ANNOTATION_FILE="${DATA_ROOT}/textvqa/TextVQA_0.5.1_val.json"

# 输出文件
OUTPUT_DIR="${DATA_ROOT}/textvqa/answers"
mkdir -p $OUTPUT_DIR

MODEL_NAME=$(basename $MODEL_PATH)
ANSWERS_FILE="${OUTPUT_DIR}/qwen_${MODEL_NAME}_tokens${RETAINED_TOKENS}.jsonl"

echo "================================================"
echo "Qwen2-VL TextVQA 评估"
echo "================================================"
echo "模型路径: $MODEL_PATH"
echo "保留tokens: $RETAINED_TOKENS"
echo "剪枝层位置: $PRUNING_LOC"
echo "问题文件: $QUESTION_FILE"
echo "图像文件夹: $IMAGE_FOLDER"
echo "输出文件: $ANSWERS_FILE"
echo "================================================"

# 运行评估
python qwen2_vl/eval/model_vqa_qwen.py \
    --attn-backend $ATTN_BACKEND \
    --model-path $MODEL_PATH \
    --question-file $QUESTION_FILE \
    --image-folder $IMAGE_FOLDER \
    --answers-file $ANSWERS_FILE \
    --temperature 0 \
    --num_beams 1 \
    --max_new_tokens 128 \
    --retained-tokens $RETAINED_TOKENS \
    --pruning-loc $PRUNING_LOC

# 评估结果
echo ""
echo "================================================"
echo "开始计算准确率..."
echo "================================================"


python qwen2_vl/eval/utils/eval_textvqa.py \
    --annotation-file $ANNOTATION_FILE \
    --result-file "$ANSWERS_FILE"

echo ""
echo "================================================"
echo "评估完成！"
echo "================================================"