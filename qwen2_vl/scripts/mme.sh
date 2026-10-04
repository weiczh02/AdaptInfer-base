#!/usr/bin/env bash
set -euo pipefail

if (( $# < 4 )); then
    echo "Usage: bash qwen2_vl/scripts/mme.sh MODEL_PATH QUESTION_FILE IMAGE_FOLDER ANSWERS_FILE [extra arguments]" >&2
    exit 2
fi
model_path="$1"
question_file="$2"
image_folder="$3"
answers_file="$4"
shift 4

python -m qwen2_vl.eval.vqa \
    --model-path "$model_path" \
    --question-file "$question_file" \
    --image-folder "$image_folder" \
    --answers-file "$answers_file" \
    --pruning-layers 0 9 19 \
    --keep-ratios 0.17 0.6 0.3 \
    "$@"
