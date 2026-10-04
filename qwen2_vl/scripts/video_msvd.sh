#!/usr/bin/env bash
set -euo pipefail

if (( $# < 5 )); then
    echo "Usage: bash qwen2_vl/scripts/video_msvd.sh MODEL_PATH VIDEO_DIR QUESTIONS_JSON ANSWERS_JSON PREDICTIONS_JSONL [extra arguments]" >&2
    exit 2
fi
model_path="$1"
video_dir="$2"
question_file="$3"
answer_file="$4"
answers_file="$5"
shift 5

python -m qwen2_vl.eval.video_qa \
    --model-path "$model_path" \
    --video-dir "$video_dir" \
    --question-file "$question_file" \
    --answer-file "$answer_file" \
    --answers-file "$answers_file" \
    --pruning-layers 0 9 19 \
    --keep-ratios 0.17 0.6 0.3 \
    --fps 4 --limit 1000 \
    "$@"
