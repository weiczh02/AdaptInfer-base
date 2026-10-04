#!/usr/bin/env bash
set -euo pipefail

if (( $# < 3 )); then
    echo "Usage: bash qwen2_vl/scripts/mmbench.sh MODEL_PATH QUESTION_TSV ANSWERS_FILE [extra arguments]" >&2
    exit 2
fi
model_path="$1"
question_file="$2"
answers_file="$3"
shift 3

python -m qwen2_vl.eval.mmbench \
    --model-path "$model_path" \
    --question-file "$question_file" \
    --answers-file "$answers_file" \
    --pruning-layers 0 9 19 \
    --keep-ratios 0.17 0.6 0.3 \
    "$@"
