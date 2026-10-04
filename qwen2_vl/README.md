# Qwen2-VL Inference

We provide AdaptInfer inference for Qwen2-VL-2B. We prune visual tokens after decoder layers **0, 9, and 19** (zero-based), using text-to-text attention to weight text-to-vision attention. The default keep ratios are **0.17, 0.6, and 0.3**, each relative to the visual tokens entering that pruning stage.

## Installation

Run from the repository root. Use a separate environment from LLaVA:

```bash
conda env create -f qwen2_vl/environment.yml
conda activate AdaptInfer-Qwen2VL
```

The environment uses Python 3.10, PyTorch 2.6.0 with CUDA 12.4 wheels, and Transformers 4.57.1. An NVIDIA GPU with a compatible driver is required for GPU inference. We use eager attention to obtain the attention weights needed for pruning.

## Inference

```bash
python -m qwen2_vl.inference \
    --model-path Qwen/Qwen2-VL-2B-Instruct \
    --image /path/to/image.jpg \
    --question "Describe the image."
```

For video inference, replace `--image` with `--video /path/to/video.mp4`. The default sampling rate is 4 FPS. We process one image or video per prompt, with batch size 1 and greedy decoding. Use `--temperature` for sampling, and `--pruning-layers` / `--keep-ratios` to change the pruning configuration. Image resolution defaults to 256–1280 visual tokens.

## Evaluation

Prepare the benchmark data and run these examples from the repository root. Replace the paths with your local files. Image questions use the existing JSONL fields `question_id`, `image`, and `text`; MMBench uses its TSV file with base64 images. Video questions use `video_name`, `question`, and `question_id`; answers use `answer` and optionally `question_id`.

```bash
MODEL_PATH="Qwen/Qwen2-VL-2B-Instruct"
DATA_ROOT="/path/to/datasets"

bash qwen2_vl/scripts/mme.sh "$MODEL_PATH" \
    "$DATA_ROOT/mme/llava_mme.jsonl" "$DATA_ROOT/mme/MME_Benchmark_release_version" outputs/mme.jsonl

bash qwen2_vl/scripts/textvqa.sh "$MODEL_PATH" \
    "$DATA_ROOT/textvqa/llava_textvqa_val_v051_ocr.jsonl" "$DATA_ROOT/textvqa/train_images" outputs/textvqa.jsonl

bash qwen2_vl/scripts/pope.sh "$MODEL_PATH" \
    "$DATA_ROOT/pope/llava_pope_test.jsonl" "$DATA_ROOT/coco/val2014" outputs/pope.jsonl

bash qwen2_vl/scripts/mmbench.sh "$MODEL_PATH" \
    "$DATA_ROOT/mmbench/mmbench_dev_20230712.tsv" outputs/mmbench.jsonl

bash qwen2_vl/scripts/video_tgif.sh "$MODEL_PATH" \
    "$DATA_ROOT/tgif/videos" "$DATA_ROOT/tgif/test_q.json" "$DATA_ROOT/tgif/test_a.json" outputs/tgif.jsonl

bash qwen2_vl/scripts/video_msrvtt.sh "$MODEL_PATH" \
    "$DATA_ROOT/msrvtt/videos" "$DATA_ROOT/msrvtt/test_q.json" "$DATA_ROOT/msrvtt/test_a.json" outputs/msrvtt.jsonl

bash qwen2_vl/scripts/video_msvd.sh "$MODEL_PATH" \
    "$DATA_ROOT/msvd/videos" "$DATA_ROOT/msvd/test_q.json" "$DATA_ROOT/msvd/test_a.json" outputs/msvd.jsonl
```

The video examples use the first 1,000 questions at 4 FPS. Append `--limit 0` to evaluate the full set. All scripts accept additional inference arguments. Use `--num-chunks N --chunk-idx I` to split inference, then merge the prediction files before scoring. MMBench also supports `--all-rounds` and `--lang cn`.

Image predictions contain `question_id`, `prompt`, and `text`. Video predictions contain `id`, `question`, `answer`, and `pred`. Missing data and inference errors stop the run so failed examples are not silently counted as predictions.

```bash
# TextVQA accuracy
python -m qwen2_vl.eval.score_textvqa \
    --annotation-file "$DATA_ROOT/textvqa/TextVQA_0.5.1_val.json" --result-file outputs/textvqa.jsonl

# POPE metrics
python llava/eval/eval_pope.py \
    --annotation-dir "$DATA_ROOT/pope/coco" \
    --question-file "$DATA_ROOT/pope/llava_pope_test.jsonl" --result-file outputs/pope.jsonl

# MME: score the converted files with the official MME evaluation tool.
python -m qwen2_vl.eval.convert_mme \
    --answers-file outputs/mme.jsonl --mme-data-path "$DATA_ROOT/mme/MME_Benchmark_release_version" \
    --output-dir outputs/mme_results

# MMBench single-round submission
python -m qwen2_vl.eval.convert_mmbench \
    --annotation-file "$DATA_ROOT/mmbench/mmbench_dev_20230712.tsv" \
    --answers-file outputs/mmbench.jsonl --output-file outputs/mmbench.xlsx
```

Video prediction files follow the Video-ChatGPT QA format and can be scored with the corresponding benchmark evaluation tools.
