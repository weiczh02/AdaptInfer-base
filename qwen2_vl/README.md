# Qwen2-VL

AdaptInfer sparse inference for Qwen2-VL-2B.

## Environment

Create the environment from the repository root. Dependencies include PyTorch 2.5.1 and Transformers 4.57.1:

```bash
conda create -n qwen python=3.10 -y
conda activate qwen
pip install -r qwen2_vl/requirements.txt
```

The attention backends are `sdpa` (default), `fa` (FlashAttention 2), and `eager`. FlashAttention is optional; if it cannot be loaded, `fa` falls back to SDPA.

## Data

Follow the [evaluation data guide](../docs/Evaluation.md) and place image benchmarks under `playground/data/eval/`. MME additionally requires `playground/data/eval/MME/MME_Benchmark_release_version/`. GQA requires the official scoring script at `playground/data/eval/gqa/eval/1_eval.py`.

For TGIF, MSRVTT, and MSVD, follow [Video-ChatGPT's data preparation](https://github.com/mbzuai-oryx/Video-ChatGPT/blob/main/quantitative_evaluation/README.md#zero-shot-question-answer-evaluation) and place the data under `playground/data/eval/GPT_Zero_Shot_QA/`. Each benchmark needs its videos, `test_q.json`, and `test_a.json`.

## Evaluation

```bash
ATTN_BACKEND=eager bash qwen2_vl/eval/scripts/mme.sh Qwen/Qwen2-VL-2B-Instruct
```

Other scripts are in [eval/scripts](eval/scripts). Set `ATTN_BACKEND=fa` or `ATTN_BACKEND=eager` to select the backend. Set `DATA_ROOT` to use a different data directory.

GPT-based video QA scoring requires your own OpenAI API key. Configure it with `--api_key` in the [Video-ChatGPT evaluator](https://github.com/mbzuai-oryx/Video-ChatGPT/blob/main/quantitative_evaluation/README.md#zero-shot-question-answer-evaluation).
