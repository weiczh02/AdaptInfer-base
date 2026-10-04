<div align="center">

<h1>AdaptInfer: Adaptive Token Pruning for Vision-Language Model Inference via Dynamical Text Guidance</h1>

<strong>Weichen Zhang</strong><sup>1,2,‡</sup>, <strong>Zhui Zhu</strong><sup>2,‡</sup>, <strong>Ningbo Li</strong><sup>2,3</sup>, <strong>Shilong Tao</strong><sup>4</sup>, <strong>Hongzi Zhu</strong><sup>5</sup>, <strong>Jingao Xu</strong><sup>1</sup>, <strong>Kebin Liu</strong><sup>2,✉</sup>, <strong>Yunhao Liu</strong><sup>2</sup>

<sup>1</sup>The University of Hong Kong · <sup>2</sup>Tsinghua University ·
<sup>3</sup>The Hong Kong University of Science and Technology <br>· <sup>4</sup>Peking University · <sup>5</sup>Shanghai Jiao Tong University

‡ Equal contribution. ✉ Corresponding author.

</div>

## News
Our paper "AdaptInfer: Adaptive Token Pruning for Vision-Language Model Inference via Dynamical Text Guidance" has been accepted to Findings of EMNLP 2026.

## Overview

![The architecture of AdaptInfer.](assests/adaptinfer_architecture.png)

## Installation

### LLaVA-1.5-7B

```bash
conda env create -f environment.yml
conda activate AdaptInfer
pip install flash-attn==2.3.3 --no-build-isolation
```

## Usage

Run the following examples from the repository root.

### LLaVA-1.5

Prepare the image benchmarks using the [evaluation data guide](docs/Evaluation.md) and place them under `playground/data/eval/`. For instance, to evaluate AdaptInfer on MME, please include `MME/MME_Benchmark_release_version/`.

```bash
conda activate AdaptInfer
bash scripts/v1_5/eval/mme.sh liuhaotian/llava-v1.5-7b
```

For Qwen2-VL-2B installation and usage, see [qwen2_vl/README.md](qwen2_vl/README.md).

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Citation

```bibtex
@misc{zhang2026adaptinferadaptivetokenpruning,
      title={AdaptInfer: Adaptive Token Pruning for Vision-Language Model Inference with Dynamical Text Guidance},
      author={Weichen Zhang and Zhui Zhu and Ningbo Li and Shilong Tao and Kebin Liu and Yunhao Liu},
      year={2026},
      eprint={2508.06084},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2508.06084},
}
```

## Acknowledgment

We sincerely appreciate the open-source efforts of [LLaVA](https://github.com/haotian-liu/LLaVA), [Qwen](https://github.com/QwenLM/Qwen2-VL), and [SparseVLM](https://github.com/Gumpest/SparseVLMs).
