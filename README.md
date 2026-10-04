<div align="center">

<h1>AdaptInfer: Adaptive Token Pruning for Vision-Language Model Inference via Dynamical Text Guidance</h1>

<strong>Weichen Zhang</strong><sup>1,2,‡</sup>, <strong>Zhui Zhu</strong><sup>2,‡</sup>, <strong>Ningbo Li</strong><sup>2,3</sup>, <strong>Shilong Tao</strong><sup>4</sup>, <strong>Hongzi Zhu</strong><sup>5</sup>, <strong>Jingao Xu</strong><sup>1</sup>, <strong>Kebin Liu</strong><sup>2,✉</sup>, <strong>Yunhao Liu</strong><sup>2</sup>

<sup>1</sup>The University of Hong Kong · <sup>2</sup>Tsinghua University ·
<sup>3</sup>The Hong Kong University of Science and Technology · <sup>4</sup>Peking University · <sup>5</sup>Shanghai Jiao Tong University

‡ Equal contribution. ✉ Corresponding author.

</div>

## Overview

![The architecture of AdaptInfer.](assests/adaptinfer_architecture.png)

## Installation

### LLaVA

```bash
conda env create -f environment.yml
conda activate AdaptInfer
pip install flash-attn==2.3.3 --no-build-isolation
```

### Qwen2-VL

```bash
conda activate qwen
pip install -r qwen2_vl/requirements.txt
```

## Usage

See [Qwen2-VL setup and evaluation](qwen2_vl/README.md). API keys must come from `OPENAI_API_KEY`; revoke any key previously hard-coded in source.

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

We extend our gratitude to the open-source efforts of [LLaVA](https://github.com/haotian-liu/LLaVA), [Qwen](https://github.com/QwenLM/Qwen2-VL), and [SparseVLM](https://github.com/Gumpest/SparseVLMs).
