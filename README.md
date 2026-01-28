<div align="center">

# AttnRL

[![Survey](https://img.shields.io/badge/Paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2509.26628)  [![Github](https://img.shields.io/badge/GitHub-000000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/RyanLiu112/AttnRL)  [![HF Paper](https://img.shields.io/badge/HF--Paper-FFD14D?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/papers/2509.26628)  

</div>



## 🔔 News

- **[2026-01-26]** 🎉 Our work is accepted by [ICLR 2026](https://openreview.net/forum?id=NCN8oUsiNf)!
- **[2025-10-21]** 📢 Our work is reported by [Synced (机器之心)](https://mp.weixin.qq.com/s/laPqnICWG-PIVIDQOTnbNQ)!
- **[2025-10-10]** ✨ Code is now available.
- **[2025-09-30]** 📄 Our paper is released on [arXiv](https://arxiv.org/abs/2509.26628).



## 🚀 Getting Started

### Installation

Clone the repository:

```bash
git clone https://github.com/RyanLiu112/AttnRL.git
cd AttnRL
```

Create a new conda environment and install the dependencies:

```bash
conda create -n attnrl python=3.10
conda activate attnrl
bash scripts/install_vllm_sglang_mcore.sh
```

### Data Preparation

The training dataset ([DeepScaleR-Preview-Dataset](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset)) is at `data/train/deepscaler_train.parquet`, which contains `40.3k` mathematical reasoning data.
The evaluation datasets are in `data/eval/` and the suffix `_${K}` indicates the number of duplicate samples for each question.

### Training

For training AttnRL with [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) backbone on 8 H100 GPUs, run:

```bash
bash recipe/attnrl/run_attnrl_r1_distill_1.5b_8k.sh
```

### Evaluation

Evaluation scripts are the same as the training scripts. `+trainer.val_only=True` should be added to perform evaluation only. We recommend setting `data.max_prompt_length=2048` and `data.max_response_length=32768`.



## 📝 Citation

If you find this work helpful, please kindly cite our paper:

```bibtex
@inproceedings{AttnRL,
    title     = {Attention as a Compass: Efficient Exploration for Process-Supervised {RL} in Reasoning Models},
    author    = {Runze Liu and Jiakang Wang and Yuling Shi and Zhihui Xie and Chenxin An and Kaiyan Zhang and Jian Zhao and Xiaodong Gu and Lei Lin and Wenping Hu and Xiu Li and Fuzheng Zhang and Guorui Zhou and Kun Gai},
    booktitle = {The Fourteenth International Conference on Learning Representations},
    year      = {2026},
    url       = {https://openreview.net/forum?id=NCN8oUsiNf}
}
```



## 💡 Acknowledgements

Our code is based on [verl](https://github.com/volcengine/verl) ([commit](https://github.com/volcengine/verl/commit/83ebd007e01de29bbe353de112d04245b4820b47)) and [TreeRL](https://github.com/THUDM/TreeRL).
Our training dataset is from [DeepScaleR-Preview-Dataset](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset) and rule-based verifier is based on [Skywork-OR1](https://github.com/SkyworkAI/Skywork-OR1), and [Archer](https://github.com/wizard-III/ArcherCodeR).
