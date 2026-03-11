<div align="center">

<h1>🦅 TALON: Test-time Adaptive Learning for On-the-Fly Category Discovery</h1>

<div>
    <strong>Accepted to CVPR 2026</strong>
</div>

<div>
    <h4 align="center">
        • <a href="https://arxiv.org/abs/2603.08075" target='_blank'>[arXiv]</a> •
        • <a href="https://huggingface.co/papers/2603.08075" target='_blank'>[Hugging Face]</a> •
    </h4>
</div>

<!-- <br>
<div>
  <img src="assets/framework.png" width="700px"/>
</div> -->

</div>

## 💡 TL;DR
> *TALON is a novel approach for on-the-fly category discovery (OCD) that utilizes a test-time adaptation framework to continuously learn from unlabeled data streams. This repository contains the official implementation of our CVPR 2026 paper.*

## 📂 Project Structure

```
TALON/
├── config.py                      <- Dataset root paths & DINO pretrain path
├── train.py                       <- Training entry point
├── test.py                        <- Evaluation entry point
├── pyproject.toml                 <- Project metadata & dependencies (uv)
│
├── data/                          <- Dataset loading modules
│   ├── cifar.py                      CIFAR-10 / CIFAR-100
│   ├── cub.py                        CUB-200-2011
│   ├── food101.py                    Food-101
│   ├── pets.py                       Oxford-IIIT Pet
│   ├── scars.py                      Stanford Cars
│   └── imagenet.py                   ImageNet-100
│
├── methods/                       <- Model implementations
│   └── talon/
│       ├── model.py                  TALONModel (backbone + learnable prototypes)
│       ├── trainer.py                Training loop, TTA, evaluation logic
│       └── utils.py                  NCM prototypes, logits, metrics
│
├── tools/                         <- General utilities
│   ├── evaluate_utils.py             Clustering accuracy (Hungarian assignment)
│   ├── losses.py                     Loss functions
│   └── train_utils.py               SmoothedValue, training helpers
│
└── checkpoints/                   <- Pretrained model weights (download below)
    ├── clip/{cub,food,scars}/
    └── dino/{cub,food,scars}/
```

## ⚙️ Dependencies and Installation

This project uses [`uv`](https://github.com/astral-sh/uv) for dependency management to ensure a clean and reproducible environment.

**Requirements:**
- Python >= 3.12
- PyTorch (CUDA 12.4)
- OpenAI CLIP / timm (DINO ViT-B/16)

```bash
# 1. Clone the repository
git clone https://github.com/ynanwu/TALON
cd TALON

# 2. Install uv (if not already installed)
# See https://github.com/astral-sh/uv for details

# 3. Install all dependencies
uv sync

# That's it! Use `uv run` to execute any script — no need to manually activate the venv.
```

## 📦 Pretrained Models

We provide pretrained checkpoints for **CUB**, **Food-101**, and **Stanford Cars** using both CLIP and DINO backbones.

📥 Download from [**Google Drive**](https://drive.google.com/drive/folders/11O7OtYzBzRT6rrSjhl3u3SOdHxu8kqIK) or [**Hugging Face**](https://huggingface.co/conrain/TALON) and place them as follows:

```
checkpoints/
├── clip/
│   ├── cub/
│   │   └── best_model.pth
│   ├── food/
│   │   └── best_model.pth
│   └── scars/
│       └── best_model.pth
└── dino/
    ├── cub/
    │   └── best_model.pth
    ├── food/
    │   └── best_model.pth
    └── scars/
        └── best_model.pth
```

## 📊 Datasets

Supported datasets and their known / novel class splits:

| Dataset | Total Classes | Known Classes | Novel Classes |
|---------|:---:|:---:|:---:|
| CIFAR-10 | 10 | 6 | 4 |
| CIFAR-100 | 100 | 80 | 20 |
| CUB-200-2011 | 200 | 100 | 100 |
| Oxford-IIIT Pet | 37 | 19 | 18 |
| Stanford Cars | 196 | 98 | 98 |
| Food-101 | 101 | 51 | 50 |
| ImageNet-100 | 100 | 80 | 20 |

Configure the dataset root paths in `config.py` before running:

```python
# config.py
CUB_ROOT = "datasets/CUB"
FOOD_101_ROOT = "datasets/Food101"
OXFORD_PET_ROOT = "datasets/OxfordPets"
SCARS_ROOT = "datasets/stanford_cars/"
CIFAR_10_ROOT = "datasets/CIFAR/"
CIFAR_100_ROOT = "datasets/CIFAR/"
IMAGENET_ROOT = "datasets/imagenet/"

# DINO backbone pretrained weights
pretrain_path = "dino_vitbase16_pretrain.pth"
```

## 🚀 Usage

### Training

```bash
# CUB with CLIP backbone
uv run train.py --dataset_name cub --backbone clip --save_dir my_experiment --device cuda:0

# Food-101 with DINO backbone, custom tau and TTA
uv run train.py --dataset_name food --backbone dino --tau 0.8 --tta_state M+P --epochs 100 --device cuda:0

# Stanford Cars with CLIP, Model TTA only
uv run train.py --dataset_name scars --backbone clip --tta_state M --save_dir scars_exp --device cuda:0
```

<details>
<summary><b>📋 Full list of training arguments</b></summary>

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--seed` | int | 1028 | Random seed |
| `--dataset_name` | str | `cub` | `pets` / `scars` / `cub` / `food` / `imagenet100` / `cifar10` / `cifar100` |
| `--backbone` | str | `clip` | `clip` or `dino` |
| `--tta_state` | str | `M+P` | `M` / `P` / `M+P` / `none` (see below) |
| `--tau` | float | 0.75 | Threshold for novel class detection (see below) |
| `--device` | str | auto | e.g. `cuda:0` (auto-selects best GPU if empty) |
| `--save_dir` | str | `test` | Directory for logs and checkpoints |
| `--train_batch_size` | int | 128 | Training batch size |
| `--eval_batch_size` | int | 64 | Evaluation batch size |
| `--num_workers` | int | 8 | DataLoader workers |
| `--prop_train_labels` | float | 0.5 | Proportion of labeled training data |
| `--epochs` | int | 100 | Total training epochs |
| `--start_epoch` | int | 0 | Resume from epoch |
| `--clip_grad` | float | None | Gradient clipping max norm |

</details>

#### 🌡️ About `--tau` (Novel Class Detection Threshold)

`tau` controls the confidence threshold for **novel class detection**. When a test sample's maximum cosine similarity to all known prototypes is below `tau`, it is identified as a **novel (unseen) class** and a new prototype is created.
#### 🔄 About `--tta_state` (Test-Time Adaptation Modes)

| Mode | What Gets Updated | Description |
|------|-------------------|-------------|
| `M` | Backbone norm layers | **Model TTA** — fine-tunes the affine parameters (weight & bias) of LayerNorm/BatchNorm in the **last transformer block**. Minimizes entropy + maximizes instance-to-prototype similarity + inter-class repulsion. |
| `P` | Class prototypes | **Prototype TTA** — updates class prototype vectors via **EMA** based on test features. Adapts the classifier without touching the backbone. |
| `M+P` | Both | **Joint TTA** — applies both Model TTA and Prototype TTA simultaneously. Typically yields the **best performance**. **(Recommended)** |
| `none` | Nothing | No adaptation. Uses fixed model and prototypes. Useful as a baseline. |

### Evaluation

```bash
# Evaluate CLIP on CUB (with default M+P TTA)
uv run test.py --dataset_name cub --backbone clip --ckpt_path checkpoints/clip/cub/best_model.pth

# Evaluate DINO on Food-101 with Prototype TTA only
uv run test.py --dataset_name food --backbone dino --tta_state P --ckpt_path checkpoints/dino/food/best_model.pth

# Evaluate without TTA (baseline)
uv run test.py --dataset_name scars --backbone clip --tta_state none --ckpt_path checkpoints/clip/scars/best_model.pth
```

## 📝 Citation

If you find this work useful for your research, please consider citing our paper:

```bibtex
@inproceedings{talon2026,
  title={TALON: Test-time Adaptive Learning for On-the-Fly Category Discovery},
  author={Wu, Yanan and Yan, Yuhan and Chen, Tailai and Chi, Zhixiang and Wu, ZiZhang and Jin, Yi and Wang, Yang and Li Zhenbo},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2026}
}
```
