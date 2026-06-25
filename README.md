# DiffVNet — Semi-Supervised Medical Image Segmentation

Official implementation for **"DiffVNet: Diffusion-based Semi-Supervised Segmentation with Fused Proxy Loss"** (BIBM 2024 submission).

---

## Method Overview

**DiffVNet** is a semi-supervised segmentation framework combining:

- **Backbone**: VNet with shared encoder and 3 separate decoders (mean teacher + diffusion branch)
- **Diffusion module**: Denoising diffusion process on latent features for pseudo-label refinement
- **FusedProxyLoss**: Two complementary unsupervised losses
  - **CDBA** (Class Distribution-Based Alignment): distributional proxy loss aligning student/teacher class prototypes
  - **SAC** (Sampling-Aware Consistency): variation vector loss on uncertain regions

```
Input CT
   └─► SharedEncoder
          ├─► Decoder₁ (supervised)
          ├─► Decoder₂ (EMA teacher)
          └─► Decoder₃ (diffusion) ─► DiffusionModule ─► refined pseudo-labels
                                                             └─► FusedProxyLoss
```

---

## Results

### Synapse 20% (13-class abdominal segmentation)

Split: 18 labeled / 54 unlabeled / 6 eval / 18 test

| Seed | DSC ↑ | ASD ↓ | HD95 ↓ | NSD ↑ |
|------|------:|------:|-------:|------:|
| seed0 | 63.74% | 1.88mm | 5.86mm | 74.93% |
| seed1 | 63.65% | 1.63mm | 5.94mm | 75.09% |
| seed666 | 60.18% | 1.95mm | 5.51mm | 71.69% |
| **3-fold mean** | **62.52%** | **1.82mm** | **5.77mm** | **73.90%** |

### AMOS 5% (15-class abdominal segmentation)

Split: 10 labeled / 206 unlabeled / 24 eval / 120 test

| Seed | DSC ↑ | ASD ↓ | HD95 ↓ | NSD ↑ |
|------|------:|------:|-------:|------:|
| seed0 | 57.10% | 14.87mm | 24.85mm | 53.73% |
| seed1 | 42.25% | 43.21mm | 54.44mm | 35.29% |
| seed666 | 39.88% | 45.24mm | 59.15mm | 31.14% |
| **3-fold mean** | **46.41%** | **34.44mm** | **46.15mm** | **40.05%** |

> **Note:** seed1/666 on AMOS 5% show DSC=0% for classes 5, 11, 12 (esophagus, adrenal glands).
> This is a known SSL pseudo-label collapse under the extreme 5%-label regime, not a code bug.
> See [results/](results/) for per-organ breakdowns.

Hyperparameters (both tasks): `--base_lr 0.01 -w 10 --lambda_cdba 0.05 --lambda_sac 0.2 --lambda_var 0.2 --vapl_warmup 500 --variation_warmup 3000`

---

## Repository Structure

```
bibm_GenSSL_ours/
├── code/
│   ├── train_diffusion.py      # Main training script
│   ├── test.py                 # Inference on test set
│   ├── evaluate_metrics.py     # Compute DSC/ASD/HD95/NSD from predictions
│   ├── DiffVNet/
│   │   ├── diff_vnet.py        # DiffVNet model architecture
│   │   └── guided_diffusion/   # Diffusion module (adapted from openai/guided-diffusion)
│   ├── utils/
│   │   ├── __init__.py         # Data I/O, test_all_case, EMA, augmentation helpers
│   │   ├── config.py           # Dataset paths, patch sizes, class counts
│   │   ├── fused_proxy.py      # FusedProxyLoss (CDBA + SAC variation loss)
│   │   ├── loss.py             # DC_and_CE_loss, SoftDiceLoss, RobustCrossEntropyLoss
│   │   └── metrics.py          # DSC, ASD, HD95, NSD computation
│   └── data/
│       ├── data_loaders.py     # DatasetAllTasks — unified Dataset for all tasks
│       ├── StrongAug.py        # 3D strong augmentation pipeline
│       ├── preprocess_synapse.py  # Synapse NIfTI → .npy preprocessing
│       └── preprocess_amos.py     # AMOS22 NIfTI → .npy preprocessing
├── data/
│   ├── amos_splits/            # Fixed train/eval/test/labeled/unlabeled split TXTs
│   └── synapse_splits/         # Fixed split TXTs for Synapse
├── results/
│   ├── amos_5p_seed{0,1,666}.md    # Per-organ AMOS results
│   └── synapse_20p_seed{0,1,666}.md # Per-organ Synapse results
├── slurm/                      # HPC SLURM job scripts (A800 GPU, XJTLU HPC)
├── requirements.txt            # Python dependencies
├── REPRODUCE.md                # Step-by-step reproduction guide
└── RESULTS_SUMMARY.md          # Full results table with checkpoint locations
```

---

## Pre-trained Checkpoints

All 6 checkpoints (~222MB each) are in the [GitHub Release v1.0-checkpoints](https://github.com/linvii77/bibm_GenSSL_ours/releases/tag/v1.0-checkpoints).

```bash
BASE="https://github.com/linvii77/bibm_GenSSL_ours/releases/download/v1.0-checkpoints"

# Synapse 20%
wget $BASE/synapse_20p_seed0_best_model.pth
wget $BASE/synapse_20p_seed1_best_model.pth
wget $BASE/synapse_20p_seed666_best_model.pth

# AMOS 5%
wget $BASE/amos_5p_seed0_best_model.pth
wget $BASE/amos_5p_seed1_best_model.pth
wget $BASE/amos_5p_seed666_best_model.pth
```

**Load a checkpoint:**
```python
import torch
from code.DiffVNet.diff_vnet import DiffVNet

# AMOS (16 classes); Synapse: n_classes=14
net = DiffVNet(n_channels=1, n_classes=16, n_filters=32)
net.load_state_dict(torch.load("amos_5p_seed0_best_model.pth", map_location="cpu"))
net.eval()
```

---

## Quick Start

See [REPRODUCE.md](REPRODUCE.md) for the complete step-by-step guide. Below is the minimal training command:

```bash
pip install -r requirements.txt

# 1. Preprocess data (one-time)
python code/data/preprocess_amos.py      # AMOS 5%
python code/data/preprocess_synapse.py   # Synapse 20%

# 2. Train (seed0, local GPU)
python code/train_diffusion.py \
  -t amos_0.05 --exp diffusion_fp --seed 0 \
  -sl labeled_5p -su unlabeled_5p \
  --base_lr 0.01 -w 10 \
  --lambda_cdba 0.05 --lambda_sac 0.2 --lambda_var 0.2 \
  --vapl_warmup 500 --variation_warmup 3000

# 3. Test
python code/test.py -t amos_0.05 \
  --exp Exp_IBSSL_AMOS_0.05/diffusion_fp --speed 0

# 4. Evaluate
python code/evaluate_metrics.py -t amos_0.05 \
  --exp Exp_IBSSL_AMOS_0.05/diffusion_fp --folds 1
```

---

## Key Implementation Notes

- **`test.py` resume logic**: `utils/__init__.py:test_all_case()` skips already-saved `.nii.gz` files — safe to re-run after a timeout.
- **SLURM time limit**: AMOS test jobs need `--time=12:00:00` (120 samples × ~3.3min/sample ≈ 6.6h + eval).
- **`evaluate_metrics.py --exp`** must NOT include the `/fold1` suffix — the script appends it internally.
- **Data splits are fixed** in `data/amos_splits/` and `data/synapse_splits/`. Do not regenerate them.
