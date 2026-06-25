# DiffVNet (GenSSL) — Experiment Results Summary

Model: **DiffVNet** — Diffusion-based VNet with shared encoder + 3 decoders  
Loss: **FusedProxyLoss** = CDBA distributional proxy loss + SAC variation vector loss  
Evaluated: 2026-06-25

---

## Synapse 20% — 3-Fold Results

Hyperparameters: `--base_lr 0.01 -w 10 --lambda_cdba 0.05 --lambda_sac 0.2 --lambda_var 0.2 --vapl_warmup 500 --variation_warmup 3000`  
Split: 18 labeled / 54 unlabeled / 6 eval / 18 test

| Seed | DSC (%) | ASD (mm) | HD95 (mm) | NSD (%) |
|------|--------:|--------:|--------:|--------:|
| seed0 | 63.74 | 1.88 | 5.86 | 74.93 |
| seed1 | 63.65 | 1.63 | 5.94 | 75.09 |
| seed666 | 60.18 | 1.95 | 5.51 | 71.69 |
| **Mean** | **62.52** | **1.82** | **5.77** | **73.90** |

### Per-organ detail (seed0)
See [results/synapse_20p_seed0.md](results/synapse_20p_seed0.md)

### Per-organ detail (seed1)
See [results/synapse_20p_seed1.md](results/synapse_20p_seed1.md)

### Per-organ detail (seed666)
See [results/synapse_20p_seed666.md](results/synapse_20p_seed666.md)

---

## AMOS 5% — 3-Fold Results

Hyperparameters: `--base_lr 0.01 -w 10 --lambda_cdba 0.05 --lambda_sac 0.2 --lambda_var 0.2 --vapl_warmup 500 --variation_warmup 3000`  
Split: 10 labeled / 206 unlabeled / 24 eval / 120 test  
Config: patch=(64,128,128), n_filters=32, 16 classes

| Seed | DSC (%) | ASD (mm) | HD95 (mm) | NSD (%) |
|------|--------:|--------:|--------:|--------:|
| seed0 | 57.10 | 14.87 | 24.85 | 53.73 |
| seed1 | 42.25 | 43.21 | 54.44 | 35.29 |
| seed666 | 39.88 | 45.24 | 59.15 | 31.14 |
| **Mean** | **46.41** | **34.44** | **46.15** | **40.05** |

> **Note on seed1/666:** class_5, class_11, class_12 collapsed to DSC=0% due to SSL
> pseudo-label failure on tiny structures (adrenal glands + esophagus) under 5% labels.
> This is a known instability of SSL at extreme low-label regimes. The labeled split is
> fixed; divergence is caused by training stochasticity (init + augmentation + pseudo-labels).

### Per-organ detail (seed0)
See [results/amos_5p_seed0.md](results/amos_5p_seed0.md)

### Per-organ detail (seed1)
See [results/amos_5p_seed1.md](results/amos_5p_seed1.md)

### Per-organ detail (seed666)
See [results/amos_5p_seed666.md](results/amos_5p_seed666.md)

---

## Model Checkpoint Locations

### Synapse 20%
| Seed | Path |
|------|------|
| seed0 (local) | `~/Desktop/A&D复现/logs/Exp_IBSSL_Synapse_0.2/diffusion_fp/fold1/ckpts/best_model.pth` |
| seed1 (HPC) | `/gpfs/work/aac/zimuzhang2302/AD_project/logs/diffusion_fp_synapse_20p_seed1_fixed/ckpts/best_model.pth` |
| seed666 (HPC) | `/gpfs/work/aac/zimuzhang2302/AD_project/logs/diffusion_fp_synapse_20p_seed666/ckpts/best_model.pth` |

### AMOS 5%
| Seed | Path |
|------|------|
| seed0 (local) | `~/Desktop/A&D复现/logs/Exp_IBSSL_AMOS_0.05/diffusion_fp/fold1/ckpts/best_model.pth` |
| seed1 (HPC) | `/gpfs/work/aac/zimuzhang2302/AD_project/logs/diffusion_fp_amos_5p_seed1/ckpts/best_model.pth` |
| seed666 (HPC) | `/gpfs/work/aac/zimuzhang2302/AD_project/logs/diffusion_fp_amos_5p_seed666/ckpts/best_model.pth` |

HPC: `zimuzhang2302@login.hpc.xjtlu.edu.cn`
