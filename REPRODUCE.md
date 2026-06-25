# Reproduction Guide — DiffVNet (GenSSL)

## Environment

```bash
conda activate dhc   # Python 3.x, PyTorch, SimpleITK, tqdm, nibabel
```

Code root: `~/Desktop/A&D复现/code/`  
HPC code root: `/gpfs/work/aac/zimuzhang2302/AD_project/code/`

---

## 1. Synapse 20% — Training

### seed0 (local)
```bash
cd ~/Desktop/A&D复现
python code/train_diffusion.py \
  -t synapse \
  --exp diffusion_fp \
  --seed 0 \
  -sl labeled_20p \
  -su unlabeled_20p \
  --base_lr 0.01 \
  -w 10 \
  --lambda_cdba 0.05 \
  --lambda_sac 0.2 \
  --lambda_var 0.2 \
  --vapl_warmup 500 \
  --variation_warmup 3000
```

### seed1 / seed666 (HPC SLURM)
```bash
# On HPC:
cd /gpfs/work/aac/zimuzhang2302/AD_project
sbatch slurm/train_synapse_20p_seed1.slurm
sbatch slurm/train_synapse_20p_seed666.slurm
```

Key SLURM flags: `--qos=4a800 --gres=gpu:a800:1 --time=2-00:00:00`

---

## 2. Synapse 20% — Testing & Evaluation

### seed0 (local)
```bash
cd ~/Desktop/A&D复现
python code/test.py -t synapse \
  --exp Exp_IBSSL_Synapse_0.2/diffusion_fp \
  --speed 0

python code/evaluate_metrics.py -t synapse \
  --exp Exp_IBSSL_Synapse_0.2/diffusion_fp \
  --folds 1
```

### seed1 / seed666 (HPC)
```bash
sbatch slurm/test_syn_s1f.slurm   # seed1
sbatch slurm/test_syn_s666.slurm  # seed666
# evaluate_metrics.py runs automatically inside the SLURM script
```

---

## 3. AMOS 5% — Training

### seed0 (local)
```bash
cd ~/Desktop/A&D复现
python code/train_diffusion.py \
  -t amos_0.05 \
  --exp diffusion_fp \
  --seed 0 \
  -sl labeled_5p \
  -su unlabeled_5p \
  --base_lr 0.01 \
  -w 10 \
  --lambda_cdba 0.05 \
  --lambda_sac 0.2 \
  --lambda_var 0.2 \
  --vapl_warmup 500 \
  --variation_warmup 3000
```

### seed1 / seed666 (HPC SLURM)
```bash
sbatch slurm/train_amos5p_seed1.slurm
sbatch slurm/train_amos5p_seed666.slurm
```

---

## 4. AMOS 5% — Testing & Evaluation

### seed0 (local)
```bash
python code/test.py -t amos_0.05 \
  --exp Exp_IBSSL_AMOS_0.05/diffusion_fp \
  --speed 0

python code/evaluate_metrics.py -t amos_0.05 \
  --exp Exp_IBSSL_AMOS_0.05/diffusion_fp \
  --folds 1
```

### seed1 / seed666 (HPC)
```bash
# IMPORTANT: SLURM time limit must be >= 12:00:00 (120 samples × ~3.3min/sample ≈ 6.6h + eval)
sbatch slurm/test_amos5p_fp_seed1.slurm
sbatch slurm/test_amos5p_fp_seed666.slurm
```

> **Critical:** `evaluate_metrics.py --exp` must NOT include `/fold1` suffix.
> The script internally appends `fold{N}/predictions/`.

---

## 5. Resume Interrupted Test Jobs

If a test job times out mid-way, add skip logic to `code/utils/__init__.py`
in `test_all_case()` before re-submitting:

```python
def test_all_case(task, net, ids_list, num_classes, patch_size, stride_xy, stride_z, test_save_path=None):
    for data_id in tqdm(ids_list):
        if os.path.exists(f'{test_save_path}/{data_id}.nii.gz'):  # <-- add this
            continue                                               # <-- and this
        image, _ = read_data(data_id, task=task, normalize=True)
        ...
```

---

## 6. Data Split Files

Located at `data/amos_splits/` and `data/synapse_splits/`:

| File | Description |
|------|-------------|
| `labeled_5p.txt` | 10 labeled samples for AMOS 5% (fixed, same for all seeds) |
| `unlabeled_5p.txt` | 206 unlabeled samples for AMOS 5% |
| `eval.txt` | 24 evaluation samples |
| `test.txt` | 120 test samples |
| `labeled_20p.txt` | Synapse 20% labeled split |

---

## 7. Expected Runtime

| Task | Per-sample inference | Total (120 test) |
|------|---------------------|-----------------|
| Synapse 20% | ~90s | ~3h |
| AMOS 5% | ~200s | ~6.6h |

Set `--time=12:00:00` in SLURM for all AMOS test jobs.

---

## 8. Pre-trained Model Checkpoints

All 6 checkpoints are available in the [GitHub Release v1.0-checkpoints](https://github.com/linvii77/bibm_GenSSL_ours/releases/tag/v1.0-checkpoints).

### Download

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

### Load for Inference

```python
import torch
from DiffVNet.diff_vnet import DiffVNet

net = DiffVNet(n_channels=1, n_classes=16, n_filters=32)  # AMOS: n_classes=16; Synapse: n_classes=14
ckpt = torch.load("amos_5p_seed0_best_model.pth", map_location="cpu")
net.load_state_dict(ckpt)
net.eval()
```

Then run `test.py` and `evaluate_metrics.py` as described in sections 2 and 4 above.
