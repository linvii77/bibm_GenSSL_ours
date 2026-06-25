#!/bin/bash
#SBATCH --job-name=ad_amos_smoke
#SBATCH --partition=gpua800
#SBATCH --qos=4a800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs/work/aac/zimuzhang2302/AD_project/logs/ad_amos_smoke_%j.out
#SBATCH --error=/gpfs/work/aac/zimuzhang2302/AD_project/logs/ad_amos_smoke_%j.err

set -euo pipefail

echo "===== A&D AMOS smoke start: $(date) on $(hostname) ====="
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"

ROOT=/gpfs/work/aac/zimuzhang2302/AD_project
cd "$ROOT"
mkdir -p logs

module load miniconda3/22.11.1-gcc-8.5.0-l4fo6ta
source activate dhc

python code/train_diffusion.py \
    -t amos \
    --exp smoke_amos_fp \
    -sl labeled_5p \
    -su unlabeled_5p \
    --max_epoch 1 \
    --lambda_cdba 0.05 \
    --lambda_sac 0.2 \
    --lambda_var 0.2 \
    --vapl_warmup 0

echo "===== A&D AMOS smoke end: $(date) ====="
