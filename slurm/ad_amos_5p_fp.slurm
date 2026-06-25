#!/bin/bash
#SBATCH --job-name=ad_amos5p_fp
#SBATCH --partition=gpua800
#SBATCH --qos=4a800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/work/aac/zimuzhang2302/AD_project/logs/ad_amos5p_fp_%j.out
#SBATCH --error=/gpfs/work/aac/zimuzhang2302/AD_project/logs/ad_amos5p_fp_%j.err

set -euo pipefail

echo "===== A&D AMOS 5% FusedProxy start: $(date) on $(hostname) ====="
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"

ROOT=/gpfs/work/aac/zimuzhang2302/AD_project
cd "$ROOT"
mkdir -p logs

module load miniconda3/22.11.1-gcc-8.5.0-l4fo6ta
source activate dhc

python code/train_diffusion.py \
    -t amos \
    --exp diffusion_fp_amos_5p \
    -sl labeled_5p \
    -su unlabeled_5p \
    --max_epoch 300 \
    --lambda_cdba 0.05 \
    --lambda_sac 0.2 \
    --lambda_var 0.2 \
    --vapl_warmup 6000

echo "===== A&D AMOS 5% FusedProxy end: $(date) ====="
