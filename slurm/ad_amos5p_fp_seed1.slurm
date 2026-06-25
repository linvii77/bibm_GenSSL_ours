#!/bin/bash
#SBATCH --job-name=amos5p_s1
#SBATCH --partition=gpua800
#SBATCH --qos=4a800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/work/aac/zimuzhang2302/AD_project/logs/ad_amos5p_fp_seed1_%j.out
#SBATCH --error=/gpfs/work/aac/zimuzhang2302/AD_project/logs/ad_amos5p_fp_seed1_%j.err

set -euo pipefail
ROOT=/gpfs/work/aac/zimuzhang2302/AD_project
cd "$ROOT"
mkdir -p logs
module load miniconda3/22.11.1-gcc-8.5.0-l4fo6ta
source activate dhc

echo "===== AMOS 5% fp seed=1 start: $(date) on $(hostname) ====="

python code/train_diffusion.py     -t amos_0.05     --exp diffusion_fp_amos_5p_seed1     -sl labeled_5p     -su unlabeled_5p     -se eval     --seed 1     --max_epoch 300     --base_lr 0.01     -w 10     --lambda_cdba 0.05     --lambda_sac 0.2     --lambda_var 0.2     --vapl_warmup 500     --variation_warmup 3000

echo "===== AMOS 5% fp seed=1 end: $(date) ====="
