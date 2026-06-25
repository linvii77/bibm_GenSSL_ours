#!/bin/bash
#SBATCH --job-name=test_a5p_s1
#SBATCH --partition=gpua800
#SBATCH --qos=4a800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs/work/aac/zimuzhang2302/AD_project/logs/test_amos5p_fp_seed1_%j.out
#SBATCH --error=/gpfs/work/aac/zimuzhang2302/AD_project/logs/test_amos5p_fp_seed1_%j.err

set -euo pipefail
ROOT=/gpfs/work/aac/zimuzhang2302/AD_project
cd "$ROOT"
module load miniconda3/22.11.1-gcc-8.5.0-l4fo6ta
source activate dhc

echo "===== test AMOS 5% seed1 start: $(date) on $(hostname) ====="

python code/test.py -t amos_0.05 --exp diffusion_fp_amos_5p_seed1 --speed 0

mkdir -p logs/diffusion_fp_amos_5p_seed1/fold1
ln -sfn "$ROOT/logs/diffusion_fp_amos_5p_seed1/predictions" logs/diffusion_fp_amos_5p_seed1/fold1/predictions
python code/evaluate_metrics.py -t amos_0.05 --exp diffusion_fp_amos_5p_seed1 --folds 1

echo "===== test AMOS 5% seed1 end: $(date) ====="
