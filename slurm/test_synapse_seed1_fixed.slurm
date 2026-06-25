#!/bin/bash
#SBATCH --job-name=test_syn_s1f
#SBATCH --partition=gpua800
#SBATCH --qos=4a800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/work/aac/zimuzhang2302/AD_project/logs/test_syn_s1f_%j.out
#SBATCH --error=/gpfs/work/aac/zimuzhang2302/AD_project/logs/test_syn_s1f_%j.err

set -euo pipefail
ROOT=/gpfs/work/aac/zimuzhang2302/AD_project
cd "$ROOT"
module load miniconda3/22.11.1-gcc-8.5.0-l4fo6ta
source activate dhc

echo "===== test synapse seed1 fixed start: $(date) ====="

python code/test.py -t synapse_0.2 --exp diffusion_fp_synapse_20p_seed1_fixed --speed 0

mkdir -p logs/diffusion_fp_synapse_20p_seed1_fixed/fold1
ln -sfn "$ROOT/logs/diffusion_fp_synapse_20p_seed1_fixed/predictions" logs/diffusion_fp_synapse_20p_seed1_fixed/fold1/predictions
python code/evaluate_metrics.py -t synapse_0.2 --exp diffusion_fp_synapse_20p_seed1_fixed --folds 1

echo "===== test synapse seed1 fixed end: $(date) ====="
