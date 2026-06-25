#!/bin/bash
#SBATCH --job-name=ad_syn_s1
#SBATCH --partition=gpua800
#SBATCH --qos=4a800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/work/aac/zimuzhang2302/AD_project/logs/ad_synapse_20p_seed1_%j.out
#SBATCH --error=/gpfs/work/aac/zimuzhang2302/AD_project/logs/ad_synapse_20p_seed1_%j.err

set -euo pipefail

echo "===== A&D Synapse 20% seed=1 start: $(date) on $(hostname) ====="
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"

ROOT=/gpfs/work/aac/zimuzhang2302/AD_project
cd "$ROOT"
mkdir -p logs

module load miniconda3/22.11.1-gcc-8.5.0-l4fo6ta
source activate dhc

python code/train_diffusion.py \
    -t synapse \
    --exp diffusion_fp_synapse_20p_seed1 \
    -sl labeled_20p \
    -su unlabeled_80p \
    --seed 1 \
    --max_epoch 300 \
    --lambda_cdba 0.05 \
    --lambda_sac 0.2 \
    --lambda_var 0.2 \
    --vapl_warmup 500

echo "===== A&D Synapse 20% seed=1 end: $(date) ====="
