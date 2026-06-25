#!/bin/bash
#SBATCH --job-name=gssl_amos5p_smoke
#SBATCH --partition=gpua800
#SBATCH --qos=4a800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/work/aac/yuangluo22/GenericSSL-main/slurm_logs/%x_%j.out
#SBATCH --error=/gpfs/work/aac/yuangluo22/GenericSSL-main/slurm_logs/%x_%j.err

set -euo pipefail

echo "===== AMOS 5% smoke test start: $(date) on $(hostname) ====="
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"

ROOT=/gpfs/work/aac/yuangluo22/GenericSSL-main
cd "$ROOT"
mkdir -p slurm_logs logs

source slurm/hpc_env.sh

export PYTHONPATH="$(pwd)/code:${PYTHONPATH:-}"
python code/train_diffusion.py \
  --exp smoke/amos_5p \
  --task amos_5p \
  -sl labeled_5p \
  -su unlabeled_5p \
  -se eval \
  --seed 0 \
  -g 0 \
  --base_lr 1e-2 \
  -w 10 \
  -ep 1

echo "===== AMOS 5% smoke test passed: $(date) ====="
