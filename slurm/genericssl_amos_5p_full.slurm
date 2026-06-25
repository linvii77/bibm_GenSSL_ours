#!/bin/bash
#SBATCH --job-name=gssl_amos5p
#SBATCH --partition=gpua800
#SBATCH --qos=4a800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=/gpfs/work/aac/yuangluo22/GenericSSL-main/slurm_logs/%x_%j.out
#SBATCH --error=/gpfs/work/aac/yuangluo22/GenericSSL-main/slurm_logs/%x_%j.err

set -euo pipefail

echo "===== AMOS 5% formal experiment start: $(date) on $(hostname) ====="
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"

ROOT=/gpfs/work/aac/yuangluo22/GenericSSL-main
cd "$ROOT"
mkdir -p slurm_logs logs

source slurm/hpc_env.sh

# AMOS 5% labeled, 1-fold, 200 epochs, GenericSSL paper hyperparams
bash train.sh -c 0 -e diffusion -t amos_5p -i '' -l 1e-2 -w 10 -n 200 -d true

echo "===== AMOS 5% formal experiment end: $(date) ====="
