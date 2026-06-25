#!/bin/bash
#SBATCH --job-name=gssl_syn_20p
#SBATCH --partition=gpua800
#SBATCH --qos=4a800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/work/aac/yuangluo22/GenericSSL-main/slurm_logs/%x_%j.out
#SBATCH --error=/gpfs/work/aac/yuangluo22/GenericSSL-main/slurm_logs/%x_%j.err

set -euo pipefail

echo "===== Job start: $(date) on $(hostname) ====="
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"

ROOT=/gpfs/work/aac/yuangluo22/GenericSSL-main
cd "$ROOT"
mkdir -p slurm_logs logs

source slurm/hpc_env.sh

# Synapse IBSSL 20% labeled, 3-fold CV, GenericSSL paper hyperparams
bash train.sh -c 0 -e diffusion -t synapse_0.2 -i '' -l 1e-2 -w 10 -n 300 -d true

echo "===== Job end: $(date) ====="
