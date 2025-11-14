#!/bin/bash
#SBATCH --job-name="sft_server_run"
#SBATCH --output="logs/sft_server_%j.out"
#SBATCH --partition=gpuA40x4
#SBATCH --mem=200G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --constraint="scratch"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --account=beaa-delta-gpu
#SBATCH --no-requeue
#SBATCH -t 48:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

source /u/tianyaosu/anaconda3/etc/profile.d/conda.sh
conda activate chatbot

MODEL_NAME="${MODEL_NAME:-gemma-3-1b-it}"
LOG_TAG="${LOG_TAG:-sft_${SLURM_JOB_ID:-local}}"
DRY_RUN="${DRY_RUN:-0}"

declare -a EXTRA_ARGS=()
if [[ "${DRY_RUN}" == "1" ]]; then
  EXTRA_ARGS+=("--dry-run")
fi

echo "[SLURM] MODEL_NAME=${MODEL_NAME}"
echo "[SLURM] LOG_TAG=${LOG_TAG}"
echo "[SLURM] DRY_RUN=${DRY_RUN}"

srun python -m training.sft_training_server \
  --model "${MODEL_NAME}" \
  --log-tag "${LOG_TAG}" \
  "${EXTRA_ARGS[@]}"