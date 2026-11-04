#!/usr/bin/env bash
# Slurm wrapper, single node. Not tested on a real cluster yet: adjust partition and resources.
# Usage: sbatch scripts/slurm_train.sbatch baseline
#SBATCH --job-name=egotac
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --output=egotac-%j.log
#SBATCH --requeue
set -euo pipefail
srun bash scripts/train_full.sh "${1:?baseline or diffusion}" "${SLURM_GPUS_ON_NODE:-8}"
