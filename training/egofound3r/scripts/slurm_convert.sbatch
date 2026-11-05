#!/usr/bin/env bash
# Usage, from the repository root: sbatch -p <partition> scripts/slurm_convert.sbatch [workers]
# Reads the DATA_* paths in cluster.env. No internet needed.
#SBATCH --job-name=egotac-convert
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=egotac-convert-%j.log
#SBATCH --requeue
set -euo pipefail
bash "$SLURM_SUBMIT_DIR/scripts/convert_all.sh" "${1:-6}"
