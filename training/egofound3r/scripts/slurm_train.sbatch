#!/usr/bin/env bash
# Usage, from the repository root:
#   sbatch -p <partition> scripts/slurm_train.sbatch baseline [--set section.key=value ...]
# Change nodes and GPUs with sbatch flags, e.g. --nodes=2 --gpus-per-node=4. The batch stays at the
# paper's effective 64: gradient accumulation fills any gap. Requeued jobs resume from latest.pt.
#SBATCH --job-name=egotac
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --output=egotac-%j.log
#SBATCH --requeue
set -euo pipefail
. "$SLURM_SUBMIT_DIR/scripts/env.sh"
VARIANT=${1:?baseline or world}; shift
export EGOFOUND3R_ROOT=$(tree_for_variant "$VARIANT")
GPUS=$("$PY" -c "import torch; print(torch.cuda.device_count())")
MASTER=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
bash scripts/download_weights.sh
"$PY" scripts/preflight.py --config configs/full.toml "$@"
ARGS=(--config configs/full.toml --set "train.variant=\"$VARIANT\"" --set "train.output_dir=\"$EGOTAC_ROOT/outputs/full_$VARIANT\"" "$@")
srun --export=ALL bash -c '. "$SLURM_SUBMIT_DIR/scripts/env.sh"; exec "$PY" -m torch.distributed.run \
  --nnodes="$SLURM_NNODES" --node_rank="$SLURM_NODEID" --nproc_per_node='"$GPUS"' \
  --master_addr='"$MASTER"' --master_port="${MASTER_PORT:-29500}" -m egotac.train "$@"' _ "${ARGS[@]}"
