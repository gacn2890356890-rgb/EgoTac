#!/usr/bin/env bash
# One-click install for the Python pinned in .python-version: environment, pinned packages, generated config.
# Rerunning is safe. Works even when the system Python has no pip, ensurepip, or virtualenv.
# Usage: bash scripts/setup.sh
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"

BASE=${EGOTAC_BASE_PYTHON:-python3}

pinned=$(cat "$REPO_DIR/.python-version")
python_version() {
  "$1" -c 'import platform; print(platform.python_version())'
}
version=$(python_version "$BASE")
if [ "$version" != "$pinned" ]; then
  echo "error: $BASE is Python $version; this environment is pinned to $pinned. Set EGOTAC_BASE_PYTHON to a $pinned interpreter." >&2
  exit 1
fi

if [ -x "$EGOTAC_VENV/bin/python" ] && [ "$(python_version "$EGOTAC_VENV/bin/python")" != "$pinned" ]; then
  echo "error: $EGOTAC_VENV uses a different Python than $pinned. Delete it and rerun." >&2
  exit 1
fi
if [ ! -x "$EGOTAC_VENV/bin/python" ]; then
  if ! "$BASE" -m venv "$EGOTAC_VENV" 2>/dev/null; then
    # Debian and Ubuntu ship Python without ensurepip unless python3-venv is installed.
    rm -rf "$EGOTAC_VENV"
    "$BASE" -m venv --without-pip "$EGOTAC_VENV"
  fi
fi
PY=$EGOTAC_VENV/bin/python
if ! "$PY" -m pip --version >/dev/null 2>&1; then
  echo "pip is missing from the environment, bootstrapping it"
  bootstrap=$(mktemp -d)
  curl -fsSL https://bootstrap.pypa.io/get-pip.py -o "$bootstrap/get-pip.py"
  "$PY" "$bootstrap/get-pip.py" --quiet "pip==25.0.1"
  rm -rf "$bootstrap"
fi

# Wheels carry their own CUDA, so only the driver has to be new enough.
if [ -z "${TORCH_CUDA:-}" ]; then
  TORCH_CUDA=cu124
  if command -v nvidia-smi >/dev/null 2>&1; then
    driver_cuda=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9]*\.[0-9]*\).*/\1/p' | head -1)
    if [ -n "$driver_cuda" ] && "$PY" -c "import sys; sys.exit(0 if tuple(map(int, '$driver_cuda'.split('.'))) < (12, 4) else 1)"; then
      TORCH_CUDA=cu118
    fi
  fi
fi
echo "installing packages with PyTorch build $TORCH_CUDA"
# The official wheels bundle CUDA; the PyPI ones would pull several more gigabytes of NVIDIA packages.
pip_install() {
  # Large wheels over a slow link time out now and then; pip resumes from its cache on the next try.
  local attempt
  for attempt in 1 2 3 4 5; do
    "$PY" -m pip install --quiet --timeout 120 --retries 10 "$@" && return 0
    echo "pip attempt $attempt failed, retrying" >&2
    sleep 15
  done
  return 1
}
torch_version=$(sed -n 's/^torch==//p' requirements.txt)
vision_version=$(sed -n 's/^torchvision==//p' requirements.txt)
pip_install --index-url "https://download.pytorch.org/whl/$TORCH_CUDA" \
  --extra-index-url https://pypi.org/simple "torch==$torch_version+$TORCH_CUDA" "torchvision==$vision_version+$TORCH_CUDA"
pip_install -r requirements.txt
"$PY" -m pip check

mkdir -p "$EGOTAC_ROOT"/{checkpoints,configs,data,outputs}
sed "s#^vggt_checkpoint_path = .*#vggt_checkpoint_path = \"$EGOTAC_ROOT/checkpoints/vggt_omega_1b_512.pt\"#" \
  "$REPO_DIR/EgoFound3R_original/configs/default.toml" > "$EGOTAC_ROOT/configs/default.toml"

EGOFOUND3R_ROOT="$REPO_DIR/EgoFound3R_world" "$PY" scripts/check_env.py
bash scripts/verify_original.sh
echo
echo "Environment ready: $EGOTAC_VENV"
echo "Next: see README.md"
