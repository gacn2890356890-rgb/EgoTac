# Shared by every script. Copy cluster.env.example to cluster.env to change any of these.
REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
[ -f "$REPO_DIR/cluster.env" ] && . "$REPO_DIR/cluster.env"
export EGOTAC_ROOT=${EGOTAC_ROOT:-$REPO_DIR/workspace}
export EGOTAC_VENV=${EGOTAC_VENV:-$REPO_DIR/.venv}
export EGOFOUND3R_ROOT=${EGOFOUND3R_ROOT:-$REPO_DIR/EgoFound3R_original}
export MANO_DIR=${MANO_DIR:-$EGOTAC_ROOT/mano}
export DATA_OAKINK2=${DATA_OAKINK2:-$EGOTAC_ROOT/data/oakink2}
export DATA_TACO=${DATA_TACO:-$EGOTAC_ROOT/data/taco}
export DATA_HOI4D=${DATA_HOI4D:-$EGOTAC_ROOT/data/hoi4d}
export DATA_H2O=${DATA_H2O:-$EGOTAC_ROOT/data/h2o}
export DATA_H2O_OBJECTS=${DATA_H2O_OBJECTS:-$DATA_H2O/object}
export DATA_HOT3D=${DATA_HOT3D:-$EGOTAC_ROOT/data/hot3d_aria}
export DATA_ARCTIC=${DATA_ARCTIC:-$EGOTAC_ROOT/data/arctic}
export PYTHONPATH=$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
PY=${EGOTAC_PYTHON:-$EGOTAC_VENV/bin/python}
cd "$REPO_DIR"

# baseline runs the untouched official tree, world runs the modified one.
tree_for_variant() {
  case "$1" in
    baseline) echo "$REPO_DIR/EgoFound3R_original" ;;
    world) echo "$REPO_DIR/EgoFound3R_world" ;;
    *) echo "error: variant must be baseline or world, got $1" >&2; return 1 ;;
  esac
}
