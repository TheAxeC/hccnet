# Source inside a SLURM allocation. Reuse the packed environment on each node.
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through sbatch}"
ENV_ARCHIVE="${HCCNET_ENV_ARCHIVE:-$HOME/hccnet-env.tar.gz}"
[ -f "$ENV_ARCHIVE" ] || { echo "Missing packed environment: $ENV_ARCHIVE. Build with the commands in code/README.md."; exit 1; }
WORK="/local/$SLURM_JOB_ID"
mkdir -p "$WORK"
trap 'rm -rf "$WORK"' EXIT
export TMPDIR="$WORK" MPLCONFIGDIR="$WORK/mpl" OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
echo "Node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
ENV_CACHE="/local/$USER/hccnet-radiologist-env"
mkdir -p "$ENV_CACHE"
(
  flock 9
  if [ ! -f "$ENV_CACHE/.ready" ]; then
    tar -xzf "$ENV_ARCHIVE" -C "$ENV_CACHE"
    PATH="$ENV_CACHE/bin:$PATH" "$ENV_CACHE/bin/conda-unpack"
    touch "$ENV_CACHE/.ready"
  fi
) 9>"$ENV_CACHE/.lock"
export PATH="$ENV_CACHE/bin:$PATH" SETUPTOOLS_USE_DISTUTILS=stdlib
python3.10 -c 'import sys; assert sys.version_info[:2] == (3,10), sys.version'
export HCCNET_CACHE_DIR="$WORK/cache" HCCNET_RESULTS_DIR="$WORK/results"
mkdir -p "$HCCNET_RESULTS_DIR/model_weights" "$HCCNET_CACHE_DIR"
