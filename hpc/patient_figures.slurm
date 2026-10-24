#!/bin/bash
#SBATCH --job-name=hccnet-patient-figures
#SBATCH --partition=main-cpu
#SBATCH --exclude=ctit088
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
PROJECT_DIR="${HCCNET_PROJECT_DIR:-$HOME/projects/hccnet}"
ENV_ARCHIVE="${HCCNET_ENV_ARCHIVE:-$HOME/hccnet-env.tar.gz}"
DATA_ROOT="${HCCNET_DATA_DIR:-/deepstore/datasets/bms/hcc_study}"
[ -f "$ENV_ARCHIVE" ] || { echo "Missing HCCNET_ENV_ARCHIVE: $ENV_ARCHIVE"; exit 1; }
[ -f "$DATA_ROOT/labels/labels.csv" ] || { echo 'Missing institutional data; restore cohort group access or set HCCNET_DATA_DIR.'; exit 1; }
WORK="/local/$SLURM_JOB_ID"
mkdir -p "$WORK/env" "$WORK/out" "$WORK/mpl"
trap 'rm -rf "$WORK"' EXIT
export TMPDIR="$WORK" MPLCONFIGDIR="$WORK/mpl" OMP_NUM_THREADS=2
# The packed HCC environment includes its matching Python interpreter.
tar -xzf "$ENV_ARCHIVE" -C "$WORK/env"
export PATH="$WORK/env/bin:$PATH" SETUPTOOLS_USE_DISTUTILS=stdlib
"$WORK/env/bin/conda-unpack"
cp "$PROJECT_DIR/patient_figures.py" "$WORK/"
cp -r "$PROJECT_DIR/radiologist_predictions" "$WORK/predictions"
# Read only the eight patients' slices from deepstore; no full-cohort staging.
python3.10 -u "$WORK/patient_figures.py" --data "$DATA_ROOT" --predictions "$WORK/predictions" --out "$WORK/out"
mkdir -p "$PROJECT_DIR/results/patient_figures"
cp -r "$WORK/out/." "$PROJECT_DIR/results/patient_figures/"
echo 'Patient figures complete'
