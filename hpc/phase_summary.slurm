#!/bin/bash
#SBATCH --job-name=hccnet-phase-summary
#SBATCH --partition=main-cpu
#SBATCH --exclude=ctit088
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
RUN="${1:?Run output directory}"
PROJECT_DIR="${HCCNET_PROJECT_DIR:-$HOME/projects/hccnet}"
source "$PROJECT_DIR/radiologist_code/hpc/radiologist_env.sh"
mkdir -p "$WORK/run"
for ARM in base arterial venous late all; do
  mkdir -p "$WORK/run/$ARM/model_diagnostics"
  cp -r "$RUN/$ARM/cohort_audit" "$WORK/run/$ARM/"
  cp -r "$RUN/$ARM/model_diagnostics/predictions" "$WORK/run/$ARM/model_diagnostics/"
done
python3.10 -u "$PROJECT_DIR/radiologist_code/phase_impact.py" "$WORK/run"
cp "$WORK/run/"*.json "$WORK/run/"*.png "$RUN/"
