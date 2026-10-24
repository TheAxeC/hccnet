#!/bin/bash
# From the cluster project root: bash radiologist_code/hpc/run_phase_impact.sh
set -euo pipefail
PROJECT_DIR="${HCCNET_PROJECT_DIR:-$HOME/projects/hccnet}"
DATA_ROOT="${HCCNET_DATA_DIR:-/deepstore/datasets/bms/hcc_study}"
ENV_ARCHIVE="${HCCNET_ENV_ARCHIVE:-$HOME/hccnet-env.tar.gz}"
[ -f "$ENV_ARCHIVE" ] || { echo "Missing HCCNET_ENV_ARCHIVE: $ENV_ARCHIVE; build the packed environment described in code/README.md."; exit 1; }
for LABEL in labels lirads; do
  [ -f "$DATA_ROOT/labels/$LABEL.csv" ] || { echo "Missing raw cohort labels; restore institutional group access or set HCCNET_DATA_DIR."; exit 1; }
done
[ -d "$DATA_ROOT/nifti" ] || { echo 'Missing raw nifti/ directory.'; exit 1; }
cd "$PROJECT_DIR"
RUN="$PROJECT_DIR/results/phase_impact/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN" logs
PREFLIGHT=$(sbatch --parsable radiologist_code/hpc/phase_preflight.slurm "$RUN")
DINO=$(sbatch --parsable --dependency="afterok:$PREFLIGHT" radiologist_code/hpc/phase_impact.slurm dino "$RUN")
DECODER=$(sbatch --parsable --time=16:00:00 --dependency="aftercorr:$DINO" radiologist_code/hpc/phase_impact.slurm decoder "$RUN")
FINETUNE=$(sbatch --parsable --time=20:00:00 --dependency="aftercorr:$DECODER" radiologist_code/hpc/phase_impact.slurm finetune "$RUN")
SUMMARY=$(sbatch --parsable --dependency="afterok:$FINETUNE" radiologist_code/hpc/phase_summary.slurm "$RUN")
printf 'run=%s\npreflight=%s\ndino=%s\ndecoder=%s\nfinetune=%s\nsummary=%s\n' "$RUN" "$PREFLIGHT" "$DINO" "$DECODER" "$FINETUNE" "$SUMMARY" | tee "$RUN/jobs.txt"
