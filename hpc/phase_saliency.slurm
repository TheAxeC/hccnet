#!/bin/bash
#SBATCH --job-name=hccnet-phase-saliency
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=ctit086,ctit088
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
PROJECT_DIR="${HCCNET_PROJECT_DIR:-$HOME/projects/hccnet}"
RUN="${1:?Pass the completed phase run directory}"
if [ -z "${SLURM_JOB_GPUS:-}" ]; then export CUDA_VISIBLE_DEVICES=""; fi
source "$PROJECT_DIR/radiologist_code/hpc/radiologist_env.sh"
cp -r "$PROJECT_DIR/radiologist_code" "$WORK/code"
cd "$WORK/code"
ARMS=(base arterial venous late all)
ARM="${ARMS[$SLURM_ARRAY_TASK_ID]}"
MODS=(T1WI)
case "$ARM" in
 arterial) MODS+=(T1A);;
 venous) MODS+=(T1V);;
 late) MODS+=(T1D);;
 all) MODS+=(T1A T1V T1D);;
esac
STEM="phase_${ARM}_pico_ft"
cp "$RUN/$ARM/model_weights/weights_fold0_${STEM}.pth" "$HCCNET_RESULTS_DIR/model_weights/"
mkdir -p "$WORK/predictions"
cp "$RUN/$ARM/model_diagnostics/predictions/${STEM}.npz" "$WORK/predictions/"
export HCCNET_PREDICTIONS="$WORK/predictions"
export HCCNET_DISPLAY_SCAN_DATES=1
export HCCNET_REVIEW_PATIENTS=PT_009,PT_034,PT_035,PT_042,PT_100,PT_200,PT_214,PT_254,PT_303,PT_315,PT_318
python3.10 -u patient_saliency.py --stem-mode original --pe-mode linear --arch pico --mod-list "${MODS[@]}" --required-modalities T1WI T1A T1V T1D --suffix "$STEM"
mkdir -p "$RUN/saliency"
cp -r "$HCCNET_RESULTS_DIR/patient_saliency/." "$RUN/saliency/"
