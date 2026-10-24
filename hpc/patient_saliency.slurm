#!/bin/bash
#SBATCH --job-name=hccnet-patient-saliency
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40
#SBATCH --exclude=ctit086,ctit088
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
PROJECT_DIR="${HCCNET_PROJECT_DIR:-$HOME/projects/hccnet}"
source "$PROJECT_DIR/radiologist_code/hpc/radiologist_env.sh"
cp -r "$PROJECT_DIR/radiologist_code" "$WORK/code"
cd "$WORK/code"
SAL_GROUPS=(dwi t1dce t1iop t1iop_only t2_only)
M="${SAL_GROUPS[$SLURM_ARRAY_TASK_ID]}"
case "$M" in
 dwi) MODS=(DWI_b0 DWI_b150 DWI_b400 DWI_b800);;
 t1dce) MODS=(T1WI T1A T1V T1D);;
 t1iop) MODS=(T1W_IP T1W_OOP T2W_TES T2W_TEL);;
 t1iop_only) MODS=(T1W_IP T1W_OOP);;
 t2_only) MODS=(T2W_TES T2W_TEL);;
esac
STEM="${M}_pico_paper_ft"
cp "$PROJECT_DIR/results/model_weights/weights_fold0_${STEM}.pth" "$HCCNET_RESULTS_DIR/model_weights/"
mkdir -p "$WORK/predictions"
cp "$PROJECT_DIR/radiologist_predictions/${STEM}.npz" "$WORK/predictions/"
export HCCNET_PREDICTIONS="$WORK/predictions"
python3.10 -u patient_saliency.py --stem-mode original --pe-mode linear --arch pico --mod-list "${MODS[@]}" --suffix "$STEM"
mkdir -p "$PROJECT_DIR/results/patient_figures/saliency"
cp -r "$HCCNET_RESULTS_DIR/patient_saliency/." "$PROJECT_DIR/results/patient_figures/saliency/"
