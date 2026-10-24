#!/bin/bash
#SBATCH --job-name=hccnet-contrast-cam
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40
#SBATCH --exclude=ctit086,ctit088
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
PROJECT_DIR="${HCCNET_PROJECT_DIR:-$HOME/projects/hccnet}"
source "$PROJECT_DIR/radiologist_code/hpc/radiologist_env.sh"
cp -r "$PROJECT_DIR/radiologist_code" "$WORK/code"
cd "$WORK/code"
cp "$PROJECT_DIR/results/model_weights/weights_fold0_t1dce_pico_paper_ft.pth" "$HCCNET_RESULTS_DIR/model_weights/"
export HCCNET_CONTRAST_CAM=1
python3.10 -m torch.distributed.run --standalone --nproc_per_node=1 saliency_localize.py --distributed --stem-mode original --pe-mode linear --arch pico --mod-list T1WI T1A T1V T1D --suffix t1dce_pico_paper_ft
mkdir -p "$PROJECT_DIR/results/patient_figures/contrast_saliency"
cp "$HCCNET_RESULTS_DIR/model_diagnostics/localize/t1dce_pico_paper_ft/"*_contrast_cam.* "$PROJECT_DIR/results/patient_figures/contrast_saliency/"
