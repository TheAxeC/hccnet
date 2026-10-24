#!/bin/bash
#SBATCH --job-name=hccnet-phase-preflight
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40
#SBATCH --exclude=ctit086,ctit088
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
RUN="${1:?Run directory}"
PROJECT_DIR="${HCCNET_PROJECT_DIR:-$HOME/projects/hccnet}"
source "$PROJECT_DIR/radiologist_code/hpc/radiologist_env.sh"
cp -r "$PROJECT_DIR/radiologist_code" "$WORK/code"
cd "$WORK/code"
python3.10 -u phase_preflight.py --arch pico --stem-mode original --pe-mode linear
cp "$HCCNET_RESULTS_DIR/preflight.json" "$RUN/"
