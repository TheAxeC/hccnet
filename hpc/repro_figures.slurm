#!/bin/bash
#SBATCH --job-name=hcc_repro_figures
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=ctit086
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#
# Regenerate the two composite manuscript figures (calibration.png, cumulative_gain.png)
# from the committed per-model predictions - the offline, no-retrain figure step of the
# reproduction. CPU-only (matplotlib Agg); the GPU request is only to land on the same
# partition/env as the training stages. Reads the finetune + baseline prediction npz
# written by the earlier stages; writes both PNGs to results/figures/.
cd $HOME/projects/hccnet
source hpc/_repro_lib.sh
stage_local
export HCC_PRED_DIR="$HOME/projects/hccnet/results/model_diagnostics/predictions"
mkdir -p results/figures
echo "##### REPRO FIGURES: regenerating composite figures from $HCC_PRED_DIR #####"; date
python3 paper_figs.py --out results/figures
RC=$?; date; echo "paper_figs.py exit code: $RC"
[ "$RC" -eq 0 ] || { echo "ERROR: figure regeneration failed"; exit 1; }
ls -la results/figures/calibration.png results/figures/cumulative_gain.png
echo "##### DONE: figures -> results/figures/ (copy into manuscript/figures/ for the deposit) #####"
