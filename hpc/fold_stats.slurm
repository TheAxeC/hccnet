#!/bin/bash
#SBATCH --job-name=hcc_fold_stats
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40
#SBATCH --exclude=ctit086
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:08:00
#SBATCH --output=logs/%x_%j.out
cd $HOME/projects/hccnet
source hpc/_repro_lib.sh
stage_local
python3 fold_stats.py
