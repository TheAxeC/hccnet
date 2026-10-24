#!/bin/bash
#SBATCH --job-name=hccnet-restart-test
#SBATCH --partition=main-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --exclude=ctit088
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
source "$HOME/projects/hccnet/radiologist_code/hpc/radiologist_env.sh"
python3.10 -u "$HOME/projects/hccnet/radiologist_code/tests/test_pretrain_restart.py"
