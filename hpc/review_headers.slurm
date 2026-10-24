#!/bin/bash
#SBATCH --job-name=hccnet-review-headers
#SBATCH --partition=main-cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
python3 "$HOME/projects/hccnet/radiologist_code/audit_review_headers.py" > "$HOME/projects/hccnet/results/patient_figures/new_outcomes/acquisition_headers.json"
