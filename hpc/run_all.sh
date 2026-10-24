#!/bin/bash
# One-command full paper reproduction. Submits every stage as an array job with
# SLURM dependencies (a linear chain - one array active at a time), then EXITS in
# seconds; SLURM runs the chain over the next few days. You can log out. Watch with
# `squeue -u $USER`; logs in logs/. If a stage fails, its afterok-dependents stay
# pending and are cancelled by SLURM (they will not run on bad inputs).
#
# Concurrency:
#   bash hpc/run_all.sh        # no cap -> uses up to your account's job limit (8)
#   bash hpc/run_all.sh 6      # cap every stage to 6 concurrent tasks (leaves 2 free)
# (A bare `sbatch hpc/repro_<stage>.slurm` likewise runs uncapped = up to 8.)
#
# Matrix: dino + decoder = 5 groups x 5 archs (25 each; the 2 constituents get their
# OWN standalone backbone, pre-trained only on their own 2 channels); finetune +
# baseline = 5 modalities x 5 archs (25 each); ablation = 6 (pico, CNN-only +
# encoder-only, self-tests); fusion = 3 self-contained chains (pico); figures = 1
# (offline, rebuilds the 2 composite figures from the predictions - no retrain).
set -e
cd "$(dirname "$0")/.."          # -> repo root (code/ ; $HOME/projects/hccnet on the cluster)
mkdir -p logs

MAXC="${1:-}"                    # optional max concurrent tasks per stage; empty = uncapped (account cap, 8)
SUF=""; [ -n "$MAXC" ] && SUF="%$MAXC"
[ -n "$MAXC" ] && echo "capping each stage to $MAXC concurrent tasks" || echo "no cap (uses up to your 8-job account limit)"

# Linear chain (each waits for the previous to finish OK). Independent stages
# (baseline, fusion) are serialized too so the cap, when set, always holds.
DINO=$(sbatch --parsable --array=0-24$SUF hpc/repro_dino.slurm);                          echo "1 dino     = $DINO  (25)"
DEC=$(sbatch  --parsable --array=0-24$SUF --dependency=afterok:$DINO hpc/repro_decoder.slurm); echo "2 decoder  = $DEC   (25)"
FT=$(sbatch   --parsable --array=0-24$SUF --dependency=afterok:$DEC  hpc/repro_finetune.slurm); echo "3 finetune = $FT    (25)"
ABL=$(sbatch  --parsable --array=0-5$SUF  --dependency=afterok:$FT   hpc/repro_ablation.slurm); echo "4 ablation = $ABL   (6, pico, self-tests)"
BASE=$(sbatch --parsable --array=0-24$SUF --dependency=afterok:$ABL  hpc/repro_baseline.slurm); echo "5 baseline = $BASE  (25)"
FUSE=$(sbatch --parsable --array=0-2$SUF  --dependency=afterok:$BASE hpc/repro_fusion.slurm);   echo "6 fusion   = $FUSE  (3, self-contained)"
FIG=$(sbatch  --parsable                  --dependency=afterok:$FUSE hpc/repro_figures.slurm);  echo "7 figures  = $FIG   (composite calibration + cumulative-gain, offline from predictions)"
# NO separate test stage: finetune/baseline/ablation/fusion now ensemble-test IN their own
# job (DISK-LEAN - the 50 fold-seed checkpoints live+die on /local; only the metric + plots +
# ONE representative fold-0 model persist to home). All stages run the reproducing config
# (linear PE + multi-channel stem) via $REPRO_ARCH in _repro_lib.sh. Stage 7 (figures) rebuilds
# the two composite figures offline from the committed predictions - no retrain, no GPU compute.

echo
echo "submitted the full reproduction (reproducing config: linear PE + multi-channel stem, disk-lean)."
echo "metrics -> results/model_diagnostics/ ; figures -> results/figures/ ; representative weights -> results/model_weights/ ; watch: squeue -u \$USER ; you can log out."
