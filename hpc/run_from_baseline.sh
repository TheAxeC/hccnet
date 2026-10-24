#!/bin/bash
# Submits the chain from baseline onward (baseline -> fusion -> test), assuming finetune
# and ablation are done. Used to resume after ablation tasks were re-run: fire it from a
# helper gated on afterok of the ablation re-run. cap arg optional.
set -e
cd "$(dirname "$0")/.."          # -> repo root ($HOME/projects/hccnet on the cluster)
mkdir -p logs

MAXC="${1:-}"                    # optional max concurrent tasks per stage; empty = uncapped
SUF=""; [ -n "$MAXC" ] && SUF="%$MAXC"
[ -n "$MAXC" ] && echo "capping each stage to $MAXC concurrent tasks" || echo "no cap (account limit, 8)"

BASE=$(sbatch --parsable --array=0-24$SUF hpc/repro_baseline.slurm);                        echo "5 baseline = $BASE  (25)"
FUSE=$(sbatch --parsable --array=0-2$SUF  --dependency=afterok:$BASE hpc/repro_fusion.slurm);   echo "6 fusion   = $FUSE  (3, self-contained)"
TEST=$(sbatch --parsable --dependency=afterok:$FUSE --job-name=hcc_test_submit --partition=main-gpu \
  --gres=gpu:ampere:1 --constraint=a40 --cpus-per-task=1 --mem=2G --time=00:05:00 \
  --output=logs/test_submit_%j.out \
  --wrap="cd \$HOME/projects/hccnet && sbatch --array=0-24$SUF hpc/repro_test.slurm")
echo "7 test     = $TEST  (helper: fires after fusion, then submits the 25 test tasks)"
echo
echo "baseline-onward chain submitted."
