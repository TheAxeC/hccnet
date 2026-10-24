#!/bin/bash
# Submits the post-DINO chain (decoder -> finetune -> ablation -> baseline -> fusion
# -> test), assuming the 15 DINO hand-offs already exist. This is the tail of
# run_all.sh with the dino stage removed, so the DINO submit-quota can be freed while
# DINO runs. Fire it from a tiny helper job gated on the DINO array:
#   sbatch --dependency=afterok:<DINO_JOBID> --wrap "cd ~/projects/hccnet && bash hpc/run_rest.sh 5"
# By the time DINO finishes the queue has drained, so the ~74 tail tasks fit under the
# 100-task submit cap. cap arg optional (empty = uncapped account limit of 8).
set -e
cd "$(dirname "$0")/.."          # -> repo root ($HOME/projects/hccnet on the cluster)
mkdir -p logs

MAXC="${1:-}"                    # optional max concurrent tasks per stage; empty = uncapped
SUF=""; [ -n "$MAXC" ] && SUF="%$MAXC"
[ -n "$MAXC" ] && echo "capping each stage to $MAXC concurrent tasks" || echo "no cap (account limit, 8)"

# decoder has no dependency: the helper's afterok:DINO already gated this whole script,
# so all 15 DINO hand-offs exist by now.
DEC=$(sbatch  --parsable --array=0-14$SUF hpc/repro_decoder.slurm);                         echo "2 decoder  = $DEC   (15)"
FT=$(sbatch   --parsable --array=0-24$SUF --dependency=afterok:$DEC  hpc/repro_finetune.slurm); echo "3 finetune = $FT    (25)"
ABL=$(sbatch  --parsable --array=0-5$SUF  --dependency=afterok:$FT   hpc/repro_ablation.slurm); echo "4 ablation = $ABL   (6)"
BASE=$(sbatch --parsable --array=0-24$SUF --dependency=afterok:$ABL  hpc/repro_baseline.slurm); echo "5 baseline = $BASE  (25)"
FUSE=$(sbatch --parsable --array=0-2$SUF  --dependency=afterok:$BASE hpc/repro_fusion.slurm);   echo "6 fusion   = $FUSE  (3, self-contained)"
# test submitted via a tiny helper after fusion, to stay under the 100-task submit cap.
TEST=$(sbatch --parsable --dependency=afterok:$FUSE --job-name=hcc_test_submit --partition=main-gpu \
  --gres=gpu:ampere:1 --constraint=a40 --cpus-per-task=1 --mem=2G --time=00:05:00 \
  --output=logs/test_submit_%j.out \
  --wrap="cd \$HOME/projects/hccnet && sbatch --array=0-24$SUF hpc/repro_test.slurm")
echo "7 test     = $TEST  (helper: fires after fusion, then submits the 25 test tasks)"
echo
echo "post-DINO chain submitted."
