#!/bin/bash
#SBATCH --job-name=hcc_ablation_test
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40
#SBATCH --exclude=ctit086
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --array=0-5
#
# Test-ONLY re-run of the 6 pre-training-ablation arms (HCCNet-Pico). Their full 50-checkpoint
# sets are still on NFS home, so NO training: just re-score to persist per-patient predictions
# and the full-ensemble metrics (via the extended test.py). Array = arm x 3 main modalities:
# 0-2 CNN-only (--partial arm), 3-5 encoder-only. NO arguments.
cd $HOME/projects/hccnet
source hpc/_repro_lib.sh
stage_local
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export HCCNET_RESULTS_DIR="$HOME/projects/hccnet/results"   # load the 50 ckpts + save on home

A=pico
M=${MAIN_GROUPS[$((SLURM_ARRAY_TASK_ID % 3))]}
MODLIST=$(modlist_for $M)
if [ $((SLURM_ARRAY_TASK_ID / 3)) -eq 0 ]; then SUFFIX="${M}_${A}_paper_cnnonly"; else SUFFIX="${M}_${A}_paper_enconly"; fi
RDZV="--nnodes=1 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:${RDZV_PORT} --rdzv_id=hcc-${SLURM_JOB_ID} --nproc_per_node=1"

N=$(ls "$HCCNET_RESULTS_DIR/model_weights"/weights_fold*_${SUFFIX}.pth 2>/dev/null | wc -l)
echo "##### ABLATION TEST [task ${SLURM_ARRAY_TASK_ID}]: $SUFFIX ($N checkpoints on home) #####"; date
[ "$N" -ge 50 ] || { echo "ABORT: expected 50 checkpoints for $SUFFIX, found $N"; exit 1; }
torchrun $RDZV test.py $REPRO_ARCH --distributed --arch $A --mod-list $MODLIST --suffix $SUFFIX --batch-size 4
date; echo "##### ABLATION TEST DONE: $SUFFIX #####"
