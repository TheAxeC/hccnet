#!/bin/bash
#SBATCH --job-name=hcc_repro_decoder
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --array=0-24
#
# Stage 2 (decoder / SOP Transformer pretraining), full sweep, NO arguments.
# Array index = 5 groups x 5 archs (constituents included). Needs the DINO hand-off.
# Submit: sbatch --dependency=afterok:<dino_jobid> repro_decoder.slurm
cd $HOME/projects/hccnet
source hpc/_repro_lib.sh
stage_local
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

G=${MAIN_GROUPS[$((SLURM_ARRAY_TASK_ID / 5))]}
A=${ARCHS[$((SLURM_ARRAY_TASK_ID % 5))]}
MODLIST=$(modlist_for $G)
TAG="${G}_${A}_paper"

NEED=$WEIGHTS_DIR/weights_dinopretrain_${TAG}.pth
if [ ! -f "$NEED" ]; then echo "ABORT: missing DINO hand-off $NEED"; exit 1; fi
echo "##### REPRO DECODER [task ${SLURM_ARRAY_TASK_ID}]: group=$G arch=$A tag=$TAG #####"; date
torchrun --nnodes=1 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:${RDZV_PORT} --rdzv_id=hcc-${SLURM_JOB_ID} --nproc_per_node=1 pretrain.py $REPRO_ARCH \
  --paper --distributed --loss-fn bce --arch $A \
  --mod-list $MODLIST --batch-size 4 \
  --run-tag $TAG --suffix ${TAG}_te
RC=$?; date; echo "torchrun exit code: $RC"
OUT=$WEIGHTS_DIR/weights_decoderpretrain_${TAG}.pth
echo "##### REPRO DECODER DONE #####"
# Exit on whether the hand-off was produced, so afterok gates on real success.
if [ -f "$OUT" ]; then echo "HANDOFF OK: $(basename $OUT)"; exit 0; else echo "ERROR: $OUT not created"; exit 1; fi
