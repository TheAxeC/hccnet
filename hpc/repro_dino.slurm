#!/bin/bash
#SBATCH --job-name=hcc_repro_dino
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=40:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --array=0-24
#
# Stage 1 (DINO CNN backbone), full sweep, NO arguments. The array index encodes
# (group, arch) = 5 groups x 5 archs (the 2 constituents get their OWN backbone).
# Submit: sbatch repro_dino.slurm
# Runs on 1 A40 GPU (~27 h/config; a whole 4-GPU node never frees up on the
# contended A40 nodes, so 1 GPU is what actually schedules). For a faster 4-GPU
# run when a node IS free:
#   sbatch --gres=gpu:4 --cpus-per-task=16 --export=ALL,NGPU=4 --time=12:00:00 repro_dino.slurm
cd $HOME/projects/hccnet
source hpc/_repro_lib.sh
stage_local
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

G=${MAIN_GROUPS[$((SLURM_ARRAY_TASK_ID / 5))]}
A=${ARCHS[$((SLURM_ARRAY_TASK_ID % 5))]}
MODLIST=$(modlist_for $G)
TAG="${G}_${A}_paper"

echo "##### GPU #####"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "##### REPRO DINO [task ${SLURM_ARRAY_TASK_ID}]: group=$G arch=$A tag=$TAG nproc=${NGPU:-1} #####"; date
torchrun --nnodes=1 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:${RDZV_PORT} --rdzv_id=hcc-${SLURM_JOB_ID} --nproc_per_node=${NGPU:-1} pretrain.py $REPRO_ARCH \
  --paper --distributed --loss-fn dino --arch $A \
  --mod-list $MODLIST --batch-size 4 \
  --run-tag $TAG --suffix $TAG
RC=$?; date; echo "torchrun exit code: $RC"
OUT=$WEIGHTS_DIR/weights_dinopretrain_${TAG}.pth
echo "##### REPRO DINO DONE #####"
# Exit on whether the hand-off was produced (a benign post-save step can make torchrun
# exit non-zero), so afterok gates on real success.
if [ -f "$OUT" ]; then echo "HANDOFF OK: $(basename $OUT)"; exit 0; else echo "ERROR: $OUT not created"; exit 1; fi
