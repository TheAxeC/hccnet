#!/bin/bash
#SBATCH --job-name=hcc_slice_stem
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:ampere:1
#SBATCH --constraint=a40
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=44:00:00
#SBATCH --output=logs/%x_%j.out
#
# STEM slice-test (the "both-old" revert): does the ORIGINAL multi-channel stem + LINEAR PE
# recover DWI/femto to Nolte's numbers? The stem lives in the CNN backbone, so this CANNOT
# reuse the sweep's DINO - it re-pretrains its OWN DINO with --stem-mode original, then
# decoder -> finetune -> test, all with --pe-mode linear (Nolte's PE too). Separate tag
# keeps the sweep untouched. Compare vs PE-only 0.902/0.536 and the sweep 0.880/0.418.
# NO arguments: sbatch hpc/repro_slice_stem.slurm
cd $HOME/projects/hccnet
source hpc/_repro_lib.sh
stage_local
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

A=femto
MODLIST="DWI_b0 DWI_b150 DWI_b400 DWI_b800"
TAG=dwi_femto_bothold
COMMON="--paper --distributed --arch $A --mod-list $MODLIST --batch-size 4 --pe-mode linear --stem-mode original"
RDZV="--nnodes=1 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:${RDZV_PORT} --rdzv_id=hcc-${SLURM_JOB_ID} --nproc_per_node=1"
echo "##### SLICE STEM-TEST: dwi/femto, ORIGINAL stem + LINEAR PE (both-old), tag=$TAG #####"; date

echo "--- DINO (original stem) ---"; date
torchrun $RDZV pretrain.py $COMMON --loss-fn dino --run-tag $TAG --suffix $TAG
[ -f "$WEIGHTS_DIR/weights_dinopretrain_${TAG}.pth" ] || { echo "ABORT: DINO hand-off missing"; exit 1; }

echo "--- decoder ---"; date
torchrun $RDZV pretrain.py $COMMON --loss-fn bce --run-tag $TAG --suffix ${TAG}_te
[ -f "$WEIGHTS_DIR/weights_decoderpretrain_${TAG}.pth" ] || { echo "ABORT: decoder hand-off missing"; exit 1; }

echo "--- finetune ---"; date
torchrun $RDZV train.py $COMMON --pretrained --run-tag $TAG --suffix ${TAG}_ft
N=$(ls $WEIGHTS_DIR/weights_fold*_${TAG}_ft.pth 2>/dev/null | wc -l)
echo "finetune checkpoints: $N/50"
[ "$N" -ge 50 ] || { echo "ERROR: finetune incomplete ($N/50)"; exit 1; }

echo "--- test ---"; date
torchrun $RDZV test.py --distributed --arch $A --mod-list $MODLIST --batch-size 4 --pe-mode linear --stem-mode original --suffix ${TAG}_ft
date; echo "##### SLICE STEM-TEST DONE (compare vs PE-only 0.902/0.536, sweep 0.880/0.418) #####"
