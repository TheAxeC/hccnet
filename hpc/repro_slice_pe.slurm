#!/bin/bash
#SBATCH --job-name=hcc_slice_pe
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:ampere:1
#SBATCH --constraint=a40
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x_%j.out
#
# PE slice-test: does reverting sqrt-PE -> linear-PE recover DWI/femto? The PE lives in
# the Transformer, NOT the CNN backbone, so this REUSES the sweep's dwi_femto DINO (copied
# under a separate tag) and only re-runs decoder -> finetune -> test with --pe-mode linear.
# A separate tag (dwi_femto_pelinear) keeps the sweep's dwi_femto_paper results untouched.
# NO arguments: sbatch hpc/repro_slice_pe.slurm
cd $HOME/projects/hccnet
source hpc/_repro_lib.sh
stage_local
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

A=femto
MODLIST="DWI_b0 DWI_b150 DWI_b400 DWI_b800"
SRC=$WEIGHTS_DIR/weights_dinopretrain_dwi_femto_paper.pth   # the sweep's DINO backbone
TAG=dwi_femto_pelinear
DINO=$WEIGHTS_DIR/weights_dinopretrain_${TAG}.pth

if [ ! -f "$SRC" ]; then echo "ABORT: source DINO $SRC missing (run the dwi/femto DINO first)"; exit 1; fi
cp -n "$SRC" "$DINO"   # reuse the CNN backbone under the PE-test tag (PE is CNN-independent)
echo "##### SLICE PE-TEST: dwi/femto, LINEAR PE, tag=$TAG (reusing dwi_femto DINO) #####"; date

echo "--- decoder (linear PE) ---"; date
torchrun --nnodes=1 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:${RDZV_PORT} --rdzv_id=hcc-${SLURM_JOB_ID} --nproc_per_node=1 pretrain.py \
  --paper --distributed --loss-fn bce --arch $A --mod-list $MODLIST --batch-size 4 --pe-mode linear --run-tag $TAG --suffix ${TAG}_te
[ -f "$WEIGHTS_DIR/weights_decoderpretrain_${TAG}.pth" ] || { echo "ABORT: decoder hand-off missing"; exit 1; }

echo "--- finetune (linear PE) ---"; date
torchrun --nnodes=1 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:${RDZV_PORT} --rdzv_id=hcc-${SLURM_JOB_ID} --nproc_per_node=1 train.py \
  --paper --distributed --pretrained --arch $A --mod-list $MODLIST --batch-size 4 --pe-mode linear --run-tag $TAG --suffix ${TAG}_ft
N=$(ls $WEIGHTS_DIR/weights_fold*_${TAG}_ft.pth 2>/dev/null | wc -l)
echo "finetune checkpoints: $N/50"
[ "$N" -ge 50 ] || { echo "ERROR: finetune incomplete ($N/50)"; exit 1; }

echo "--- test (linear PE) ---"; date
torchrun --nnodes=1 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:${RDZV_PORT} --rdzv_id=hcc-${SLURM_JOB_ID} --nproc_per_node=1 test.py \
  --distributed --arch $A --mod-list $MODLIST --batch-size 4 --pe-mode linear --suffix ${TAG}_ft
date; echo "##### SLICE PE-TEST DONE (compare AUROC/AUPRC vs sweep dwi_femto ft 0.880/0.418) #####"
