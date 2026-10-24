#!/bin/bash
#SBATCH --job-name=hcc_p3_finetune
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#
# PAPER-EXACT fine-tuning (arXiv:2501.10733): 400 steps, 5-fold CV x 10 seeds
# (= 50 runs), effective batch 32, 5% warm-up, dropout 0.2, weight decay 1e-5,
# seq-len 8 -- all set by --paper. ~4 h + one-time caching.
#
# Usage: sbatch stage3_finetune_paper.sbatch [GROUP] [ARCH]   (default dwi femto)
GROUP="${1:-dwi}"; ARCH="${2:-femto}"
case "$GROUP" in
  dwi)   MODLIST="DWI_b0 DWI_b150 DWI_b400 DWI_b800" ;;
  t1dce) MODLIST="T1WI T1A T1V T1D" ;;
  t1iop) MODLIST="T1W_IP T1W_OOP T2W_TES T2W_TEL" ;;
  *) echo "unknown GROUP '$GROUP' (use dwi|t1dce|t1iop)"; exit 1 ;;
esac
TAG="${GROUP}_${ARCH}_paper"
export PATH="$HOME/envs/hccnet/bin:$PATH"
cd $HOME/projects/hccnet
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
NEED=$HOME/models/hccnet/weights_decoderpretrain_${TAG}.pth
if [ ! -f "$NEED" ]; then echo "ABORT: missing decoder hand-off $NEED"; exit 1; fi
echo "##### GPU #####"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
echo "##### PAPER STAGE 3 FINETUNE: group=$GROUP arch=$ARCH tag=$TAG (50 runs) #####"; date
torchrun --standalone --nproc_per_node=1 train.py \
  --paper --distributed --pretrained --arch $ARCH \
  --mod-list $MODLIST --batch-size 4 \
  --run-tag $TAG --suffix ${TAG}_ft
RC=$?; date; echo "torchrun exit code: $RC"
echo "checkpoints:"; ls $HOME/models/hccnet/weights_fold*_${TAG}_ft.pth 2>/dev/null | wc -l
echo "##### PAPER STAGE 3 DONE #####"
