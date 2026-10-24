#!/bin/bash
#SBATCH --job-name=hcc_p4_test
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#
# Evaluate the paper-length fine-tune ensemble on the held-out test set.
#
# Usage: sbatch stage4_test_paper.sbatch [GROUP] [ARCH]   (default dwi femto)
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
N=$(ls $HOME/models/hccnet/weights_fold*_${TAG}_ft.pth 2>/dev/null | wc -l)
if [ "$N" -eq 0 ]; then echo "ABORT: no fine-tune checkpoints for tag ${TAG}_ft"; exit 1; fi
echo "##### PAPER STAGE 4 TEST: group=$GROUP arch=$ARCH (ensembling $N checkpoints) #####"; date
torchrun --standalone --nproc_per_node=1 test.py \
  --distributed --arch $ARCH \
  --mod-list $MODLIST --batch-size 4 \
  --suffix ${TAG}_ft
date; echo "##### PAPER STAGE 4 DONE #####"
