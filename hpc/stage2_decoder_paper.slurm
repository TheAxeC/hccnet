#!/bin/bash
#SBATCH --job-name=hcc_p2_decoder
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#
# PAPER-EXACT decoder/transformer pretraining (arXiv:2501.10733): 8000 steps,
# effective batch 32, 5%/10% warm-up, dropout 0.2, seq-len 8 -- via --paper. ~3 h.
#
# Usage: sbatch stage2_decoder_paper.sbatch [GROUP] [ARCH]   (default dwi femto)
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
NEED=$HOME/models/hccnet/weights_dinopretrain_${TAG}.pth
if [ ! -f "$NEED" ]; then echo "ABORT: missing DINO hand-off $NEED"; exit 1; fi
echo "##### GPU #####"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
echo "##### PAPER STAGE 2 DECODER: group=$GROUP arch=$ARCH tag=$TAG #####"; date
torchrun --standalone --nproc_per_node=1 pretrain.py \
  --paper --distributed --loss-fn bce --arch $ARCH \
  --mod-list $MODLIST --batch-size 4 \
  --run-tag $TAG --suffix ${TAG}_te
RC=$?; date; echo "torchrun exit code: $RC"
OUT=$HOME/models/hccnet/weights_decoderpretrain_${TAG}.pth
if [ -f "$OUT" ]; then echo "HANDOFF OK: $(basename $OUT)"; else echo "ERROR: handoff $OUT not created"; fi
echo "##### PAPER STAGE 2 DONE #####"
