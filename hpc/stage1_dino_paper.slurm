#!/bin/bash
#SBATCH --job-name=hcc_p1_dino
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#
# PAPER-EXACT DINO backbone pretraining (arXiv:2501.10733): 32000 steps,
# effective batch 128, 1600 warm-up, seq-len 8 -- all set by --paper.
#
# Usage: sbatch stage1_dino_paper.sbatch [GROUP] [ARCH]
#   GROUP in {dwi, t1dce, t1iop}        (default dwi)
#   ARCH  in {femto, pico, nano, tiny}  (default femto)
#
# GPUs: defaults to 4-GPU DDP (~7 h; accum adapts via accum=eff_batch//batch//num_devices,
#   pretrain.py:426, so effective batch stays 128 regardless of GPU count). For the 1-GPU
#   path (~27 h) submit with:
#     sbatch --gres=gpu:1 --cpus-per-task=8 --export=ALL,NGPU=1 stage1_dino_paper.sbatch ...
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
echo "##### GPU #####"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "##### PAPER STAGE 1 DINO: group=$GROUP arch=$ARCH tag=$TAG nproc=${NGPU:-4} #####"; date
torchrun --standalone --nproc_per_node=${NGPU:-4} pretrain.py \
  --paper --distributed --loss-fn dino --arch $ARCH \
  --mod-list $MODLIST --batch-size 4 \
  --run-tag $TAG --suffix $TAG
RC=$?; date; echo "torchrun exit code: $RC"
OUT=$HOME/models/hccnet/weights_dinopretrain_${TAG}.pth
if [ -f "$OUT" ]; then echo "HANDOFF OK: $(basename $OUT)"; else echo "ERROR: handoff $OUT not created"; fi
echo "##### PAPER STAGE 1 DONE #####"
