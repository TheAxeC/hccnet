#!/bin/bash
#SBATCH --job-name=hcc_smoke
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:ampere:1
#SBATCH --constraint=a40
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#
# Fast end-to-end SMOKE TEST: one config (femto, dwi), tiny step counts (NOT --paper),
# full chain DINO -> decoder -> finetune -> encoder-only ablation -> test. Validates
# env, data loading, the new code paths (compact arch + nhead, channel-agnostic stem,
# sqrt PE, family-aware sampling, encoder-only) and the checkpoint hand-offs in minutes.
# Submit: sbatch hpc/repro_smoke.slurm
cd $HOME/projects/hccnet
source hpc/_repro_lib.sh
stage_local
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
A=femto
MODLIST=$(modlist_for dwi)
TAG=smoke_dwi_femto
COMMON="--distributed --arch $A --mod-list $MODLIST --batch-size 1 --effective-batch-size 8 --seq-length 8"
DINO_CROPS=""   # default 72/48 crops (tiny crops make degenerate feature maps that crash cuDNN)

echo "##### SMOKE 0: functional check of the new code (no data needed) #####"; date
python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
from models.mednet import MedNet, PositionalEncoding
from models.convnext3d import convnext3d_compact, ChannelAgnosticStem
from data.transforms import contrast_families
# compact backbone -> H=448, MedNet with explicit nhead=4 (head dim 112)
net = MedNet(convnext3d_compact(in_chans=1, num_classes=1), num_classes=1, num_layers=4, nhead=4)
img = torch.randn(2, 3, 1, 72, 72, 72); pad = torch.zeros(2, 3); pos = torch.tensor([[1., 2., 3.], [1., 2., 0.]])
with torch.no_grad():
    y = net(img, pad_mask=pad, pos=pos)
stem = ChannelAgnosticStem(56, kernel_size=3, stride=3)
o1 = stem(torch.randn(1, 1, 72, 72, 72)); o4 = stem(torch.randn(1, 4, 72, 72, 72))
assert o1.shape == o4.shape, "channel-agnostic stem shape mismatch"
print("compact+nhead out", tuple(y.shape), "| stem 1ch==4ch", tuple(o1.shape),
      "| t1iop families", contrast_families(['T1W_IP', 'T1W_OOP', 'T2W_TES', 'T2W_TEL']))
print("FUNCTIONAL CHECK OK")
PY
RC=$?; echo "functional check exit: $RC"; [ $RC -ne 0 ] && { echo "ABORT: functional check failed"; exit 1; }

echo "##### SMOKE 1: DINO (30 steps) #####"; date
torchrun --standalone --nproc_per_node=1 pretrain.py $COMMON $DINO_CROPS --loss-fn dino --num-steps 30 --warmup-steps 3 --run-tag $TAG --suffix $TAG
echo "exit: $?"
echo "##### SMOKE 2: decoder (20 steps) #####"; date
torchrun --standalone --nproc_per_node=1 pretrain.py $COMMON --loss-fn bce --num-steps 20 --warmup-steps 2 --run-tag $TAG --suffix ${TAG}_te
echo "exit: $?"
echo "##### SMOKE 3: finetune (20 steps, 2 folds, 1 seed) #####"; date
torchrun --standalone --nproc_per_node=1 train.py $COMMON --pretrained --num-steps 20 --warmup-steps 2 --k-folds 2 --num-seeds 1 --run-tag $TAG --suffix ${TAG}_ft
echo "exit: $?"
echo "##### SMOKE 4: encoder-only ablation (10 steps) #####"; date
torchrun --standalone --nproc_per_node=1 train.py $COMMON --pretrained --encoder-only --num-steps 10 --warmup-steps 1 --k-folds 2 --num-seeds 1 --run-tag $TAG --suffix ${TAG}_enconly
echo "exit: $?"
echo "##### SMOKE 5: test #####"; date
torchrun --standalone --nproc_per_node=1 test.py $COMMON --suffix ${TAG}_ft
echo "exit: $?"
date; echo "##### SMOKE DONE #####"
