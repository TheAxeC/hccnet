#!/bin/bash
#SBATCH --job-name=hcc_repro_fusion
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40
#SBATCH --exclude=ctit086
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --array=0-2
#
# Multi-modal fusion (HCCNet-Pico), NO arguments, DISK-LEAN. Each fusion config gets its OWN
# self-contained DINO -> decoder -> finetune -> test chain in ONE job (own DINO, within-contrast
# augmentation). Because the chain is self-contained its DINO/decoder hand-offs + 50 fold-seed
# checkpoints NEVER leave the job, so route ALL model I/O to the node-local /local NVMe (via
# HCCNET_RESULTS_DIR) and discard it on exit; only the ensemble test metric + plots (+ ONE
# representative fold-0 model for saliency) persist to NFS home. This keeps the heavy checkpoint
# writes OFF the storage3 home server (UTwente admin request 2026-07-30). Idempotent via a sentinel.
# Batch size is the paper's 4: the earlier fusion GPU-OOM was NOT the batch but the DINO
# CacheDataset caching the whole dataset on the GPU (device=cuda) - fixed in preprocessing.py
# (DINO cache -> CPU RAM), so 8-12ch fusion no longer overflows the A40 at cache-build.
cd $HOME/projects/hccnet
source hpc/_repro_lib.sh
stage_local
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

F=${FUSES[$SLURM_ARRAY_TASK_ID]}
A=pico
MODLIST=$(modlist_for $F)
TAG="${F}_${A}_paper"
RDZV="--nnodes=1 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:${RDZV_PORT} --rdzv_id=hcc-${SLURM_JOB_ID}"

HOME_RES="$HOME/projects/hccnet/results"
SENT="$HOME_RES/done/${TAG}_fusion.done"
mkdir -p "$HOME_RES/done" "$HOME_RES/model_diagnostics" "$HOME_RES/model_weights"
[ -f "$SENT" ] && { echo "SKIP: $TAG fusion already done"; exit 0; }

# Route ALL fusion model I/O (hand-offs + 50 checkpoints + metrics) to node-local /local NVMe.
export HCCNET_RESULTS_DIR="/local/${SLURM_JOB_ID}"
LW="$HCCNET_RESULTS_DIR/model_weights"; mkdir -p "$LW"
echo "##### REPRO FUSION [task ${SLURM_ARRAY_TASK_ID}]: config=$F arch=$A tag=$TAG (all model I/O on /local) #####"; date

echo "--- fusion DINO ---"; date
torchrun $RDZV --nproc_per_node=${NGPU:-1} pretrain.py $REPRO_ARCH \
  --paper --distributed --loss-fn dino --arch $A --mod-list $MODLIST --batch-size 4 --run-tag $TAG --suffix $TAG
[ -f "$LW/weights_dinopretrain_${TAG}.pth" ] || { echo "ABORT: fusion DINO hand-off missing"; exit 1; }

echo "--- fusion decoder ---"; date
torchrun $RDZV --nproc_per_node=1 pretrain.py $REPRO_ARCH \
  --paper --distributed --loss-fn bce --arch $A --mod-list $MODLIST --batch-size 4 --run-tag $TAG --suffix ${TAG}_te
[ -f "$LW/weights_decoderpretrain_${TAG}.pth" ] || { echo "ABORT: fusion decoder hand-off missing"; exit 1; }

echo "--- fusion finetune ---"; date
torchrun $RDZV --nproc_per_node=1 train.py $REPRO_ARCH \
  --paper --distributed --pretrained --arch $A --mod-list $MODLIST --batch-size 4 --run-tag $TAG --suffix ${TAG}_ft
N=$(ls $LW/weights_fold*_${TAG}_ft.pth 2>/dev/null | wc -l)
echo "checkpoints (/local): $N/50"
[ "$N" -ge 50 ] || { echo "ERROR: fusion $TAG incomplete ($N/50)"; exit 1; }

rm -f "$HOME_RES/model_diagnostics"/test_metrics_*${TAG}_ft.txt
echo "--- fusion test ---"; date
torchrun $RDZV --nproc_per_node=1 test.py $REPRO_ARCH \
  --distributed --arch $A --mod-list $MODLIST --batch-size 4 --suffix ${TAG}_ft

# Persist ONLY the metric + plots + one representative model to NFS home; discard the 50 ckpts on /local.
cp -r "$HCCNET_RESULTS_DIR/model_diagnostics/." "$HOME_RES/model_diagnostics/" 2>/dev/null
[ -f "$LW/weights_fold0_${TAG}_ft.pth" ] && cp "$LW/weights_fold0_${TAG}_ft.pth" "$HOME_RES/model_weights/" && echo "kept representative weights_fold0_${TAG}_ft.pth"
ls "$HOME_RES/model_diagnostics"/test_metrics_*${TAG}_ft.txt >/dev/null 2>&1 || { echo "ERROR: no fusion test metric written for $TAG"; exit 1; }
touch "$SENT"
date; echo "##### REPRO FUSION DONE: $TAG (metric+plots+1 representative on home; rest discarded on /local) #####"
