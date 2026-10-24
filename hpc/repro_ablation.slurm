#!/bin/bash
#SBATCH --job-name=hcc_repro_ablation
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40
#SBATCH --exclude=ctit086
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --array=0-5
#
# NOTE: ctit086 is a 120G-RAM a40 node (the others are 250G) and this cluster does NOT
# schedule on memory (AllocMem=0 on every node), so --mem is ignored for placement and
# several 80G jobs stacked on the 120G node node-OOM (SIGKILL/-9). Excluding ctit086 keeps
# these heavy cache-building jobs on the 250G nodes.
#
# Pre-training ablation (HCCNet-Pico), NO arguments, DISK-LEAN. Array index = arm x 3 main
# modalities: tasks 0-2 = CNN-only (DINO backbone, random encoder, --partial); tasks 3-5 =
# encoder-only (SOP encoder, random backbone, --encoder-only). The 50 fold-seed checkpoints
# go to node-local /local (via HCCNET_RESULTS_DIR) and are discarded on exit; only the test
# metric + plots + ONE representative model persist to NFS home, keeping checkpoint writes off
# the storage3 home server (UTwente admin request 2026-07-30). The read-only cross-job hand-off
# it loads (DINO for CNN-only, decoder for encoder-only) is copied from home to /local.
# Submit: sbatch --dependency=afterok:<decoder_jobid> repro_ablation.slurm
cd $HOME/projects/hccnet
source hpc/_repro_lib.sh
stage_local
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

A=pico
M=${MAIN_GROUPS[$((SLURM_ARRAY_TASK_ID % 3))]}
MODLIST=$(modlist_for $M)
RUNTAG="${M}_${A}_paper"
RDZV="--nnodes=1 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:${RDZV_PORT} --rdzv_id=hcc-${SLURM_JOB_ID} --nproc_per_node=1"
if [ $((SLURM_ARRAY_TASK_ID / 3)) -eq 0 ]; then
  FLAG="--partial"; SUFFIX="${M}_${A}_paper_cnnonly"; NEEDNAME="weights_dinopretrain_${RUNTAG}.pth"
else
  FLAG="--encoder-only"; SUFFIX="${M}_${A}_paper_enconly"; NEEDNAME="weights_decoderpretrain_${RUNTAG}.pth"
fi

HOME_RES="$HOME/projects/hccnet/results"
HOME_WEIGHTS="$HOME_RES/model_weights"
SENT="$HOME_RES/done/${SUFFIX}.done"
mkdir -p "$HOME_RES/done" "$HOME_RES/model_diagnostics" "$HOME_WEIGHTS"

# Idempotent skip: a sentinel, OR a legacy-complete arm (50 checkpoints already on home from a
# pre-disk-lean run) so previously-finished ablations are never needlessly recomputed.
[ -f "$SENT" ] && { echo "SKIP: $SUFFIX done (sentinel)"; exit 0; }
if [ "$(ls $HOME_WEIGHTS/weights_fold*_${SUFFIX}.pth 2>/dev/null | wc -l)" -ge 50 ]; then
  echo "SKIP: $SUFFIX already complete (50/50 on home, legacy)"; touch "$SENT"; exit 0
fi

NEED="$HOME_WEIGHTS/$NEEDNAME"
[ -f "$NEED" ] || { echo "ABORT: missing hand-off $NEED"; exit 1; }

# Route model I/O to /local NVMe; seed the read-only hand-off there for train.py to load.
export HCCNET_RESULTS_DIR="/local/${SLURM_JOB_ID}"
LW="$HCCNET_RESULTS_DIR/model_weights"; mkdir -p "$LW"
cp "$NEED" "$LW/"
echo "##### REPRO ABLATION [task ${SLURM_ARRAY_TASK_ID}]: mod=$M arch=$A $FLAG out=$SUFFIX (50 ckpts on /local) #####"; date

torchrun $RDZV train.py $REPRO_ARCH --paper --distributed --pretrained $FLAG --arch $A \
  --mod-list $MODLIST --batch-size 4 --run-tag $RUNTAG --suffix $SUFFIX
RC=$?; date; echo "torchrun exit code: $RC"
N=$(ls $LW/weights_fold*_${SUFFIX}.pth 2>/dev/null | wc -l)
echo "checkpoints (/local): $N/50"
[ "$N" -ge 50 ] || { echo "ERROR: $SUFFIX train incomplete ($N/50, torchrun RC=$RC)"; exit 1; }

rm -f "$HOME_RES/model_diagnostics"/test_metrics_*${SUFFIX}.txt
echo "--- ablation test ---"; date
torchrun $RDZV test.py $REPRO_ARCH --distributed --arch $A --mod-list $MODLIST --batch-size 4 --suffix $SUFFIX

# Persist ONLY the metric + plots + one representative model to NFS home; discard ckpts on /local.
cp -r "$HCCNET_RESULTS_DIR/model_diagnostics/." "$HOME_RES/model_diagnostics/" 2>/dev/null
[ -f "$LW/weights_fold0_${SUFFIX}.pth" ] && cp "$LW/weights_fold0_${SUFFIX}.pth" "$HOME_WEIGHTS/" && echo "kept representative weights_fold0_${SUFFIX}.pth"
ls "$HOME_RES/model_diagnostics"/test_metrics_*${SUFFIX}.txt >/dev/null 2>&1 || { echo "ERROR: no ablation test metric for $SUFFIX"; exit 1; }
touch "$SENT"
date; echo "##### REPRO ABLATION DONE: $SUFFIX (metric+plots+1 representative on home; rest discarded on /local) #####"
