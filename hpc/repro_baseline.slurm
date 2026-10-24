#!/bin/bash
#SBATCH --job-name=hcc_repro_baseline
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40
#SBATCH --exclude=ctit086
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --array=0-24
#
# Randomly-initialized baseline (no pre-training) + ensemble-test, DISK-LEAN self-contained.
# Array index = 5 modalities x 5 archs (= 25). Same as repro_finetune but random init (no
# decoder hand-off, --run-tag = its own suffix). The 50 fold-seed checkpoints live on /local
# and are discarded; only the ensemble metric + plots (-> home) and ONE representative fold-0
# checkpoint persist. NO arguments. Idempotent via a per-config sentinel.
cd $HOME/projects/hccnet
source hpc/_repro_lib.sh
stage_local
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

M=${FT_MODS[$((SLURM_ARRAY_TASK_ID / 5))]}
A=${ARCHS[$((SLURM_ARRAY_TASK_ID % 5))]}
MODLIST=$(modlist_for $M)
SUFFIX="${M}_${A}_paper_base"
RDZV="--nnodes=1 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:${RDZV_PORT} --rdzv_id=hcc-${SLURM_JOB_ID} --nproc_per_node=1"

HOME_RES="$HOME/projects/hccnet/results"
HOME_WEIGHTS="$HOME_RES/model_weights"
SENT="$HOME_RES/done/${SUFFIX}.done"
mkdir -p "$HOME_RES/done" "$HOME_RES/model_diagnostics"
[ -f "$SENT" ] && { echo "SKIP: $SUFFIX already done"; exit 0; }

# Route ALL model I/O to node-local NVMe (checkpoints + metrics).
export HCCNET_RESULTS_DIR="/local/${SLURM_JOB_ID}"
LW="$HCCNET_RESULTS_DIR/model_weights"; mkdir -p "$LW"
echo "##### BASELINE+TEST [task ${SLURM_ARRAY_TASK_ID}]: mod=$M arch=$A (50 ckpts transient on /local) #####"; date

torchrun $RDZV train.py $REPRO_ARCH --paper --distributed --arch $A \
  --mod-list $MODLIST --batch-size 4 --run-tag $SUFFIX --suffix $SUFFIX
RC=$?; date; echo "train RC: $RC"
N=$(ls $LW/weights_fold*_${SUFFIX}.pth 2>/dev/null | wc -l)
echo "checkpoints (/local): $N/50"
[ "$N" -ge 50 ] || { echo "ERROR: $SUFFIX incomplete ($N/50, train RC=$RC)"; exit 1; }

rm -f "$HOME_RES/model_diagnostics"/test_metrics_*${SUFFIX}.txt
echo "--- ensemble test (reads /local, writes metrics/plots -> /local) ---"; date
torchrun $RDZV test.py $REPRO_ARCH --distributed --arch $A --mod-list $MODLIST --batch-size 4 --suffix $SUFFIX

cp -r "$HCCNET_RESULTS_DIR/model_diagnostics/." "$HOME_RES/model_diagnostics/" 2>/dev/null
[ -f "$LW/weights_fold0_${SUFFIX}.pth" ] && cp "$LW/weights_fold0_${SUFFIX}.pth" "$HOME_WEIGHTS/" && echo "kept representative weights_fold0_${SUFFIX}.pth"
ls "$HOME_RES/model_diagnostics"/test_metrics_*${SUFFIX}.txt >/dev/null 2>&1 || { echo "ERROR: no test metric written for $SUFFIX"; exit 1; }
touch "$SENT"
echo "##### DONE: $SUFFIX (metric+plots+1 representative on home; 49 ckpts discarded on /local) #####"; date
