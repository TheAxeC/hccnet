#!/bin/bash
#SBATCH --job-name=hcc_wholecohort
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --array=0-3
#
# Score the WHOLE cohort (243 patients), not the 58 held-out patients, at every clinician-set
# month horizon. The development patients trained the model, so this describes the cohort and
# does not validate on unseen data; the clinical manuscript reports it as such.
#
# One array task per single sequence, each at the size the test-set sweep picked. Only fold 0
# is retained for these configurations, so this is a single model where the test-set numbers
# are a fifty-model ensemble.
#
# Output goes to results_allpatients/, which carries a model_weights symlink to the real
# weights directory, so the existing held-out predictions are never overwritten.
cd $HOME/projects/hccnet
source hpc/_repro_lib.sh
stage_local
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

MODS=(dwi t1dce t1iop_only t2_only)
SIZES=(tiny pico femto nano)
M=${MODS[$SLURM_ARRAY_TASK_ID]}
A=${SIZES[$SLURM_ARRAY_TASK_ID]}
MODLIST=$(modlist_for $M)
SUFFIX="${M}_${A}_paper_ft"
OUT="$HOME/projects/hccnet/results_allpatients"

mkdir -p "$OUT"
ln -sfn "$WEIGHTS_DIR" "$OUT/model_weights"

N=$(ls $WEIGHTS_DIR/weights_fold*_${SUFFIX}.pth 2>/dev/null | wc -l)
if [ "$N" -eq 0 ]; then echo "ABORT: no checkpoints for $SUFFIX"; exit 1; fi
echo "##### WHOLE COHORT [task ${SLURM_ARRAY_TASK_ID}]: mod=$M arch=$A suffix=$SUFFIX ($N ckpt) #####"; date

torchrun --nnodes=1 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:${RDZV_PORT} \
  --rdzv_id=hcc-${SLURM_JOB_ID} --nproc_per_node=1 test.py $REPRO_ARCH \
  --distributed --arch $A --mod-list $MODLIST --batch-size 4 --suffix $SUFFIX \
  --results-dir "$OUT" --score-all-patients --save-prefixes --month-horizons

date; echo "##### WHOLE COHORT DONE #####"
