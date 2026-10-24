#!/bin/bash
#SBATCH --job-name=hcc_wholecohort_v2
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --array=0-6
#
# Score the WHOLE cohort at every horizon, saving the patient id beside every score so subgroup
# and paired analyses can be run on the re-run route.
#
# Tasks 0-3 are the four single sequences at the size the test-set sweep picked; tasks 4-6 are
# the three combined-sequence models at pico, the only capacity they were trained at. Only fold 0
# is retained for these configurations, so each is a single model.
#
# Horizons: the original cluster tables (1, 3, 6, 12, 24, 36 months) and the ruleA tables built by
# analysis/build_horizon_tables.py in the clinical project. ruleA_12/24/36 sit beside the originals
# so the difference between the two table rules can be measured before ruleA_48/60 are used.
# The ruleA tables live in labels_derived/, never in the shared dataset folder.
#
# Output goes to results_allpatients_v2/, which carries a model_weights symlink to the real
# weights directory, so no earlier output is overwritten.
cd $HOME/projects/hccnet
source hpc/_repro_lib.sh
stage_local
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export HCC_EXTRA_LABEL_DIR="$HOME/projects/hccnet/labels_derived"

MODS=(dwi t1dce t1iop_only t2_only fuse_dwi_t1dce fuse_dwi_t1iop fuse_full)
SIZES=(tiny pico femto nano pico pico pico)
M=${MODS[$SLURM_ARRAY_TASK_ID]}
A=${SIZES[$SLURM_ARRAY_TASK_ID]}
MODLIST=$(modlist_for $M)
SUFFIX="${M}_${A}_paper_ft"
OUT="$HOME/projects/hccnet/results_allpatients_v2"
HORIZONS="1,3,6,12,24,36,ruleA_12,ruleA_24,ruleA_36,ruleA_48,ruleA_60"

mkdir -p "$OUT"
ln -sfn "$WEIGHTS_DIR" "$OUT/model_weights"
for H in ruleA_12 ruleA_24 ruleA_36 ruleA_48 ruleA_60; do
  if [ ! -f "$HCC_EXTRA_LABEL_DIR/labels_${H}months.csv" ]; then
    echo "ABORT: missing $HCC_EXTRA_LABEL_DIR/labels_${H}months.csv"; exit 1
  fi
done

N=$(ls $WEIGHTS_DIR/weights_fold*_${SUFFIX}.pth 2>/dev/null | wc -l)
if [ "$N" -eq 0 ]; then echo "ABORT: no checkpoints for $SUFFIX"; exit 1; fi
echo "##### WHOLE COHORT v2 [task ${SLURM_ARRAY_TASK_ID}]: mod=$M arch=$A suffix=$SUFFIX #####"; date

torchrun --nnodes=1 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:${RDZV_PORT} \
  --rdzv_id=hcc-${SLURM_JOB_ID} --nproc_per_node=1 test.py $REPRO_ARCH \
  --distributed --arch $A --mod-list $MODLIST --batch-size 4 --suffix $SUFFIX \
  --results-dir "$OUT" --score-all-patients --save-prefixes --month-horizons \
  --horizon-files "$HORIZONS"

date; echo "##### WHOLE COHORT v2 DONE #####"
