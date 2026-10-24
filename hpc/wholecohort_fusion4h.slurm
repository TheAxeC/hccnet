#!/bin/bash
#SBATCH --job-name=hcc_wholecohort_fusion
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --array=4-6
#
# Score the WHOLE cohort with the three combined-sequence models (tasks 4-6, pico, the only
# capacity they were trained at; only fold 0 is retained, so each is a single model), saving the
# patient id beside every score. The four single sequences are not rescored: their horizon ids are
# rebuilt exactly from the first run by analysis/recover_horizon_ids.py in the clinical project.
#
# Horizons: the next scan and 1, 2 and 3 years (the original cluster tables 1, 12, 24, 36 months).
# The task numbers keep the MODS indexing of wholecohort_v2.slurm.
#
# Output goes to results_allpatients_v2/, which carries a model_weights symlink to the real
# weights directory, so no earlier output is overwritten.
cd $HOME/projects/hccnet
source hpc/_repro_lib.sh
stage_local
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

MODS=(dwi t1dce t1iop_only t2_only fuse_dwi_t1dce fuse_dwi_t1iop fuse_full)
SIZES=(tiny pico femto nano pico pico pico)
M=${MODS[$SLURM_ARRAY_TASK_ID]}
A=${SIZES[$SLURM_ARRAY_TASK_ID]}
MODLIST=$(modlist_for $M)
SUFFIX="${M}_${A}_paper_ft"
OUT="$HOME/projects/hccnet/results_allpatients_v2"
HORIZONS="1,12,24,36"

mkdir -p "$OUT"
ln -sfn "$WEIGHTS_DIR" "$OUT/model_weights"

N=$(ls $WEIGHTS_DIR/weights_fold*_${SUFFIX}.pth 2>/dev/null | wc -l)
if [ "$N" -eq 0 ]; then echo "ABORT: no checkpoints for $SUFFIX"; exit 1; fi
echo "##### WHOLE COHORT fusion [task ${SLURM_ARRAY_TASK_ID}]: mod=$M arch=$A suffix=$SUFFIX #####"; date

torchrun --nnodes=1 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:${RDZV_PORT} \
  --rdzv_id=hcc-${SLURM_JOB_ID} --nproc_per_node=1 test.py $REPRO_ARCH \
  --distributed --arch $A --mod-list $MODLIST --batch-size 4 --suffix $SUFFIX \
  --results-dir "$OUT" --score-all-patients --save-prefixes --month-horizons \
  --horizon-files "$HORIZONS"

date; echo "##### WHOLE COHORT fusion DONE #####"
