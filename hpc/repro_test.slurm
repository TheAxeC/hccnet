#!/bin/bash
#SBATCH --job-name=hcc_repro_test
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --array=0-24
#
# Evaluate the fine-tuned AND baseline ensembles on the held-out test set, full
# sweep, NO arguments. Array index = 5 modalities x 5 archs. test.py $REPRO_ARCH now reports
# Accuracy/Precision/Recall/F1, AUPRC/AUROC, ECE/MCE/Brier, the cumulative-gain MAE,
# and writes reliability + cumulative-gain plots. Needs the finetune + baseline
# checkpoints. Submit: sbatch --dependency=afterok:<finetune_jobid>:<baseline_jobid> repro_test.slurm
cd $HOME/projects/hccnet
source hpc/_repro_lib.sh
stage_local
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

M=${FT_MODS[$((SLURM_ARRAY_TASK_ID / 5))]}
A=${ARCHS[$((SLURM_ARRAY_TASK_ID % 5))]}
MODLIST=$(modlist_for $M)

for SUFFIX in "${M}_${A}_paper_ft" "${M}_${A}_paper_base"; do
  N=$(ls $WEIGHTS_DIR/weights_fold*_${SUFFIX}.pth 2>/dev/null | wc -l)
  if [ "$N" -eq 0 ]; then echo "skip $SUFFIX (no checkpoints)"; continue; fi
  echo "##### REPRO TEST [task ${SLURM_ARRAY_TASK_ID}]: mod=$M arch=$A suffix=$SUFFIX ($N ckpts) #####"; date
  torchrun --nnodes=1 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:${RDZV_PORT} --rdzv_id=hcc-${SLURM_JOB_ID} --nproc_per_node=1 test.py $REPRO_ARCH \
    --distributed --arch $A --mod-list $MODLIST --batch-size 4 --suffix $SUFFIX
done
date; echo "##### REPRO TEST DONE #####"
