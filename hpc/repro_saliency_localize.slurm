#!/bin/bash
#SBATCH --job-name=hcc_saliency_localize
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40
#SBATCH --exclude=ctit086
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --array=0-4
cd $HOME/projects/hccnet
source hpc/_repro_lib.sh
stage_local
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export HCCNET_RESULTS_DIR="$HOME/projects/hccnet/results"
SAL_MODS=(dwi t1dce t1iop t1iop_only t2_only); A=pico
M=${SAL_MODS[$SLURM_ARRAY_TASK_ID]}
MODLIST=$(modlist_for $M); SUFFIX="${M}_${A}_paper_ft"
RDZV="--nnodes=1 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:${RDZV_PORT} --rdzv_id=hcc-${SLURM_JOB_ID} --nproc_per_node=1"
echo "##### LOCALIZE [task $SLURM_ARRAY_TASK_ID]: $SUFFIX #####"; date
torchrun $RDZV saliency_localize.py $REPRO_ARCH --paper --distributed --arch $A --mod-list $MODLIST --suffix $SUFFIX
date; echo DONE
