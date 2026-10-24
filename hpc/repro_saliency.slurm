#!/bin/bash
#SBATCH --job-name=hcc_saliency
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40
#SBATCH --exclude=ctit086
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --array=0-2
#
# Input-gradient saliency on the kept fold-0 models (NO training, inference + one backward
# pass). Array = 3 configs: the strongest DW-MRI (tiny), the standalone T2 surprise (nano),
# and the combined T1 IOP & T2 (pico). Reads the representative fold-0 checkpoint + writes
# heatmaps to NFS home (model_diagnostics/saliency/<suffix>/). NO arguments.
cd $HOME/projects/hccnet
source hpc/_repro_lib.sh
stage_local
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export HCCNET_RESULTS_DIR="$HOME/projects/hccnet/results"   # load fold-0 + write saliency on home

SAL_MODS=(dwi t2_only t1iop)
SAL_ARCHS=(tiny nano pico)
M=${SAL_MODS[$SLURM_ARRAY_TASK_ID]}
A=${SAL_ARCHS[$SLURM_ARRAY_TASK_ID]}
MODLIST=$(modlist_for $M)
SUFFIX="${M}_${A}_paper_ft"
RDZV="--nnodes=1 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:${RDZV_PORT} --rdzv_id=hcc-${SLURM_JOB_ID} --nproc_per_node=1"

echo "##### SALIENCY [task ${SLURM_ARRAY_TASK_ID}]: mod=$M arch=$A suffix=$SUFFIX #####"; date
torchrun $RDZV saliency.py $REPRO_ARCH --paper --distributed --arch $A --mod-list $MODLIST --suffix $SUFFIX
date; echo "##### SALIENCY TASK DONE #####"
