#!/bin/bash
#SBATCH --job-name=hccnet-phase-impact
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40
#SBATCH --exclude=ctit086,ctit088,ctit092
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=72:00:00
#SBATCH --array=0-4%2
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
STAGE="${1:?dino, decoder, or finetune}"
RUN="${2:?Run output directory}"
PROJECT_DIR="${HCCNET_PROJECT_DIR:-$HOME/projects/hccnet}"
source "$PROJECT_DIR/radiologist_code/hpc/radiologist_env.sh"
cp -r "$PROJECT_DIR/radiologist_code" "$WORK/code"
cd "$WORK/code"
ARMS=(base arterial venous late all)
ARM="${ARMS[$SLURM_ARRAY_TASK_ID]}"
MODS=(T1WI)
case "$ARM" in
  arterial) MODS+=(T1A);;
  venous) MODS+=(T1V);;
  late) MODS+=(T1D);;
  all) MODS+=(T1A T1V T1D);;
esac
TAG="phase_${ARM}_pico"
DEST="$RUN/$ARM"
mkdir -p "$DEST/model_weights" "$DEST/cohort_audit"
COMMON=(--distributed --stem-mode original --pe-mode linear --arch pico --mod-list "${MODS[@]}" --required-modalities T1WI T1A T1V T1D --batch-size 4 --seed 1234 --run-tag "$TAG")
LAUNCH=(python3.10 -m torch.distributed.run --nnodes=1 --rdzv_backend=c10d --rdzv_endpoint="127.0.0.1:$((20000 + SLURM_JOB_ID % 40000))" --rdzv_id="$SLURM_JOB_ID" --nproc_per_node=1)
case "$STAGE" in
  dino)
    export HCCNET_PRETRAIN_RESTART="$DEST/dino_restart.pth"
    "${LAUNCH[@]}" pretrain.py "${COMMON[@]}" --paper --loss-fn dino --suffix "$TAG"
    cp "$HCCNET_RESULTS_DIR/model_weights/weights_dinopretrain_${TAG}.pth" "$DEST/model_weights/"
    ;;
  decoder)
    export HCCNET_PRETRAIN_RESTART="$DEST/decoder_restart.pth"
    cp "$DEST/model_weights/weights_dinopretrain_${TAG}.pth" "$HCCNET_RESULTS_DIR/model_weights/"
    "${LAUNCH[@]}" pretrain.py "${COMMON[@]}" --paper --loss-fn bce --suffix "${TAG}_te"
    cp "$HCCNET_RESULTS_DIR/model_weights/weights_decoderpretrain_${TAG}.pth" "$DEST/model_weights/"
    ;;
  finetune)
    cp "$DEST/model_weights/weights_decoderpretrain_${TAG}.pth" "$HCCNET_RESULTS_DIR/model_weights/"
    "${LAUNCH[@]}" train.py "${COMMON[@]}" --paper --pretrained --suffix "${TAG}_ft"
    COUNT=$(find "$HCCNET_RESULTS_DIR/model_weights" -name "weights_fold*_${TAG}_ft.pth" | wc -l)
    [ "$COUNT" -eq 50 ] || { echo "Expected 50 checkpoints; found $COUNT"; exit 1; }
    "${LAUNCH[@]}" test.py "${COMMON[@]}" --seq-length 8 --suffix "${TAG}_ft" --save-prefixes
    cp -r "$HCCNET_RESULTS_DIR/model_diagnostics" "$DEST/"
    cp "$HCCNET_RESULTS_DIR/model_weights/weights_fold0_${TAG}_ft.pth" "$DEST/model_weights/"
    ;;
  *) echo "Unknown stage $STAGE"; exit 2;;
esac
cp "$HCCNET_RESULTS_DIR/cohort_audit/"*.json "$DEST/cohort_audit/"
echo "$STAGE complete: $ARM"
