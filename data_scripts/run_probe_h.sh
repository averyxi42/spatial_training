#!/usr/bin/env bash
# The frozen-h code probe, run the way it has to be run to mean anything.
#
# Three fixes over the first attempt, each of which invalidated the previous number:
#   SPLIT BY ROW   turns inside one episode are heavily correlated, so a turn-level
#                  split leaves val turns with siblings in train while the in-run eval
#                  number it is compared against is on held-out EPISODES.
#   --max-turns    matches the training run. DataConfig defaults to 16, which computes
#                  h from a 16-turn window; the in-run head sees up to 200.
#   --mem-fraction 200-turn sequences do not fit in the 0.10 default.
#
# Fewer rows than the 16-turn version at a similar turn count, because each row now
# yields ~55 turns instead of ~16. 800 episodes still leaves 160 for the val split.
set -euo pipefail

REPO=/Projects/spatial_training_tok
DATA=/Projects/spatial_training/data
DUMP=/Projects/spatial_training/dump
RUN=${RUN:-run_code_v4_mlp_warm}
CKPT=${CKPT:-$DUMP/pose_injection/$RUN/checkpoint-800}
TOKENIZER=${TOKENIZER:-$DUMP/tokenizer/dual_fsq_40x40/tokenizer.pt}
# objectnav_nopose, the component the 4.578 in-run CE is measured on.
DS=${DS:-$DATA/v2_25hz_obs2.5hz/formatted_nopose}
GPU=${GPU:-7}
TAG=${TAG:-objectnav_rowsplit}

LOG=$DUMP/tokenizer/probe_h_${TAG}.log
exec > >(tee "$LOG") 2>&1
echo "=== $(date -Is)  probe $CKPT on $DS (gpu $GPU)"

cd "$REPO"
CUDA_VISIBLE_DEVICES=$GPU /workspace/conda/envs/longnav_vlm/bin/python \
  data_scripts/probe_code_from_h.py \
    --checkpoint "$CKPT" \
    --tokenizer "$TOKENIZER" \
    --dataset "$DS" \
    --rows ${ROWS:-800} \
    --max-turns 200 \
    --mem-fraction 0.35 \
    --steps 12000 --lr 3e-4 \
    --in-run-ce 4.578 \
    --cache $DUMP/tokenizer/probe_h_${TAG}.pt \
    --out $DUMP/tokenizer/probe_h_${TAG}.json
