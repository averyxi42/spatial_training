#!/usr/bin/env bash
# v6: run_code_v4_mlp_warm's recipe with SOFT CODE LABELS -- and nothing else changed.
#
# THE QUESTION. v4 trained the 1600-way code head with a one-hot cross-entropy, which scores
# a 4-degree miss like a 90-degree one and leaves the head's output geometry only weakly
# metric (output-row Spearman vs the per-tick metric +0.24 at ck12000; the decoder-side code
# embeddings, trained by a metric loss, sit at +0.67/+0.76). v6 replaces the one-hot target
# with a Gaussian kernel in the flow head's own per-tick metric (docs/CODE_SOFT_LABELS.md).
# Everything that is not the target -- data, mixture, warm start, LoRA, LRs, schedule,
# steps, seed -- is v4's, so a difference is attributable to the label alone.
#
# What it is judged on (docs/CODE_SOFT_LABELS.md, "Metrics" and "Experiment plan"):
#   * eval_objectnav_nopose_code_mdist_expected  -- expected per-step error of the SAMPLED
#     policy in action_scales units; the number soft labels should move
#   * eval_objectnav_nopose_code_stationary_excess -- must not rise (stationary-bleed check)
#   * eval_objectnav_nopose_code_ce              -- the HARD CE, comparable to v4's 3.625;
#     allowed to rise slightly
#   * dump/audits/code_head_geometry.py <ckpt>   -- output-row metric organisation, at
#     checkpoint-2000 and the final one, against v4's +0.12 (ck800) / +0.24 (ck12000)
#   * a sample101 eval at gap 10 / dt 0.04, code_mode sample T=0.5 (the decode the v4
#     sample101 pair selected), against v4's own sample101 numbers -- never on pose RMSE
#
# SIGMA. 0.10 in per-tick action_scales units: the target spreads over ~3.3 cells and the
# true code keeps ~0.69 of the mass (measured on dual_fsq_40x40; nearest-neighbour spacing
# 0.19, stationary<->creep 0.22). 0.08 / 0.12 are the sweep neighbours if this one is
# ambiguous. --code-label-metric tick_diff is the default and is stated explicitly anyway.
#
# GPUS. v5 (run_code_v5_cont24k) occupies 0-3 at the time of writing; this script makes no
# attempt to check. Set GPUS to four free devices, or wait for v5. --nproc_per_node is
# derived from GPUS so the two cannot disagree.
#
# SHELL TRAP inherited from v3/v4/v5: no `#` comment may appear inside the backslash-continued
# command below -- a comment line terminates the continuation and every flag after it,
# including --output-dir, is silently dropped.
set -euo pipefail

REPO=/Projects/spatial_training_tok
DATA=/Projects/spatial_training/data
DUMP=/Projects/spatial_training/dump
TOKENIZER=${TOKENIZER:-$DUMP/tokenizer/dual_fsq_40x40/tokenizer.pt}
WARM_START=${WARM_START:-$DUMP/tokenizer/code_flow_40x40_cycle2/code_flow_head.pt}
CODE_PRIOR=${CODE_PRIOR:-$DUMP/tokenizer/code_log_prior.npy}
SIGMA=${SIGMA:-0.10}
RUN_NAME=${RUN_NAME:-run_code_v6_soft$(printf '%s' "$SIGMA" | tr -d '.')}
OUT=${OUT:-$DUMP/pose_injection/$RUN_NAME}
MAX_STEPS=${MAX_STEPS:-12000}
GPUS=${GPUS:-0,1,2,3}
NPROC=$(awk -F, '{print NF}' <<<"$GPUS")

for f in "$TOKENIZER" "$WARM_START" "$CODE_PRIOR"; do
  [ -f "$f" ] || { echo "missing input: $f" >&2; exit 1; }
done
if [ -d "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then
  echo "output dir $OUT is not empty; pick another RUN_NAME/OUT rather than overwrite" >&2; exit 1
fi

LOG=${LOG:-$DUMP/pose_injection/${RUN_NAME}.log}
mkdir -p "$(dirname "$LOG")"
# The script owns its own logging (see v5: a tee pipeline between tmux and torchrun leaves
# the elastic launcher unreaped and makes kills hang).
exec > >(tee "$LOG") 2>&1
echo "=== $(date -Is)  $RUN_NAME -> $OUT"
echo "=== soft labels sigma=$SIGMA metric=tick_diff; warm start $WARM_START; max_steps $MAX_STEPS; gpus $GPUS ($NPROC procs)"
cd "$REPO"
echo "=== git: $(git rev-parse --short HEAD) $(git status --short | wc -l) uncommitted file(s)"

# The env's torchrun, not whatever is first on PATH.
TORCHRUN=${TORCHRUN:-/workspace/conda/envs/longnav_vlm/bin/torchrun}
CUDA_VISIBLE_DEVICES=$GPUS "$TORCHRUN" --nproc_per_node="$NPROC" --master_port=29574 \
  data_scripts/train_flow_matching_sft_code.py \
    --tokenizer "$TOKENIZER" \
    --code-init-from "$WARM_START" \
    --code-loss-weight 1.0 \
    --code-head-kind mlp \
    --code-prior "$CODE_PRIOR" \
    --code-max-grad-norm 1.0 \
    --code-label-sigma "$SIGMA" \
    --code-label-metric tick_diff \
    --mixture-datasets objectnav_nopose=$DATA/v2_25hz_obs2.5hz/formatted_nopose:2 \
    --mixture-datasets pointnav=$DATA/pointnav_long_full/formatted_pose:2 \
    --mixture-length 86093 \
    --mixture-seed 0 \
    --modality-specs $REPO/src/longnav/conf/modality_specs/v3_pose.json \
    --max-turns 200 \
    --eval-per-component \
    --k-samples 8 \
    --lora-r 128 --lora-alpha 256 --lora-dropout 0.05 \
    --lr 1e-5 --head-lr 1e-4 --weight-decay 1e-4 --max-grad-norm 1.0 \
    --lr-scheduler cosine --warmup-ratio 0.01 \
    --max-steps "$MAX_STEPS" --grad-accum 1 --seed 42 \
    --logging-steps 1 --save-steps 200 --eval-steps 200 \
    --wandb-project pose_injection --run-name "$RUN_NAME" \
    --output-dir "$OUT"
