#!/usr/bin/env bash
# CONTINUATION of run_code_v4_mlp_warm from its final checkpoint-12000.
#
# THE QUESTION. v4's ObjectNav code CE flattened over its last ~2000 steps (3.628 at
# 10000, 3.624 at 11400, 3.625 at 12000) and that was read as convergence. But its LR
# schedule is cosine over max_steps=12000, so the LR was ~0 exactly where the curve went
# flat -- annealing and convergence are perfectly confounded in that run. Re-raising the
# LR on the same weights separates them:
#
#   loss falls again  -> v4 was ANNEALED, not converged; the plateau was the schedule
#   loss stays flat   -> v4 really had reached the limit of this setup
#
# EVERY HYPERPARAMETER IS v4's, deliberately, except the three that have to change:
#
#   --init-from      the new thing. Loads trainable weights ONLY -- no optimizer state,
#                    no scheduler, no step counter, no RNG, no dataloader order (that is
#                    --resume-from, which would restore the annealed LR and defeat the
#                    entire test). Runs `warm_start`, which loads `head` and `normalizer`
#                    strictly; the code slot is attached during build() so code_head.* and
#                    code_mixer.* (13 tensors, verified present in the blob) come across.
#   --max-steps      24000, double v4. The point is a GENTLER cosine, not more steps at
#                    the same aggressive decay: over 24000 the LR is still ~85% of peak at
#                    step 5000 where v4 was already down to ~57%. At the measured 9.24
#                    s/it this is ~62 h on 4 GPUs. Halve it if that is too long -- the
#                    test reads out well before the end, since a schedule-driven plateau
#                    breaks within the first few thousand steps.
#   RUN_NAME/OUT     new, so v4's checkpoints and wandb history are untouched.
#
# --code-init-from IS DELIBERATELY ABSENT. It warm-starts the decoder and both code tables
# from the c-only PROTOTYPE and asserts r(h) is still zero -- both meaningless here, where
# every one of those weights is being loaded from a fully trained checkpoint instead. It
# would run at build() and then be overwritten by --init-from, so keeping it would be
# harmless but actively misleading in the log.
#
# EXPECT A LOSS SPIKE at the start. Re-raising the LR on converged weights perturbs them
# before it improves them; judge the run against v4's 3.625 after a few hundred steps, not
# at step 1.
#
# SHELL TRAP inherited from v3/v4: no `#` comment may appear inside the backslash-continued
# command below -- a comment line terminates the continuation and every flag after it,
# including --output-dir, is silently dropped.
set -euo pipefail

REPO=/Projects/spatial_training_tok
DATA=/Projects/spatial_training/data
DUMP=/Projects/spatial_training/dump
TOKENIZER=${TOKENIZER:-$DUMP/tokenizer/dual_fsq_40x40/tokenizer.pt}
INIT_FROM=${INIT_FROM:-$DUMP/pose_injection/run_code_v4_mlp_warm/checkpoint-12000}
# Prior-initialises the code head's bias at build(). Inert here -- --init-from overwrites
# it with the trained bias -- but kept so the build path is identical to v4's.
CODE_PRIOR=${CODE_PRIOR:-$DUMP/tokenizer/code_log_prior.npy}
RUN_NAME=${RUN_NAME:-run_code_v5_cont24k}
OUT=${OUT:-$DUMP/pose_injection/$RUN_NAME}
MAX_STEPS=${MAX_STEPS:-24000}
GPUS=${GPUS:-0,1,2,3}

if [ ! -f "$INIT_FROM/turn_vector_head.pt" ]; then
  echo "no checkpoint at $INIT_FROM" >&2; exit 1
fi

LOG=${LOG:-$DUMP/pose_injection/${RUN_NAME}.log}
mkdir -p "$(dirname "$LOG")"
# The script owns its own logging. Launching as `tmux new-session "bash script | tee"`
# puts a pipeline between tmux and torchrun, which leaves the elastic launcher unreaped
# and makes kills hang for minutes.
exec > >(tee "$LOG") 2>&1
echo "=== $(date -Is)  $RUN_NAME -> $OUT"
echo "=== warm start from $INIT_FROM, max_steps $MAX_STEPS, gpus $GPUS"

cd "$REPO"
# The env's torchrun, not whatever is first on PATH: the bare name resolves to a python
# without transformers and every rank dies on import.
TORCHRUN=${TORCHRUN:-/workspace/conda/envs/longnav_vlm/bin/torchrun}
CUDA_VISIBLE_DEVICES=$GPUS "$TORCHRUN" --nproc_per_node=4 --master_port=29573 \
  data_scripts/train_flow_matching_sft_code.py \
    --tokenizer "$TOKENIZER" \
    --init-from "$INIT_FROM" \
    --code-loss-weight 1.0 \
    --code-head-kind mlp \
    --code-prior "$CODE_PRIOR" \
    --code-max-grad-norm 1.0 \
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
