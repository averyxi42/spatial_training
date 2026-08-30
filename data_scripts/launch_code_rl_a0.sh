#!/usr/bin/env bash
# Path A0 code-only RL: code_rl_held128 on 4 GPUs (default 4,5,6,7), quad resources.
# docs/CODE_RL_PLAN_V2.md sections 2/6.5; the yaml's own header documents the recipe.
#
# THE WORKTREE PIN, load-bearing: the longnav_vlm editable install resolves `longnav` to
# /Projects/spatial_training/src (flowsde), which has neither CodeFlowHead nor this yaml.
# PYTHONPATH is exported so the DRIVER and every Ray actor (separate processes that
# re-import from scratch; env_vars merge leaves PYTHONPATH inherited) import THIS worktree
# -- the tests/conftest.py convention, LATENT_RL_ENV.md "Two landmines".
set -euo pipefail

REPO=/Projects/spatial_training_tok
GPUS=${GPUS:-4,5,6,7}
RUN_NAME=${RUN_NAME:-code_rl_a0_held128}
RESOURCES=${RESOURCES:-quad}   # 4 VLMs + 5 sims; hapo n_rollout 16 divides 4

# Refuse a launch onto busy devices rather than OOM into someone's job.
for g in ${GPUS//,/ }; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
  if [ "$used" -gt 2000 ]; then
    echo "GPU $g has ${used} MiB in use; refusing to launch (set GPUS to free devices)" >&2
    exit 2
  fi
done

LOG=/Projects/spatial_training/dump/flow_rl/${RUN_NAME}_launch.log
mkdir -p "$(dirname "$LOG")"
exec > >(tee "$LOG") 2>&1
cd "$REPO"
echo "=== $(date -Is)  $RUN_NAME  gpus=$GPUS resources=$RESOURCES"
echo "=== git: $(git rev-parse --short HEAD) $(git status --short | wc -l) uncommitted file(s)"

export PYTHONPATH=$REPO/src${PYTHONPATH:+:$PYTHONPATH}
CUDA_VISIBLE_DEVICES=$GPUS /workspace/conda/envs/longnav_vlm/bin/python -m longnav.scripts.train_rl \
  +resources=$RESOURCES +training=hapo \
  +experiment=code_rl_held128 \
  task.run_name="$RUN_NAME"
