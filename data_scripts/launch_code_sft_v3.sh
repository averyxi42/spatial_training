#!/usr/bin/env bash
# The (c, r(h))-conditioned SFT run, modelled on the standard v3 recipe. NO RTC.
#
# docs/CODE_CONDITIONED_POLICY.md sections 2 and 5. RTC is deliberately absent: it is
# introduced as a fine-tune at ~75% of compute AFTER a normal run, which is how the
# shipped RTC checkpoint (run_cotrain_v3_rtc_ft9000_geo) was made from v3 ck9000.
#
# Every hyperparameter below is read back from run_cotrain_v3_nopose_mix's own
# training_args.bin and adapter_config.json, so this is a replication of the recipe and
# not a reconstruction of it. Passed explicitly rather than left to defaults, so the
# recipe is legible here instead of scattered across argparse.
#
# The modality spec is read back from v3's own checkpoint and COMMITTED at
# src/longnav/conf/modality_specs/v3_pose.json -- it is part of the recipe, so it
# does not belong in dump/ where it is one cleanup away from being unreproducible.
#
# ONE THING NOT RECOVERED EXACTLY: the mixture. v3's epoch counter puts its stream at
# 86,096 examples, but objectnav (39,061) + pointnav (5,062) sums to 44,123, so v3 set
# --mixture-length explicitly and/or weighted the sources unevenly. The length is
# reproduced exactly; the 1:1 ratio is inferred from the flag's own documented example
# and is the one number here that could differ from v3.
set -euo pipefail

REPO=/Projects/spatial_training_tok
DATA=/Projects/spatial_training/data
DUMP=/Projects/spatial_training/dump
TOKENIZER=${TOKENIZER:-$DUMP/tokenizer/dual_fsq_40x40/tokenizer.pt}
RUN_NAME=${RUN_NAME:-run_code_v3_nopose_mix}
OUT=${OUT:-$DUMP/pose_injection/$RUN_NAME}
GPUS=${GPUS:-0,1,2,3}

cd "$REPO"
# The env's torchrun, not whatever is first on PATH: the bare name resolves to a
# python without transformers and every rank dies on import.
TORCHRUN=${TORCHRUN:-/workspace/conda/envs/longnav_vlm/bin/torchrun}
CUDA_VISIBLE_DEVICES=$GPUS "$TORCHRUN" --nproc_per_node=4 --master_port=29571 \
  data_scripts/train_flow_matching_sft_code.py \
    --tokenizer "$TOKENIZER" \
    --code-loss-weight 1.0 \
    --mixture-datasets objectnav=$DATA/v2_25hz_obs2.5hz/formatted_pose:1 \
    --mixture-datasets pointnav=$DATA/pointnav_hm3d_2p5hz_clamp/formatted_pose:1 \
    --mixture-length 86096 \
    --mixture-seed 0 \
    --modality-specs $REPO/src/longnav/conf/modality_specs/v3_pose.json \
    --k-samples 8 \
    --lora-r 128 --lora-alpha 256 --lora-dropout 0.05 \
    --lr 1e-5 --weight-decay 1e-4 --max-grad-norm 1.0 \
    --lr-scheduler cosine --warmup-ratio 0.01 \
    --max-steps 12000 --grad-accum 1 --seed 42 \
    --logging-steps 1 --save-steps 200 --eval-steps 200 \
    --wandb-project pose_injection --run-name "$RUN_NAME" \
    --output-dir "$OUT"
