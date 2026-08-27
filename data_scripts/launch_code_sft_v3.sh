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
# OBJECTNAV_POSE IS DROPPED, its weight folded into objectnav_nopose (2:2 rather than
# 1:1:2). v3 trained objectnav both with and without pose, but every run after it
# (v10/v11/v12) dropped the pose-conditioned objectnav stream -- their eval components are
# objectnav_nopose / onpolicy / pointnav. Keeping the total objectnav weight identical
# means the stream composition is unchanged (50% objectnav, 50% pointnav); only the source
# of the objectnav half changes. --mixture-length is now explicit at 86093 because the
# default is the sum of source sizes, which two sources no longer reproduce.
#
# THE MIXTURE, recovered from newer runs rather than inferred. v3/v11/v12 all log
# per-component eval keys naming exactly THREE components -- objectnav_nopose,
# objectnav_pose, pointnav -- so the mix carries objectnav both with and without pose
# conditioning (hence "nopose_mix"), plus a LONG pointnav corpus.
#
#   39,061 + 39,061 + 7,971 = 86,093 rows, and HF's ceil(86093/4) = 21524 steps/epoch
#   gives 12000/21524 = 0.557517190113363 against v3's logged 0.5575171901133619 --
#   exact to the last digit, so --mixture-length was left at its default (sum of sizes).
#
#   The 1:1:2 ratio is fixed by turn counts: objectnav rows average 54.6 turns and
#   pointnav rows are a fixed 198, so 1:1:2 predicts a stream mean of 126.3 against v3's
#   measured 124.0, where 1:1:1 gives 102.4 and 1:1:3 gives 140.7.
# --head-lr 1e-4 IS A DELIBERATE DEVIATION FROM v3, which left it unset (so v3 trained its
# fresh decoder and readout at --lr 1e-5). The code head is a from-scratch module that v3
# never had, and its LR was never tuned with one present; the audit measured ~1.4 nats at
# step 406 from the 10x step. The cost is that the warm-started decoder and the readout
# share that group and also move 10x faster than v3 moved them, so this is no longer a
# strict v3 replication.
#
# TWO FLAGS THAT ARE NOT OPTIONAL AND WERE MISSED ON THE FIRST LAUNCH:
#   --max-turns 200        max_turns_per_sample DEFAULTS TO 16, which silently trains on
#                          16-turn windows -- a different recipe, and ~8x faster, which is
#                          how it was caught. v3's samples average 123.94 turns with a max
#                          of 198; objectnav rows average 44 and pointnav rows are a fixed
#                          198, so v3 ran with no effective cap. (That 44/198 split also
#                          corroborates the 1:1 mixture ratio: (44+198)/2 = 121 vs 123.94.)
#   --eval-per-component   otherwise eval is ONE blended number over the mixture, and a run
#                          that forgets ObjectNav while improving PointNav looks flat.
#
# AND A SHELL TRAP: no `#` comment may appear inside the backslash-continued command below.
# A comment line terminates the continuation, and every flag after it -- including
# --output-dir -- is dropped, which is what killed the second launch attempt.
set -euo pipefail

REPO=/Projects/spatial_training_tok
DATA=/Projects/spatial_training/data
DUMP=/Projects/spatial_training/dump
TOKENIZER=${TOKENIZER:-$DUMP/tokenizer/dual_fsq_40x40/tokenizer.pt}
# The c-only prototype: its decoder and both code tables load name-for-name, and it was
# trained with context tokens 4-7 held at zero -- which is exactly what r(h) emits at its
# zero init, so step 0 reproduces it. MUST be the DIFFERENTIAL-space (cycle2) checkpoint;
# the pose-space variant scored better but predicts in another target representation than
# the shipped codec.normalize = scale(decompose_chunk(.)).
WARM_START=${WARM_START:-$DUMP/tokenizer/code_flow_40x40_cycle2/code_flow_head.pt}
# Turn-weighted joint code log-marginal over the 1:1:2 mixture (H(c) = 4.838 nats vs
# ln(1600) = 7.378). Initialises the code head bias; a TRAINABLE bias is worth almost
# nothing because Adam moves a parameter by at most lr per step and the log-marginal
# spans ~7 nats, so the value is entirely in the starting point.
CODE_PRIOR=${CODE_PRIOR:-$DUMP/tokenizer/code_log_prior.npy}
RUN_NAME=${RUN_NAME:-run_code_v4_mlp_warm}
OUT=${OUT:-$DUMP/pose_injection/$RUN_NAME}
GPUS=${GPUS:-0,1,2,3}

cd "$REPO"
# The env's torchrun, not whatever is first on PATH: the bare name resolves to a
# python without transformers and every rank dies on import.
TORCHRUN=${TORCHRUN:-/workspace/conda/envs/longnav_vlm/bin/torchrun}
CUDA_VISIBLE_DEVICES=$GPUS "$TORCHRUN" --nproc_per_node=4 --master_port=29571 \
  data_scripts/train_flow_matching_sft_code.py \
    --tokenizer "$TOKENIZER" \
    --code-init-from "$WARM_START" \
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
    --max-steps 12000 --grad-accum 1 --seed 42 \
    --logging-steps 1 --save-steps 200 --eval-steps 200 \
    --wandb-project pose_injection --run-name "$RUN_NAME" \
    --output-dir "$OUT"
