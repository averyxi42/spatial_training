#!/usr/bin/env bash
# Two slower-observation corpora from the same 25 fps recordings as v2_25hz.
#
#                        v2_25hz (v8)   obs2.5hz (v9)   obs1hz (v10)
#   obs_stride_frames               5             10             25
#   obs_interval                0.2 s          0.4 s          1.0 s
#   action_chunk_len               10             20             30
#   chunk_duration              0.4 s          0.8 s          1.2 s
#   overlap_fraction              0.5            0.5          0.167
#   lookahead past next obs     0.2 s          0.4 s          0.2 s
#   executor --gap                  5             10             25
#
# The hard constraint is chunk_duration >= obs_interval: below it the
# controller runs out of setpoints before the next observation lands. All
# three satisfy it. The three do NOT agree on overlap_fraction -- 1 Hz would
# need action_chunk_len 50 (action dim 150) for that -- so read a v10-vs-v9
# difference as "rate and overlap", not rate alone. Chosen deliberately to
# keep the action dimension bounded.
#
# THE TICK RATE NEVER CHANGES. A setpoint is always 0.04 s apart, so every
# evaluation runs --dt 0.04. What changes is --gap, which must equal
# obs_stride_frames. A run that keeps --gap 5 on these corpora executes a
# fifth of each chunk and discards the rest -- that presents as a slow
# policy, not as a misconfiguration.
#
# Built with THIS repo's builder. data/v2_25hz was built by one that lived
# only in the retired continuous_demos fork; the fork is merged in now, and
# a build from the pre-merge habitat copy silently omits eight per-row
# metadata columns and is not comparable. See habitat CLAUDE.md.
set -euo pipefail

HAB=/Projects/habitat_physical_nav
PY=/workspace/conda/envs/vln/bin/python
SRC=/Projects/habitat_physical_nav/recordings     # recordings (data, not code)
ROOT=/Projects/spatial_training/data

build () {          # $1 = name   $2 = obs_stride_frames   $3 = action_chunk_len
  local OUT="$ROOT/$1"
  echo "=== $(date +%H:%M:%S) $1: stride=$2 chunk_len=$3 -> $OUT"
  "$PY" "$HAB/scripts/build_action_chunk_episode_dataset.py" \
    --data-dir "$SRC" \
    --output-dir "$OUT/chunks" \
    --native-fps 25.0 \
    --obs-stride-frames "$2" \
    --action-chunk-len "$3" \
    --val-scene-count 5 \
    --obs-spacing-mode fixed \
    --num-proc 8

  "$PY" /Projects/spatial_training/data_scripts/format_action_chunk_dataset.py \
    --in-dir "$OUT/chunks" \
    --out-dir "$OUT/formatted_pose" \
    --split train --val-split validation \
    --modality-marker '<pose>' \
    --modality-column obs_poses \
    --num-proc 8
  echo "=== $(date +%H:%M:%S) $1 done"
}

build v2_25hz_obs2.5hz 10 20
build v2_25hz_obs1hz   25 30
echo "=== $(date +%H:%M:%S) both corpora built"
