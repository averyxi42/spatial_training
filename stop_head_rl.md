# RL stop-head diagnosis and study

## Scope and guardrails

This study uses the fixed HM3D `eval32.txt` split and the existing exploration
reward.  It runs four independent two-GPU jobs, not one eight-GPU job with four
interacting policies.  All arms write to `rl_stop_experiments/stop_head_rl_20260913`.
The old Ray heads on ports 26380 and 26381 are deliberately left untouched.

The common initialization is `hm3d_stop_rl_exploration_v12_shadow_r6/checkpoints/checkpoint_339`.
The fixed evaluation peak was cycle 336; `checkpoint_339` is the closest retained
numbered/rolling weight, three training cycles later.  Optimizer and scheduler state
are reset.  This small but real gap is recorded rather than treated as byte-identical.

## Evidence before changing the objective

The historical fixed-eval records show that a late rolling checkpoint is not the
selection authority:

| run | selected fixed-eval point | SR | OSPL | mean steps | last logged point |
| --- | ---: | ---: | ---: | ---: | --- |
| standard shadow r6 | 248 | 0.6875 | 0.5469 | 76.2 | cycle 992: SR 0.7188, OSPL 0.3844 |
| exploration shadow r6 | 336 | 0.7188 | 0.5590 | 54.3 | cycle 624: SR 0.5625, OSPL 0.3064 |

Thus the navigation policy can regress well after a good fixed-eval point even while
the process remains healthy.  The new jobs start near the stronger exploration peak,
use the same fixed 32 episodes every eight cycles, save per-metric best snapshots, and
retain an atomic rolling `latest` checkpoint separately.

### Stop-loss audit

The trainable binary probe has three terms over the same history-conditioned VLM
readout: per-decision BCE, ordered first-arrival survival loss, and a shadow Bernoulli
REINFORCE term.  It is not a single-frame classifier: rollout inference retains the VLM
history through the KV cache and the training forward consumes that same growing
sequence.

Before this study, `StateProbe.losses()` exposed the three component values *and* their
sum as `probe/stop_loss`; the generic train loop added all returned values.  Each stop
component was therefore counted twice.  This was corrected so only `probe/stop_loss`
is optimized; BCE, first-pass and shadow values are logging-only component metrics.

For arms A/B, every 16 train forwards now also log the norm and pairwise cosine of each
component's gradient with respect to the shared readout hidden state.  This is a cheap
conflict diagnostic at the actual history representation.  It is not a claim about
full-parameter gradient cosine, but a negative cosine here is direct evidence that the
heads ask the common history state to move in opposing directions.

### Two action-head changes

`action_path_length_m` is the sum of XY segment lengths in the decoded cumulative SE(2)
chunk, including the origin-to-first-waypoint segment.  It is deliberately measured on
the decoded controller command, not on the latent SDE chain or a raw tensor norm.

During a post-goal tail the environment can award
`post_goal_stillness_reward * max(0, 1 - path_length / scale)`.  The action transition
is unmasked only in arms intended to learn that signal.  A trajectory-length STOP calls
the same environment `policy_stop` endpoint as the binary head, but its sampled action
is kept in PPO's response mask: otherwise the action head would have no gradient for
the stopping decision it made.  Explicit STOP receives +1 inside the success radius and
-1 outside it in B/C/D; this makes early action-head stops identifiable rather than merely
ending future progress.

## Experiment matrix

Each arm: 2 VLM workers + 2 simulators on its own GPU pair, 360 driver cycles
(`total_optimization_steps=180`, `grad_accum_steps=4`, `n_rollout=2`), exploration
reward enabled, reset optimizer, fixed eval32 every eight cycles, checkpoint every 20
cycles plus rolling latest.

| arm | GPUs | execution | state probe | post-goal action training | purpose |
| --- | --- | --- | --- | --- | --- |
| A | 0,1 | oracle-success + shadow | BCE + first-pass + shadow RL | 4 steps, stillness reward | historical control with a direct stationary-chunk reward |
| B | 2,3 | binary head hard-stop at 0.95 | BCE + first-pass; no shadow RL | masked | test true deployment semantics for the existing head |
| C | 4,5 | decoded path <= 0.10 m | disabled | 4 steps, same stillness reward | action-head STOP with practical 10 cm threshold |
| D | 6,7 | decoded path <= 0.01 m | disabled | 4 steps, same stillness reward | stricter action-head STOP control |

## Preflight and monitoring

Before updates, cycle-0 fixed eval is the threshold calibration.  The saved per-episode
rows include termination reason, stop mode, and mean decoded action-path length.  C/D's
actual stop rate is the decisive calibration; a mean alone is not enough.

Hourly checks inspect process liveness, GPU placement, fixed-eval coverage, newest
checkpoint, train/eval trend, stop precision/recall, path-stop rate, gradient diagnostics,
reference-density drift, and disk headroom.  The automatic drift guard aborts when
`ref_log_ratio_p95 > 8`.

Manual early-stop/revise triggers after two fixed evals beyond cycle 16:

1. SR < 0.25 and OSPL < 0.15 in both evaluations;
2. B/C/D policy-stop rate >= 0.25 with precision < 0.50 in both evaluations;
3. C or D has <= 0.5% trajectory stops through cycle 32 (the threshold is not testing
   the claimed mechanism), or > 40% stops with poor precision;
4. non-finite losses, failed fixed-eval UID coverage, repeated actor failures, or disk
   headroom below 100 GB.

No conclusion will use a W&B run heartbeat, MP4 count, or a connected Ray head as proof
of progress.

## Live log

### 2026-09-13 — setup

- Fixed-eval history and source checkpoint audited as above.
- GPU 0–7 were idle at preflight; 433 GB remains on `/mnt/nvme_scratch` and 666 GB on
  `/home/ubuntu/Projects`.
- Existing Ray heads on 26380/26381 are idle but retained; study heads use 26410–26413.
- Source/config unit gate after the changes: pending final config composition and launch.
