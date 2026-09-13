# RL stop-head diagnosis and study

## Scope and guardrails

This study uses the fixed HM3D `eval32.txt` split and the existing exploration
reward.  It runs four independent two-GPU jobs, not one eight-GPU job with four
interacting policies.  All four experiments write to `rl_stop_experiments/stop_head_rl_20260913`.
The old Ray heads on ports 26380 and 26381 are deliberately left untouched.  The study
uses one dedicated Ray head on 26410 with custom per-experiment resources, so the four two-GPU
jobs remain isolated without attempting four incompatible heads on one host.

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

For experiments A/B, every 16 train forwards now also log the norm and pairwise cosine of each
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
is unmasked only in experiments intended to learn that signal.  A trajectory-length STOP calls
the same environment `policy_stop` endpoint as the binary head, but its sampled action
is kept in PPO's response mask: otherwise the action head would have no gradient for
the stopping decision it made.  Explicit STOP receives +1 inside the success radius and
-1 outside it in B/C/D; this makes early action-head stops identifiable rather than merely
ending future progress.

## Experiment matrix

Each experiment: 2 VLM workers + 2 simulators on its own GPU pair, 360 driver cycles
(`total_optimization_steps=180`, `grad_accum_steps=4`, `n_rollout=2`), exploration
reward enabled, reset optimizer, fixed eval32 every eight cycles, checkpoint every 20
cycles plus rolling latest.

| experiment | GPUs | execution | state probe | post-goal action training | purpose |
| --- | --- | --- | --- | --- | --- |
| A | 0,1 | oracle-success + shadow | BCE + first-pass + shadow RL | 4 steps, stillness reward | historical control with a direct stationary-chunk reward |
| B | 2,3 | binary head hard-stop at calibrated 0.47 | BCE + first-pass; no shadow RL | masked | test true deployment semantics for the existing head |
| C | 4,5 | decoded path <= 0.10 m | disabled | 4 steps, same stillness reward | action-head STOP with practical 10 cm threshold |
| D | 6,7 | decoded path <= 0.01 m | disabled | 4 steps, same stillness reward | stricter action-head STOP control |

## Four no-update diagnostics before the study

No 360-cycle experiment starts until these four readings have completed on the frozen c339
checkpoint.  They use one eight-GPU Ray head with four isolated two-GPU custom-resource
pairs and write under `diagnostics/`.  They are deliberately short: their job is to reject
a bad intervention, not to substitute for the actual RL experiments.

1. **Gradient interference.** Four real exploration rollouts are packed as their complete
   history and forwarded without an optimizer step.  BCE, first-pass, and shadow-RL are
   differentiated separately.  We record loss, norm and pairwise cosine for the STOP head,
   LoRA parameters and the shared readout hidden state.  Frozen backbone parameters are
   reported as structurally frozen rather than pretending they received a gradient.
2. **Full-history calibration.** A shadow rollout over the fixed `calibration16` half of
   eval32 retains every KV-history decision: probability, pre-action distance, stop label,
   action-path length and history length.  We compute AP/AUROC/Brier/ECE plus first-stop
   precision, recall and lag; one threshold is selected there only.
3. **Termination semantics.** On disjoint `test16`, run the same checkpoint/UID/ODE seed
   under shadow, deterministic-seeded sampled hazard, and the calibration-selected hard
   threshold.  This distinguishes calibration error from the absorbing-stop exposure gap.
4. **Minimal loss ablation reading.** On the exact full-history gradient batches, compare
   the combined gradient implied by BCE+first-pass+shadow, BCE+first-pass, BCE only, and
   BCE+first-pass with the backbone gradient gate set to zero.  This is an objective-surface
   screen, not a claim of learned performance; the four 360-cycle experiments are the learned
   performance test.

The calibration/test partition is fixed before looking at STOP scores; neither per-test
result nor the main 32-episode monitoring series is used to change the threshold.

## Monitoring after the four experiments start

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

### 2026-09-13 — setup and completed frozen-checkpoint diagnostics

- Fixed-eval history and source checkpoint audited as above.
- GPU 0–7 were idle at preflight; 433 GB remains on `/mnt/nvme_scratch` and 666 GB on
  `/home/ubuntu/Projects`.
- Existing Ray heads on 26380/26381 are idle but retained.  The new 26410 head exposes
  four named VLM/simulator resource pairs, one pair per experiment.
- Targeted source/config gate: 27 passed, 2 skipped.  No A–D 360-cycle training process
  has started.
- The diagnostic trace path now disables MP4 capture explicitly.  It also repaired two
  runtime contracts found before any conclusion: reset may legitimately have no finite
  goal distance, and a `policy_stop` terminal observation must provide the normal
  `just_reached`/post-goal fields expected by trajectory packing.
- The initial gradient replay exposed a separate diagnostic-only sequence error: its
  targets were still collator-padded to 150 decisions while the packed VLM history had
  38 real decisions.  The replay now applies the exact `response_mask` selection used by
  `run_training_epochs`; a subsequent parameter-group aggregation bug was also repaired.
  The final reading below is from the repaired, no-update full-history replay.

#### 1. Component-gradient interference — completed

Four real exploration rollouts were collected; two reached the success region and were
replayed as complete 22- and 115-decision histories.  The following norms are already
loss-weighted and therefore comparable within each row:

| parameter group | history | BCE norm | first-pass norm | shadow norm | notable cosine |
| --- | ---: | ---: | ---: | ---: | --- |
| STOP head | 22 | 4.0740 | 0.000207 | 0.2084 | BCE/shadow +0.787; FP/shadow -0.393 |
| STOP head | 115 | 0.1521 | 0.001960 | 0.4128 | BCE/shadow +0.107; FP/shadow -0.876 |
| LoRA | 22 | 1.5909 | 0.000040 | 0.1090 | BCE/shadow +0.728; FP/shadow -0.558 |
| LoRA | 115 | 0.1007 | 0.000333 | 0.1715 | BCE/shadow +0.555; FP/shadow -0.510 |

At the common hidden readout the first-pass norm is only `6.2e-7` and `3.7e-6`, versus
`2.54e-2` and `1.29e-3` for BCE.  Thus first-pass and shadow often point in opposite
directions, but first-pass is four or more orders smaller and is not the material source
of interference at c339.  Shadow is substantial (and exceeds BCE on the 115-step STOP
head/LoRA replay), but its BCE cosine is positive on both readings; this is evidence of
large extra optimization pressure, not proof of destructive BCE conflict.  The 6.65M
other trainable probe parameters have zero gradient because this STOP-only objective does
not touch them.

The objective-surface ablation gives the same decision cheaply: combined STOP-head norms
for `BCE+FP+shadow` are 4.240 and 0.454, versus 4.074 and 0.153 for `BCE+FP`; LoRA is
1.672 and 0.242 versus 1.591 and 0.101.  `BCE+FP` is numerically indistinguishable from
`BCE` here.  Setting the probe gradient gate to zero would make the LoRA contribution
exactly zero by construction, not demonstrate a learned-performance gain.  Therefore the
main study retains first-pass and uses A versus B to test shadow/absorbing-stop semantics
instead of deleting first-pass speculatively.

#### 2. Full-history calibration — completed

The precommitted calibration16 split has 1,333 decision points and 54 in-radius positives.
Full-KV-history STOP scoring gives AP 0.6264, AUROC 0.8438, Brier 0.02360 and ECE(10)
0.02326.  Maximizing episode utility (TP minus FP) selects **0.47**, with 7 correct first
stops, 1 premature stop, 3 reach-without-stop cases and 5 navigation failures.  This
threshold was selected before opening test16.

#### 3. Termination-semantics test — completed

All rows use c339, exactly the fixed test16 UIDs and UID-seeded ODE sampling.

| execution | SR | OSPL | mean steps | actual terminal distribution |
| --- | ---: | ---: | ---: | --- |
| shadow | 0.6250 | 0.4231 | 62.6 | 10 reached, 4 max-step, 2 escaped |
| sampled live hazard | 0.5000 | 0.2923 | 40.3 | 10 policy-stop (5 successful), 3 reached, 2 escaped, 1 max-step |
| deterministic physical, calibrated 0.47 | 0.5000 | 0.3537 | 48.4 | 9 policy-stop (5 successful), 3 reached, 2 escaped, 2 max-step |

On test16 the calibrated threshold's first-stop categories are 5 correct, 4 premature,
3 reach-without-stop and 4 navigation failures.  Its held-out per-decision AP is 0.3592
and AUROC 0.8998, but the true absorbing execution remains 12.5 SR points and 6.9 OSPL
points below shadow.  This separates two effects: calibration improves the sampled
execution's OSPL, but it does not eliminate the shadow-to-live exposure gap.

An implementation audit found that physical STOP already obeyed
`rollout.stop_prob_threshold`; an independent checkpoint-side threshold had only been
used when reporting the counterfactual confusion matrix.  The report path now uses the
same rollout threshold for physical STOP.  The first physical pass and a repeat with both
settings at 0.47 have identical per-UID terminal steps/results; the repeat verifies the
live result and the repair prevents future metric-only disagreement.

#### Decision for the four learning runs

The diagnostics support the requested matrix without a long preliminary ablation: keep A
as the shadow/stillness control; use B as the direct test of real absorbing binary STOP;
and keep C/D as action-head STOP alternatives.  Do not interpret the head's good
full-history AUROC as deployment success: four of 16 held-out episodes still stop early,
and live episode outcomes—not frame AP—are the authority.  Each experiment will retain both its
metric-best checkpoints and rolling `latest`, with calibration thresholds recorded beside
future selected checkpoints rather than assumed universally equal to 0.95.

#### Launch isolation correction

The first four drivers were stopped during bootstrap before any rollout or optimizer step:
although their optimizer/scheduler reset flags were false, the generic launcher also read
`rl_state.pt` and inherited cycle 340 plus a 256-episode advantage buffer.  That is a
valid resume behavior but invalid for this independent study.  `resume_driver_state` now
defaults to true for ordinary resumes and is explicitly false for all stop-study experiments;
the relaunch starts at cycle 0 with c339 model/STOP-head weights only.  The relaunch gate
again passed 27 tests (2 skipped).

A further launch-time gate found a missing `maybe_save_eval_bests` import after C/D had
completed their cycle-0 evals but before any optimizer update.  All four drivers were
then stopped, the import was repaired and covered by a training-driver import gate plus
17 related tests (2 skipped).  The resulting r1 output and W&B runs are retained as
invalid audit artifacts, including C's 0.031/0.025 and D's 0.094/0.058 cycle-0 SR/OSPL;
they are not study results.  Their eight cloud W&B entries (r1/r2 only) were deleted after
the local logs had captured the failure evidence, so the project now retains only meaningful
historical runs and the active r3 runs.

The r2 reset gate itself passed, but C/D then exposed a second bootstrap-only issue: an
action-head STOP configuration has no binary state probe, while the generic training
forward still passed it binary STOP targets.  The worker raised before each update, so
r2 has no valid learned checkpoint and is also retained only as an audit artifact.  The
worker now drops probe-only training inputs when no probe is configured; the regression
test exercises that exact path, and the relevant unit suite passes 28 tests (2 skipped).
All four formal learning runs therefore use fresh `r3` names and directories.

### r3 initial execution gate

All four r3 drivers started from fresh state, completed the same cycle-0 fixed `eval32`,
saved a metric-best checkpoint and a rolling latest checkpoint, and created separate W&B
runs.  These are execution baselines from identical c339 weights, not learned results:

| experiment | real termination rule | SR | OSPL |
| --- | --- | ---: | ---: |
| A | shadow / oracle success | 0.6875 | 0.3865 |
| B | binary head, physical threshold 0.47 | 0.5938 | 0.3508 |
| C | decoded action path <= 0.10 m | 0.0312 | 0.0250 |
| D | decoded action path <= 0.01 m | 0.0938 | 0.0583 |

C/D have also completed valid rollout, backward and rolling-latest saves through cycle 7;
their prior probe-free crash does not recur.  At cycle 8, C reached SR/OSPL 0.0625/0.0405
and D reached 0.0938/0.0598.  Thus the current action-path distribution already places
substantial mass below even 0.01 m, producing premature physical STOP.  This is an
important negative result for the proposed direct thresholds, but it is not yet an
early-stop decision: the predeclared gate requires two fixed evaluations beyond cycle 16.

### C/D early-stop conclusion

The two post-cycle-16 fixed evaluations met the early-stop rule for both action-path
variants, so C and D were deliberately stopped and their GPU pairs released.  This is not
an infrastructure failure: each run completed forward/backward updates, fixed-eval UID
coverage, checkpoint writes and W&B synchronization before the decision.

| experiment | eval 24: SR / OSPL | eval 32: SR / OSPL | eval-32 stop rate | eval-32 stop precision |
| --- | ---: | ---: | ---: | ---: |
| C, path <= 0.10 m | 0.0625 / 0.0400 | 0.0312 / 0.0250 | 0.9688 | 0.0000 |
| D, path <= 0.01 m | 0.0625 / 0.0400 | 0.0938 / 0.0623 | 0.9062 | 0.0345 |

The two absolute thresholds therefore fail for the same structural reason: a single
decoded action can be nearly stationary far from the object, and the absorbing endpoint
turns that transient into an irreversible false stop.  The post-goal stillness reward
cannot repair it while most rollouts end in one or two decisions.  C's per-metric best
snapshots remain at its fixed-eval checkpoints and its rolling latest is
`.latest_cycle_38`; D's current numbered latest is `checkpoint_39`, with the final
emergency snapshot `checkpoint_40_crash`.  Both W&B runs are kept, tagged
`early-stopped`, `negative-result`, and `action-path-stop`, with the reason attached;
they are retained as meaningful negative evidence rather than deleted as failed garbage.

### A/B first learned fixed evaluation

The remaining two runs passed cycle 8 with nonzero exploration reward, valid PPO updates,
fixed-eval coverage and both checkpoint classes.  One learned evaluation is only a health
gate, not evidence of a trend:

| experiment | cycle 0 SR / OSPL | cycle 8 SR / OSPL | cycle-8 stop precision / recall |
| --- | ---: | ---: | ---: |
| A, shadow + stillness | 0.6875 / 0.3865 | 0.6562 / 0.3849 | 0.8182 / 0.0968 (shadow diagnostic) |
| B, real binary STOP at 0.47 | 0.5938 / 0.3508 | 0.5938 / 0.3505 | 0.7143 / 0.2381 |

B is not suffering C/D-style mass premature stopping: its cycle-8 false stops are 4
decisions, while 32 positive opportunities are missed.  The immediate stop-head question
for B is therefore recall and sequential decision timing, not a lower global threshold;
the fixed calibration split remains the only source for any threshold change.

At cycle 16, A is SR/OSPL 0.6562/0.3637 and B is 0.5938/0.3948.  B's OSPL and SPL
improve materially while SR stays fixed, but its STOP precision/recall (0.6875/0.2558)
is almost unchanged from cycle 0 (0.7059/0.3000).  That is evidence of more efficient
successful trajectories, not evidence that the head has solved the missed-stop problem.
A's OSPL decline after one additional point is likewise insufficient for a trend claim.
Neither run has a non-finite metric, reference-density fuse, fixed-eval coverage failure,
or checkpoint-write failure.

At cycle 24, A recovers to SR/OSPL 0.6562/0.4429 and records its current best OSPL/SPL
snapshot.  B temporarily falls to 0.5000/0.3374, with STOP precision 0.4667, but the
cycle-32 confirmation returns to 0.5938/0.3676 with precision/recall 0.7059/0.3158.
This is not a sustained collapse: retain B's cycle-16 best OSPL/SPL snapshot and continue
training, rather than restarting from a one-eval dip or changing its calibrated threshold.

### Action-path semantic diagnostic after C/D

The released C/D GPU pairs ran two no-update, no-W&B full-history shadow traces from c339:
calibration16 selects a rule and test16 is used once for validation.  Every action recorded
its decoded path length and online stop target, so this measures the user-proposed signal
without the confound that an early physical STOP removes its later history.

Small action-path length is not a STOP signal: its AUROC for stop opportunities is 0.4935
on calibration16 and 0.4588 on test16.  The median first-action path is only 0.00320 m on
calibration16 and 0.00278 m on test16, which explains why both 0.10 m and 0.01 m actuate
near the start of an episode.  A calibration-only grid over 9 thresholds, 4 consecutive
action counts and 4 minimum-history values (144 rules) chose the degenerate 0.0005 m rule
that never stops (TP-FP = 0); no candidate had positive TP-FP utility, and its one-time
test16 result also has zero recall.  Do not relaunch C/D as another absolute threshold or
a simple consecutive-small-action rule.  A viable action-only method would need a learned
history-conditioned stopping utility, which is functionally a replacement STOP model;
the current actionable learning comparison remains A versus B.
