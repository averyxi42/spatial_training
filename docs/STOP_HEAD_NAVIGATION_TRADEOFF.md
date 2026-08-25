# Why co-training a stop head degraded navigation (v8, 2026-08-24)

`run_cotrain_v8_stop` warm-started from the deployed SFT policy
(`run_cotrain_v3_nopose_mix/checkpoint-12000`) and added an episode-stop head on the
state probe. The head improved steadily. **Navigation degraded monotonically**, measured
closed-loop on `sample400` at the benchmark budget (175 policy steps / 70 s), paired by
episode with identical packing and seeds:

| checkpoint | oracle success @70 s |
|---|---|
| v3 ck12000 (warm start) | 0.72 |
| v8 ck1400 | 0.68 |
| v8 ck1800 | 0.63 |
| v8 ck2000 | 0.52 |

ck1800 vs base −7.2 pts (p = 0.012); ck2000 vs base −15.9 (p < 1e-4); ck1800 vs ck2000
−8.9 (p = 0.009). **Every offline metric stayed flat throughout** — objectnav flow RMSE
0.033 → 0.032, `turn_loss` within a few percent of the warm start — which is why the
training monitor showed green for two thousand steps while the policy went inert.

## The mechanism: three compounding factors

1. **The stop objective never converges; the flow objective was already converged.**
   `turn_loss` sits at ~0.13 for the whole run. `probe_stop_loss` oscillates around 0.55
   with no downward trend past step ~400 (`pos_weight` 5.45 on a hard discrimination).
   One head is done learning and the other supplies gradient forever.

2. **The flow loss is diluted by probe-only rows.**
   `TurnVectorSFTTrainer._get_num_items_in_batch` (`vector_sft.py:1300`) sums `num_turns`
   over EVERY row and gathers it across ranks; the flow loss divides by that global count
   (`vector_sft.py:936`). Probe-only rows are zeroed in the *numerator* by
   `action_weight = 0`, but their turns remain in the *denominator*. At v8's mixture
   (48 / 198 / 175 turns per objectnav / pointnav / on-policy row, 1:1:1 by row) the
   probe-only component is ~42% of all turns, so the action gradient is scaled ~0.58x
   while the probe loss -- a per-rank masked mean the denominator never touches -- is
   undiluted.

3. **Gradient clipping converts that imbalance into winner-take-most.**
   `max_grad_norm = 1.0` in both runs. v3's raw grad norm was ~0.54 and was NEVER
   clipped. v8's is 4.6-5.1 **every step** (`state_probe` alone contributing 0.64 of the
   post-clip total), so every update is rescaled ~0.21x and the surviving unit-norm
   update points mostly along the stop gradient.

Compounded, the action-imitation component of each weight update is roughly
**0.58 x 0.21 ~= 12% of what v3's recipe delivered**, while the stop gradient sets the
direction. A warm-started policy anchored by an eighth of its former imitation signal and
pushed continuously by an unconverging auxiliary objective drifts -- monotonically.

## How the damage expresses

**The policy commands less, uniformly and increasingly.** Commanded displacement per tick
0.0229 -> 0.0201 -> 0.0187 (base -> ck1400/1800 -> ck2000); commanded settle fraction
0.121 -> 0.239; median turn rate 47 -> 32 deg/s. Tracking undershoot is constant at 0.89,
so this is the policy, not the controller. Flip rates are unchanged: **inert, not
erratic**. A uniform 25-30% slowdown reproduces ~6-8 oracle points against the baseline's
arrival-time distribution -- all of ck1800's deficit, which is why ck1800 fully recovers
when given 140 s. ck2000 adds an extensive margin (closest approach 2.25 -> 3.05 m):
episodes that stop making progress at all.

**There is a closed-loop amplifier.** Mean stop-belief beyond 8 m from the goal grows
0.11 -> 0.22; pre-arrival speed drops 2-3x at `stop_prob > 0.8`, and the share of such
steps grows 5.6% -> 10.5%. Slow motion makes the context resemble the on-policy corpus's
post-arrival meandering, which raises stop-belief, which slows it further.

**The weights are not globally damaged.** Offline on demonstration frames, ck2000 and
ck2600 command normal motion (6.27 / 6.70 against ground truth 6.58) and flow RMSE is
flat. The attenuation exists ONLY on the policy's own rollout distribution.

## Ruled out

* **Prompt-format mismatch.** v8 trains on `"Observation:"` while the harness renders
  `"Observation {step}:"`. A paired forward-pass discriminator over four checkpoints found
  a format effect of −0.05 to +0.15 on chunk magnitudes of ~6.5, with no growth over
  training.
* **Pose injection.** Paired on one fixed checkpoint (ck1800, pose on vs off, all else
  identical): ±1.3 points, p ~ 0.74. Not the cause of any of the deficit. See the
  correction in `SAMPLE101_EVALS.md`.
* **The world-size probe-loss inflation** (`_probe_ddp_scale`). Confirmed working and
  active in v8, from the run's own blended `eval_loss` matching `flow + 0.25 x stop`
  rather than `flow + stop`. It IS part of v4's story: v4 ran at effective ~0.4 probe
  pressure and sat ~5 points below v3 by ck1000 -- a dose-response across three runs.
* Mixture/task-balance shift, escapes and mesh holes, LR schedule -- all flat or identical
  to v3, which ran 12000 steps without this.

## Addendum (2026-08-24): what the v9 restart taught us

**Full detachment makes the head drift.** v9 ran `--probe-grad-scale 0`: eval stop loss
rose monotonically (0.292 -> 0.330 by step 800), recall collapsed (0.718 -> 0.559), and
`pred_rate` fell BELOW both components' base rates and kept falling. With the backbone
deaf to the stop objective, action training moves the representation under the head, and
a 2-layer MLP cannot track it. The head needs a small say in the representation; v10 runs
at grad_scale 0.1 (a true 0.1 -- the world-size inflation is fixed).

**Detachment alone does not decouple the objectives, because the CLIPPER still sees
both.** Even at grad_scale 0, the probe head's own parameters enter the single global
`max_grad_norm` clip; in v9 they drove pre-clip norms to 3.8 and scaled the action update
to ~0.26x on those steps -- v8's mechanism in a milder form, inside the run built to
remove it. Fix: `--probe-clip-norm`, which clips probe parameters SEPARATELY and excludes
them from the global norm (implemented in `train_flow_matching_sft_value.py`,
`_install_split_clip`). Two objectives, two budgets. The reported `grad_norm` stays the
primary group's, comparable to probe-free runs.

**The denominator fix is now implemented** (`_get_num_items_in_batch` override: only
action-bearing turns count), restoring the ~1.7x of action-anchor strength that
probe-only rows' turns divided away. It matters most the moment the head is re-attached,
because a diluted anchor is what a backbone-reaching auxiliary gradient overpowers.

**Changing pos_weight over a warm-started head causes a recalibration transient.** The
v8-ck800 head carries the +1.70-logit bias of pos_weight 5.45 training; at pos_weight 1.0
the negative-dominant gradient sheds that bias, and with one-row-per-rank batches (label
distributions alternate between all-negative rows and 93%-positive near-goal rows) the
descent overshoots. Expect the first few hundred steps of stop metrics after such a
change to be transient, and judge the head only after they settle.

v10 (`cotrain_v10_launch.sh`): grad_scale 0.1 + split clip + action-turn denominator +
pos_weight 1.0 + PointNav stop supervision off. First measured steps: primary grad norm
0.41-0.75 (v3 territory, unclipped) with the probe's own clip engaging at its separate
1.0 -- both guards doing their jobs simultaneously.

## What to do instead

* **Detach the head.** `--probe-grad-scale 0` makes the probe loss reach only the head's
  own parameters (proven exactly zero into the backbone by
  `tests/test_gradient_isolation.py` I6). `stop_head.py`'s own motion-stop head has
  carried a `stop_grad` mode for exactly this reason since it was written: "no weight of
  this loss can damage the motion objective". The head reads a probe hidden and is two
  linear layers -- it is not obvious it needs backbone gradient at all.
* **Fix the denominator** so only action-bearing turns count
  (`num_turns` where `action_weight > 0`). Otherwise any future run mixing probe-only rows
  into a token-normalised loss inherits the dilution, scaled by how many turns those rows
  carry -- and on-policy rows carry 175 against ObjectNav's 48, so they dominate far more
  than their row share suggests.
* **Watch the raw gradient norm against `max_grad_norm`.** v3 never clipped; v8 clipped
  every step. A co-training run whose grad norm jumps 8x over its warm start is not
  running the recipe that produced the warm start, whatever the flags say.
* **Offline loss will not warn you.** Flow RMSE was flat while oracle success fell 20
  points. Only closed-loop rollouts detect this.
