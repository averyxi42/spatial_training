# held128: results at cycle ~400, and the offline variance/mastery analyses

**2026-08-16.** Findings from `flow_sde_a09_held128` (the first run with a genuinely
held-out eval set -- see `TRAINING_EVAL_SET.md`) at ~400 cycles / 102 complete eval
passes, plus the GPU-free analyses run alongside it. Analysis scripts:
`dump/eval_system/analysis_2026-08-16/`. All rollout-log numbers come from
`dump/flow_rl/<run>/rollout/*/*/{summary,sequence}.json`.

## 1. Held-out eval: efficiency is real, success is not (yet)

26 pinned episodes, ODE, one pass per 4 cycles, reconstructed by per-episode
occurrence index (each episode's k-th serving = pass k; no snapshot selection --
every statistic uses every pass in its window). Series mean success 0.679, pass sd
0.0587 against the fixed-checkpoint floor of 0.0448.

| metric | slope/pass | naive t | Newey-West(L=10) t | block-bootstrap p | pass-level adj. t |
|---|---|---|---|---|---|
| oSPL | +0.00069 | +3.86 | **+5.06** | **<0.0005** | +3.49 |
| steps | -0.094 | -3.31 | **-3.59** | **0.0005** | -3.15 |
| success | +0.00020 | +1.02 | +1.33 | 0.20 | +0.85 |

Thirds (33 passes each): oSPL 0.392 / 0.430 / 0.441; steps 93.0 / 88.9 / 86.3;
success 0.669 / 0.681 / 0.685. Paired per-episode half2-half1: oSPL +0.043
(t=+3.40, 17/5 episodes up), steps -6.4 (t=-3.85), success +0.009 (t=+0.62).
Lag-1 autocorrelation is mild (+0.17 oSPL/steps, ~0 success), so the naive tests
were not badly inflated; the robust treatments agree.

**The success null now has power.** At 102 passes the paired half-split se is
0.0136, so a ck391-sized +0.05 effect would have shown at t~3.7. Observed +0.009;
2-se upper bound ~+0.036. The run has bought efficiency on never-trained episodes
(last-third oSPL 0.441 vs SFT baseline 0.382 +- 0.052), not success. Individually
moved episodes match the reward-horizon story: two mid-range episodes converted up,
the one far episode (12.9 m tv_monitor) degraded 0.36 -> 0.15.

**Pre-registered decision rule** (fixed before looking again, because both prior
over-readings came from choosing thresholds after seeing the curve): at ~160
passes, read the Newey-West t on the success slope. >=2: the run is buying success,
keep going. <=1: efficiency-only; bank the result and pivot to the value co-train.
Between: one more day, once.

## 2. What "faster" is, mechanically

Within-episode late-early differencing on training servings (identity fully
removed; median 67 servings/episode): success +0.036 (t=+4.13), steps -5.3
(t=-4.42), return +0.29 m (t=+3.03), min_m -0.092 (t=-2.23).

Speed decomposition (earlier snapshot at ~15 servings/ep, all still consistent):
steps migrated out of settle (<0.05 m/step, -1.5 pp, t=-2.5) and creep
(0.05-0.35 m, -0.8 pp, t=-3.0) into cruise (>=0.6 m, +2.2 pp, t=+3.4); the p95
step displacement did not move. The displacement distribution has a hard rail at
~0.71 m/policy-step (~1.78 m/s executed; the 2 m/s corpus cap = 0.8 m/step is
never touched -- max observed 0.768 over 134k steps, zero exceedances). No speed
cap is needed; nothing approaches the existing one.

## 3. Offline variance studies (all from existing rollout logs)

Scripts: `speed_gamma_analysis.py`, `bandwidth_sweep.py`, `buffer256_sweep.py`.
Metric: corr(r_t, A_t) -- the fraction of the whitened advantage that is per-step
mechanical credit; since whitening fixes gradient scale, the noise-channel SNR
scales with it and (corr_ref/corr)^2 is the batch multiplier to hold per-step SNR.

* **Gamma dilution** (1,440 a=0.9 servings; 5-fold CV by episode; reproduced at
  the real procedure -- 256-episode buffers, in-sample fit, as
  `run_rollout_cycle` does; identical numbers, the 1-D kernels saturate long
  before 256 episodes):

  | gamma | time kernel | distance kernel | batches vs current |
  |---|---|---|---|
  | 0.95 | corr 0.402 | 0.434 | 1.00x / 0.86x |
  | 0.97 | 0.314 | 0.352 | 1.64x / 1.30x |
  | 0.99 | 0.190 | 0.234 | 4.5x / 2.9x |
  | 1.00 | 0.102 | 0.148 | 15x / 7.4x |

* **Bandwidth is a non-factor.** Sweeping sigma over two orders of magnitude moves
  CV Var(A)/Var(R) by <2% within either family; the config's effective defaults
  (~6.8 steps, 0.5 m) were already each family's optimum. The family ordering
  (distance > time at every bandwidth and gamma) is not a bandwidth artifact.

* **The distance-kernel estimator dominates the time kernel everywhere** (12%
  cheaper at gamma 0.95, ~30% at 0.99) and is already registered
  (`reinforce_plus_plus_distance_kernel`, sigma 0.5). Legitimate baseline:
  d_t is pre-action state (privileged info is allowed in baselines). The 2-D
  time x distance kernel removes more total variance (0.84 at gamma 0.99) but its
  corr(r,A) equals distance-alone -- the extra removal is orthogonal to the
  per-step channel.

* **Ceiling of all static baselines:** Var(A)/Var(R) >= 0.85 at gamma 0.95 even
  for the 2-D kernel. The residual is per-episode divergence, reachable only by a
  state-conditioned value function.

* **Grouping (GRPO-style per-episode time-aligned baseline)** -- `grouping_sweep.py`
  (full-LOO, upper bound) and `grouping_clean.py` (split-peer, uncontaminated: the
  naive corr(dr,A) shares peer noise between baseline and metric and overstates
  grouping): at the current pool (k~2 servings in buffer) grouping *hurts*
  (Var(A) +48%); it breaks even at pool <=32 and only leads on total variance at
  pool ~16 stacked on distance (0.69 vs 0.83 at gamma 0.99) -- while **never**
  beating distance-alone on the clean per-step metric at any k. Mechanism: the
  episode "fixed effect" mostly is not fixed; what episode identity predicts,
  distance already carries, and the rest is divergence that does not transfer
  between servings.

## 4. Mastery: variance collapse through policy convergence

The one mechanism that demonstrably removes divergence variance. On the 24-episode
run (`flow_sde_freezehead_a09`, the ck391 run, ~398 servings/episode),
serving-order quartiles: success 0.839 -> 0.900 (t=+4.14), within-episode
Var(return) 9.89 -> 4.53 (t=-3.81, 21/24 episodes down), Var(success) 0.085 ->
0.048 (t=-4.45). Roughly a halving of the advantage-noise floor -- worth ~2x
effective batch at fixed whitened gradient scale, and it compounds.

held128 has now entered the same regime: at median 67 servings/episode,
within-episode Var(success) -0.018 (t=-3.47), Var(return) Q1->Q4 11.15 -> 8.98
(paired t=-2.45, 63/33 down). At ~15 servings it was unresolved (t=-0.77) -- the
effect needs repetition depth, as predicted.

**Pool-size rate comparison (matched depth, bins of 17 servings/episode):**

| run | servings 1-17 | 18-34 | 35-51 | success rate/serving |
|---|---|---|---|---|
| pool24 Var/Var0 | 1.000 | 1.290 | 0.781 (t=+0.46 at bin4) | +0.38e-3 |
| pool128 Var/Var0 | 1.000 | 0.811 | 0.704 (t=-2.34 at bin3) | +1.44e-3 |

At matched servings-per-episode, the 128-pool run is mastering *at least as fast
per serving* as the 24-pool run did -- nominally ~4x faster on success, and the
24-pool showed no resolvable variance decline at this depth at all (its collapse
happened late, servings ~300+). **Do not lean on the ratio**; see the caveats in
the analysis note below. What it does support, weakly: no evidence of per-serving
interference from the larger pool.

Caveats on the cross-run comparison, in order of severity: (1) the a09 rollout
directory interleaves ODE eval passes with training servings with no arm marker --
on a 24-episode pool eval runs every other cycle, so the contamination is large,
flattens its serving-order trends, and could alone explain its flat early bins;
(2) ceiling: pool24 started at success 0.84 vs pool128's 0.70 -- room-to-improve
differs by ~2x and "rate" is not comparable across ceilings; (3) possibly
different base-model era (pre/post merged base -- unverified); (4) different
episode/scene composition; (5) two pool sizes, one run each -- no exponent is
fittable. The arm-marker instrumentation (queued with the noise-corr logging)
removes (1) for all future runs.

## 5. What follows

1. Success verdict per the pre-registered rule (~160 passes).
2. Estimator switch to `distance_kernel` on any next run: free 12-30%.
3. Gamma falsifier, if run: distance kernel + n_rollout 16 -> ~48 at 0.99
   (or 0.97 at ~1.35x), else it confounds horizon with a drowned channel.
4. Value co-train: the distance-to-goal annotation tooling is built
   (`habitat_physical_nav scripts/annotate_distance_to_goal.py`, sidecar design,
   navmesh choice must match the env: current lineage is `dataset`). Offline gate:
   held-out EV must beat the distance kernel on existing rollouts before entering
   a run at GAE lambda=1.

## Verdict at the pre-registered 160-pass point (2026-08-16, cycle 648)

The rule fired the CONTINUE branch, against the analyst's stated expectation: success
slope Newey-West t = **+2.84** over 162 passes (rule: >=2 continue). Success half1
0.679 -> half2 0.691, last-20-pass mean **0.706**; oSPL still rising (NW t +4.92,
last-20 0.452); steps still falling (t -3.34). The "plateau" read at 130 passes was
premature -- passes 130-162 resumed climbing, consistent with lagged conversion
(efficiency gains converting to success within budget). Training side still
accelerating (within-episode success t +5.1, speed t +7.3). Run continues toward the
deep-mastery endpoint; next scheduled read at ~cycle 1150 (~1 day) with the same
robust-trend machinery. Curriculum run deferred while this curve is still paying.
