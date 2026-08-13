# The pre-RL bake: recipe, evidence, and what to watch

Status: recipe fixed, not yet launched. Branch `latent_rl`. Companion to `LATENT_RL.md`
(the design) and `LATENT_RL_ENV.md` (the RL environment and policy head).

The pilot is `dump/pose_injection/run_reinitdiv_clean` (1500 steps, frozen trunk). This
document is the long run built from it, and — equally important — the list of things that
were investigated and deliberately NOT changed.

## What the evidence actually supports

Every number below is on the fixed 101-episode sample101 set at a 70 s budget unless said
otherwise, paired against the deterministic baseline (v3: oracle 0.663, oSPL_fix 0.348).

| intervention | behavioural spread ratio |
|---|---|
| baseline (deterministic checkpoint, latent bolted on) | 0.028 |
| free bits + KL warmup | 0.031 |
| auxiliary `c -> chunk` regression | 0.032 |
| **+ decoder re-initialisation** | **0.139** |
| reinit + aux + free bits | 0.162 |
| **reinit + diversity** (the pilot) | **0.854** |

So free bits, KL warmup and the auxiliary loss each bought **nothing** on their own.
Re-initialising the velocity field bought 5x at unchanged reconstruction, and the diversity
term multiplied that. Everything else in the search was noise.

**The attribution result is the one that justifies the whole programme.** Three runs sharing
a common `z_0` stream with different `c` streams, over the 48 mid-dispersion episodes:

* within-episode return variance, `z_0` common (only `c` varies): **0.0844**
* within-episode return variance, `z_0` free: 0.1029
* => **`c` accounts for 82% of within-episode variance**
* oracle: mean draw 0.611, **best-of-3 under `c`-only selection 0.875**
* deterministic floor (c at mean, same `z_0` stream): 0.695 mean return, so simulator
  nondeterminism is not inflating this

Good actions demonstrably exist **inside the current exploration ball**. RL in `c`-space is
therefore a local selection problem, not a discovery problem, and most of the risk between
here and the RL run is about not destroying that property.

The cost, stated plainly: at a single unselected draw the latent checkpoint scores 0.545 /
0.257 against 0.663 / 0.348, McNemar p = 0.058 and oSPL 95% CI [-0.159, -0.023].

## The recipe

```
--latent-cvae --latent-sigma0 0.002 --latent-beta 0.001
--latent-free-bits 0.01 --latent-kl-warmup 300
--latent-reinit-decoder
--latent-diversity 1.0 --latent-diversity-clamp 40.0
--max-steps 12000 --eval-per-component --eval-max-samples 64
(no --latent-freeze-trunk, no --latent-diversity-target, no --latent-sigma-floor)
```

Mixture unchanged from the baseline (`objectnav_nopose : objectnav_pose : pointnav = 1:1:2`),
so the comparison stays honest.

Changes from the pilot, and only these: **the trunk is unfrozen** (`h` was optimised as a
point predictor for a deterministic decoder and cannot co-adapt to carry a stochastic `c`
while frozen), **12000 steps instead of 1500**, **`--eval-per-component`** so the three
components are separable and comparable against the baseline run, **no diversity controller**
— it destroys the x-axis of the parity/spread trade curve, so we pin its saturation value
instead — and **no `--latent-sigma-floor`**, which at 0.001 sits below `sigma0 = 0.002` and
never binds once diversity is doing work.

### The diversity weight is 1.0, and reading it off a log gives 0.05

The first launch of this bake passed `--latent-diversity 0.05` and was killed at step 87. It
ran the term at **1/20 strength**: `sigma` was flat at ~0.002 where the pilot was at 0.20 by
step 80, and `sigma/h_std` was 0.005 against the pilot's 0.117.

The pilot's controller ramped the weight from its 0.05 seed to the `diversity_weight_max`
ceiling of **1.0 by step 148**, and held it there for 90% of the run. The training log prints
`div_w: 0.05` at that ceiling, because `sum_div_w` is written as `_div_w * N` and drained over
`n_rows = N * n_ticks` — the same x20 deflation every other latent metric carries (below).
`--latent-diversity` is the true weight, undeflated. Copying the logged number into the flag
costs exactly a factor of 20, and the failure is silent: every metric stays well-formed and
only the growth *rate* of `sigma` gives it away.

The deflation is deliberately **not** being fixed. The pilot's log is the only calibration
reference this run has, and renormalising mid-family would make the two incomparable for the
sake of tidiness — which this document already names as the more expensive mistake.

## What the diversity term is, precisely

```
E|| A(mu + sigma*eps_1) - A(mu + sigma*eps_2) ||  /  ||c1 - c2||_detached
```

Two draws from the prior, decoded under a **shared** `z_0`, maximise the spread of the
decoded chunks. The denominator is detached, so the gradient flows only through the
numerator: it is A-spread maximisation with per-sample normalisation, **not** a Jacobian
penalty.

That distinction is worth holding onto because an earlier version *was* Jacobian-based — a
fixed 0.001 probe along a unit direction — and it failed for a specific reason: at 1024 dims
the prior-scale separation is `sigma * sqrt(2048) ~ 22.6` against `||c|| ~ 50`, i.e. a ~45%
perturbation of the latent, and the decoder is measurably nonlinear over that ball (the tau
sweep is sub-linear: 10x tau buys 5.3x, then 2.6x). A 0.001 probe measured a local derivative
at a scale ~90x smaller than deployment. The current form degenerates into a Jacobian-norm
regulariser only in the limit `sigma -> 0`, which is precisely the state we must avoid.

### Why `sigma` is NOT detached, despite the term buying scale

At matched absolute perturbation the 30x spread gain decomposes into ~13x exploration-scale
inflation and ~2.3x genuine decoder responsiveness. Detaching `sigma` in the draw would point
the term at responsiveness alone — and would be a mistake here, because **the KL is inert**
(below) and reconstruction is the only other force on `sigma`, pointing down. The diversity
numerator is the only thing holding `sigma` up. Detach it and `sigma` drifts to zero over
12000 steps, with two consequences: the decoder is trained at `c ~= mu` and so learns no
noise robustness, meaning tau-scaled exploration at RL time decodes off-distribution; and the
detached denominator shrinks with `sigma`, the ratio spikes past the clamp, and the diversity
gradient silently zeroes.

## This is not a CVAE, and the framing has consequences

Measured on the pilot: `kl_raw` peaked at ~0.6 nats against a free-bits allowance of
`1024 x 0.01 = 10.24` nats, so **the KL contributed essentially zero gradient for the entire
run**. Below the free-bits floor a dimension is charged a constant and receives no gradient
at all, so the posterior, `beta`, free bits and the warmup were decorative. `delta_mu` was
negligible on essentially every dimension.

What we have in function is a **deterministic regressor with conditioning-noise injection
plus a finite-difference sensitivity regulariser** — equivalently, an SFT-pretrained
diagonal-Gaussian policy over a learned 1024-d action embedding.

That is idle for the RL math (`pi(c|o) = N(mu, tau^2 sigma^2)` has an exact log-prob whatever
produced `sigma`) and load-bearing in four places:

1. **`sigma` is a knob, not a measurement.** Its magnitude *and* its per-dimension shape are
   artifacts of the reconstruction-versus-diversity balance. Do not read its anisotropy as
   information.
2. **The "mean decode is safe because `c` carries the mode" argument fails** — `c` does not
   carry the mode. Whether mean or sampled decoding is better is an empirical question per
   checkpoint (on the pilot: indistinguishable, p = 1.000, but 21% of episodes flip).
3. **Do not interpret `c` semantically.** That the `c`-ball spans success and failure is all
   RL needs; it is not evidence of intent disentanglement. Any reading of `c`-directions as
   "behaviours" is reading the regulariser's fingerprint.
4. **`kl_raw` becomes a train/deploy mismatch gauge.** Today the gap is zero because the
   posterior is dead. The bake makes waking it more likely (unfrozen trunk, 8x the steps,
   free bits letting `delta_mu` grow to the floor free of charge). If it wakes, the decoder is
   calibrated on `q` while RL samples `p`. Watch it as a mismatch signal, not a compression
   signal.

**The CVAE machinery is retained anyway.** It is inert, it costs little, and the pilot is the
only configuration with end-to-end evidence behind it. Stripping it is tidiness, and tidiness
has been the more expensive mistake in this work.

## `sigma` does not converge — it is a relaxation oscillator

Observed on both runs, and the single most important dynamical fact about this objective.
The bake took `sigma` from 0.0125 to **79.4** and back to ~1.4 in about 40 steps; the pilot
peaked at **3541** at step 688 and returned to 0.663 by 1500. Step-to-step swings of 10x
persist afterwards (over 40 consecutive steps: min 0.457, max 79.4, median 2.33).

**Why the long flat phase then the explosion — they are one process.** The term is
`D = ||A(mu + sigma*e1) - A(mu + sigma*e2)|| / ||sigma*(e1-e2)||_detached` and the parameter
is `log sigma`. While the decoder is locally linear over the ball the numerator is
`~ sigma*||J de||`, so:

* `D ~ ||J de|| / ||de||`, **independent of `sigma`** — which is why `dadc` sat pinned at 0.36
  for 60 steps looking inert;
* `dD/d log sigma = sigma * dD/d sigma = D` — **a constant**.

A constant gradient on `log sigma` makes `log sigma` grow linearly, i.e. `sigma` grows
*exponentially* at a rate proportional to `w * lr`. Nothing triggers at the moment of
"explosion"; it was always exponential and only became visible on a linear axis. Two
corollaries that already cost time: at 1/20 weight the exponent is 20x smaller (which is what
the mis-specified first launch looked like), and two exponentials at different rates diverge
without bound even when plotted against *cumulative LR* — so a rate difference cannot be
distinguished from a schedule difference by curve-matching.

It ran faster than exponential because `D` itself rose 0.36 -> 1.70, and `D` is the gradient:
a positive feedback loop. It terminates when `sigma` is large enough that the decoder
saturates over the ball, the numerator stops tracking `sigma`, and `D` collapses (1.70 ->
0.0002). Reconstruction then pulls `sigma` back. Hence oscillation, not convergence.

**Nothing in the objective bounds `sigma` from above.** Two guards that look like they would
and do not:

* `kl_shared_sigma` is `sum dmu^2 / (2 sigma^2)` — monotonically *decreasing* in `sigma`. The
  KL does not restrain growth, it mildly rewards it. (It is inert anyway under free bits.)
* `--latent-diversity-clamp` clamps the *ratio*. It does NOT bind while `sigma` is large (the
  ratio is smallest exactly then), but it **does** bind during the sensitivity spike that
  precedes takeoff: in v1 at step 76, raw `dadc_clipfrac` was 0.02727 against the 0.05
  ceiling, i.e. **54.5% of pairs clipped**, over a narrow window (steps ~70-86). An earlier
  version of this document claimed the clamp "cannot bind" on the strength of sampling every
  ninth step and seeing zeros. It binds, hard, and for clipped pairs the diversity gradient is
  exactly zero — so the clamp is a partial brake acting at the spike, not an inert guard.

Reconstruction is the only opposing force and it only acts after the excursion.

### Correction: what actually drives the takeoff (external review, verified in code)

The "constant gradient on `log sigma` => single-rate exponential" story above is geometrically
right but **dynamically wrong, refuted by our own numbers**: induction runs at 0.018 nats/step
(0.002 -> 0.0125 over ~100 steps) and takeoff at 0.49 nats/step (0.0125 -> 79.4 over 18), a
**27x acceleration**. No single rate fits both, and the LR ramp cannot close the gap. Three
corrections, each checked against the source:

1. **`h` is NOT detached in the diversity block** (`hf = context.float()`,
   `flow_matching_head.py:1104`). The diversity gradient therefore flows into the backbone,
   which can satisfy the term by moving `h` — not merely by growing `sigma`.
2. **The decoder LayerNorms the context.** `ctx_tok = context.view(N, C, d)` is a reshape with
   no projection (`:665`) into a `norm_first=True` encoder (`:578`), so the first operation
   touching a context token is a LayerNorm. The decode is therefore ~invariant to token
   *scale* — which is why rmse held flat through a 60% `h_std` collapse — but a perturbation
   `sigma*eps` arriving through that LayerNorm has effective size `~sigma / std(token)`.
   **Shrinking `h` at fixed `sigma` inflates `D` mechanically, with no decoder weight change
   at all.** So the `h_std` decline is not drift along a direction reconstruction is
   indifferent to; it is a direction the objective actively rewards and reconstruction cannot
   see. `sigma/h_std` is not merely the convenient invariant — it is the only scale the
   decoder can physically perceive.
3. **The rate is set by the log-sigma head's fan-in, not by `w * lr`.** Under Adam a
   persistent gradient saturates near +-`lr` per parameter, so the bias alone gives 1e-4
   nats/step — 5000x too slow for the observed takeoff. The observed rate is reachable only
   through `to_log_sigma.weight`'s 1024 coordinated entries projected back through `h`. `w`
   enters through the Adam SNR (signal ∝ `w`, noise largely `w`-independent), which is why a
   20x smaller weight still produced a ~20x slower takeoff even though Adam normalises
   magnitude away.

The overshoot is **integrator windup into a dead zone**: past saturation the diversity
gradient vanishes, and at very large `sigma` the reconstruction gradient does too (the flow
loss plateaus at its unconditional value), so `log sigma` coasts on momentum. Predicted lag of
10-20 steps from D-crash to `sigma`-peak; **measured 20 in v1**. There is no universal
overshoot constant — it is `exp(slope x lag)`, and both are history-dependent.

**`rmse` never sees `sigma`.** `metrics_ctx = latent.decode(pieces["mu"])` (`:1072`) — metrics
report the deployed mean path. So "reconstruction pulls `sigma` back" is inferred, not
observed; the restoring force lives in `fm_loss_sum`, which is the series to read.

**`z_0` is shared across the whole batch**, not just within a pair (`:1115-1116`,
`randn(1,T,D).expand(N,...)`). Sharing within the pair is the point; sharing across rows is
incidental and means the diversity gradient's `z_0` noise does not batch-average. Per-row
`z_0` (still shared within the pair) is free variance reduction — but by (3) it would *raise*
the Adam SNR and therefore **speed up** the takeoff, so it must not be done before the damping
is in.

### Consequences, and levers if this needs fixing

`sigma` is the RL exploration scale (`sigma_rl = tau * sigma_p`), so an oscillating `sigma`
means **checkpoint selection samples a phase, not a value**. The pilot's ck1000 had
`sigma = 1.80` and ck1500 `0.663`, a 2.7x difference — very likely why measured spread peaked
at ck1000 while `dA/dc` rose monotonically. **Record `sigma` alongside every scored
checkpoint**; a parity or spread number without it is not interpretable.

Not acted on for this bake (it is one step from the validated pilot, and the pilot shipped a
usable checkpoint through the same oscillation). Ordered by how surgical:

1. **Record and select on `sigma`** — zero risk, do this regardless.
2. **A `sigma` ceiling**, mirroring the existing `sigma_floor` clamp in `LatentSplit` — one
   line, bounds exploration scale without touching the objective's shape.
3. **Lower LR or an EMA on `to_log_sigma`** — damps the oscillator, objective unchanged.
4. **Decouple responsiveness from scale.** The root cause is that one term sets both. Measure
   spread at a *fixed absolute* perturbation so the term trains decoder responsiveness only,
   and set `sigma` by a separate explicit rule. This is the principled fix and the biggest
   change.
5. **A KL that bites** — with shared `sigma` it cannot. A non-shared `sigma_q`, or a prior at
   fixed scale, restores a real penalty on growth.

## Risks recorded but not acted on

Raised in review, deliberately left alone so the bake stays one step from the validated
pilot. Each needs a decision only if its monitor fires.

1. **Fixed diversity weight over a run where reconstruction falls ~10x.** The term's share of
   the objective drifts. This is why the controller existed; the controller is worse.
2. **A new gaming path through the unfrozen trunk.** `W_mu` is identity-initialised, so
   nothing separates `h`'s scale from `c`'s: the term could satisfy spread by inflating
   `h_std` rather than by improving the decoder. Monitor `h_std_perdim`.
3. **Reinit decoder plus unfrozen trunk from step 0** sends random-decoder gradients through
   the readout into a converged backbone. Staging the unfreeze would avoid it.
4. **Diversity is applied uniformly across the 1:1:2 mixture**, but PointNav's conditional
   given the goal is near-deterministic. Forcing sensitivity there may cost precision on half
   the data. `--eval-per-component` is what will show this.
5. **No pilot calibration transfers.** `tau = 2`, the spread ratios, `sigma/h_std`, and the
   mid-dispersion 48 are all properties of the pilot checkpoint. Re-measure on the bake.

## Monitors, and what each one means

**Every threshold below is in TRUE units, i.e. the logged value times 20.** An earlier version
of this table stated four of them against logged values and one against a number the metric
cannot reach; all five are corrected here.

| quantity | healthy (true units) | what a move means |
|---|---|---|
| **`sigma / h_std_perdim`** | **0.4-0.6** | **the only gauge-invariant scalar here, and the one to watch.** It is the latent's noise-to-signal: the observation's share of `c`'s variance is `1/(1+r^2)`, so r=0.7 is 67%, r=1.4 is 34%, r=2 is 20%. Multiply by `tau` for what RL actually sees |
| `delta_mu_over_sigma` | <~0.15 | **the train/deploy mismatch gauge** — how far `q` sits from `p` in units of sigma. Use this, NOT `kl_raw` |
| `kl_raw` | ~10-30 nats total | free bits is a **per-dimension** 0.01-nat floor, so the "10.24-nat allowance" is just 1024 dims sitting on it. Read `kl_raw/1024` against 0.01: the pilot ran 0.011, i.e. marginally active. **Not** a mismatch gauge |
| `dadc_clipfrac` | ~0 | **maximum possible logged value is 0.05** (a count over N turns drained by N*20). 0.05 = fully clipped. A table saying "1.0 = clamp binds" describes an unreachable state |
| `div_w` | equals `--latent-diversity` | logged value is the true weight / 20. Movement means the controller is on; it should not be |
| `h_std_perdim` | **not a gate** | gauge-dependent: `h -> h/k` with a `k`x sharper decoder is behaviourally identical, and v1 fell 60% with rmse flat. Only its ratio to `sigma` means anything |
| per-component eval | all three tracked | a component regressing under a blended number is the failure `--eval-per-component` exists to catch |

### v1 outcome: stopped at step 752, and what it taught

Run `run_bake_latent_v1` (`--latent-diversity 1.0`) was stopped because **all twelve
per-component eval metrics regressed monotonically and were accelerating** (e.g. pointnav
`mae_dtheta` 0.0204 -> 0.0248 -> 0.0307 at steps 250/500/750), while the spread probe on its
ck500 read far above what we need. Probe, identical settings, 32 observations x 16 samples on
`formatted_pose` validation:

| checkpoint | ratio vs noise (tau=1) | tau=2 | vary-`c` head std | noise-arm head std | `c` share of heading variance |
|---|---|---|---|---|---|
| **bake v1 ck500** | **1.920** | 2.232 | 0.348 | **0.181** | **78.7%** |
| pilot ck1000 (shipped) | 0.906 | 1.243 | 0.305 | 0.336 | 45.1% |
| pilot ck1500 | 0.830 | 1.193 | 0.279 | 0.336 | 45.1% |

Two things worth keeping. First, the pilot ck1000 measurement (0.906) is close to the 0.854
recorded above, so probe settings are comparable across runs and this table can be extended.
Second, v1's gain came from **both** arms — more variation from `c` and roughly half the
variation from flow noise. The second is the RL-relevant one, since `z_0` is environment noise
the policy cannot select on.

**The lesson is that the pilot's weight was not transferable.** The pilot ran frozen, where the
only route to spread was growing `sigma`. Unfrozen, the gauge `h -> h/k` with a sharper decoder
is a second and cheaper route that reconstruction cannot see, so the same weight buys about
twice the spread — and charges precision for it. v2 runs at 0.3.

**Every latent metric in the training log is 20x too small.** They are per-turn sums divided
by `n_rows = turns * n_ticks`. An earlier conclusion that `sigma` was 6-120x `h_std` came from
compounding that with a second error (reading `h_std` off the same log rather than measuring
it: it is 1.58 on real states, not 0.107). Measured directly, `sigma` is 0.4-0.6x `h_std` at
every checkpoint. Fix the normalisation or apply the factor by hand, but do not read these
raw.

## Evaluation hygiene

* The 101-episode sample101 set must stay **out of RL training**.
* **Never report gains on the mid-dispersion 48.** That set was selected on outcome variance
  and is a diagnostic, not a benchmark.
* `dA/dc` is **not** a proxy for the behavioural gate: it rose monotonically through
  checkpoint 1500 while measured spread peaked at 1000. Score checkpoints with the spread
  probe and a parity eval, not with the training metric.
* Report parity at `--latent-mode mean` and at `sample`; they measure different policies and
  on the pilot they were indistinguishable.

## Sequence

1. Bake (12000 steps, ~30 h on 4 GPUs).
2. Spread probe at checkpoints 4000 / 8000 / 12000, at native and matched perturbation.
3. Parity eval (sample101, `--latent-mode mean`, 175 steps, `--navmesh dataset`) on the best.
4. Re-run the attribution with common random numbers (`--policy-seed` common, `--latent-seed`
   varied) on the bake, since no pilot calibration transfers.
5. Only then RL.
