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

| quantity | healthy | what a move means |
|---|---|---|
| `kl_raw` | stays ~0.5 nats | a rise = the posterior woke; train/deploy mismatch |
| `sigma` (x20 the logged value) | 0.4-0.6x `h_std` | collapse = no exploration; inflation = term buying scale |
| `h_std_perdim` | stable | rising = the term is inflating `h` instead of improving the decoder |
| `dadc_clipfrac` | ~0 | 1.0 = the clamp binds, diversity gradient is a constant, term is dead |
| `div_w` | fixed at 0.05 | any movement means the controller is on; it should not be |
| per-component eval | all three tracked | a component regressing under a blended number is the failure `--eval-per-component` exists to catch |

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
