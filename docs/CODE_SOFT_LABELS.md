# Soft code labels: putting the metric back into the bins

Status: **built (2026-08-30), not yet run.** `--code-label-sigma`, `--code-label-metric` on
`data_scripts/train_flow_matching_sft_code.py`; `code_distance_matrix` / `soft_code_targets`
in `longnav/utils/code_conditioned_head.py`; `tests/rl_math/test_code_soft_labels.py`.

## The problem it addresses

The code head is a 1,600-way categorical over cells that partition a **metric** space (a
0.8 s chunk of body motion), but the objective was a one-hot cross-entropy, which treats
the cells as **nominal**: a prediction one 4-degree cell off the true heading is penalised
exactly like one 90 degrees off. Measured consequences on `run_code_v4_mlp_warm`:

* ObjectNav strict θ accuracy 0.256 against a 0.14 majority baseline, while within-one-cell
  is 0.43 -- most "errors" are neighbours;
* the head's probability mass fragments across metric neighbours it cannot separate from
  one frame (prediction entropy 3.4 nats, ~30 effective codes), which is what `sample`
  executes and what `argmax` breaks ties over.

"Metric vs bins" is the right description. The fix keeps the bins as the support (the
categorical, and with it sampling for RL) and puts the metric into the **target**: the
label for a true code `c*` is a Gaussian kernel over cells,

    q(c) ∝ exp( − d(c, c*)² / 2σ² ),        loss = CE(p(·|h), q)

and `σ → 0` recovers the one-hot exactly (the tests pin this). This is the histogram-loss
/ HL-Gauss construction (Imani & White 2018; Farebrother et al. 2024), which gives
regression-like gradients while retaining a full distribution.

## The distance: per-tick, in the flow head's own units, equal tick weights

`d(c, c′)` is computed once from the frozen tokenizer (`code_distance_matrix`):

1. decode each cell's **centroid chunk** -- the physical anchor-relative `(T, 3)` chunk the
   flow decoder is asked to reproduce for that code;
2. take **per-tick body-frame differentials** with `decompose_chunk` -- the same quantity
   the vanilla flow loss regresses;
3. divide by `action_scales = (0.03, 0.03, 0.05)` -- the same exchange rate between
   translation and rotation the vanilla head was trained under (3 cm per tick weighs as
   much as 0.05 rad per tick; at the chunk level ~1 rad ≈ 0.6 m);
4. RMS over **all ticks and dims with equal weights**.

### Why per tick and not per trajectory

A trajectory-level distance (endpoint, or cumulative pose over the chunk) is dominated by
the tail: cumulative magnitude grows ~19x from tick 0 to tick 19, and the tail is exactly
the part of the chunk the harness does **not** execute (`gap = 10` of 20 ticks, then
replan). The tokenizer's own FSQ geometry was fit on cumulative chunks and inherits that
bias -- its θ cells are 4° apart in *final* heading and 20 pairs of cells differ only in
turn-profile shape. Differentials remove the bias by construction: each tick's error is
its own term, no tick accumulates the others.

### Why equal tick weights, not a prefix emphasis

The vanilla flow loss weights ticks equally and produced a policy that navigates, so equal
weights are the known-good reference. The bar is "do not accidentally favour the tail",
and per-tick differentials already meet it; an explicit prefix weighting was measured
(prefix-only, prefix 1.0 / tail 0.4) and barely moved the neighbour structure of the 1,600
centroids, so it buys nothing and adds a knob. There is deliberately no tail-weight option.

### What the metric says about the codebook (dual_fsq_40x40, `dump/tokenizer`)

Under this metric, on the 1,600 joint centroids (`code_distance_matrix`, body-frame
differentials via `decompose_chunk`):

* median nearest-neighbour distance **0.19** units (p10 0.14, p90 0.24); a θ step at fixed
  xy is 0.20, an xy step at fixed θ is 0.24 -- under the vanilla exchange rate the two
  factors are resolved about equally, i.e. the codebook is not lopsided in these units;
* the θ "shape twins" (same final heading, different timing) are **not** neighbours: a turn
  at tick 3 and one at tick 15 differ at every tick. They only looked interchangeable
  under the endpoint metric. The per-tick metric is the honest one;
* the stationary xy cell's nearest non-stationary neighbours (the 0.2--0.3 m creep cells)
  sit at ~0.22 units, so a kernel of that order puts real mass on stationary ↔ creep --
  the transition the policy has to make to leave a look-around;
* cells within 0.2 units of a typical cell: ~1; within 0.3: ~7.

Effective support of the target (exp of its entropy) by σ, measured on this matrix:

| σ | eff. support (cells) | mass on the true code (median) |
|---|---|---|
| 0.05 | 1.1 | 1.00 |
| 0.08 | 1.7 | 0.89 |
| **0.10** | **3.3** | **0.69** |
| 0.12 | 6.5 | 0.49 |
| 0.15 | 16 | 0.27 |
| 0.20 | 50 | 0.10 |

(An earlier per-factor, anchor-frame analysis quoted spacings of 0.3--0.4; the joint
body-frame matrix the loss actually uses is tighter, and these numbers supersede it.)

## Configuration

| flag | default | meaning |
|---|---|---|
| `--code-label-sigma S` | `0.0` | kernel width in metric units. `0` = one-hot CE, bit-identical to every earlier run. Start at **0.10** (~3 cells, true code keeps ~0.7 of the mass); sweep 0.08 / 0.10 / 0.12. Above ~0.15 the target is spread over 16+ cells and the true code is no longer dominant. |
| `--code-label-metric M` | `tick_diff` | `tick_diff` as above. `grid` = FSQ lattice L1 (what `code_within1_*` reports), a non-metric **control**, not a recommendation. |

## Metrics, and how to read a soft-label run

The pre-existing suite measures the head on the **grid** (`code_acc*`, `code_within1_*`,
`code_grid_l1_*`, `code_top5`) plus its shape (`code_pred_entropy`, `code_confidence`,
`code_pred_used` / `code_pred_perplexity` / `code_pred_top1_share` for collapse). None of
those can see the quantity a soft label optimises, so a soft-label run can lose strict
accuracy while improving, or keep it while scattering mass. The following are logged for
**every** code run with a tokenizer (σ = 0 included), so a soft run has a hard baseline:

| metric | what it is | how to read it |
|---|---|---|
| `code_ce` | **hard** CE against the one-hot, unchanged | the one number comparable across σ; expect it to *rise* slightly under soft labels (the head is no longer asked to put all mass on one cell) |
| `code_soft_ce` | the trained objective, `CE(p, q)` | must fall; `code_soft_ce − H(q)` is the KL to the target and is the honest training loss |
| `code_target_entropy` | `H(q)` in nats, ≈ constant per σ | sanity: `exp` of it is the effective support (0.10 → ~3.3 cells). If it drifts, the tokenizer or `action_scales` differ from what σ was chosen for |
| `code_top5` | unchanged definition | expect a slight DIP under soft labels (measured at ck1800: objnav 0.377 vs v4's 0.389, pointnav 0.716 vs 0.751): the smooth head's top-5 is one mode plus its neighbours (redundant), the hard head's spans distinct modes, and hit-rate rewards diversity. A widening gap argues for smaller σ, not larger |
| `code_mdist_argmax` | `d(argmax, true)` in action-scale units per tick | the greedy policy's per-step error in the flow head's units; should fall even where `code_acc` does not |
| `code_mdist_expected` | `Σ_c p(c) d(c, true)` | **the number to watch**: the expected per-step error of the *sampled* policy. Lower is better regardless of σ; this is where soft labels should pay |
| `code_mass_near` | p-mass within σ (or 0.2 at σ = 0) of the true code | did the head learn the neighbourhood? Under soft labels this should exceed the hard run's, while `pmass_target` is allowed to be lower |
| `code_pmass_target` | `p(true code)` | under soft labels sits below 1 by design (~0.7 at σ = 0.10 if the head matched the target exactly); much lower than the target's own value means the head is under-fit, not smoothed |
| `code_stationary_tgt` / `_argmax` / `_expected` | share of stationary-xy codes in the teacher, in argmax, and in `p` (expectation) | the closed-loop-relevant bias, watched from SFT |
| `code_stationary_excess` | `_expected − _tgt` | **> 0 = the head over-predicts standing still** relative to the teacher. This is the offline precursor of the look-around loop; soft labels should move it toward 0, not away |

Failure signatures: `code_soft_ce` falls but `code_mdist_expected` does not → the head is
fitting the kernel's shape, not the metric (σ too large, check `code_target_entropy`);
`code_mass_near` up but `code_stationary_excess` up too → the kernel is bleeding mass
into the stationary cell from its creep neighbours (σ too large for the stationary↔creep
spacing of ~0.22); `code_pmass_target` ≪ the target's own peak with `code_pred_entropy`
high → under-fitting, same as a hard run. Per-component (`eval_objectnav_nopose_*`,
`eval_pointnav_*`) versions exist under `--eval-per-component`, as before.

The kernel matrix is a non-persistent buffer (`code_head.label_dist`), recomputed from the
tokenizer at build time; checkpoints written before it existed load strictly, and
`meta["fm_code"]` records `label_sigma` / `label_metric` for provenance only.

## What it does at rollout

Nothing directly -- `sample`/`argmax` are unchanged. Indirectly, mass that was fragmented
across near-identical cells is pulled together, so `sample` draws jitter within a mode
instead of across the vocabulary and `argmax` ties are broken by metric proximity. It does
not reintroduce mode averaging (`code_mode="mean"` is the construction that does, and it
paralyses; see `LIVE_ROLLOUT_CADENCE.md`).

## How to judge it

Not on open-loop pose RMSE (it picked the worst decode). On a matched-cadence
(`--gap 10 --dt 0.04`) multi-episode eval, and on `code_within1_theta` / `code_ce` on the
held-out split as the training-side proxies.

## Experiment plan: does the metric reach the policy head's output geometry?

**Why this matters for RL.** A policy-gradient step on a sampled code `c` moves `W₂[c]`
(code-specific: 1,600 free rows, no spillover by construction) and the shared hidden `z`
toward `W₂[c] − E_p[W₂]`, which lifts every code whose output row is *aligned* with `c`'s.
So how much RL credit on "turn left ~40° and drive" generalises to the neighbouring cells is
set by how metrically the output rows are organised after SFT — an SFT property, measured
before any RL is run. The decoder-side code embeddings, trained by a metric loss (velocity
MSE), are the reference for what "metric" looks like in this model.

**Measurement.** `dump/audits/code_head_geometry.py <checkpoint_dir>...`: Spearman between
pairwise cosine similarity of the output rows and per-tick metric closeness
(`code_distance_matrix`), plus mean cosine of each row to its 5 metric-nearest cells vs 5
random cells; same for `emb_xy` / `emb_theta` against the per-factor metric.

**Baseline (hard one-hot CE, `run_code_v4_mlp_warm`).**

| checkpoint | policy-head output rows: ρ / cos(5 nearest) / cos(5 random) | decoder code embeddings: ρ xy / ρ θ |
|---|---|---|
| ck800 | +0.12 / +0.04 / 0.00 | +0.66 / +0.74 |
| ck1800 | +0.16 / +0.08 / 0.00 | +0.66 / +0.75 |
| ck12000 | +0.24 / +0.15 / 0.00 | +0.67 / +0.76 |

**Soft labels (`run_code_v6_soft010`, σ = 0.10), same recipe and seed — measured while training:**

| checkpoint | policy-head output rows: ρ / cos(5 nearest) | decoder embeddings | hard `code_ce` (v4 at same step) | `within1_θ` (v4) | `stationary_excess` |
|---|---|---|---|---|---|
| ck800 | **+0.17 / +0.07** | +0.66 / +0.74 | 4.578 (4.578) | 0.306 (0.299) | −0.096 |
| ck1800 | **+0.22 / +0.12** | +0.66 / +0.75 | 4.337 (4.301) | 0.350 (0.338) | −0.004 |

Early reading (2026-08-30, step 1800 of 12000): the rows become metric ~1.5x faster under
soft labels — v6 at 1800 matches v4 at 12000 — with the expected signature elsewhere (hard CE
+0.04, within-1 up, predictions softer, stationary bias converging on the teacher's). Where
it plateaus is the open question; re-measure at ck6000 and final.

Open-loop decode error at ck1800, same 9,623 held-out turns, real decoder
(`dump/audits/pose_err_*_ck1800.json`): argmax dθ 0.626 (v4) vs 0.637 (v6), dx 0.313 vs
0.310; sampled T=1 dθ 0.635 vs 0.637 — **no accuracy cost**. Sampled-policy stationary rate
0.483 (v4) vs **0.438 (v6)** against the teacher's 0.440: the soft head's sampled policy is
calibrated on stop-vs-go where the hard head over-stands. Head entropy at T=1 is 4.21 nats
(v6) vs 3.42 (v4), so the tempered policy (T ≈ 0.5) matters more for v6 at rollout.

Reading: the decoder already carries a metric embedding of the codes; the policy head does
so only weakly, and only because the data's smoothness in `h` induces it slowly. Under
one-hot CE nothing asks two output rows to co-activate. Under the soft target, `p` is asked
to put mass on `c*`'s metric neighbours every step, which is the signal that correlates
their rows — the same mechanism that made the decoder side metric.

**Prediction to test.** After a σ = 0.10 run (same recipe, same steps), the output-row ρ
moves from ~0.24 toward the decoder's ~0.7 and the 5-nearest cosine rises well above 0.15,
at no cost in `code_mdist_expected`. If ρ does not move, soft labels changed the loss value
but not the representation, and RL credit will still land on single cells; the RL-side
kernel spreading below becomes necessary rather than optional.

**Run at:** the soft-label run's ck2000 (to compare with the hard run's early geometry) and
its final checkpoint; report the two numbers next to the table above.

## Baseline to beat (hard one-hot CE, `run_code_v4_mlp_warm` ck12000, sample101, gap 10)

| policy | success | oracle | SPL | oSPL | path | coll/m |
|---|---|---|---|---|---|---|
| vanilla cotrain_v3 ck12000 | 0.545 | 0.663 | 0.258 | 0.434 | 19.8 | 0.51 |
| hybrid greedy | 0.257 | 0.584 | 0.081 | 0.298 | 31.4 | 0.87 |
| hybrid sample T=0.5 | 0.267 | 0.535 | 0.110 | 0.321 | 27.5 | 0.70 |

Full record: `dump/audits/code_sft_conditioning_audit_2026-08-30.md` §10. Evaluate the
soft-label checkpoint with the same command (both decodes; greedy and T=0.5 tied here) and
read the paired wins/losses, not just the means.
