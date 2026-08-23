# RTC training: conditioning the chunk on the commitment

Status: design approved, implementation on branch `rtc`. Companion to
`habitat_physical_nav/docs/LATENCY_MASKING.md`, which owns the harness-side
schedule this document builds against; the split is deliberate and mirrored
there (§Scope). Method follows the training-time variant of real-time chunking
([arXiv:2512.05964], Black/Ren/Equi/Levine), not the original inference-time
inpainting ([arXiv:2506.07339]) — see §8 for why the inference-time algorithm
is deliberately not built.

**The sentence-long version:** while a chunk is being generated, the robot
executes `d` actions it already has; the new chunk is generated *conditioned on
those `d` actions* by feeding them to the flow head as already-denoised rows,
and the model is trained for exactly that by randomly pretending, on every
training example, that the first `d` rows are already decided.

---

## 1. One new parameter

| symbol | value | status |
| --- | --- | --- |
| `d` | 0 .. `H − gap` (= 0..10) | **the new parameter**: commitment length = assumed inference delay, in ticks |
| `H` | 20 (`fm_n_ticks`) | existing, checkpoint-carried |
| `gap` | 10 | existing: ticks per decision, fixed by the 2.5 Hz decision clock over 25 Hz control (`dt` = 0.04 s) |

The paper's "execution horizon `s`" is a free variable only because its runtime
re-triggers inference as fast as latency allows. Our decision clock is fixed by
design — observations are budgeted at 2.5 Hz regardless of model speed — which
freezes `s` at `gap`. Notation mapping for readers of the paper: `s ≡ gap`, and
the paper's `τ = 1` = clean is this repo's `t = 0` = clean (see §6, the trap).

At `d = H − gap = 10` the full 400 ms of budgeted overlap is spent on masking
latency; the paper's measured real-world latencies translate to `d = 3..4` at
25 Hz, so `d ≤ 5` is the regime that matters first and `d_max = 10` is the
ceiling training should cover.

## 2. Glossary

Two different counts appear in any discussion of this mechanism and must never
blur:

* **tick** — one 25 Hz control step. **decision interval** — the `gap` ticks
  between consecutive observations.
* **tail** — the `H − gap` rows a chunk carries past its own interval,
  unexecuted.
* **commitment** — the first `d_k` rows of the tail, chosen at observation
  time `k`.
* **fresh span** — chunk `k`'s rows `[d_k, gap)`, executed in its own interval.
* **consumed rows** (per chunk) — fresh span plus next commitment,
  `(gap − d_k) + d_{k+1}`: a *variable* count in `[gap − d_max, gap + d_max]`.
* **waste** — the `(H − gap) + d_k − d_{k+1}` tail rows never consumed.

Interval statements (the tiling below; per-step reward/budget/step-count
currency) are exact by construction and `d`-invariant. Per-chunk statements
(consumption, waste) vary with neighboring `d`s. "Executes `gap` actions" with
no qualifier is either a tautology or wrong depending on which count is meant —
do not write it.

## 3. The reciprocal schedule

Commitment is a property of the **next request**, not of the chunk. Nothing is
committed when a chunk is generated; the chunk merely carries a tail. At
decision `k` (tick `k·gap`) the scheduler holds the previous chunk's full tail,
draws `d_k` from the delay source, slices `tail[:d_k]` as the commitment, and
sends observation and prefix together. `d_k` must be fixed before inference
starts — the prefix crosses with the request — and cannot usefully be drawn
earlier than observation time, so **observation emission is the single draw
site**.

The interval is then exactly tiled: `d_k` committed ticks followed by
`gap − d_k` fresh ticks, for any `{d_k}` sequence — no tick unassigned, none
double-booked. The first decision has no tail, so `d_0 = 0`, stated rather than
falling out of the loop structure.

The alternative "forward view" implementation — each step executes
`chunk[d : d+gap]` contiguously — needs the *next* decision's `d` at execution
time and therefore only works for constant `d`. The reciprocal implementation
subsumes it (identical trajectory when `d` happens to be constant), so only the
reciprocal one is built.

## 4. Two-phase interval execution

The commitment is fully determined at observation time — tail held, `d_k`
drawn, setpoints known — so executing it needs no policy input. That is
precisely why it can overlap inference. The env exposes the interval as two
phases:

* `begin_interval()` — execute the commitment (`d_k` ticks; no-op at
  `d_k = 0`).
* `step(chunk_k)` — execute the fresh span, retain the new tail, draw
  `d_{k+1}`, return the next observation plus prefix.

Default mode fuses both into the single `step()` call — bit-identical
trajectory, no new failure modes. Opportunistic-overlap mode fires
`begin_interval` without awaiting it, before inference; Ray actor task ordering
serializes it ahead of `step` on the same actor, which yields wait-don't-toss
semantics in both directions for free (slow policy → env idles; slow env →
chunk queues). Wall-clock saved per interval is `min(inference time, commitment
sim time)`. The trajectory is identical either way, which is the point of the
next section.

## 5. The decoupling principle

**Training is never coupled to wall-clock latency.** `d` is always the
schedule's `d_assumed` — configured or sampled, never measured (the
`d_assumed` / `d_actual` split and the reasons a measured latency cannot be the
conditioning delay are LATENCY_MASKING.md §3, not restated here). During SFT
and RL, real model latency changes nothing about the trajectory: the delay is a
scheduling fiction and the env waits for the policy.

Overrun — skip one decision: hold position for the rest of the interval, toss
the late chunk, tail empties so the next prefix is `d` zeros — is
**deployment-time behavior**. In training and eval it exists only as *injected
fault* (a rate knob on the scheduler), never as a consequence of actual
latency. The fallback arithmetic and the reasons hold-position is the default
are LATENCY_MASKING.md §5.

Deployment then chooses `d` from real measured latency — legal precisely
because training randomized over the whole `[0, d_max]` range. That is the
compatibility this design buys: the training recipe fixes no latency, and the
deployed system picks its point on the curve after measuring its own.

## 6. The algorithm: training-time conditioning

Regular flow matching here (`flow_matching_head.py`): `x_t = t·noise +
(1−t)·a`, target velocity `u_t = noise − a`, **`t = 1` is noise and `t = 0` is
data**, Euler integration with `dt = −1/K`. The upgrade is a pure extension —
no new parameters, no behavior change when no prefix is passed:

1. **Per-tick flow time.** `time` becomes `(N,)` or `(N, T)`. The sinusoidal
   time featurization is a fixed parameter-free function and the fusion MLP
   already runs per token; today's single per-row time is embedded once and
   broadcast to all ticks only because all ticks are equally noisy. The change
   replaces that broadcast with a per-(row, tick) evaluation of the same
   function. This is the paper's "per-token flow timestep", which in a
   concat-conditioned decoder costs a reshape.
2. **Prefix rows are clean.** Training: prefix rows of `x_t` are the
   ground-truth actions with per-tick time `0.0` (⚠ the paper writes `1.0`
   because its time axis is reversed — transcribing Algorithm 1 literally pins
   the prefix at pure noise, runs fine, and prints a plausible wrong number).
   Inference: prefix rows are pinned to the committed actions before the loop
   and after every Euler/SDE update, time `0.0` at those rows.
3. **Loss on the postfix only**, as a mask multiplier. There is no DDP branch
   hazard: `d ≤ H − gap < H` guarantees a nonempty postfix, so every parameter
   receives gradient on every example (prefix-position `tick_embed` rows
   included, via bidirectional attention from postfix outputs).
4. **`d` is sampled online per training example** from a configured
   distribution — no dataset column, no rebuild. The prefix target is the
   chunk's own first `d` rows: chunks are cut at stride `gap`, so the previous
   chunk's tail rows and the current chunk's first rows are the same underlying
   frames, and per-tick body differentials are anchor-free, so the same numbers
   regardless of which chunk they are read from. The residual train/deploy gap
   — deployment conditions on the model's own previous *sampled* rows, training
   on ground truth — is the paper's accepted shift.

One implementation, three consumers: the mask and the per-tick time vector are
constructed in a single helper in `flow_matching_head.py`, called by the SFT
loss, by `euler_integrate`, and by the SDE head. The training-time masking is
literally the same code in SFT and RL.

## 7. What the RL scorer masks, and why

The flow-SDE head's organizing invariant — sampler and scorer share the one
transition density `_sde_transition` — survives because prefix conditioning
changes the transition, not the algorithm around it: prefix rows have
`mu = x_t` (frozen, zero variance contribution) and per-tick time `0.0`; the
Gaussian log-density is summed over postfix elements only, mask as multiplier.
Prefix rows carry no probability mass because nothing about them was sampled —
they are conditioning, and scoring them would corrupt the PPO ratio with terms
the policy cannot influence.

Consequences that must hold and are tested:

* The stored chain keeps its `(K+1)·T·3` contract — the prefix is
  conditioned-in, never prepended; prefix rows are constant across all `K+1`
  blocks.
* The scorer reconditions on the **stored** `prefix_actions` from the
  trajectory, never on rows re-derived from executed poses — the 1/σ²-amplified
  chain density does not forgive even a compose/decompose round-trip
  difference. Gauges: `chain/abs_log_ratio_mean` (≈0.01 nats at epoch 0),
  `chain/rollout_seam_gap`, `chain/h_drift_from_init`.
* `d = 0` reproduces today's sampler bit-for-bit under the same seed.

## 8. Delay distributions, per phase

| phase | `d` source | rationale |
| --- | --- | --- |
| SFT | per-example, uniform `[0, d_max]`, optional extra mass at 0 | cover the whole range deployment may pick from; the zero-prefix / freezing hazard (below) argues for deliberate `d = 0` and rest-prefix exposure |
| RL rollout | per-decision, uniform `[0, d_max]`, separate RNG stream | match what SFT saw; a separate stream so changing the delay policy does not perturb the flow head's noise draws |
| eval | `fixed`, swept `d ∈ {0..10}` | the headline figure: success vs `d`; flat-ish is the acceptance criterion |
| deployment | `fixed` from measured latency (`ceil(latency / dt)`) | the point of the whole exercise |

**Zero-prefix hazard.** The overrun fallback conditions the next chunk on `d`
zeros — "continue from rest". This policy family has a measured freezing
attractor (the `--boundary-policy` results: near-stationary training signal →
self-reinforcing standstill), so resume-from-rest must be trained, not hoped
for, and post-overrun recovery needs its own instrument
(ticks-until-motion-resumed), because an overrun costing one interval and one
triggering a 200-tick freeze are indistinguishable in aggregate metrics.

**Deliberately not built:** inference-time inpainting (pseudoinverse guidance
inside the integration loop). It would add per-step backprop latency at
deployment — the cost the paper exists to remove — and its guidance term would
have to live inside `_sde_transition` where it breaks the cheap sampler/scorer
symmetry that training-time conditioning preserves. The paper reports parity or
better for training-time at realistic delays.

## 9. Provenance

Schedule, fallback arithmetic, delay-source taxonomy, invariants:
`habitat_physical_nav/docs/LATENCY_MASKING.md` (verified against
`executor.py` / the corpus table there). Algorithm: [arXiv:2512.05964]
(training-time conditioning, per-token timestep, postfix loss, uniform delay
sampling; comparative results cited, not reproduced). Freezing measurements:
the `--boundary-policy` work, from prior corpus experiments, not re-measured
here. The reciprocal formalization (§2–§4) was derived in planning and checked
by hand over variable-`d` sequences; its tests are the tiling and `d = 0`
identity tests named in the implementation plan.

[arXiv:2512.05964]: https://arxiv.org/abs/2512.05964
[arXiv:2506.07339]: https://arxiv.org/abs/2506.07339

## Scope

This document covers the model and training side: the conditioning algorithm,
its RL form, and the delay distributions. The harness schedule, delay sources,
overrun semantics and what runs must record are LATENCY_MASKING.md's subject.
The scheduler implementation shared by both rollout paths lives in
`objectnav_eval` (one schedule, no copies).
