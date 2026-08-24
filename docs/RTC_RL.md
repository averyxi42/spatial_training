# RTC under RL: the conditioned chain, and who gets credit

Status: design, agreed in discussion 2026-08-24; nothing built. Companion to
`docs/RTC_TRAINING.md` (the SFT-side conditioning, built), `docs/FLOW_SDE_RL.md`
(the flow-SDE machinery this extends), and
`habitat_physical_nav/docs/LATENCY_MASKING.md` (the schedule). This document is
also the implementation plan (section 6); the design decisions in sections 1-5
are the contract that plan implements.

**Flow-time convention, stated once and used everywhere below.** `t` is flow
time in THIS repo's reversed axis: **`t = 1` is noise, `t = 0` is data**
(`flow_interpolate`; openpi convention). Committed prefix rows sit at
**`t = 0`** — clean, finished actions. The RTC paper's Algorithm 1 writes
`τ = 1` for clean because its axis points the other way; transcribing it
literally pins the prefix at pure noise and prints a plausible wrong number.
`γ` is the RL discount per decision step and has nothing to do with `t`; the
symbol `τ` is not used below at all, to keep the two clocks unconfusable.

---

## 1. The decision-step MDP

At the fixed 2.5 Hz decision clock (`gap` = 10 ticks of 0.04 s; chunk length
`H` = 20; commitment length `d ∈ [0, H − gap]`):

* **State** `s_k = (o_k, c_k)`: the observation AND the commitment — the `d_k`
  committed differential rows. The commitment is state, not action: at
  decision time it is already fixed, it will execute regardless of what the
  policy emits, and the policy conditions on it.
* **Action** `a_k`: the **postfix rows** `[d_k, H)` of chunk `k` — the only
  rows the policy chooses. Rows `[d_k, gap)` execute in interval `k` (the
  fresh span); rows `[gap, H)` are the tail, from which the NEXT commitment
  `c_{k+1} = a_k[gap : gap + d_{k+1}]` is sliced.
* **Transition**: interval `k` executes `c_k` then the fresh span; then the
  exogenous draw `d_{k+1}` decides how much of `a_k`'s tail becomes binding.
  The `d`-draw randomness lives in the TRANSITION KERNEL, not in the policy.
* **Reward** `r_k`: per-interval, as today. Well-defined for any `{d_k}`
  because the interval tiling is exact — every interval is `gap` ticks.

Assumptions this rests on, each already a standing design principle:

1. The decision clock is fixed; `d` never derives from wall-clock during
   training (RTC_TRAINING.md section 5 — the delay is a scheduling fiction).
2. `d_k` is exogenous and policy-independent: drawn env-side at observation
   emission from a configured law, on a stream separate from the policy's
   noise (LATENCY_MASKING.md section 4 discipline).
3. The commitment is a deterministic function of the PREVIOUS action's tail
   and the exogenous draw — never of anything the current action does.

Under these, `(s_k, a_k, r_k)` is an ordinary MDP and the existing PPO/GAE
stack applies **without structural change**. The rest of this document is the
fine print of that sentence: two places where "no change" is a theorem with
conditions (sections 2 and 4), and one place where an attractive change is
wrong (section 3).

## 2. The prefix-conditioned SDE chain

Formalization: **the SDE runs on the postfix subspace; prefix rows are
conditioning inputs to the drift, not chain state.** Per integration step of
`sample_chain_np`:

* Per-tick flow time: `t = 0` at prefix rows (clean — reversed axis, see the
  convention note), the shared integration time elsewhere. Built by the SAME
  helpers the SFT loss uses (`prefix_mask_from_len`, `prefix_time` in
  `flow_matching_head.py`) — one construction site, no re-derivation.
* Velocity over the full block: `v = decoder(h, x, t)`. Attention is how the
  postfix reads the clean prefix; that IS the conditioning.
* Update, then **re-pin prefix rows**; in `_sde_transition`, equivalently and
  more honestly: `mu = where(prefix_mask, x, mu)` — the prefix "transition"
  is a Dirac.
* Per-SDE-transition log-density: sum over **postfix elements only** (mask as
  multiplier).

Why the density mask is FORCED, not hygiene: at a prefix row the computed
Gaussian mean `mu = x + dt(v − ½σ²·score)` does not equal the pinned value —
the field's output at a clean row estimates `noise − a` for an unrealized
noise, so the Gaussian evaluated there is a garbage finite number that depends
on `h` and therefore does NOT cancel between old and new policy in the ratio.
Pinning `mu` and masking the density makes each transition exactly
Dirac × diagonal-Gaussian; the Dirac contributes identically to sampler and
scorer (i.e., nothing). Prefix rows are also STATE (section 1) — scoring them
would compute log-density of the state under the policy.

Contracts this preserves and the tests that pin them:

* **One-forward reproduction**: `chain_log_prob_batch(h, chain, positions,
  prefix_actions, prefix_len)` reproduces the rollout density exactly from
  stored quantities. The stored chain keeps its `(K+1)·T·3` shape (prefix is
  conditioned-in, never prepended); prefix rows are constant across all `K+1`
  blocks, and the scorer ASSERTS they equal the stored prefix.
* **`d = 0` bit-identity** with today's sampler under the same seed.
* Sampler and scorer share `_sde_transition` — any RTC term lives inside it
  or the PPO ratio is silently wrong (FLOW_SDE_RL.md's organizing invariant).
* `force_ode` (eval) composes with the prefix: eval/deploy is
  prefix-conditioned pure ODE, logprob 0.

## 3. What is scored: the full postfix, tail included — a rejected alternative, recorded

It is tempting to also mask the density of tail rows that end up unexecuted
(ex post, rows `[gap + d_{k+1}, H)` never touched anything), on the argument
that their score terms have zero expected gradient and only add variance.
**That argument is wrong for this chain, and the scheme is rejected.** The
per-transition Gaussian is elementwise given `x_t`, but `x_t` evolves through
the velocity field, whose attention mixes ALL rows: noise injected at a tail
row at transition `i` moves the EXECUTED rows at transition `i+1`. Tail noise
is therefore part of the transition kernel of the executed rows — not
payoff-irrelevant auxiliary randomness — and a ratio that ignores how the
tail-row distribution shifted between policies is a biased importance weight,
not a marginal. Note the pre-RTC status quo already scores the full block
including the (then always-discarded) tail; RTC only shrinks the unexecuted
region. The asymmetry with the prefix is exact: prefix rows carry NO
randomness (Dirac), so excluding them is exact; tail rows carry randomness
that feeds forward, so excluding them is wrong.

(At the shipped config the "provably never executed" region is empty anyway:
`gap + d_max = 20 = H`. The argument above is recorded so the scheme is not
re-derived as a "free variance reduction" if `d_max` ever shrinks.)

## 4. Returns: split at the splice, re-time with γ

`r_k` accrues over `d_k` committed ticks (caused by `a_{k−1}`'s tail) plus
`gap − d_k` fresh ticks (caused by `a_k`). With a commitment-blind critic
(`V(o_k)` — the prefix conditions the flow decoder, never the backbone), the
committed part of `r_k` leaks into advantages as variance and, through the
GAE bootstrap, some bias.

**The scheme.** Split each interval's reward at the splice point (one extra
geodesic query per step, at tick `d_k`): `r_k = r_k^commit + r_k^fresh`.
Re-time:

    r̃_k = r_k^fresh + γ · r_{k+1}^commit

**The identity that justifies it.** The re-timed return telescopes:

    R̃_k = Σ_j γ^j r̃_{k+j} = Σ_i γ^i (r^fresh_{k+i} + r^commit_{k+i}) − r^commit_k
        = R_k − r^commit_k

So the whole scheme is EXACTLY "subtract from each step's return the one
reward component its action cannot influence" — a pure baseline correction:
unbiased, variance-reducing, objective-preserving. **γ is the unique weight
for which this identity holds**; any other weight silently re-weights
committed progress and changes the objective (weight 1 ≈ +3% at γ = 0.97).

Points of care, each argued in the 2026-08-24 discussion:

* **Do not DELETE the commit reward instead of re-timing it.** Dropping
  `r^commit` without reassignment removes committed progress from every
  step's return: nobody is credited for the tail, and the policy is taught to
  defer progress into fresh spans — a perverse incentive, not a
  simplification.
* **γ discounts ARRIVAL time, not emission time.** The commitment is emitted
  in the same forward pass as the fresh span, but its reward arrives one
  interval boundary later; a delayed effect of an action being discounted is
  what a discounted objective means. The lump-level discount error (γ^1 vs
  tick-exact γ^{1.0..1.5}) is ≤ ~1.5% at γ = 0.97 and is shared with the
  UN-retimed scheme — re-timing introduces no new timing error.
* **Critic corollary**: a critic regressed on re-timed returns learns
  `Ṽ(o) = V(o) − E[r^commit | o]` — the re-timing automatically strips the
  commitment-determined component the commitment-blind critic could not
  model. This is why re-timing is the DEFAULT and prefix-conditioning the
  critic (feeding `(d, prefix)` into the value-head input) is a measured
  refinement: after a baseline run, regress value residuals on `d`; add the
  conditioning only if the residuals say so.
* **Edge cases**: `d_0 = 0` (no committed reward at episode start — the
  boundary term of the identity is zero); the final chunk's tail never
  executes (no `r_{K+1}^commit` — correctly, nothing arrived); early
  termination inside an interval splits at whatever executed (the env owns
  the split, so oracle/budget cuts are respected); truncation bootstrapping
  (`γ·V` on the capped step) is unchanged and now bootstraps the re-timed
  value, consistently.
* **Estimator compatibility**: every registered estimator consumes per-step
  rewards; the re-timing is a pure pre-transform in `rl_core` behind a config
  flag, so re-timed vs raw is a one-flag ablation, not an architectural fork.

## 5. `d`: supply and storage

**Supply** (env-side, unchanged from the settled schedule design): the env
actor owns the `ChunkScheduler` (imported from `objectnav_eval` — build on,
not copy); `d_k` is drawn at observation emission from `rtc_*` env config on
a separate stream seeded `seed + episode`; the obs carries `(prefix, d)` to
the worker. Default RL law: **`exp(0.8)` over `[0, 10]`, matching the SFT
checkpoint's law** — the policy's conditioned behavior gets RL gradient over
the same `d` distribution SFT trained. A fixed deployment-`d` fine-tune is an
ablation, not the default; `d = 0`-only RL is the named trap (RL would
improve only the unconditioned policy and reintroduce the train/deploy shift
at `d > 0`). Startup validates `d_max ≤ min(H − gap, fm_config.rtc_delay_max)`.
Overrun injection stays OFF during training (deployment behavior, simulated
only as deliberate disturbance — a later experiment). Opportunistic overlap
(`begin_interval` fired un-awaited before inference; Ray actor ordering
serializes it) is wall-clock only, config-gated, off by default.

**Storage** (the rollout buffer): two per-step fields, and they are REQUIRED,
not convenience:

| field | shape/type | why |
| --- | --- | --- |
| `prefix_len` | int64 | rebuilds the mask and per-tick time in the scorer |
| `prefix_actions` | `(d_max, 3)` float32, zero-padded | the PHYSICAL differentials the sampler consumed, verbatim |

Neither is recoverable at training time: re-deriving prefix rows from the
stored chain violates the scorer-sees-the-sampler's-exact-input rule (the
1/σ²-amplified density does not forgive round-trips), and `d` is not
inferable from the chain (a zero prefix from an overrun is a legal value, not
a sentinel). Cost: 31 floats per step. They thread through exactly three
call sites — `sample_chain_np` (rollout), `postprocess_episode`'s
old-log-prob recompute, and `rl_loss`'s chain branch — which also covers the
ref-logprob/KL path for free (same scorer). `collate_trajectories` pads
shape-agnostically; no change expected there.

## 6. Implementation structure and order

| component | home | change |
| --- | --- | --- |
| env schedule | `longnav/env/objectnav_continuous.py` | `ChunkScheduler` in the actor; strict `len(chunk) == H` (replaces `== gap`); obs carries `(prefix, d)`; reward split `(r_commit, r_fresh)` in `info` (always on); `begin_interval()`; `rtc_*` config fields (defaults off ⇒ bit-identical) |
| head sampling | `longnav/utils/flow_sde_policy.py` | `sample_chain_np(h, prefix)`; `_sde_transition(prefix_mask)` (mu pinned, density masked, per-tick `t` via the shared helpers); `chain_log_prob_batch(..., prefix_actions, prefix_len)` + prefix-row constancy assert; `decode_action` returns the full `(H, 3)` chunk (env owns slicing); `force_ode` composes |
| rollout plumbing | `longnav/utils/rollout_core.py` | prefix from obs into the sampler; full chunk to env; buffer fields; `begin_interval` hasattr hook (overlap, off by default) |
| returns | `longnav/utils/rl_core.py` | the re-timing transform `r̃_k = r^fresh_k + γ·r^commit_{k+1}`, one flag, applied before any estimator |
| training | `longnav/utils/vlm_worker.py` `rl_loss` | thread the two buffer fields into the scorer |
| config | `conf/env_configs.py` (+ experiment YAML) | `rtc_delay_source/dist/base/max/seed`; re-timing flag; overlap flag |

**Order of work**, chosen so each phase lands something checkable:

1. **Env + eval parity.** Everything in the env row plus the ODE-path prefix
   in `sample_chain_np` (`force_ode`), interleaved eval + `scripts/eval.py`
   inheriting (and the known `eval.py` missing-`set_ode_sampling` fix while
   there). Deliverable: the RL stack evaluates the published RTC checkpoint
   and REPRODUCES the harness numbers at `d = 0` and `d = 5` — cross-path
   parity is itself the test, and this phase alone delivers the Ray-parallel
   eval path.
2. **Scorer + buffer.** `_sde_transition` masking, `chain_log_prob_batch`
   reconditioning, buffer fields through all three call sites.
3. **Smoke + gauge recalibration.** Extend `tests/test_flow_sde_policy.py`
   (recompute-matches-rollout with a random prefix; zero-noise-limit-is-ODE
   with prefix; prefix-row constancy across blocks; `d = 0` chain
   bit-identity); dummy-env rollout for the buffer keys; one GPU cycle at
   fixed `d = 3`. The seam gauges (`chain/abs_log_ratio_mean` ≈ 0.01 nats,
   `chain/rollout_seam_gap`, `chain/h_drift_from_init`) must have their
   expected bands RE-ESTABLISHED at cycle 0 — the postfix-masked density sums
   fewer elements, so the historical bands do not transfer.
4. **Decide the critic refinement** from the baseline run's diagnostics
   (value-residual-vs-`d` regression, plus the probe counterfactual block
   `train_loop` already logs), then launch the real run from the current
   `flow_sde_*` config line plus the `rtc_*` fields.

## 7. Open decisions

* RL delay law: `exp(0.8)` default (matches SFT) — confirm, or fixed
  deployment-`d`.
* Re-timing flag default: ON (the identity argues for it) — confirm.
* Critic prefix-conditioning: decide from phase-3/4 measurements, not now.
* Overlap default: OFF until parity is demonstrated.
* Overrun-as-disturbance training: deferred; the scheduler supports injection
  whenever that experiment is wanted.

## Provenance

The MDP formalization, the Dirac/garbage-mean argument for the forced density
mask, the tail-masking rejection (attention feed-forward of tail noise), and
the re-timing subtraction identity `R̃_k = R_k − r^commit_k` with γ-uniqueness
were derived in the 2026-08-24 design discussion; the discount-lumping error
bound (≤ ~1.5% at γ = 0.97) was computed there against tick-exact
discounting. The schedule, delay-source discipline and storage contract carry
over from `LATENCY_MASKING.md` and `RTC_TRAINING.md`, both verified against
the built SFT/eval implementation. Nothing in this document is implemented
yet; section 6 is the plan.
