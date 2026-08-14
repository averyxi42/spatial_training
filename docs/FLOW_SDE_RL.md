# Flow-SDE RL: the hybrid-sampler policy

Status: **design finalised, implementation next.** This document merges the project design
(`FLOW_SDE_RL.md` as of f4df635) with the independently written piRL-style spec
(`reference/FLOW_SDE_PIRL_SPEC.md`); the resolution of every conflict is recorded in
`reference/FLOW_SDE_PIRL_COMPARISON.md` and baked in here, so this document supersedes both for
implementation. Companions: `LATENT_RL.md` (the latent-Gaussian design), `LATENT_RL_ENV.md` (the
RL environment, reused unchanged), `LATENT_RL_BAKE.md` (what the latent bake measured and cost).

Two documents derived the same core structure independently — collapsed Gaussian log-probs over
denoising transitions, one env-step advantage broadcast uniformly with no inner-step GAE or
discounting, transitions stored and `mu` recomputed under current theta, PPO with a learned critic
rather than GRPO (whose group baseline assumes prompt-style resets; robotics has dynamics), eval
integrator identical to deploy. That convergence is the main evidence the structure is right.
Where they disagreed, the disagreement and the pick are stated inline in one sentence each, so a
future reader does not reopen them from scratch.

## Why not the latent

The latent-Gaussian construction (`c = h + sigma*eps`, decoder subsumed into the environment)
works and has a shipped checkpoint. Its cost is that **it makes us do RL on a worse policy.**

| | oracle | oSPL_fix |
|---|---|---|
| deterministic baseline (`run_cotrain_v3_nopose_mix/checkpoint-12000`) | 0.663 | 0.348 |
| the latent pilot that shipped | 0.545 | 0.257 |

That gap is not a tuning failure. `LATENT_RL_BAKE.md` derives it: at the sigma equilibrium the
model permanently pays reconstruction under effective context noise `s* = sigma/h_std ~ 0.7-1.0`,
and for a near-deterministic conditional that noise models nothing and is pure loss. PointNav —
near-deterministic given the goal — is the worst-hit component at 4.1x `mae_dx` against the
deterministic baseline. **The cost is structural to any scheme that injects exploration noise
upstream of the decoder during SFT**, so it is not fixable by patching the diversity term.
Flow-SDE pays none of it: SFT is untouched, and RL runs on the strongest checkpoint we have.

### The measurement that also ruled out the simpler alternative

An intermediate option was considered and rejected: keep the deterministic decode, treat the
flow's base noise `z_0` as exogenous logged noise, and add a Gaussian in *action* space
(`pi(a|o,z_0) = N(ODE(h,z_0), Sigma)`). Exact log-prob, no SFT change, hours of work.

It fails on a measurement. `z_0`-induced behavioural spread, noise arm only, 32 observations x 16
samples on `formatted_pose` validation:

| checkpoint | `z_0` terminal heading std | disp std | mean path len |
|---|---|---|---|
| **v3 deterministic** | **0.737 rad (42 deg)** | 0.307 m | 0.230 m |
| latent pilot ck1000 | 0.336 | — | — |
| latent bake v1 ck500 | 0.181 | — | — |
| latent bake v2 ck500 | 0.173 | — | — |

The deterministic checkpoint's endpoint scatter from `z_0` alone **exceeds its own mean path
length**. Treating it as exogenous conditioning means `z_0`-induced outcome variation is
uncorrelated with the action-Gaussian's residual, so it contributes advantage noise to every
update while being reducible only *indirectly*, at the eps-credit rate — a variance floor that
swamps any deliberate `Sigma`. (Stated carefully because the stronger claim — "irreducible, PPO
can never reduce it" — is false and would be the attackable part of a sound decision: the mean map
`ODE(h, z_0)` is trainable per `(o, z_0)`, so the `z_0`->action Jacobian does shrink over
training; it is the *rate* that is bad, not the possibility. **Rung 1 is therefore retained as the
fallback in the other direction** if chain-ratio machinery proves unmanageable.)

This measurement is also load-bearing for a design choice below: `z_0` is **unusually large for
us** relative to what this literature assumes, which makes the ODE-vs-SDE fidelity gap a
first-class gauge here rather than a formality, and is the strongest standing argument for raising
`N` later. (Incidentally, the table records that the latent diversity term did its job: it moved
variation out of `z_0` and into `c`, 0.737 -> 0.17.)

## The policy

**RL here is a peer of SFT, not a layer bolted on top.** The probability-flow ODE and its
corresponding SDE share marginals at every `t` by construction, so sampling stochastically
reproduces the action distribution the SFT model already defines — same prior, same velocity
field, same parameters, reached by a stochastic path. RL **initialises at the SFT policy itself**
and the conversion costs nothing. That is the whole contrast with the latent, where RL acted on a
variable SFT never had, bought by a bottleneck that changed the model class and charged `phi(s*)`
forever.

The sampler is **hybrid**: of the `K = 10` denoising steps (matching `num_inference_steps` in the
checkpoint's `fm_config`, `flow_matching_head.py:473`), **`N` are SDE steps and `K - N` are plain
ODE steps**. `N` defaults to **1**, tunable 1-3 initially. The `N` stochastic positions are drawn
**uniformly at random per chunk, without replacement**, from `{0 .. K-1-n_exclude_last}` with
`n_exclude_last >= 1`. Only the `N` stochastic transitions carry a Gaussian density; the chunk
log-prob is their sum; one env-step advantage from GAE multiplies that sum uniformly. `z_0` and
the ODE steps are treated as dynamics.

This is exact, not an approximation, *for the MDP whose action is the stored stochastic
transitions*: PPO with ratio `pi_new/pi_old` over the `N` Gaussians is a correct policy-gradient
estimator, and it marginalises correctly to the induced chunk distribution. Two statistical costs,
both accepted: the conditional chain density stands in for the intractable marginal `log p(a|o)`
(variance, inherent to the whole literature), and at small `N` the ratio is **blind to the policy
change carried by the `K - N` deterministic steps** — `v_theta` is one network shared across all
`t`, so it moves everywhere while only `N` transitions are ratio-visible. The trust region
therefore does not cover the whole policy change; the unclipped drift gauge in the instrument
panel exists for exactly this, and **PPO epochs per batch must never be raised on ratio health
alone.**

> **Reversal, recorded:** the previous revision of this document made all `K` transitions
> stochastic (`N = K`) and rejected step subsetting because a *fixed late window* (DPPO-style
> last-K) surrenders mode control — early high-noise steps decide which mode the sample lands in.
> That objection does not apply to *random* positions over a shared `v_theta`: every `t` receives
> gradient over training, so mode control is preserved while the per-chunk ratio stays tight.
> `N = K` is retained as a diagnostic bound (known unstable at scale in this literature), not a
> target. Fixed last-K remains rejected — given up mode control, the rung-1 action Gaussian is
> simpler and should be preferred instead.

**Not DPPO's two-level MDP.** Credit assignment stays at the environment step: no per-denoising
advantages, no inner GAE, no inner discounting. There is no environment feedback interleaving the
chain to justify finer credit — every inner step contributed to the same chunk. The analogy is
thinking-trace RL: uniform credit across the trace, GAE across turns. The critic sees
**observation features only** — never denoising latents, never denoising time. (Our framework
satisfies this structurally: the value head reads `hidden[:, logits_to_keep]` at
`vlm_worker.py:634-643`, and the chain never reaches it.) We are likewise safe from the
flattened-buffer trap — GAE's gamma/lambda silently decaying across inner steps when denoising
transitions are appended to the trajectory's time axis — but only because the chain is stored as
one flat vector per env step, so the buffer's time axis is env steps *by construction*. Keep it
that way; that property is what keeps `rl_core.py` untouched.

## Conventions and the SDE step — pin these first

**The convention warning comes before the formulas because a sign error here does not crash — it
silently trains on a wrong distribution.** Codebases disagree on whether `t=0` or `t=1` is noise
and on the sign of `v`; every identity below flips with the convention. The ground truth for this
project is the checkpoint's own sampling loop, `euler_integrate` (`flow_matching_head.py:349`):

- `t = 1` is pure noise, `t = 0` is the clean chunk; integration runs `t: 1 -> 0`.
- `dt = -1/K` (negative), update `x <- x + dt * v(x, t)`.
- The SFT target field is `u_t = noise - actions`, i.e. `v_theta(x_t, t, h) ~= E[eps - x_data]`,
  with interpolation `x_t = (1-t)*actions + t*noise`.

This matches the external spec's stated convention, verified by deriving both identities from the
interpolation: `x_t + (1-t)*v = eps` and `x_t - t*v = x_data` hold exactly for the ideal field.
**Do not copy formulas from any other codebase without re-deriving them against
`euler_integrate`;** the marginal-preservation test below is the executable check.

From `x_t` and `v = v_theta(x_t, t, h)`:

```
eps_hat = x_t + (1 - t) * v            # predicted noise component
x0_hat  = x_t - t * v                  # predicted clean chunk (Tweedie estimate)
score   = -eps_hat / max(t, t_min)     # grad log p_t(x_t),  t_min ~= 1e-3
```

Noise schedule (Flow-GRPO-style; `a` is THE noise hyperparameter, start small):

```
sigma_t = a * sqrt(t / (1 - t))        # evaluate at t clamped away from both 0 and 1;
                                       # the clamp is a config constant beside `a`
```

The SDE Euler-Maruyama step, written in this codebase's own signs (`dt = -1/K`):

```
drift      = v - 0.5 * sigma_t**2 * score   # MINUS: see the post-mortem below
mu         = x_t + dt * drift
sigma_step = sigma_t * sqrt(|dt|)      # per-step Gaussian std for the policy
x_next     ~ Normal(mu, sigma_step**2 * I)
```

ODE steps are `x_next = x_t + dt * v` — no noise, no correction, no log-prob.

> **CORRECTED 2026-08-14.** This section originally asserted `drift = v + (sigma^2/2)*score`
> with the note "the external spec's reverse-time step under the sign flip". That was the
> bug: the external spec's convention (tau decreasing) already matched this codebase's, so
> no flip was needed — the drift sign is a dynamical claim that must come from the
> Fokker–Planck derivation, not a convention heuristic, and the originally designated
> arbiter (the small-a marginal test) is blind to the resulting O(a^2) error. Full
> post-mortem at the end of this document; the analytic high-noise guard test is the
> arbiter now.

**Wiring rules — each one is a real observed failure mode in this literature:**

1. **`sigma` appears in BOTH terms or NEITHER.** The `0.5*sigma^2*score` drift correction and the
   `sigma*sqrt(|dt|)` noise are two halves of one identity, and they scale differently
   (`sigma^2` vs `sigma`). A "noise_level" knob that scales one without the other is silently
   wrong; grep for any code path touching one alone.
2. **Do NOT `detach()` the score.** It contains `v_theta`, so gradient flows through the
   correction term in `mu`. Detaching changes the gradient estimator into a different algorithm.
3. **ODE steps get NO correction.** The correction cancels injected noise; with no noise there is
   nothing to cancel, and adding it moves the deterministic path away from the pretrained model's.
4. **The `1/t` singularity is why `n_exclude_last` exists.** At `K = 10` the last position sits at
   `t = 0.1` (score factor ~10); exclude the final 1-2 positions from stochastic selection AND
   floor `t` at `t_min` inside the score regardless. This is a deliberate, principled
   non-uniformity — do not "fix" it back to full-range uniform.
5. **The position set is not an input.** The network only ever sees the current `t`, which flow
   models already condition on; do not add index-set inputs.

**`sigma_t`'s usable range is bounded on both sides.** Above: the shared-marginal property holds
for the *exact* score, and ours is recovered from an approximate `v` that flow matching fit on the
interpolation path — larger `a` leans harder on that approximation in states the SFT never
visited, so past some point the sampler stops reproducing the SFT distribution **before RL has
done anything**. Below: at `N = 1` and small `a`, injected noise is a small share of the policy's
total stochasticity (`z_0` contributes unit variance per dim and is exogenous), so the
ratio-visible exploration can be too weak to carry learning signal — the rung-1 advantage-noise
floor partially persists at small `N`, which is the accepted price of tight ratios and the
standing argument for raising `N` if learning stalls with a healthy instrument panel. The
pre-training sweeps below measure both ends before any RL step is taken.

## Likelihood bookkeeping

- Chunk log-prob = **sum of the `N` per-step Gaussian log-probs**, and nothing else. `z_0`'s
  density is parameter-free and cancels in the ratio: **stored, never summed**. ODE transitions
  are dynamics: no density.
- **Recompute under current theta.** The PPO ratio needs `logprob_new` at the SAME transitions:
  at each stored stochastic position `k`, take the stored `x_k`, run `v_theta(x_k, t_k, h)` fresh,
  rebuild `mu` (a function of theta, including the score term), and evaluate the stored `x_{k+1}`.
  Storing only `logprob_old` and re-sampling is wrong; reusing `mu_old` is wrong (freezes theta).
  **Stored latents are data**: do not re-integrate the deterministic prefix under new theta —
  that changes the conditioning states and the estimator.
- **Storage.** `actions_continuous` stores the **full flattened chain including `z_0`** —
  `(K+1) * 20 * 3 = 660` floats per env step — plus a small `sde_positions` key. The spec stores
  only the `N` transitions; we keep the full chain because it is fixed-size (collation stays
  untouched, below), it preserves `decode_action` = last element, and it keeps the invariant the
  actuator seam's comment states: *what is stored for the PPO ratio stays the thing the log-prob
  was computed over*. `sigma_step` is not stored — it is a pure function of `(position, config)`.
- **Clipping.** The joint ratio `exp(sum logprob_new - sum logprob_old)` with one clip per chunk
  is the default. At `N = 1` joint and per-step clipping coincide; they diverge as `N` grows
  (joint-ratio dispersion grows with `N`), so per-step clipping is retained as an option — it
  lives in the `rl_loss` branch (site 3), where it must **bypass `policy_loss_fn`**
  (`vlm_worker.py:942` consumes `(B, S)` log-probs with no per-transition axis): a contained
  loss-path fork, not a head-internal knob. Pick one, document it, and keep it fixed across `N`
  sweeps.
- **fp32 accumulation, as a launch gate.** At `N = 1` the sum has 60 terms (20 ticks x 3 dims per
  transition), 180 at `N = 3`. bf16 summation loses ~2 decimal digits on a quantity whose
  *differences* must resolve to fractions of a nat; `chain_log_prob` computes and sums in float32
  regardless of model dtype. Invisible when missing, and no shape check catches it.
- **Two lr couplings (do not sweep confounded).** Per-chunk gradient magnitude grows ~`N` (`N`
  log-prob terms share one advantage): sweeping `N` at fixed lr measures step size, not
  exploration — this project scales lr ~`1/N` for the 1-3 sweep and widens the clipfrac
  tolerance, rather than dividing the summed log-prob by `N` (the spec allows either; one is
  chosen here so configs stay comparable). Separately, smaller `sigma` gives larger log-prob
  gradients (~`1/sigma^2`): changing `a` requires retuning lr. Never sweep `a` and `N` in one
  study.
- **RNG hygiene.** `z_0` and each per-step SDE noise must be independent fresh draws, and
  parallel env workers must not share correlated streams — correlated draws silently bias
  advantages (and wreck any group-baseline variant outright). Rollout sampling in this framework
  is worker-side; seed per worker, and keep the head's torch generator separate from
  `np.random`'s global state.

## Eval and deploy: pure ODE, shared code path

Evaluation and deployment run the **same sampler code with `N = 0`** — the plain ODE, identical
integrator config to the checkpoint's own. Not a separate reimplementation: divergence between
train-time ODE steps and eval ODE steps is a classic source of phantom train/eval gaps.

`ode_eval_success` is **the** metric — all checkpoint selection, early stopping, and reporting.
`sde_train_success` is diagnostic only, and the **`ode - sde` gap is the fidelity gauge**: a
widening gap means `v` is drifting from something the ODE faithfully samples — respond by
lowering `a` or `N`, not by switching the deploy sampler.

> **Reversal, recorded:** the previous revision required deploying with the SDE, arguing that RL
> modifies `v` under no constraint keeping it a valid velocity field, so ODE and SDE marginals may
> diverge after training. That argument was sound *for `N = K`* and largely dissolves at small
> `N`: when 9 of 10 steps are already the ODE, the training sampler is nearly the deploy sampler
> by construction. Small `N` buys back deterministic evaluation — worth more here than to most
> projects, because checkpoint selection in the latent programme was repeatedly confounded by
> stochastic metrics. The underlying risk did not vanish; it moved into the `ode - sde` gauge,
> which is why that gauge is mandatory and not decorative. Never checkpoint-select on stochastic
> rollout metrics. Given our outsized `z_0` scatter, this gauge is more load-bearing here than in
> the literature this recipe comes from.

## What this touches

Every contract below was read at the source, not assumed. Line numbers are as of this writing and
should be re-checked; the *claims* are what matter.

### Not touched, and why that is structural

| component | why |
|---|---|
| `rl_core.py` — collation, GAE, advantages, `response_mask` | `pad_sequence(batch_first=True)` at `:52` is shape-agnostic; its only reshape is `if padded.dim()==3 and padded.shape[-1]==1: squeeze(-1)`. The flattened 660-float chain does not trigger it. **Trap:** `sde_positions` at `N = 1` has shape `(steps, 1)` and DOES trigger it, collating to `(B, S)` while `N = 2` gives `(B, S, 2)` — the key changes rank with `N`. The head's recompute must tolerate both ranks; nothing else reads the key. |
| the dict passthrough, at **both** `vlm_worker.py:644-645` (`_forward_embeds`) and `:888-889` (the wrapper's forward, which `_training_forward` reaches through `ddp_model`) | `policy_stats = action_head(hidden[:, logits_to_keep])` — whatever dict the head returns *becomes* `policy_stats`. `{"h": ...}` flows through both untouched. |
| `infer_step` `vlm_worker.py:474-477` | maps generically over the head's dict (`{k: v.squeeze(0).float().cpu().numpy()}`), so `{"h": ...}` needs no fourth branch site — but **dict values must be tensors**, the comprehension is unconditional. |
| `env/objectnav_continuous.py`, `ChunkExecutor`, reward, screening | consume a chunk; head-agnostic. |
| value head, returns, `trainers.py` | operate on env steps only — the uniform-credit choice is what keeps these still. |
| the entire discrete path | separate branch throughout. |

### Capability dispatch, NOT a new `policy_head.type`

`policy_head_config['type']` is load-bearing in three places: a hard validator at
`vlm_worker.py:59` (`not in ["discrete","continuous"]` raises), the wrapper construction at
`:777`, and the `_forward_embeds` gate at `:644`. Adding a `"flow_sde"` string means editing the
validator and the wrapper, and makes every future `== "continuous"` comparison a place to forget
the third case. Instead **keep `type == "continuous"` and dispatch on the head's capabilities**
(`getattr(head, "chain_log_prob", None)`), the idiom the actuator seam at `rollout_core.py:204`
already established. A head without `chain_log_prob` behaves exactly as today.

Net: **one new module, three branch sites, zero config or validation changes.**

### The three branch sites

1. `rollout_core.py:161` — sampling. The capability check **must precede** the existing
   `mu = policy_out["mu"]`, which `KeyError`s on `{"h": ...}`.
2. `rollout_core.py:374` — the `old_log_prob` recompute. The dtype anchor on `:375`
   (`dtype=policy_stats['mu'].dtype`) is part of the same `KeyError` surface — fixing the branch
   while leaving the anchor produces a second, more confusing failure.
3. `vlm_worker.py:925` — `rl_loss`.

### The new module: `longnav/utils/flow_sde_policy.py`

```
forward(hidden)                    -> {"h": ...}       # satisfies the policy_stats contract
sample_chain(h, N, rng)            -> (chain, positions, chunk, logprob)   # hybrid sampler;
                                                       # N=0 is the eval/deploy ODE, same loop
chain_log_prob(h, chain, positions)-> per-step tensor  # the PPO recompute, fp32, eval-pinned
decode_action(flat)                -> chunk            # the EXISTING seam; = last chain element
```

Compute wiring: the backbone runs **once** per env step (it already does — `h` is the cached
feature); the velocity expert runs `K` times at rollout and `N` times in the update, so update
cost is sublinear in `K`. Expert activation memory in the update scales with `N`; at ~1M
parameters this is noise here, but it is the term that grows if `N` does.

## Failure modes designed for, not discovered

1. **Dropout in the velocity field silently corrupts the PPO ratio.** `old_log_prob` is
   *recomputed from a fresh forward* at `rollout_core.py:374`, not carried from rollout; the
   velocity net has `dropout=0.1` and training runs under `model.train()`. Both recompute paths
   must pin the velocity net to eval — inside the head's methods, caller-independent (the
   `LatentIntentHead` / `decode_action` precedent). Worse than the earlier instance: it corrupts
   the log-prob, not just the action.
2. **The entropy bonus `KeyError`s — make the raise deliberate.** `rl_loss:960` evaluates
   `self._continuous_entropy(policy_stats['log_std'])` whenever `entropy_bonus` is set, and this
   head has no `log_std`. Raise with a message saying chain entropy is schedule-fixed and the
   bonus is meaningless here. Do not "fix" it by making `sigma_t` learnable: a learnable noise
   scale is an unopposed entropy bonus on `log sigma`, and `LATENT_RL_BAKE.md` documents where
   that ends (sigma 3541 in one run). Fixed schedule in v1; adaptive sigma is a gated upgrade,
   co-annealed with lr and clip because of the `1/sigma^2` sensitivity.
3. **`use_oracle_action` / DAgger must hard-error.** `rollout_core.py:165-178` substitutes an
   env-provided action and scores it under the policy; `rl_loss:935-941` carries DAgger handling.
   An oracle action is a *chunk*; no unique chain ends at it, and any bridge-sampled chain yields
   a density that is not the marginal — silently wrong credit. The sampling branch raises if
   `use_oracle_action` is set with a chain head.
4. **`continuous_action_clip_low/high` must not touch chain latents.** Note the existing path
   computes the log-prob of the *clipped* action under the *unclipped* Gaussian
   (`rollout_core.py:181-186`) — a pre-existing inconsistency not to replicate.
5. **`logprob_reduction`** ('sum'/'mean') must be set deliberately, not inherited: the summed
   quantity here is `N * 60` terms and its scale interacts with the clip range and lr.

## Instrument panel (log every iteration, full cadence)

Full cadence is not a nicety: the every-Nth-step sampling artifact produced false "clamp never
binds" and "h_std accelerating" readings **three separate times** in the latent programme.

- `ode_eval_success` — THE metric; all selection on this and nothing else.
- `sde_train_success` — diagnostic; the `ode - sde` gap is the fidelity gauge (widening => lower
  `a` or `N`).
- Clip fraction + joint-ratio histogram — staleness gauge for the `N` ratio-visible terms.
- **Unclipped drift gauge** — KL or mean displacement between old and new per-step means on a
  fixed probe batch of stored latents. Mandatory at small `N`: the `K - N` deterministic steps
  drift invisibly to the ratio, so ratio health *understates* true policy drift. Do not raise PPO
  epochs per batch on ratio health alone.
- Executed-action noise proxy (high-frequency energy of chunks) vs `N` — the trigger for the
  CPS-style coefficient-preserving step (upgrade path, not v1).

## Validation before the first RL launch, in order

The latent programme's most expensive lesson: its exploration/fidelity trade was first measured
after a 30-hour bake. Here everything below runs with **no RL training at all**, and the first
item needs no simulator.

1. **Marginal-preservation test (first — catches every sign/coefficient/convention bug in one
   shot).** Freeze `checkpoint-12000`. Sample ~1k chunks with the pure ODE and with the hybrid
   sampler at several `(N, a)`. Compare executed-chunk distributions: per-dim mean/std plus MMD or
   sliced-Wasserstein. The hybrid must match the ODE within sampling error at small `a` and
   degrade smoothly as `a` grows; a mismatch at small `a` is a score-sign/coefficient/convention
   bug. This supersedes running the navigation sweep first — the sweep measures the downstream
   consequence, which is dearer and confounded.
2. **Noise sweep, inference only.** Roll `checkpoint-12000` through `eval_objectnav_policy.py`
   with the hybrid sampler at the surviving `(N, a)` grid on the fixed 101-episode sample101 set
   (the `--gap` and budget traps in `SAMPLE101_EVALS.md` apply unchanged). Read oracle success and
   oSPL_fix against the known 0.663 / 0.348: the upper bound on `a` is where fidelity visibly
   pays; the lower bound is where SDE rollouts are indistinguishable from ODE rollouts in
   *behavioural spread* — under that, the policy owns too little randomness to learn from.
3. **Determinism unit test.** `chain_log_prob(h, chain, positions)` twice under `model.train()`
   must be bit-identical; likewise `sample_chain` under a fixed generator. Proves the eval-pin
   lives inside the head and is caller-independent.
4. **The rollout seam (tolerance, not equality).** The external spec's recompute test demands
   `logprob_new == logprob_old` exactly when theta is unchanged; in this framework that identity
   holds **only for identical `(h, chain)` inputs**. Across the seam it cannot: `old_log_prob`
   comes from a **full forward** while rollout used a **kv cache** (`rollout_core.py:354-355`
   says so), and the training forward adds unmerged LoRA and gradient checkpointing — three
   numerical paths over the same weights. So: assert per-step
   `|rollout_logprob - old_log_prob|` under a tolerance in nats. This is the **only** check that
   can see rollout-side dropout, because it alone compares a sampled quantity against a
   recompute; the step-zero training ratio (item 5) compares two recomputes and is structurally
   blind to it.

   **The floor is real, small, and measured — but only at low dimension.** Training-run
   trajectory dicts are never persisted (`dump/rl_training/*/dbg/result_*.pkl` holds episode
   outcomes only), and every past RL run was discrete, so there is no log archaeology to lean on.
   The one real continuous fixture,
   `tests/forward/fixtures/continuous_dummy_rpp/component_a_traj_batch.pt` (`action_dim = 2`,
   112 valid steps), measures: mean `|gap|` 0.095 nats against mean `|log_prob|` 11.03 (~1.9%),
   median 0.106, max 0.211 — **but the signed mean is +0.064 vs std 0.105 (86% one-sided, ~6.5
   SE from zero)**: a systematic component, so it does **not** extrapolate as zero-mean
   `sqrt(dim)` noise. At `N = 1` (60 terms), `sqrt` scaling gives ~0.5 nats and linear ~2.9; at
   `N = 3`, ~0.9 and ~8.6. Linear-at-`N=3` would crowd a PPO clip range, so **measure at the real
   dimension before relying on the check** — extending
   `tests/forward/scenarios/continuous_dummy_rpp.yaml` to the chunk dimension is cheap since the
   fixture machinery exists. This floor does **not** contaminate the PPO ratio itself (both of
   its legs are full forwards); it bounds only this seam check.
5. **The training seam.** Mean `|log ratio|` at step zero under the same calibrated tolerance —
   catches train-mode dropout in the training forward. Items 3, 4, 5 are three different seams;
   none subsumes another.

## The continuous path is missing diagnostics the literature assumes — a pattern

Two diagnostics this recipe's literature leans on are **dead code for every continuous head**:

- **KL-to-reference**: `rollout_core.py:389` reads
  `if self.rl_algo_config.use_ref and self.policy_head_config['type'] != "continuous"`.
  Flow-GRPO pairs its techniques with a reference-KL anchor; here restoring it is real work, not
  a config flag. Not needed to launch — but if early runs drift destructively off the SFT policy,
  this is the standing remedy, and it must be revived *before* ever adopting denoising reduction.
- **`train/rollout_kl_divergence`**: `vlm_worker.py:972` has the same `!= "continuous"` guard —
  and that metric is *exactly* the rollout-vs-recompute discrepancy validation item 4 needs,
  already computed and logged for discrete heads. Dropping the guard would give item 4 as a free
  logged metric, **but verify first that `compute_full_kl_penalty` means anything on a summed
  Gaussian log-prob** — it is written for categorical logits.

Two gates with one shape is a pattern, not a coincidence: assume other continuous-path gaps exist
and check for the guard before relying on any metric's presence.

## The latent programme is paused, not abandoned

`run_bake_latent_v2` was stopped rather than run to 12000; its `checkpoint-500` is on disk and
measured. Recorded so the latent arm is not starved unmeasured by momentum:

| checkpoint | spread ratio (tau=1) | `z_0` heading std | note |
|---|---|---|---|
| latent pilot ck1000 | 0.906 | 0.336 | the checkpoint that shipped |
| bake v1 ck500 (w=1.0) | 1.920 | 0.181 | reconstruction regressing in all 12 metrics |
| bake v2 ck500 (w=0.3) | 1.562 | 0.173 | degrading more slowly; best latent artifact |

**Pre-committed comparison.** Once a flow-SDE RL number exists, run the latent arm from v2 ck500
under matched scenes, budgets, seeds and reward, with dead-dim masking on (579 of 1024 latent
dims are decoder-dead; unmasked they contribute pure ratio variance). Flow-SDE is the control,
the latent the treatment — running the latent first would have produced a number interpretable
only against itself, and conflated "RL improves navigation" with "RL recovers bake damage". A
further latent bake (v3: `hf` detached, diversity probe normalised by `h_std.detach()`) is
justified **only** if the comparison shows structured `c`-space exploration winning despite its
weaker start.

## Deliberately not done

- **Denoising reduction** (collecting at fewer steps than inference): the collection policy
  measurably stops being the inference policy, and the KL-to-reference anchor the technique is
  normally paired with is dead code here (`:389`). Revive `use_ref` first if ever revisited.
- **Fixed last-K stochastic windows** (DPPO-style): surrenders mode control; see the reversal
  note — random positions get the variance benefit without that price.
- **Per-denoising-step advantages** (DPPO's two-level MDP): the escalation if the collapsed ratio
  proves too high-variance even per-step-clipped; a trustworthy denoising-time value signal is a
  prerequisite that does not exist.
- **GRPO**: its group baseline assumes cheap identical-state resets and terminal-only reward;
  this env has dynamics and dense reward. PPO with the learned critic.
- **Learnable or annealed `sigma_t`** in v1: see failure mode 2. Gated on `ode_eval` progress if
  ever, with lr and clip co-annealed.
- **Marginalising `z_0`**: intractable; conditioning on it is the coherent choice.
- **A new `policy_head.type`**: capability dispatch instead, for the reasons above.
- **CPS-style coefficient-preserving step**: the upgrade if executed actions grow "buzzy" as `N`
  or `a` rises (the noise-proxy gauge is its trigger); same Gaussian log-prob structure, kills
  noise accumulation by construction.
- **Entropy-adaptive position selection** and **pathwise gradients through the deterministic
  segments**: known upgrade paths, in that order, neither before the instrument panel exists.
- **Pinning a state-dependent noise scale, for the record**: the latent's `sigma` became strongly
  state-dependent by training (`to_log_sigma.bias` moved 0.006 nats in 500 steps while the
  zero-initialised weight grew to std 0.0037, so `W_s.h ~ +6` carried everything). Whatever
  state-dependence that encodes was learned; a fixed schedule discards the analogue here by
  construction. It is the reason not to assume a scalar noise scale is free — and the first place
  to look if the fixed schedule underperforms in heterogeneous scenes.

## Post-mortem: the drift-correction sign (2026-08-14)

`_sde_transition` shipped with `mu = x + dt*(v + (sigma^2/2)*score)`. The correct
reverse-time drift under this codebase's t:1->0, dt<0 convention is
`v - (sigma^2/2)*score` (derive via Fokker-Planck in s=1-t, or see piRL Eq. 8). The "+"
variant is the marginal-preserving drift for the OPPOSITE integration direction: with
dt<0 it pushes mass away from the score at every step, so instead of cancelling the
injected diffusion it doubles it -- the sampler converges to a systematically
over-dispersed policy (~4-6x terminal std on an analytic Gaussian flow at a=0.8).

Why months of verification never caught it, recorded so the next error of this class is
found sooner:
1. Distortion scales as a^2: ~1.5% at the configured a=0.15, invisible to the
   marginal-preservation test, which checked the a->0 limit where both signs are exact.
2. Every internal-consistency check is sign-invariant BY DESIGN: sampler and scorer
   share the transition function, so ratios, seams, recompute equality and toy-env
   learning all certified that we faithfully trained SOME policy -- never WHICH policy.
3. The deviation sweeps had no ground truth for how much deviation is correct.

THE LESSON: internal consistency cannot validate distributional correctness. Every
sampler needs at least one test against an EXACT analytic solution, in the regime where
the studied terms are LARGE (here: high noise). That guard now exists
(tests/test_flow_sde_policy.py::TestHighNoiseMarginalPreservation, fails at 5.84x under
the old sign).

Strategic consequence: the working noise regime documented by piRL (a~0.5; a~0.2 fails)
was unreachable -- distortion grew quadratically exactly along the escape route -- and
the low-noise instability symptoms piRL predicts (clip spikes, low-sigma gradient
blowups) were treated here by LOWERING lr instead of raising `a`, because raising `a`
was genuinely broken. Fixed 2026-08-14; all prior SDE-sampled runs drew from the
distorted sampler (evals were pure ODE and are unaffected).

Exact mechanism (independent audit): piRL Eq. 9 writes mu = A + [v + (sigma^2/2tau)
eps_hat]*delta, self-consistent ONLY when delta is read as d(tau) < 0. Reading delta as
positive flips both terms; transcribing just the correction term while keeping the local
x + dt*v produces exactly the shipped `+`. A half-transcription across a sign boundary --
the specific trap to name in review: any formula imported from a paper must be imported
WHOLE or re-derived whole, never term by term.

Wrong-sign terminal-spread inflation is a curve, not a number (analytic Gaussian, K=10):
1.10x / 2.01x / 6.42x at a = 0.15 / 0.5 / 1.0, growing as exp(O(a^2)).


## Audit corrections (2026-08-14, independent re-derivation)

1. **The last-step exclusion rationale was refuted.** The score's 1/t cancels exactly in
   the drift: (sigma_t^2/2)*score = -a^2*eps_hat/(2(1-t)), monotonically shrinking toward
   t=0 (coefficient 0.139 at t=0.1 vs 1.250 at t=0.9 for a=0.5); terminal marginals are
   identical for n_exclude_last in {0,1,2,3}. The singular end is t=1. `n_exclude_last`
   is a variance choice; 0 is legal and matches RLinf's default.
2. **First-step noise rule changed to RLinf's.** sigma_t = a*sqrt(t/max(1-t, 1/K)):
   first-step injected noise equals `a` exactly at every K, making our `a` commensurable
   with piRL's sweep (their a~0.5 works, a~0.2 fails). The previous t<=0.95 clamp gave
   a*sqrt(19/K) -- K-dependent, 1.38x RLinf's at K=10.
3. **The schedule now has ONE implementation** (`_sigma`), read by the transition, the
   scorer and the position weights alike.
4. **The partial likelihood is an approximation, named as such.** Summing only the n
   stochastic factors conditions on the stored deterministic sources, whose positions
   are themselves theta-dependent; the ratio is a conditional likelihood, not the chain
   density. This is the standard DDPO/Flow-GRPO estimator and RLinf does the same.

## Evidence invalidated by the sign bug (register)

- The (a, N) chunk-deviation sweeps and the credited-channel probe's SDE arms sampled the
  wrong-sign noise process; only the probe's ODE arm stands. Re-run post-fix before use.
- The "1x flat / 10x coherent harm" lr-ladder inference is CONFOUNDED: at 10x we
  optimized a mis-specified density; the harm no longer evidences channel blindness (H2
  demoted). The signfix A/B (identical config, sign the only variable) is the clean test.
- Behavioral conclusions at a=0.15 (motion statistics, parity, ODE eval baselines) remain
  VALID: distortion there was percent-level and all evals ran pure ODE.

## Why the bug was self-sealing (append to the lessons)

At a=0.15 the distortion was invisible in every observable we watched; at the a>=0.5
where it becomes visible, degradation would have read as "high noise destroys fidelity --
stay low", CONFIRMING the setting that kept the bug hidden and learning impossible. A
faithful-looking initialization was not evidence against the bug; it was one of the
bug's effects on our inference. Corollary for practice: when a hyperparameter direction
is theoretically motivated but empirically "fails", verify the failure is not produced by
a defect that scales along exactly that direction.

See docs/RLINF_COMPARISON.md for the external-reference recipe and the v5 replication
spec this all feeds into.

## Post-mortem: the erosion lived in the backbone, not the head (2026-08-14)

Question: in the collapsed both-halves-training runs (lr10x twins at 2e-5; pirl_lr at
5e-6, eval 0.917 -> 0.625 over 36 cycles with h_drift settling at -6.1%), which module
carried the degradation? Measured on pirl_lr2's ck7 vs ck31 (the decline window), offline,
no rollouts:

| channel | evidence |
|---|---|
| head weights | readout rel-drift 1.0e-4, velocity field 2.5e-4 over the window -- 15-38x LESS than the LoRA's 3.8e-3 (which accelerated: SFT->ck7 1.3e-3, ->ck31 4.5e-3) |
| head as a function | mu shift on matched (h, x, t) probes: 0.6-3.7% of sigma_k per step (mean 2.2%) |
| head, chunk-level | full ODE decode from identical h + z0: ck31 head vs ck7 head moves chunks **0.376 cm** mean -- inside the 0.24-0.88 cm band the SDE exploration noise itself occupies |
| backbone, chunk-level | a 6.1% h-shrink alone (the canary's reading, same head) moves chunks **0.577 cm** mean / 1.228 cm terminal -- and this is a LOWER bound: h_absmean is blind to directional drift at constant scale, so the true backbone channel is strictly larger |

Verdict: **the head stayed essentially the SFT head; the backbone drifted and dragged h
with it.** The policy died through the h-channel, exactly the channel the frozen-head +
merged-base + ref-KL arm (flow_sde_freezehead_overfit32) pins, gauges, and can leash.
Caveats: synthetic-h probes (Gaussian at the measured absmean scale) and ck7 as the early
reference; the fully decisive version is the checkpoint-swap eval (final backbone + init
head vs init backbone + final head), which needs the fleet and is superseded in practice
by the freeze-head arm itself -- if that arm survives the 10x lr that killed both twins,
the attribution is confirmed interventionally.
