# Flow-SDE RL: the collapsed-chain policy

Status: **design agreed, not yet implemented.** Companion to `LATENT_RL.md` (the latent-Gaussian
design), `LATENT_RL_ENV.md` (the RL environment, which this reuses unchanged) and
`LATENT_RL_BAKE.md` (what the latent bake measured and cost).

This document is the second attempt at the same problem — *give a flow-matching policy a
sampleable action space with a tractable log-prob, so PPO/GRPO can run* — and it is written
after the first attempt was measured. Read "Why not the latent" before changing anything here,
because most of the design is a reaction to specific costs that were paid, not to theory.

## Why not the latent

The latent-Gaussian construction (`c = h + sigma*eps`, decoder subsumed into the environment)
works and has a shipped checkpoint. Its cost is that **it makes us do RL on a worse policy.**

| | oracle | oSPL_fix |
|---|---|---|
| deterministic baseline (`run_cotrain_v3_nopose_mix/checkpoint-12000`) | 0.663 | 0.348 |
| the latent pilot that shipped | 0.545 | 0.257 |

That gap is not a tuning failure. `LATENT_RL_BAKE.md` derives it: at the sigma equilibrium the
model permanently pays reconstruction under effective context noise `s* = sigma/h_std ~ 0.7-1.0`,
and for a conditional that is near-deterministic that noise models nothing and is pure loss.
PointNav — whose conditional given the goal is nearly deterministic — is the worst-hit component
at 4.1x `mae_dx` against the deterministic baseline. **This cost is structural to any scheme that
injects exploration noise upstream of the decoder during SFT**, so it is not fixable by patching
the diversity term.

Flow-SDE pays none of it: the SFT objective is untouched, and RL runs on the strongest checkpoint
we have.

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
swamps any deliberate `Sigma`. The collapsed chain makes the same noise part of the action and
credits it exactly.

(Stated carefully because the stronger claim — "irreducible, PPO can never reduce it" — is false
and would be the attackable part of an otherwise sound decision. Under rung 1 the mean map
`ODE(h, z_0)` is trainable per `(o, z_0)`, so the `z_0`->action Jacobian does shrink over
training; it is the *rate* that is bad, not the possibility. **Rung 1 is therefore retained as the
fallback in the other direction**: if chain-ratio variance proves unmanageable, an action-space
Gaussian around the `z_0`-conditioned decode is the simpler policy to retreat to.)

(The table also records, incidentally, that the diversity term did its job: it moved variation out
of `z_0` and into `c`, 0.737 -> 0.17.)

## The policy

**RL here is a peer of SFT, not a layer bolted on top of it.** The probability-flow ODE and its
corresponding SDE share marginals at every `t` by construction, so sampling the chain reproduces
exactly the action distribution the SFT model already defines — the same prior, the same velocity
field, the same parameters, reached by a stochastic path instead of a deterministic one. RL
therefore **initialises at the SFT policy itself**, and the conversion costs nothing. That is the
whole difference from the latent, where RL acted on `c` — a variable SFT never had — and obtaining
it required inserting a bottleneck that changed the model class and charged `phi(s*)` forever.

Convert the probability-flow ODE to an SDE so each denoising step has a tractable Gaussian
transition, and **treat the whole chain as one action**:

```
log pi(Z | o) = sum_{t=1..T}  log N( z_t ; z_{t-1} + v(z_{t-1}, t, h)*dt , sigma_t^2 )
```

with `Z = (z_1 .. z_T)`, `z_T` the executed chunk, and one env-step advantage from GAE applied
uniformly across the chain.

This is exact, not an approximation. The environment consumes only `z_T`, but the remaining
coordinates are simply part of the action; PPO with ratio `pi_new(Z)/pi_old(Z)` is a correct
policy-gradient estimator for that MDP and marginalises correctly to the induced distribution over
chunks. The only statistical cost is that the chain's density is used in place of the intractable
marginal `log p(a|o)`, which adds variance — inherent to the construction and accepted throughout
this literature.

`sigma_t` here is the **per-step transition std**, with `dt` and any `sqrt(dt)` factor already
absorbed. The SDE framing is motivation only; the invariant that matters is that `sample_chain`
and `chain_log_prob` use the *identical* transition density, which the round-trip test in
Validation asserts. A convention mismatch between them is a silent wrong-ratio bug of exactly the
class failure mode 1 describes.

**`sigma_t` is bounded above by fidelity, not just by taste.** It is the dial that sets how much
of the policy's stochasticity the policy actually owns: `z_0` contributes unit variance per dim
and the injected noise contributes ~`sum_t sigma_t^2`, and only the latter is part of the action.
Larger `sigma_t` therefore hands RL a larger share — but the shared-marginal property above holds
for the *exact* score, and ours is recovered from an approximate velocity field
(`dx = [v + (sigma^2/2)*score]dt + sigma dW`). The larger `sigma_t`, the more the sampler leans on
that approximation and the further its marginals drift from the ODE's. Past some point the sampler
stops reproducing the SFT distribution **before RL has done anything**. The usable range is where
injected noise dominates `z_0` yet fidelity holds, which is exactly what the pre-training sweep in
Validation measures — so that sweep fixes the schedule, bounds the policy's share of the
randomness, and checks the peer property empirically rather than by appeal to the theorem.

A second, distinct cost: PPO's clip acts on the **joint** ratio, and joint KL upper-bounds
marginal KL — so the trust region is conservative, and a clip range calibrated on a
single-Gaussian head binds far sooner when log-ratio variance is ~`T`x higher. Expect elevated
clipfrac at unchanged *true* policy movement. Log clipfrac at full cadence from step zero (the
every-Nth-step sampling artifact has now bitten three times in this project) and treat a larger
epsilon, or the per-step clip, as calibration rather than as a response to instability.

**This is deliberately NOT DPPO's two-level MDP.** DPPO makes each denoising step its own MDP
timestep with its own advantage. Here credit assignment stays at the environment step. Flow-step
granular credit is asking for instability, and there is no environment feedback interleaving the
chain to justify it — every inner step shares the chunk's advantage because every inner step
contributed to the same chunk. The analogy is thinking-trace RL: uniform credit across the trace,
GAE across turns.

### Credit granularity and clipping granularity are separate axes

Collapsing to one ratio means the ratio is a product of `T` terms, so its log-variance is ~`T`x a
single Gaussian's, and the PPO clip becomes an all-or-nothing decision for the whole chain — one
unlucky step can gate the entire update.

Uniform *credit* (correct, and the design) does not require a single *clip*. Per-step clipping is
retained as an option, defaulting off. It lives inside the `rl_loss` branch (site 3), where it
must **bypass `policy_loss_fn`** and compute the clipped objective from per-transition ratios —
`policy_loss_fn` consumes `(B, S)` log-probs (`vlm_worker.py:942`) and has no per-transition axis.
So it is a contained loss-path fork, not a head-internal knob.

### `z_0` is excluded from the sum

`z_0 ~ N(0,I)` does not depend on `h`, so its log-prob is parameter-independent and cancels in the
PPO ratio. It is **stored** (the replay needs it) but **not summed**. This is not an optimisation;
including it would add a constant to both numerator and denominator and also risk implying we are
optimising the initial-noise distribution, which we are not.

## What this touches

Every contract below was read at the source, not assumed. Line numbers are as of this writing and
should be re-checked, but the *claims* are what matter.

### Not touched, and why that is structural

| component | why |
|---|---|
| `rl_core.py` — collation, GAE, advantages, `response_mask` | `pad_sequence(batch_first=True)` at `:52` is shape-agnostic; its only reshape is `if padded.dim()==3 and padded.shape[-1]==1: squeeze(-1)`. **Flatten the chain to `(steps, T*n_ticks*3)` and this file needs no change** — a trailing dim of 660 does not trigger the squeeze. |
| the dict passthrough, at **both** `vlm_worker.py:644-645` (`_forward_embeds`) and `:888-889` (the wrapper's forward, which is what `_training_forward` reaches through `ddp_model`) | `policy_stats = action_head(hidden[:, logits_to_keep])` — whatever dict the head returns *becomes* `policy_stats`. A head returning `{"h": ...}` flows through both untouched. |
| `infer_step` `vlm_worker.py:474-477` | maps generically over the head's dict: `{k: v.squeeze(0).float().cpu().numpy() for k, v in policy.items()}`. So `{"h": ...}` needs no fourth branch site — but **the dict's values must be tensors**, since that comprehension is applied unconditionally. |
| `env/objectnav_continuous.py`, `ChunkExecutor`, reward, screening | consume a chunk; head-agnostic |
| value head, returns, `trainers.py` | operate on env steps only — the uniform-credit choice is what keeps these still |
| the entire discrete path | separate branch throughout |

### Use capability dispatch, NOT a new `policy_head.type`

`policy_head_config['type']` is load-bearing in three places: a hard validator at
`vlm_worker.py:59` (`not in ["discrete","continuous"]` raises), the wrapper construction at
`:777`, and the `_forward_embeds` gate at `:644`. Adding a `"flow_sde"` string means editing the
validator and the wrapper, and it makes every future `== "continuous"` comparison a place someone
can forget the third case.

Instead **keep `type == "continuous"` and dispatch on the head's capabilities**, which is the
idiom this codebase already uses for the actuator seam at `rollout_core.py:204`
(`getattr(head, "decode_action", None)`). A head that does not define `chain_log_prob` behaves
exactly as today — the same property that made the actuator seam safe to add.

Net: **one new module, three branch sites, zero config or validation changes.**

### The three branch sites

1. `rollout_core.py:161` — sampling. The capability check **must precede** the existing
   `mu = policy_out["mu"]`, which would `KeyError` on a head returning `{"h": ...}`.
2. `rollout_core.py:374` — the `old_log_prob` recompute. Note the dtype anchor on the line above
   (`dtype=policy_stats['mu'].dtype`, `:375`) is part of the same `KeyError` surface: fixing the
   branch while leaving the anchor gets you a second, more confusing failure.
3. `vlm_worker.py:925` — `rl_loss`.

### The new module: `longnav/utils/flow_sde_policy.py`

```
forward(hidden)         -> {"h": ...}                      # satisfies the policy_stats contract
sample_chain(h)         -> (chain, chunk, logprob)         # Euler-Maruyama over sigma_t
chain_log_prob(h, chain)-> scalar                          # the PPO recompute
decode_action(flat)     -> chunk                           # the EXISTING seam; = last element
```

`decode_action` is not new work: `rollout_core.py:204` already routes through it, and for this
head it is the last chain element reshaped. `actions_continuous` stores the flattened chain, which
preserves the invariant that site's comment already states — *what is stored for the PPO ratio
stays the thing the log-prob was computed over.*

## The deploy-time sampler must ALSO be the SDE

The harness today evaluates with the deterministic ODE. **It must not, once RL has run.**

The peer property above holds because `v` is a valid flow-matching velocity field: the ODE and the
SDE share marginals *for such a field*. RL modifies `v` under no such constraint — after training
it is a drift function, not the velocity field of any probability path. So the shared-marginal
property **holds at initialisation and is not preserved by training**, and the ODE's and SDE's
marginals may diverge arbitrarily. Training the SDE and evaluating the ODE therefore deploys a
policy that was never optimised, and the gap grows with how much RL moved `v`.

Deploy with the SDE at the *same* schedule used for training. Consequences to accept, both benign:
the eval path needs the SDE sampler, and eval stays stochastic — which it already is, since `z_0`
is drawn fresh every step. Consequence to watch: pre- and post-RL evaluations must use the same
sampler, or the comparison measures the sampler change rather than the learning.

## Denoising steps: all of them, and the same count as deployment

`T = 10`, matching `num_inference_steps` in the checkpoint's `fm_config`. All 10 transitions carry
gradient; `z_0` does not, as above.

**Denoising reduction — collecting at fewer steps than inference uses — is rejected**, though it is
standard in this literature (Flow-GRPO). Euler discretisation error already perturbs the
shared-marginal property; fewer steps worsens it, so the collection policy measurably is not the
inference policy. Flow-GRPO pairs the technique with a KL-to-reference anchor, and `use_ref` is
dead code for continuous heads here (`rollout_core.py:389`) — so adopting it would mean taking the
technique without the mitigation it is designed to be used with. If it is ever revisited, revive
`use_ref` first.

**Training only the last K steps (DPPO-style) is also rejected**, for a different reason. It is
coherent — an exact policy gradient for the MDP that treats `z_{T-K}` as state — and it cuts ratio
variance proportionally. But early steps (`t` near 1, high noise) set coarse structure and *which
mode* the sample lands in, while late steps refine detail. Restricting gradient to late steps buys
variance reduction by surrendering mode control, which is the exact capability that distinguishes
this design from an action-space Gaussian. Given that up, the action-space Gaussian is simpler and
should be preferred instead.

**The variance lever is per-step clipping, not step reduction** — it bounds each transition's
contribution while leaving gradient flowing to every step, so early-step mode control survives.

## Failure modes designed for, not discovered

1. **Dropout in the velocity field will silently corrupt the PPO ratio.** `old_log_prob` is
   *recomputed from a fresh forward* at `rollout_core.py:374`, not carried from rollout. The
   decoder has `dropout=0.1` and training runs under `model.train()`, so the recompute would use a
   different dropout mask than the rollout and the ratio would be wrong at step zero — with no
   error and no shape mismatch. **Both recompute paths must pin the velocity net to eval.** This
   is the same class of bug already hit once with `decode_action`, where `LatentIntentHead` had to
   force decoder eval mode; here it is worse, because it corrupts the log-prob rather than only
   the action.
2. **The entropy bonus `KeyError`s — it does not silently no-op.** `rl_loss:960` evaluates
   `self._continuous_entropy(policy_stats['log_std'])` whenever `entropy_bonus` is set, and a
   `{"h": ...}` head has no `log_std`. That is *nearly* the right behaviour by accident; make it
   deliberate. The flow-SDE branch must **raise with a message saying chain entropy is
   schedule-fixed and the bonus is meaningless here**, rather than crashing obscurely on a missing
   key. Do not "fix" it by making `sigma_t` learnable: a learnable noise scale reintroduces exactly
   the unopposed-entropy-bonus blowup that `LATENT_RL_BAKE.md` documents for the latent `sigma`,
   which reached 3541 in one run.
2b. **`use_oracle_action` / DAgger is structurally incompatible and must hard-error.**
   `rollout_core.py:165-178` substitutes an env-provided action and scores it under the policy, and
   `rl_loss:935-941` carries explicit DAgger handling. An oracle action is a *chunk*; **no unique
   chain ends at it**, and any bridge-sampled chain yields a density that is not the marginal — so
   the credit would be silently wrong rather than merely approximate. The sampling branch must
   raise if `use_oracle_action` is set with a chain head.
3. **`continuous_action_clip_low/high` must not be applied to chain elements.** Note the existing
   continuous path computes the log-prob of the *clipped* action under the *unclipped* Gaussian
   (`rollout_core.py:181-186`) — a pre-existing inconsistency that should not be replicated here.
4. **Log-prob dimensionality.** The chain sums `T * n_ticks * 3` terms (10*20*3 = 600 at current
   settings), comparable to the latent head's 1024, so this is not new territory — but
   `logprob_reduction` ('sum'/'mean') must be set deliberately rather than inherited.
5. **`sigma_t = 0` has no density.** The deterministic ODE limit is not a policy. The noise
   schedule is a real design parameter, and the trade it controls — exploration against fidelity to
   the SFT policy — is measurable *before any training* by rolling out at a sweep of schedules.

## Validation before the first RL launch

The latent programme's most expensive lesson was that the exploration/fidelity trade was only
measured after a 30-hour bake. Here it is measurable with **no training at all**:

1. **Noise-schedule sweep, inference only.** Roll out `checkpoint-12000` through the existing
   `eval_objectnav_policy.py` harness with SDE sampling at several `sigma_t` scales, on the fixed
   101-episode sample101 set (see `SAMPLE101_EVALS.md` — the `--gap` and budget traps there apply
   unchanged). Read oracle success and oSPL_fix against the known 0.663 / 0.348. This directly
   answers "what does exploration cost in policy quality" and picks the schedule.
2. **Log-prob correctness.** A two-step toy SDE with known transitions, checked numerically. Note
   the round-trip identity holds **exactly only for identical `(h, chain)` inputs** — across the
   rollout/recompute seam it is a tolerance check, item 3b. Writing "exactly" there guarantees a
   spurious failure that then gets waived, which is how tolerance checks die.
3. **Three separate ratio checks. None subsumes another.** The single step-zero assertion this
   section originally proposed was both impossible and blind:
   *impossible*, because `old_log_prob` is computed by a **full forward** while the rollout used a
   **kv cache** (`rollout_core.py:354-355` says so explicitly), and `log_prob` additionally runs
   under `ddp_model` with unmerged LoRA and gradient checkpointing — three different numerical
   paths over the same weights, so the ratio is approximately 1 and never identically 1;
   *blind*, because `log_prob` and `old_log_prob` are **both recomputes**. If `sample_chain` ran
   with the velocity net in train mode at rollout, the actions came from a different policy than
   either recompute describes — and their ratio is still exactly 1. The check cannot see the very
   bug it was aimed at.
   - **3a. Determinism unit test.** `chain_log_prob(h, chain)` called twice under `model.train()`
     must be bit-identical, and likewise `sample_chain` under a fixed generator. This proves the
     eval-pin lives *inside* the head's methods (the `LatentIntentHead` precedent) and is
     caller-independent.
   - **3b. The rollout seam.** Assert per-step `|rollout_logprob − old_log_prob|` is under a
     tolerance in nats. This is the only check that sees rollout-side dropout, because it is the
     only one comparing a *sampled* quantity against a recompute.
     **The tolerance must be measured, not mined.** `rollout_logprobs` is written into the
     trajectory dict (`rollout_core.py:255`) but the dict is never persisted — `dump/rl_training/
     */dbg/result_*.pkl` holds episode outcomes only (`success`, `spl`, `distance_to_goal`,
     `pos_rots`, `oracle_action`), and the tensors are consumed by the update. Every past RL run
     was also **discrete**, so even persisted they would be per-token log-probs over a six-way
     vocab rather than a 600-term sum — the wrong magnitude to extrapolate from. Budget a short
     dedicated rollout with a continuous head, dumping both columns, before the flow-SDE code
     depends on this number.
   - **3c. The training seam.** Mean `|log ratio|` at step zero under the same calibrated
     tolerance, catching train-mode dropout in the training forward.
4. **fp32 accumulation, as a launch gate.** A bf16 sum of 600 log-prob terms carries ~2 decimal
   digits of relative error on a quantity whose *differences* must resolve to fractions of a nat.
   `chain_log_prob` must compute and sum transitions in float32 regardless of model dtype. Cheap,
   invisible when missing, and it corrupts the ratio in a way no shape check catches.

## The latent programme is paused, not abandoned

`run_bake_latent_v2` was stopped rather than run to 12000. Its `checkpoint-500` is on disk and
measured, and the latent arm remains resumable. Recorded so it is not starved unmeasured by
momentum:

| checkpoint | spread ratio (tau=1) | `z_0` heading std | note |
|---|---|---|---|
| latent pilot ck1000 | 0.906 | 0.336 | the checkpoint that shipped |
| bake v1 ck500 (w=1.0) | 1.920 | 0.181 | reconstruction regressing in all 12 metrics |
| bake v2 ck500 (w=0.3) | 1.562 | 0.173 | degrading more slowly; best latent artifact |

**Pre-committed comparison.** Once a flow-SDE RL number exists, run the latent arm from v2 ck500
under matched scenes, budgets, seeds and reward, with dead-dim masking on (579 of 1024 latent dims
are decoder-dead, and unmasked they contribute pure ratio variance). Flow-SDE is the control; the
latent is the treatment. Running the latent first would have produced a number interpretable only
against itself, and would additionally have conflated "RL improves navigation" with "RL recovers
bake damage" — which is the argument that decided the ordering.

A further latent bake (v3, with `hf` detached and the diversity probe normalised by
`h_std.detach()`) is justified **only** if that comparison shows the structured `c`-space
exploration winning despite its weaker start.

## Deliberately not done

- **Per-denoising-step advantages (DPPO's two-level MDP).** See above; retained as the escalation
  if the collapsed ratio proves too high-variance, and the per-step *clipping* option is the
  cheaper intermediate.
- **Marginalising `z_0`.** Intractable; conditioning on it is the coherent choice.
- **A new `policy_head.type`.** Capability dispatch instead, for the reasons above.
- **KL-to-reference.** The standard anchor in this literature (Flow-GRPO carries one) and it is
  **dead code for every continuous head**: `rollout_core.py:389` reads
  `if self.rl_algo_config.use_ref and self.policy_head_config['type'] != "continuous"`. Not needed
  to launch, but if the first runs drift destructively off the SFT policy, restoring it is real
  work rather than a config flag.

  **This is a pattern, not an isolated gap.** `vlm_worker.py:972` gates
  `train/rollout_kl_divergence` the same way — and that metric is *exactly* the rollout-vs-recompute
  discrepancy that validation 3b needs, already computed and logged for discrete heads. Two
  diagnostics are silently unavailable to the continuous path, so assume there are more and check
  before relying on any metric's presence. Dropping the `!= "continuous"` guard at `:972` would give
  3b for free as a logged metric rather than a bespoke script, **but verify first that
  `compute_full_kl_penalty` means anything on a summed Gaussian log-prob** — it is written for
  categorical logits and may not transfer.
- **Pinning `sigma` state-independently.** Note for the record that the latent's `sigma` became
  strongly state-dependent by training: `to_log_sigma.bias` moved only 0.006 nats in 500 steps
  (1.3e-5/step, under the 1e-4 `head_lr`), while `to_log_sigma.weight` went from *exactly zero* to
  std 0.0037, so `log sigma = W_s . h + b_s` with `b_s` pinned near `log(0.002)` and `W_s . h ~ +6`.
  Whatever state-dependence that encodes was learned, so discarding it is a trade rather than a
  simplification. It does not arise for flow-SDE with a fixed schedule, but it is the reason not to
  assume a scalar noise scale is free.
