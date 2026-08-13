# The piRL spec vs our design: what to adopt

Compares `FLOW_SDE_PIRL_SPEC.md` (written independently, with **no knowledge of this project**)
against `../FLOW_SDE_RL.md`. Written after the external spec arrived; conclusions here supersede
`FLOW_SDE_RL.md` where they conflict, and that document should be updated to match.

## Independent convergence — worth noting before the disagreements

Reached separately, in both documents:

- collapsed/summed Gaussian log-probs over denoising transitions, one env-step advantage
  broadcast uniformly, **no inner-step GAE or discounting**;
- PPO with a learned critic rather than GRPO (they add the reason: GRPO's group baseline assumes
  prompt-style resets, robotics has dynamics);
- store the transitions and **recompute `mu` through the velocity net under current theta** —
  storing `logprob_old`, or reusing `mu_old`, is wrong;
- the joint ratio's dispersion grows with the number of stochastic steps, and per-step clipping is
  the alternative;
- eval integrator must be *exactly* the deployment integrator, sharing a code path.

Two documents deriving the same structure from different starting points is the main evidence that
the structure is right.

## Adopt: five things the spec has that we lacked

1. **The score-correction identities, concretely.** We wrote `dx = [v + (sigma^2/2)*score]dt +
   sigma dW` and never said how `score` is obtained. Theirs: `eps_hat = x_t + (1-t)*v`,
   `x0_hat = x_t - t*v`, `score = -eps_hat / max(t, t_min)`. With two rules we had not considered:
   **sigma must appear in the drift correction and the noise or in neither** (they are halves of
   one identity, and note `sigma^2` vs `sigma` — a "noise_level" knob that scales one and not the
   other is silently wrong); and **do not `detach()` the score**, because it contains `v_theta` and
   detaching changes the estimator into a different algorithm.
2. **`n_exclude_last`.** The score carries `1/t`, singular as `t -> 0`. Exclude the final 1-2
   positions from stochastic selection and floor `t` at ~1e-3 regardless. We had not noticed this
   at all.
3. **Two learning-rate couplings.** Gradient magnitude grows ~`N` (N terms share one advantage), so
   sweeping `N` at fixed lr measures step size, not exploration — divide by `N` or scale lr ~1/N,
   and say which. Separately, smaller sigma gives larger log-prob gradients (~1/sigma^2), so
   changing the noise scale requires retuning lr. Do not sweep both at once.
4. **The critic must see observations only** — never denoising latents or denoising time.
5. **The marginal-preservation test (their 8.1)** is better than our noise-schedule sweep and
   should run *first*: freeze the SFT checkpoint, sample ~1k chunks under pure ODE and under the
   hybrid at several `(N, a)`, compare chunk distributions (per-dim mean/std plus MMD or
   sliced-Wasserstein). It must match at small `a` and degrade smoothly. This isolates the sampler
   with no simulator at all, and catches every sign/coefficient/convention bug in one shot. Our
   sweep measures the downstream navigation consequence, which is what we ultimately care about but
   is far more expensive and confounded — run theirs first, ours second.

Also adopt their RNG hygiene note (`x_0` and per-step noise must be independent fresh draws;
correlated streams across parallel envs bias advantages) and their warning about flattening
denoising steps into the trajectory buffer, where GAE's gamma/lambda would silently decay across
inner steps. **We are safe from the latter by construction** — the chain is stored as one flat
660-vector per env step, so the buffer's time axis is env steps — but only by accident, and it is
worth stating.

## Conflict 1: `N = K` (ours) versus small `N` with random positions (theirs)

We decided to make **all** 10 transitions stochastic. They call `N=K` "known unstable at scale —
treat as a diagnostic bound, not a target," and start at `N=1` (the proven piRL config).

**They are right, and our objection does not apply to their scheme.** We rejected step reduction
because early high-noise steps set *which mode* the sample lands in, so restricting gradient to a
fixed late window surrenders mode control. But their positions are **sampled uniformly at random
per chunk**, and `v_theta` is one shared network across all `t` — so over training every `t`
receives gradient. Mode control is preserved. Our argument was against *fixed* last-K (DPPO), and
they are not proposing that.

So: `N` becomes a knob, `N=1` the starting point, `N=K` retained as the diagnostic bound it is.

**Their own cost, which they flag and we should not lose:** at small `N`, ratio health
*understates* true policy drift, because the `K-N` deterministic steps still change (`v_theta` is
shared) while contributing nothing to the ratio. The trust region therefore does not cover the
whole policy change. Their patch is an unclipped drift gauge — KL or mean displacement between old
and new per-step means on a fixed probe batch of stored latents. **Do not raise PPO epochs per
batch on ratio health alone.**

## Conflict 2: deploy with the SDE (ours) versus the ODE (theirs)

We argued the deploy sampler must be the SDE, because RL modifies `v` under no constraint keeping
it a valid velocity field, so the ODE's and SDE's marginals may diverge after training.

That reasoning is sound **for `N=K`** and largely dissolves at small `N`: when 9 of 10 steps are
already the deterministic ODE, the training sampler *is* nearly the deploy sampler, so the mismatch
is small by construction rather than by argument. Adopting small `N` therefore also buys back
**deterministic evaluation**, which is worth more to us than to most projects — checkpoint
selection in the latent programme was repeatedly confounded by stochastic metrics.

Their discipline is the right one: `ode_eval_success` is THE metric and the only thing checkpoint
selection ever sees; `sde_train_success` is diagnostic, and the **gap between them is the fidelity
gauge** — a widening gap means lower `a` or lower `N`.

## Where our document still has the edge

The spec has no knowledge of our framework, so these remain ours to carry:

- **Their test 8.2 is sound in concept but needs a tolerance here, not an equality.** It asks that `logprob_new == logprob_old`
  exactly when theta is unchanged. In this framework `old_log_prob` is recomputed by a **full
  forward** while the rollout used a **kv cache** (`rollout_core.py:354-355`), and the training
  forward adds unmerged LoRA and gradient checkpointing. Their check needs our tolerance treatment
  — see `FLOW_SDE_RL.md` validation 3a/3b/3c.
- **Dropout.** They never mention it. Our velocity field has `dropout=0.1` and training runs under
  `model.train()`, so both recompute paths must pin it to eval or the ratio is corrupt at step
  zero.
- **fp32 accumulation** over the log-prob sum.
- **The three continuous-head exclusions** (`use_ref` at `rollout_core.py:389`,
  `train/rollout_kl_divergence` at `vlm_worker.py:972`, and training-run trajectory tensors not
  persisted -- though `tests/forward/fixtures/continuous_dummy_rpp/` does store a real continuous
  `traj_batch` with both log-prob columns, which is where the seam tolerance was measured).
  Their instrument panel assumes these diagnostics exist; here two of them are gated off.
- **`z_0` is unusually large for us.** They keep `z_0` as the policy's real deployed stochasticity,
  which is reasonable when it is modest. Ours measures **0.737 rad terminal heading std, with
  endpoint scatter exceeding the mean path length** — so under an ODE deployment that scatter *is*
  the deployed behaviour. This does not change the ordering decision, but it makes the
  `ode_eval` vs `sde_train` gap more important for us than for them, and it is the strongest
  remaining argument for eventually pushing `N` up.

## Net change to the plan

`N` is a knob defaulting to 1, positions uniform at random with `n_exclude_last >= 1`; deploy and
eval with pure ODE, sharing the sampler code path; `ode_eval_success` is the selection metric;
add the unclipped drift gauge and the marginal-preservation test before training. Everything in
`FLOW_SDE_RL.md` about the touch points, the recompute hazards, dropout, fp32 and the continuous-
head exclusions stands unchanged.
