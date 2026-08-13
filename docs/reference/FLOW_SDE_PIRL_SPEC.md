# EXTERNAL REFERENCE — Hybrid Flow-SDE RL Fine-Tuning for Flow-Matching VLAs

**Provenance: written by an independent session with NO context of this project.** Stored
verbatim as a reference. It is not this project's design — see `../FLOW_SDE_RL.md` for that, and
`FLOW_SDE_PIRL_COMPARISON.md` for where the two agree, differ, and which we adopted.

Its base recipe is piRL-style Flow-SDE + PPO (arXiv 2510.25889), generalised with a tunable `N` =
number of stochastic denoising steps per chunk. piRL implements the `N=1` special case.

---

## 0. System overview

```
SFT flow-VLA checkpoint (e.g., pi0/pi0.5-style: VLM backbone + flow-matching action expert)
        |
        v
Parallel sim rollouts --> chunks sampled with HYBRID sampler (N SDE steps, K-N ODE steps)
        |                                |
        v                                v
Env rewards --> GAE on ENV MDP only --> PPO update on the N Gaussian log-probs per chunk
        |
        v
Eval + checkpoint selection: PURE ODE sampling, always
        |
        v
Deploy: pure ODE (identical integrator config to eval)
```

Fixed design decisions (do not re-litigate in code):
- Policy stochasticity for RL ratios comes ONLY from the N noisy denoising steps.
- Advantage is computed per ENV step (per chunk), broadcast uniformly to all N inner log-prob
  terms. No inner-step discounting. No inner-step GAE. Critic sees observation only, never
  denoising latents.
- sigma schedule is FIXED (not learned, not annealed) in v1.
- Deployment/eval = deterministic ODE. The SDE machinery is training scaffolding.

---

## 1. Notation & conventions (pin these first — #1 source of silent bugs)

- `K` = total denoising steps per action chunk (e.g., 10). `dt = 1/K` (uniform grid assumed; if
  non-uniform, thread actual per-step `dt[k]` everywhere below).
- Time convention used in this spec: `t=1` is pure noise, `t=0` is clean action. Sampling
  integrates t from 1 -> 0. Latent at time t is `x_t`; chunk = `x` at t=0, shape
  `[H, action_dim]` flattened.
- Velocity convention: model trained so that `v_theta(x_t, t, obs) ~= E[eps - x_data | x_t]`, i.e.
  the Euler update is `x_{t-dt} = x_t - dt * v_theta(x_t, t, obs)`.

> **GOTCHA (convention flip):** pi0-family and Flow-GRPO-family codebases disagree on whether t=0
> or t=1 is noise, and on the sign of v. EVERY formula below (Euler sign, score identity, sigma
> schedule argument) flips with the convention. Before writing any sampler code, print the
> pretrained model's own sampling loop and derive the two identities in section 3 from it. Do not
> copy formulas across codebases. A sign error in the score term does not crash — it silently
> trains on a wrong distribution. Unit test in 8.1 catches this.

---

## 2. Hybrid sampler (rollout-time action generation)

Per chunk:

```python
def sample_chunk(obs_features, N, K, sigma_schedule, rng):
    # 1. Draw stochastic positions
    valid_positions = range(0, K - n_exclude_last)   # see GOTCHA below
    sde_idx = rng.choice(valid_positions, size=N, replace=False)  # uniform, no replacement

    x = rng.normal(size=chunk_shape)                 # x_0-noise: KEEP. This is the
                                                     # policy's real stochasticity and
                                                     # survives to deployment.
    logprob_sum = 0.0
    for k in range(K):                               # t goes 1 -> 0
        t = 1 - k * dt
        v = velocity_expert(x, t, obs_features)      # backbone features CACHED, computed
                                                     # once per chunk (see section 7 compute)
        if k in sde_idx:
            mu, sigma_step = sde_step_mean_and_std(x, v, t, dt, sigma_schedule)
            noise = rng.normal(size=chunk_shape)
            x_next = mu + sigma_step * noise
            logprob_sum += gaussian_logprob(x_next, mu, sigma_step)  # store per-step too
            # STORE for the update: (x, t, x_next, sigma_step)  -- see section 4 recompute note
        else:
            x_next = x - dt * v                      # plain Euler ODE step. No noise,
                                                     # no score correction, no logprob.
        x = x_next
    return x, sde_idx, stored_transitions
```

- `sde_step_mean_and_std` implements section 3.
- **GOTCHA (last-step exclusion):** the score correction contains a `1/t` factor -> singular as
  t->0. Exclude the final 1-2 denoising positions from `valid_positions`
  (`n_exclude_last = 1 or 2`) AND floor `t` at `t_min ~= 1e-3` inside the score computation
  anyway. This is a deliberate, principled non-uniformity; don't "fix" it back to full-range
  uniform.
- **GOTCHA (do not reuse noise):** `x_0` draw and per-step SDE noise must be independent fresh
  draws. Sharing an RNG stream across parallel envs such that draws correlate will silently wreck
  GRPO-style baselines and quietly bias PPO advantages.
- **GOTCHA (eval path):** the eval sampler is the same loop with `N=0`. It must reuse THIS code
  path (flag), not a separate reimplementation — divergence between train-ODE-steps and
  eval-ODE-steps is a classic source of phantom train/eval gaps.

---

## 3. The SDE step: mean, std, and the score correction

The two identities (under the section 1 convention). From `x_t` and `v = v_theta(x_t, t)`:

```
eps_hat = x_t + (1 - t) * v          # predicted noise component
x0_hat  = x_t - t * v                # predicted clean chunk (Tweedie estimate)
score   = -eps_hat / max(t, t_min)   # grad log p_t(x_t)
```

sigma schedule (Flow-GRPO-style; `a` is THE noise hyperparameter, start small, e.g. the value the
reference piRL config uses):

```
sigma_t   = a * sqrt(t / (1 - t))    # clamp t away from both 0 and 1
```

SDE Euler-Maruyama step (reverse time):

```
drift      = v + 0.5 * sigma_t**2 * score        # SIGN of the score term depends on
                                                 # convention; derive, don't copy (section 1)
mu         = x_t - dt * drift
sigma_step = sigma_t * sqrt(dt)                  # per-step Gaussian std for the policy
```

Per-step policy: `x_{t-dt} ~ Normal(mu, sigma_step**2 * I)`.

**Wiring rules (each one is a real observed failure mode):**

1. **sigma appears in BOTH terms or NEITHER.** The `0.5*sigma^2*score` drift and the
   `sigma*sqrt(dt)` noise are two halves of one identity. If a config flag scales noise (e.g.,
   "noise_level"), it must scale the drift correction consistently (sigma^2 vs sigma — quadratic
   vs linear!). Grep for any code path where one is touched without the other.
2. **The correction is theta-dependent.** `score` contains `v_theta`, so gradients flow through
   the correction term in `mu`. Do NOT `detach()` the score when computing the log-prob for the
   PPO ratio. (Detaching changes the gradient estimator; it is a different algorithm.)
3. **ODE steps get NO correction.** The correction cancels injected noise; where there's no noise
   there's nothing to cancel. Adding it to ODE steps changes the deterministic path away from the
   pretrained model's.
4. **Coefficient-preservation option (upgrade, not v1):** if executed actions look "buzzy"/noisy
   as N or `a` grows (visible as reward degradation with rising N at fixed everything else), swap
   this step for CPS-style resampling: recombine `x0_hat`/`eps_hat` with the scheduler's exact
   interpolation coefficients and a partial fresh-noise mix (knob eta), instead of Euler+noise.
   Same Gaussian log-prob structure; kills noise accumulation by construction.

---

## 4. Likelihood bookkeeping for PPO

- Chunk log-prob = **sum of the N per-step Gaussian log-probs**. ODE steps contribute nothing
  (they are treated as dynamics).
- **GOTCHA (recompute under current theta):** during PPO epochs, the ratio needs `logprob_new`
  under current theta at the SAME transitions. You must store
  `(x_t, t, x_next, obs_features_or_obs, position k)` for each of the N stochastic transitions at
  rollout time, and recompute `mu` (a function of theta!) through the velocity expert in the
  update loop. Storing only `logprob_old` and re-sampling is wrong; storing `mu_old` and reusing
  it is wrong (that freezes theta).
- **GOTCHA (position conditioning):** the ratio is per sampled positions — the network never sees
  which positions were chosen (it only ever sees the current `t`, which flow models already
  condition on). Do not add index-set inputs to the network.
- Ratio for clipping: use the JOINT ratio `exp(sum logprob_new - sum logprob_old)` per chunk (one
  clip per chunk, advantage is per-chunk), OR per-step ratios each clipped with the shared chunk
  advantage (piRL-style). Pick one, document it, and keep it fixed across N sweeps — the two
  behave differently as N grows (see section 6 scaling).

---

## 5. Advantages, critic, and the outer/inner split

- **Env MDP only.** One transition per executed chunk: `(o_t, chunk_t, r_t, o_{t+1})`. GAE runs
  over this sequence with normal gamma, lambda. Denoising steps NEVER appear as timesteps in GAE,
  in the critic, or in any discounting.
- **Critic input = observation features only** (same frozen/cached backbone features the policy
  uses is fine; piRL ablations favored a ~4-layer MLP head). Never feed denoising latents `x_t` or
  denoising time into the critic.
- **Broadcast:** the single chunk advantage `A_t` multiplies each of the N per-step log-prob
  gradients identically. Implemented naturally as `A_t * (sum_k logprob_k)`. No per-step
  weighting, no inner gamma.
- **GOTCHA (accidental inner discounting):** if you adapt an existing PPO impl by flattening
  denoising steps into the trajectory buffer "because the shapes fit," GAE's gamma/lambda will
  silently decay across inner steps and you'll be running a biased variant. The rollout buffer's
  time axis must be ENV steps. Inner transitions live on a separate axis of the same buffer row.
- Algorithm: PPO with learned critic, NOT GRPO, unless you specifically have cheap identical-state
  resets and terminal-only rewards. (GRPO's group baseline assumes prompt-style resets; robotics
  has dynamics. piRL's PPO choice is the right default.)

## 6. The N knob: semantics, scaling, and tuning

`N in {1..K-n_exclude_last}`, uniform positions w/o replacement, resampled per chunk.

- `N=1` == piRL hybrid (start here; it's the proven config). `N=K` == full Flow-SDE (known
  unstable at scale — treat as a diagnostic bound, not a target).
- **What N trades:** more N = more exploration (compounded perturbations) + more gradient terms
  per env sample; less N = tighter ratios, cleaner executed actions, smaller train/deploy gap.
  Expect a modest interior optimum, task-dependent.
- **GOTCHA (confounded sweeps):** per-chunk gradient magnitude grows ~N (N log-prob terms share
  one advantage) and joint-ratio dispersion grows with N. Sweeping N with frozen lr/clip measures
  "bigger effective steps," not "more exploration." Either (a) divide the summed log-prob term by
  N, or (b) scale lr proportional to 1/N and widen tolerance on clip fraction, and say which in
  the config.
- **GOTCHA (sigma-lr coupling, independent of N):** smaller sigma => larger log-prob gradients
  (~1/sigma^2) => needs smaller lr. If you change `a`, retune lr. Do not sweep both simultaneously
  in one study.

## 7. Compute wiring

- **Backbone once, expert per step:** obs -> VLM backbone forward ONCE per chunk; cache features;
  action expert (small) runs K times at rollout, and N times in the update (only stochastic
  transitions are recomputed). Update-time cost is therefore ~sublinear in N w.r.t. total model
  FLOPs — but expert ACTIVATION MEMORY in the update scales with N; budget it.
- If fine-tuning the backbone too: its backward is shared across the N terms (one obs), gradient
  just accumulates — cost still ~once per chunk.
- Rollouts: massively parallel sim envs; this recipe presumes cheap rollouts. If rollouts are the
  bottleneck you are in the wrong paradigm (that's the RECAP/off-policy regime).

## 8. Tests & diagnostics (build these BEFORE training)

**8.1 Marginal-preservation unit test (catches sign/coefficient bugs):**
Freeze the SFT checkpoint. Sample M=1k chunks with pure ODE and with the hybrid sampler at several
(N, a). Compare distributions of executed chunks (per-dim mean/std + MMD or sliced-Wasserstein).
Hybrid must match ODE within sampling error at small `a`, degrade smoothly as `a` grows. If it's
off at small `a` -> score-term sign/coefficient/convention bug. This single test would have caught
every section 3 gotcha.

**8.2 Log-prob recompute test:** `logprob_new == logprob_old` exactly when theta unchanged (same
stored transitions). Catches recompute-path drift (section 4).

**8.3 Instrument panel (log every iteration):**
- `ode_eval_success` — THE metric. All checkpointing/selection/early-stopping on this.
- `sde_train_success` (or return) — diagnostic only. Gap `ode - sde` is the fidelity gauge:
  widening gap => lower `a` or N, or more denoising steps.
- Clip fraction + joint-ratio histogram (per-chunk product) — staleness gauge for the N noisy
  terms.
- **Unclipped drift gauge:** KL or mean-displacement between old/new per-step means on a fixed
  probe batch of stored latents. NEEDED because at small N, ratio health understates true policy
  drift (the K-N deterministic steps drift invisibly to the ratio). Do not raise PPO
  epochs-per-batch on ratio health alone.
- Executed-action noise proxy (e.g., high-freq energy of chunks) vs N — early warning for noise
  accumulation (section 3.4).

**8.4 Eval discipline:** eval integrator = deployment integrator exactly (same K, same solver,
N=0, same code path). Never checkpoint-select on stochastic rollout metrics.

## 9. Deliberate v1 simplifications (known upgrade paths, in order)

1. Fixed sigma schedule -> adaptive/annealed sigma gated on `ode_eval` progress (co-anneal lr +
   clip; the 1/sigma^2 sensitivity is why). Skip unless the train/eval gap is the demonstrated
   bottleneck.
2. Uniform index sampling -> entropy-adaptive position selection (E-GRPO-style) once per-step
   entropy telemetry exists. Image-domain "weight early steps" priors may not transfer to
   manipulation; don't import them blind.
3. Score-function-only gradients -> pathwise/value-gradient terms through deterministic segments
   (recovers the N<K truncated-gradient bias; converges toward Q-VGM-style methods).
   Research-grade.
4. Uniform broadcast -> stepwise credit, only if/when a trustworthy denoising-time value signal
   exists for actions. Not before.
5. Euler+noise SDE step -> CPS-style coefficient-preserving step, triggered by the section 8.3
   noise proxy.

## 10. Scope

Valid for: sim-based, on-policy, sharpening a strong SFT flow-VLA with dense-enough reward and
parallel envs. Out of scope by design: real-robot-data-bottlenecked training (RECAP-style
conditioning owns that regime), from-scratch RL, hard-exploration tasks (would want larger N /
full chain despite costs).
