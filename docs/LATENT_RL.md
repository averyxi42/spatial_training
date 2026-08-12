# Latent RL: a stochastic intent variable at the backbone/head interface

Status: design agreed, not yet built. Branch `latent_rl`.

## Why here, and not anywhere else

The flow head sees **nothing but `c`**. This is not an accident of implementation, it is
enforced: `FlowActionDecoder._attn_mask` puts `-inf` on the context-queries x
non-context-keys quadrant, and the context block is `c` reshaped (`context.view(N, 8, 128)`,
no projection). No image patch, no language token, no pose value reaches the decoder except
through `c`.

That makes `c` the only place in the stack where a single vector is *sufficient* for the
action. pi0-style per-layer cross-attention would distribute the policy across backbone and
head and leave no such point. **Keep the bottleneck.**

The plan is to invert where the stochasticity lives:

| | today | here |
|---|---|---|
| `c` given `o` | deterministic | `N(mu(o), diag sigma(o)^2)` -- **the RL policy** |
| chunk given `c` | stochastic (flow noise) | deterministic at RL time (fixed `z_0`, fixed steps) |

Execution-level stochasticity is SFT scaffolding. Once the intent level owns exploration it
is retired, which also makes `c -> chunk` **differentiable**, so a critic gradient can reach
the policy.

## The formalization

**Objects.**

* `o` -- the conversation up to and including the current turn, as the backbone sees it
  (images, text, `<pose>` values). Operationally: the pooled final-hidden readout.
* `A` -- the action chunk, `(20, 3)` cumulative relative planar poses `[dx, dy, dtheta]` in
  the emitting pose's frame, in `FlowActionCodec`'s **scaled** space.
* `h = f_theta(o) in R^1024` -- the **deterministic readout**: the backbone's pooled state
  through the readout MLP. This is the vector v3 calls `c` and feeds straight to the head.
* `c in R^1024` -- the **latent intent**, a random variable drawn from a distribution
  parameterised by `h`. In v3 there is no such thing; `h` went to the decoder directly.

Keeping these two apart is not pedantry. `h` is a function of `o`, so conditioning on `h`
conditions on (part of) `o`; `c` is *drawn*, so nothing that produces `c` may condition on
it. The one-symbol collapse is how `q(c | o, A)` gets accidentally written as a function of
`c`, which is circular.

**Generative model.**

```
p(A | o) = INT p_psi(A | c) p_phi(c | o) dc
```

* `p_phi(c | o) = N(mu_phi(o), diag sigma_phi(o)^2)` -- the **prior**. At RL time this *is*
  the policy `pi(c | o)`.
* `p_psi(A | c)` -- the flow decoder. We have a **sampler** (Euler, t=1 -> 0) and a training
  objective (conditional flow matching), not a density.

**Inference model.** `q_xi(c | o, A) = N(mu_q, diag sigma_q^2)`. Exists only during SFT;
never loaded at RL or eval.

**Objective.**

```
L = E_{c ~ q_xi} [ L_FM(A ; c) ]  +  beta * KL( q_xi(c|o,A) || p_phi(c|o) )
```

### Three things about this that are true and one that isn't

1. **It is not an ELBO.** `L_FM` is a velocity-field regression, not `-log p_psi(A|c)`. This
   is ACT's construction (L1 reconstruction + `beta`*KL) with flow matching in the
   reconstruction slot. Consequence: **`beta` has no transferable scale** and must be swept
   per architecture. The KL is still in nats, so diagnostics *on the KL* remain meaningful
   even though `beta` is a bare exchange rate.

2. **The prior is learned and conditional**, not `N(0, I)`. Minimising KL over `phi` at fixed
   `q` drives `p` toward the aggregate posterior `E_A[q(c|o,A)]` -- which is exactly what we
   need, because at RL time we sample from `p` and require those samples to be in-distribution
   for the decoder. It also means `beta` controls the `q`/`p` **mismatch**, not the absolute
   rate of `c`.

3. **The collapse loophole is architecturally closed.** The escape "set `q = p` and let the
   flow model the multimodality itself" requires the decoder to model the marginal chunk
   distribution over all observations, because `p_psi` cannot see `o` (point 1 of this
   document). `L_FM` will not tolerate that. This guarantees `c` carries `mu(o)` -- already
   true today. It does **not** guarantee the `A`-dependent residual lands in `c`; that is
   decided by `beta` alone.

4. **What is not established** (and, per the 2026-08-12 review, is not a gate): that the
   information migrating into `c` is *mode identity* rather than *geometry precision*. The
   head's own docstring names the competing hypothesis. This predicts how much RL will buy,
   not whether the interface works. It is measured, not assumed -- see Acceptance.

### At RL time

* policy `pi(c|o) = p_phi(c|o)`, diagonal Gaussian, 1024 dims
* actuator `A = D_psi(c ; z_0, n_steps)`, `z_0` and `n_steps` fixed => deterministic and
  differentiable in `c`
* exploration temperature `tau`: `sigma_rl = tau * sigma_phi`, applied at rollout, requiring
  no retraining. This is why `sigma` is **not floored during SFT** -- flooring corrupts the
  measurement of the quantity we most want to read.

## Architecture

### Unchanged (v3, verified)

```
Qwen3-VL-2B (hidden 2048, LoRA r=128 a=256, vision tower frozen)
  -> pool over assistant content span (pool_mode=mean, train_content_len=1)   (2048,)
  -> readout MLP  2048 -> [1024] -> 1024, layer_norm=True                     (1024,) = h
  -> view(N, 8, 128)                        <- reshape only, NO projection
  -> FlowActionDecoder  d_model 128, 4 layers, 4 heads, ff 512, pre-norm
       mask: context bidirectional among itself + sees nothing else;
             action block bidirectional among itself + attends context
  -> FlowActionCodec    Euler t=1->0, 10 steps at inference, K=8 (t, eps) per example
                        owns action_scales (0.03, 0.03, 0.05)
```

`<pose>` modality (planar_se2, `masked_scatter`) is orthogonal and unchanged.

### Added

**`LatentSplit`** -- `Linear(1024 -> 2048)` -> `(mu, log_sigma)`, inserted between the
readout MLP and the codec.

```
W_mu    = I (1024x1024)      b_mu    = 0
W_sigma = 0                  b_sigma = log sigma_0
```

**`PosteriorEncoder`** -- `(stopgrad(h), A) -> delta_mu`, output layer zero-initialised.
See below.

`fm_context_dim` stays 1024, so `FlowActionDecoder`'s
`n_context_tokens * d_model == context_dim` assertion and the entire head config are
untouched. **No head change at all.**

### The init contract

With `W_mu = I`, `b_mu = 0`, zero-init posterior, and `sample_c="mean"`, the warm-started
model is **numerically identical to v3**. This is a test, not an aspiration:
`tests/test_latent_split_parity.py` must assert bit-identity of the decoded chunk against
v3 before any training runs. It catches, in one assertion, every plumbing error: where the
split sits, checkpoint loading, and the K-expansion order below.

**`sigma_0` is measured, not chosen** -- take the per-dim std of `c` over the validation
split and set `sigma_0` to 1-3% of it.

**Sample `c` once, then `repeat_interleave` for the K=8 flow samples.** Expanding the
context before sampling yields 8 different intents per example, and the gradient reaching
`sigma` then means something else entirely (averaging over intents rather than over
integration noise). Invisible if wrong.

## Posterior design

The newest piece, and the one with the most freedom. Decisions:

### `q` shifts the mean only; `sigma` is shared

**Run 1 fixes `sigma_q = sigma_p`.** `q` emits `delta_mu` and nothing else. The KL collapses
to a well-conditioned quadratic:

```
KL = SUM_i  delta_mu_i^2 / (2 * sigma_p,i^2)
```

Why this is the right first run, beyond simplicity: it gives `sigma_p` a **closed
interpretation and a two-sided gradient**. `L_FM` pushes `sigma_p` down (noise hurts
reconstruction). The KL pushes it up (`d/d sigma_p [delta_mu^2 / 2 sigma_p^2] < 0`, so wider
`sigma_p` makes a given mean-shift cheaper). The balance puts

> `sigma_p` at the scale of the mean-shift that knowing `A` induces

i.e. exactly *"how much does the realised chunk disagree with what `o` predicted"* -- the
quantity the acceptance test wants to read. It cannot collapse to zero unless `delta_mu` is
zero, which happens only if `A` adds nothing given `o`.

Free `sigma_q` (run 2) is not wrong -- KL's `log(sigma_p/sigma_q)` term already blocks
`sigma_q -> 0` -- but it adds a second interacting knob with no benefit until run 1's number
is in hand.

### `q` conditions on `h`, not on `o` directly, and not on `c` ever

`q_xi(c | o, A)` is entitled to all of `o`. It is given `h` instead, and that is a decision
with a reason rather than a shortcut.

`p` depends on `o` **only** through `h` -- by construction, since `mu_p` and `sigma_p` are
maps of `h`. So any feature of `o` that `h` drops is a feature `p` cannot express. A `q` fed
the full 2048-d pooled state could push `mu_q` along such a direction; `p` could not follow;
the residual would show up as KL, and the KL is what sets `sigma_p`. The result reads as
"the outcome is unpredictable here" when the truth is "the prior was not shown the relevant
feature." Since `sigma_p` *is* the exploration distribution at RL time, that would mean
exploring along directions that encode an architectural information gap. **Aligning `q`'s
information set with `p`'s is what keeps the KL a measurement of what `A` adds.**

The cost is bounded and already measured: the readout MLP discards 0.0007 R^2 against the
pooled 2048-d state for chunk prediction, so `h` is very nearly sufficient for this purpose.
That number was taken on v3, where `h` was optimised for prediction alone; re-take it once
`h` is also carrying `sigma_p`.

`q` never sees `c`. `c` is drawn from the distribution `q` parameterises, so a `q` that read
it would be circular. (An earlier draft of this document wrote `q(stopgrad(c), A)`, which is
that mistake, produced by using one symbol for the readout and for the latent.)

### `q` is a residual on `p`, and `h` is detached into it

```
mu_q = mu_p + delta_mu( stopgrad(h), A ),      delta_mu output layer zero-init
```

**This is not extra information.** `mu_p = W_mu h + b_mu` is a deterministic function of `h`,
which `q` already conditions on, so adding it to `q`'s output changes the parameterisation
and not the hypothesis class -- `mu_q` is a function of `(h, A)` either way. What it changes
is the optimisation geometry: `q` learns the *correction* that knowing `A` implies, instead
of re-deriving the prediction that the whole backbone exists to make.

It also **couples `q` to `p` during training**, which is the more important effect. Without
the residual, `q` and `p` are two separately-parameterised networks chasing each other, and
any lag between them appears in the KL as information that is not there. The residual makes
"`q` has nothing to add" representable exactly, as `delta_mu = 0`, rather than approximately.

* **Residual + zero init => `KL = 0` exactly at step 0.** A freshly-initialised posterior
  otherwise makes `KL(q||p)` large and meaningless, and its gradient drags `mu` off the v3
  solution to chase noise. This is the same discipline as `W_mu = I` on the prior branch,
  applied to the other side.
* **This inverts the usual KL warmup.** Standard VAE practice anneals `beta` *up* from zero
  to prevent posterior collapse when `q` starts far from `p`. Here `q` starts *at* `p`, so
  there is nothing to anneal away: constant `beta` is the default. Say so in the launch
  script, or "we skipped KL warmup" reads as an oversight.
* **`stopgrad(h)`** so `q`'s parameters -- which are discarded at RL -- cannot shape the
  trunk. The trunk's objectives stay exactly `L_FM` (through `c ~ q`, which still reaches
  `mu_p`) and the KL's `p`-side pull. Note the alternative is *benign* rather than harmful
  (the trunk cannot see `A`, so `q`'s gradient could only make `mu_p` a better predictor of
  `A`, which `L_FM` already wants) -- detaching is a cleanliness choice and is worth keeping
  as an ablation switch.

**The one real hazard the residual introduces**, and it points at the failure mode we care
about: zero-init plus weight decay is a standing pull toward `delta_mu = 0`, which is
`sigma_p -> 0`, which is the degenerate solution. It is not an information problem, it is a
regulariser pointed the wrong way. Two guards, both cheap:

* **exclude `delta_mu`'s output layer from weight decay** (the run's `--weight-decay` is
  1e-4 and applies to everything by default);
* **log `||delta_mu|| / sigma_p` from step 0.** If it rises off zero in the first few hundred
  steps and settles, the residual is doing its job. If it decays back toward zero while the
  FM loss is still falling, `beta` is too large or decay is winning, and the spread gate will
  fail later for a reason that was visible on step 300.

### `q` is deliberately under-parameterised

`q`'s capacity is a **rate limiter that complements `beta`**. Capacity spent modelling `A`
in detail is capacity to over-migrate. Proposed:

```
A_scaled (20x3 -> 60) -> LayerNorm -> Linear(60, 256) -> GELU
concat with Linear(1024, 256)(stopgrad(h))
  -> GELU -> Linear(512, 256) -> GELU -> Linear(256, 1024)   [zero-init]
```

`A` goes in **whole and unfeaturised**. Handing `q` only the terminal displacement and
heading would hard-code the mode abstraction we have explicitly declined to assume. Give it
all 60 numbers, let `beta` decide what fits, and *measure* what got through.

### Invariants

* `q` sees `o` **only** through `c`, and `A` only for the current chunk. Never the next
  observation, never a goal absent from `o`. Automatic under the signature above; pin it
  with a test that `q`'s forward takes no other tensor.
* `q` is always exercised (every example has an `A`), so `ddp_find_unused_parameters=False`
  is safe. The repo has been bitten here before -- see the `zero_touch` note in
  `flow_matching_head`'s modality section -- so it is worth an explicit assertion.
* `save_pretrained` stores `q` (SFT resume needs it); the RL and eval paths must load
  without it.

## Acceptance

Two-sided. **Parity alone is the wrong gate: `beta -> inf` passes it perfectly** by
reproducing the deterministic model, which is precisely the failure mode.

| gate | metric | requirement |
|---|---|---|
| parity | sample101 oracle / oSPL_fix at `sample_c="mean"`, 175 steps | within noise of v3 (0.663 / 0.348) |
| spread | behavioural spread over `c ~ p` at fixed flow seed | `>> 0`, and `>=` the spread from varying the flow seed at fixed `c` |

Spread is measured **behaviourally** (terminal heading std, terminal displacement std),
never as `||delta c||`. The head is `norm_first=True` over 128-dim blocks, so per-block mean
and scale are stripped from what the action tokens read at layer 1 and re-enter only through
a second-order residual path: roughly 16 of the 1024 directions are strongly attenuated and
a norm-based proxy reads them as real.

`beta` is then swept as a 1-D problem: **the largest `beta` that keeps spread above
threshold.**

### Log from step 0

* mean `KL` (nats/example), and the per-dim KL histogram
* `#{dims : KL > 0.01 nats}` -- the **effective dim**, an outcome, not a hyperparameter
* whether the active set is stable across states or state-dependent (decides whether the RL
  masking below is available at all)
* `sigma_p` percentiles; `||delta_mu|| / sigma_p`
* parity: chunk RMSE at `sample_c="mean"` against cached v3 predictions

## The RL phase (forward reference, not yet designed)

Reward is settled: standard Habitat shaped reward, no terminal term, oracle/heuristic
termination. Autonomous stopping is explicitly not blocking.

The 1024-dim Gaussian is only a problem for ratio-based methods, and there are three
independent escape hatches, all decidable *after* the effective-dim measurement above:

1. **Mask** the `KL ~ 0` dims at RL time (fix at prior mean, drop from the log-prob sum).
   Free, preserves parity. Requires a *state-stable* active set.
2. **Per-dim importance ratios** instead of the 1024-term product -- the same fix
   sequence-level RLHF applies at the token level. Biased surrogate; works in practice.
3. **Ratio-free**: advantage-weighted regression on `c`, or TD3-style with a critic
   `Q(o, c)` exploiting the differentiability of `D_psi`.

Because all three are post-hoc, **PPO compatibility must not constrain the SFT run.** Design
run 1 for parity + spread only.

Phase-1 actor scope: the readout MLP + split (~5.2M params), LoRA and backbone frozen. The
actor then ships over the existing policy-bridge socket at ~20 MB, so the two-environment
split (`habitat_sim` vs `flash-attn`) stops being an obstacle.
