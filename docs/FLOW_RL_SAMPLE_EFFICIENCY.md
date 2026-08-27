# Why flow-SDE RL is sample-hungry: what we concluded

Status: analysis and theory, 2026-08-26/27. **Nothing here is implemented.** Two
runs were launched off the back of it (section 7); everything else is a record of
reasoning, including the parts that turned out to be wrong, because the wrong
turns are the ones a reader is most likely to repeat.

Companion to `docs/FLOW_SDE_RL.md` (the machinery), `docs/RTC_RL.md` (the RTC
formulation) and `docs/BLIND_ROWS.md` (a separate data-quality issue found on the
way).

---

## 1. The question

Continuous flow-SDE RL needs a small fixed episode pool (<=128) to converge, where
the discrete-action RL did not. Hypothesis under investigation: **the flow's
sampling noise is a nuisance the policy gradient cannot marginalise over, and
repeating episodes partially substitutes for that missing marginalisation.**

## 2. The estimator, and where its variance comes from

Readout `h = f_θ(o)`; base noise `z_0 ~ N(0,I)`; `n` of `K` denoising transitions
are stochastic (`N(μ_θ, σ_k)`), the rest are deterministic ODE steps; the executed
chunk is `compose(z_K)`. The implemented gradient is

```
∇J ≈ E[ A · ∇θ Σ_{k∈P} w_k log N(z_{k+1}; μ_θ(z_k,t_k), σ_k) ]
```

This is the DDPO-style **augmented (denoising) MDP**: the action is the latent
chain, not the executed chunk. It is unbiased, and because reward depends on the
chain only through `compose(z_K)`, `J_augmented ≡ J_marginal` — the two objectives
are the same function of θ, so the formulation introduces no spurious optima. It
changes variance, not the optimum.

**Measured on the real head:** regressing the chain score on the executed
endpoint gives **held-out R² = 0.099** — ~90% of score variance is unexplained by
the action actually taken. (An initial probe reporting R² = 1.000 was vacuous:
60 samples against 60 regressors. The held-out split is the real number. In a
linear equal-noise toy the score genuinely *is* a sufficient statistic of the
endpoint — verified analytically to 4e-16 — but the real non-linear head is not
in that regime.)

Also measured: score variance scales ~linearly in `sde_n` (0.34 / 0.60 / 0.96
per-dim at n = 1/2/3) while log-prob conditioning *improves* with n (spread
0.96 -> 0.46). `sde_n` is therefore an unmanaged variance/conditioning knob.

## 3. Multi-sample schemes: what is and isn't available

SFT amortises `k_samples=8` head passes per backbone forward. **That trick does
not transfer**, and the naive version is a trap.

| scheme | valid | note |
|---|---|---|
| average the score over M chains, one reward | **NO** | **gradient attenuated by 1/M** — measured shrinkage 0.121 vs 1/M = 0.125 predicted, while apparent variance fell 203 -> 16. No error, no NaN, silently rescales the learning rate. |
| IWAE marginal log-prob inside a PPO ratio | **NO** | ratio of two bounds is not a bound of the ratio; bias is sign-indefinite |
| extra head passes at training time | valid but **null** | the buffer holds the chain that *acted*; re-drawn chains have no reward, and re-scoring the stored chain is deterministic (`lp_selfdiff ~ 0.002`) |
| RLOO / group baseline over M **executed** chains | yes | **~2.4x variance reduction** measured at matched budget; costs M env steps |
| vine branching | yes, but | one-step branches need `V(s')` per branch — **a backbone forward per branch**, i.e. the cost we were trying to avoid; continuing branches is worse |
| antithetic / CRN coupling | yes | free, unbiased, untested here |
| Rao-Blackwell over the chain posterior given the endpoint | yes in principle | would remove ~90% of score variance; needs a **non-linear diffusion bridge** posterior — no closed form. Open problem in the literature |
| pathwise gradient with a Q-critic | yes | see section 6 — the one that removes the problem rather than baselining around it |

Why the SFT analogue is null: in SFT each of the K draws carries **its own
supervised target**, so averaging K losses averages K unbiased estimates of one
gradient. In RL a chain's score is only meaningful multiplied by **that chain's**
reward, and a reward costs an env step.

Literature: [Flow-GRPO](https://arxiv.org/abs/2505.05470) uses the same ODE->SDE
construction and, per its
[technical overview](https://www.alphaxiv.org/overview/2505.05470), noise "is not
marginalized but rather conditioned upon", with the group used **only** for the
advantage baseline. [DPPO](https://proceedings.iclr.cc/paper_files/paper/2025/file/c0749c39aaff9e9e4c91f7118bf21b1e-Paper-Conference.pdf)
is the denoising-MDP formulation; [related analysis](https://arxiv.org/pdf/2509.25050)
describes per-step likelihood maximisation as a **"noisy target" estimator —
unbiased but higher-variance, substantially slowing convergence**, which is the
published form of the hypothesis in section 1.

## 4. What marginalisation is actually for (flow-matching side)

The flow-matching optimum is a conditional expectation,
`v*(x,t,c) = E[ε − a | x_t = x, c]`, so learning it is a regression whose target
is reached only via averaging **or** coverage:

* **multimodal `p(a|c)`** — the target genuinely varies at fixed `(x_t,t)`, so the
  field is recoverable only by averaging. Literal marginalisation.
* **Dirac `p(a|c)`** — the target is *deterministic* given `(x_t,t)`
  (`u = (x_t − a*)/t`), so nothing needs averaging; what is missing is **coverage**
  of the field between `z_0` and `a*`.

Effective samples per condition:
`n_eff(c) ≈ n_repeats(c) + Σ_{c'≠c} w(c,c')·n(c')`, with three suppliers:

1. **repeats** — same condition, fresh `(t, ε)`;
2. **cross-state pooling of content** — the "number line" case: neighbours in the
   embedding metric whose *targets are also similar*. Requires **both** that the
   encoding induces smoothing **and** that the target is smooth in that metric;
   satisfying only the first (e.g. digit identity as a scalar) gives
   **interference**, not marginalisation;
3. **cross-state pooling of form** — every condition shares the functional shape
   `v = (x − a)/t`. This does not raise `n_eff`; it *lowers the dimension* of the
   per-condition problem.

**One-hot conditioning defeats (2) by construction**: gradient flows only into the
active embedding column, so an MLP cannot do index-based generalisation. The trunk
still shares *form*, never *content*. (Caveat: embedding dim must be >= N, or rows
cannot be near-orthogonal and you get forced interference instead of independence.)

### A claim we made and retracted

We claimed that because `a = x_t − t·u_t` recovers the target exactly, **one SFT
sample per state suffices**. That is wrong, and the counter-example is decisive:
one "dog" caption with one image, never seen again, cannot teach a text-to-image
model to generate that image — it would mean learning a Dirac from a single
`(x, t, v_t, c)` tuple. The error was sliding from *the information is present in
the sample* to *the model acquires it*: the loss constrains `v_θ` at exactly one
point, and a capacity-rich network can zero that point's loss while behaving
arbitrarily elsewhere in `(x,t)` for that condition. Exploiting the sample
requires the trunk's shared form to be so rigid that only `a` remains free — which
is a generalisation assumption, i.e. the very thing being claimed unnecessary.

Empirically this matches the diffusion memorisation literature: replication is
driven by **duplication**, and single-occurrence images are generally not
reproducible ([Somepalli et al.](https://openreview.net/pdf?id=F9qCNPSzSY),
[Carlini et al.](https://arxiv.org/pdf/2301.13188),
[theory survey](https://arxiv.org/html/2410.02467v1)). Note also that standard
text-to-image training revisits every image across epochs with fresh `(t,ε)` —
that **is** repeated visits, i.e. the regime under discussion forbids what real
training relies on.

**Corrected position.** In the strict never-revisit, no-sharing limit, SFT fails
too. What survives is quantitative: per exposure, SFT's signal is a *d*-dimensional
vector that exactly identifies `a*` given the form, while RL's is a scalar that
identifies the right direction only in expectation. SFT needs fewer exposures —
not one. Replay is a further SFT-only advantage: a stored target stays valid
across epochs, whereas PPO's trust region caps how many epochs a sample survives.

Corollary worth stating: for a Dirac target a **direct regression** `c -> a` needs
one sample per condition. Flow matching must learn a whole velocity field to
represent the same point mass. That gap is the **sample-complexity premium paid
for the ability to represent multimodal conditionals** — worth it when the
conditional is genuinely multimodal (which is why this project adopted the flow
head), pure cost when it is not.

## 5. Why pooling is especially thin in *our* RL setting

The policy's conditioning is the accumulated conversation: `past_key_values` grows
across the episode and is cleared only per episode. So the state is the **history
of observations**, not the current observation.

* **Repeats mostly don't happen.** Re-running an episode reproduces only its
  *initial* state; from step 1 the history diverges with the policy's own actions.
  "169 repeats of episode i" is really 169 repeats of its first state and
  progressively fewer effective repeats thereafter — so the marginalisation-by-
  repetition route is far thinner than episode counts suggest.
* **Repeats buy RL less even when they occur.** SFT revisiting a state gets fresh
  `(t,ε)` against a *constant* `a*`. RL revisiting a state gets fresh chain noise
  **and a fresh return realisation**, because the advantage depends on everything
  that happens afterwards — typically the larger variance of the two.
* **Similar-looking states can carry incoherent credit.** Pooling is valid only if
  states are equivalent; histories that look alike but sit differently relative to
  the goal have different optimal actions, so pooling injects the interference
  failure mode rather than marginalisation. In SFT the analogous case is benign —
  the model just learns the conditional distribution, which is what flow matching
  is for.

**Implied knob (untested):** context length controls the size of the effective
state space and therefore the collision rate, hence how much pooling is available
at all. `context_reset_every` / `policy_memory_seconds` exist on the eval-bridge
side; the RL configs currently set none of it.

## 6. The RL-theory reading, and where the credit actually goes

Standard results line up with the flow-matching picture, reached independently:

* **Tabular / PAC-MDP:** concentration is **per `(s,a)`**, so stochastic
  transitions require repeats.
* **Deterministic dynamics escape that** ([Wen & Van Roy](https://web.stanford.edu/~bvr/pubs/OCP_MOR.pdf)):
  one observation of `(s,a) -> (s',r)` is the truth, given a function class of
  bounded Eluder dimension. Habitat's physics is in this class — but this exempts
  only the *physics* layer.
* **Function approximation is the formal substitute for revisits** — Eluder
  dimension, Bellman rank, coverability; and the closest analogue to our case, the
  **contextual bandit with unique contexts**, has regret depending on `log|F|`,
  *not* on the number of contexts. Generalisation does the work, exactly as in
  section 4.

### Where the noise sits, stated correctly

Two distinct sources, and they are not the same kind of thing:

* **The σ transitions are ordinary policy stochasticity.** Fixed σ with learnable
  μ is a standard stochastic policy; the density is exact in what was sampled, the
  augmented MDP is well-posed, and **no marginalisation is required for
  correctness** — Rao-Blackwellisation would only reduce variance.
* **`z_0` is not policy stochasticity in any useful sense.** It is drawn before
  the policy acts, θ has no influence on it, and its law does not depend on θ.
  Placing it in the augmented MDP's **initial-state distribution** is the coherent
  reading, and then the classical requirement applies: knowing whether the drift
  is good at a state means averaging the return over `z_0`, which needs repeats or
  generalisation.

### The mechanism that makes this concrete

Per training sample, gradient reaches `μ_θ` **only at the `n` stochastic
positions**. `sample_chain_np` accumulates density only there and
`chain_log_prob_batch` differentiates only there. Therefore:

* `z_0` — a full 60-dim draw that largely selects **which trajectory shape** is
  produced — receives **zero** credit;
* the other `K − n` ODE steps (7 of 10 at the shipped config) are data and receive
  **zero** gradient;
* all learning flows through `μ_θ` at `n = 3` sampled `(x,t)` points.

In the σ -> 0 limit the log-prob is identically zero and there is **no gradient at
all**, though the policy still acts and still varies — entirely via `z_0`. So the
variable generating action diversity is precisely the one carrying no learning
signal, and the σ transitions exist largely to manufacture a differentiable
density rather than to explore.

**This reframes the fix.** The problem is not that the gradient is noisy; it is
that the estimator credits the wrong variable. Reparameterising —
`∇θ Q(s, F_θ(z_0, s))` with `z_0` held fixed as environmental randomness — credits
the drift for what `z_0` actually produced and delivers gradient to the *whole*
field instead of three points of it. `pin_flow_noise` and
`generate_differentiable` exist to make that possible, and `docs/LATENT_RL_ENV.md`
already lists "TD3-style with a critic `Q(o,c)` exploiting the differentiability
that pinning `z_0` buys". Cost: an action-conditioned critic (today's head is
`V(o)`), leaving PPO, and backprop through K decoder steps.

## 7. What was launched, and one premise that collapsed

**The small-pool doctrine's evidence does not survive re-analysis.** It rested on
a 2026-08 fullpool run (8000 episodes) showing no gain — but that run stopped at
**55 cycles**, and difficulty-controlled regressions show the *successful*
held128 run is equally flat at 55 cycles (success t = +0.17; its significance
arrives only by ~900 cycles, t = +5.0). Matched-horizon numbers:

| | fullpool (55 cyc) | held128 (first 55 cyc) |
|---|---|---|
| mean_reward | t = +0.22 | t = +1.99 |
| success | t = +0.70 | t = +0.17 |
| progress fraction | t = +0.60 | t = +0.15 |

Neither shows a credible gain. The comparison was **underpowered, not
informative**. Caveats on even these numbers: the fullpool reward was progress
plus escape penalty only (`slack = collision = success_reward = 0`), so side
metrics are not in the objective and cannot evidence "learning"; `start_m` control
does not handle scene composition; and nothing here survives a multiple-comparison
correction.

Hence **`flow_sde_rtc_fullset`** — a literal twin of `flow_sde_rtc_held128`
differing in exactly two lines (`train_uids`, `run_name`), running to several
hundred cycles. The 26 eval uids are subtracted from the 7,403-episode training
list so the held-out set stays honest while remaining reachable by the eval shard.
Audited confounds: identical code (no `.py` changed between the two launches),
near-identical pool difficulty (start_m mean 7.17 vs 6.97) and category mix; the
irreducible one is that a larger pool inherently means a less predictable return
distribution, so a worse time-kernel baseline is confounded with the manipulation
— mitigate by comparing the logged baseline-quality metrics between arms.

## 8. If the toy experiment is ever run

Image generation substitutes cleanly for the robot (no simulator; same flow head,
same conditioning, same scalar-vs-vector supervision) with `r = −‖x − x*‖²` and
SFT on `x*`, so both share an optimum. Two things do change: **dimensionality
inflates the pathology** (the chain density sums over output dims), and **convnets
generalise far more than our transformer-over-20-ticks**, which risks engineering
away the effect.

Design points that matter more than the domain:

* The regime is set by **visits per state**, not by encoding. Ten digit classes are
  revisited thousands of times and cannot test the hypothesis *whatever* the
  encoding; our navigation ratio is ~2.9 visits/episode (fullset) vs ~169
  (held128) and needs matching.
* Dial generalisation with **RBF/Fourier features of a continuous condition,
  bandwidth σ** — large σ pools across neighbours, small σ approaches one-hot.
  Same task, same targets, same cardinality, one continuous knob.
* Predicted dissociation: **SFT nearly flat in σ; RL degrading sharply as σ -> 0.**
  The measurable is exposures-to-learn as a function of σ for each method — a
  ratio between two curves, which is falsifiable in a way "RL fails, SFT doesn't"
  was not.

## 9. Ranked, if the thread is picked up again

1. **`sde_n` sweep** (1 vs 3) — existing config knob, no code, directly trades the
   quantity in question.
2. **Antithetic / CRN coupling** — free, unbiased, low expected payoff.
3. **Pathwise gradient + action-conditioned critic** — the formulation that credits
   `z_0`'s consequences and the whole field. Substantial work; leaves PPO.
4. **Vine + RLOO** — valid, ~2.4x variance reduction, but costs a backbone forward
   per branch and must answer `FLOW_SDE_RL.md`'s standing GRPO objection.
5. **Rao-Blackwell over the chain posterior** — highest payoff, no tractable
   estimator, research project.

**Standing warning for anyone implementing in this direction:** the naive
"average the flow head over M samples like SFT does" attenuates the gradient by
1/M with no error and a *lower* apparent variance. First test of any such change
must be that the estimator's mean is unchanged at M > 1.
