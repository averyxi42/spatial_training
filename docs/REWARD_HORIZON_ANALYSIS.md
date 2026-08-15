# What the dense geodesic reward actually teaches, and where it goes silent

**2026-08-15.** An analysis of the reward/discount pair used by every flow-SDE RL run to
date, from the per-step `reward` and `distance_to_goal` arrays already sitting in the
rollout artifacts. It was produced by a subagent tasked to *attack* the hypothesis below,
not to confirm it.

## Why this exists

The programme spent months arguing about the "credited channel" — whether injected
exploration noise reaches the backbone at all — on the strength of the h2 probe, which
measured whether noise flips **episode outcomes**. That question is malformed as a test of
credit, and so was the return-variance follow-up run on 2026-08-15. Take the reward on the
real line with a fixed goal direction: `r_t = eps_t`. Then `E[R] = 0` for *any* noise
scale, so mean return, return variance and flip rate are uninformative **by construction** —
while `r_t = eps_t` exactly, so `Cov(eps_t, A_t)` is near-maximal. Perfect per-step credit,
zero episode-level footprint. Our geodesic-delta reward is this, locally.

So per-step credit was never plausibly blind, every episode-aggregate diagnostic measured
closed-loop re-convergence instead, and the open question became a different one:

> Given that per-step credit is near-perfect and near-trivial, what does this reward +
> discount actually teach, and is it anything the SFT policy does not already do?

## Verification status

The two structural findings were **independently reproduced** by the coordinator on a
different, cleaner episode set — the 416-episode `sft_baseline_fullpool_evalset` probe
(one fixed SFT checkpoint, 26 episodes x 16 passes, arms verified by reproducing
`h2_summary.json`'s success means to 4 dp) — against the agent's 1,334-episode
`flow_sde_a09_fullpool/rollout` set:

| median per episode | successes (n=265) | failures (n=151) | agent's set |
|---|---|---|---|
| steps | 49 | 175 (cap) | 47 / 175 |
| reward mass forward / back | +7.1 / −1.5 | +11.7 / −10.6 | +6.5/−1.2, +12.8/−11.8 |
| net | +5.6 m | +1.1 m | +4.6 / +0.7 |
| `\|r\|` < 0.05 m | 53% of steps | 62% | 53% |
| wrong-direction steps (Δd > 0.25 m) | 2 | 17 | 2 / 19 |
| max excursion above start | +0.24 m | +1.96 m | +0.10 |
| best moment, steps before end | 0 | **74** (γ⁷⁴ = 0.022) | 78 (γ⁷⁸ = 0.018) |

Differences are sampling between the two sets; the structure is identical. **Not**
independently verified: the sample400 conversion analysis in section 3 below (the 48
converted / 91 unconverted episode split and its path-length figures).

---

## The agent's report, as delivered

### 1. What r_t actually looks like

Parsed per-step `reward` and `distance_to_goal` from 1,334 fullpool rollout episodes
(`flow_sde_a09_fullpool/rollout` — note this directory mixes ~936 training servings at
a=0.9 with ~390 in-training ODE eval passes; I therefore replicated everything on the clean
208-episode ODE arm of the baseline probe, and the numbers agree, so I quote the big set
with the probe in parentheses).

**Successes** (n=889; median 47 steps, start 4.9 m): positive reward mass 6.5 m, negative
1.2 m, net 4.6 m. 53% of steps carry |r| < 0.05 m (turning, settling), and half the positive
mass sits within 14 steps of termination on a 47-step episode. Median wrong-direction steps
(one-step geodesic rise > 0.25 m): **2**. Maximum excursion above the start distance:
**0.10 m** (probe: 0.19 m). So a successful episode is a near-greedy descent from the first
step — there is no search phase in successes at all.

**Failures** (n=445; all at the 175-step cap, start 9.6 m, min_m median **5.45 m**):
positive mass 12.8 m, negative mass 11.8 m, net 0.7 m. This is the finding, and it amends
the framing: the exploration phase is not trained on *approximately no* signal — it is
trained on **~25 m of large, nearly perfectly self-cancelling signal**. 19 wrong-direction
steps per episode, each punished at the step it occurs; every transient distance increase —
including the ones correct search requires, like entering an unexplored room on the wrong
side — is charged in full at entry and refunded at exit. The reward in failures is dense,
loud, and sums to nothing. One structural note: `F_oracle` is empty by construction — in
this env `reached` terminates the episode, so training failures are episodes that *never got
near the goal*, median closest approach 5.45 m.

### 2. The horizon arithmetic

gamma = 0.95 per policy step: half-life 13.5 steps, effective horizon ~20 steps = 8 s of sim
time. Against that, the defensible commitment proxies from the distance curves:

- Successes: last wrong-direction step at median **24 steps before the end**
  (gamma^24 = 0.29) — even the little correction they contain sits at the edge of the horizon.
- Failures: best approach (argmin) at median **78 steps before the end**
  (gamma^78 = 0.018), and since min_m never gets below ~5.5 m, the wrong-region commitment
  necessarily happened earlier still — ≥100 steps from any consequence, gamma^100 = 5.9e-3.
  Honest limitation: distance curves cannot localize "the decision" more sharply than this;
  these are bounds, not identifications.

So yes: the event that separates success from failure is invisible to this discount, by two
orders of magnitude.

### 3. The attack — where the hypothesis survives, and two places it is wrong

**Survives, strengthened by the held-out data.** If the reward/horizon pair only teaches
≤20-step behaviour, the sample400 gains should be motion-quality gains, not search gains.
They are. On the 48 baseline-oracle-failures that ck391 converts: baseline burned 175 steps,
31.8 m of path, best approach 4.34 m; ck391 arrives (min 0.06 m) in 81.5 steps and
**22.4 m — less path than the baseline spent failing**, at 0.275 vs 0.182 m/step. On the 91
far failures it does *not* convert: ck391 travels 43.0 m (vs 30.9) and improves min_m from
7.71 to only 7.05 — where search is actually required, 39% more coverage buys nothing,
because the coverage is undirected. And the payload is bounded: ck303 → ck391 is 88 cycles
at stable k2 ≈ 2e-2 with zero further held-out gain — the pair had already taught everything
it had. Overall speed 0.194 → 0.254 m/step; the gain currency (oSPL, mid-tercile
conversions, far-tercile oSPL delta +0.01–0.04) is all local polish extending reach under a
fixed 70 s budget.

**Wrong in emphasis, and it matters for the fix.** The exploration phase is not
signal-starved; it is **signal-cancelled, with the sign against search**. A reward that were
merely silent during search would be neutral to it; this one actively pays greedy descent
against the *true* goal location the agent cannot know, so during search it is per-step noise
plus a systematic penalty on exactly the excursions search consists of. Raising gamma does
not fix that — it converts per-step credit into return-to-go credit, i.e. episode-outcome
credit, and we have now measured that regime: per-episode return scatter has a z0-only floor
of ~3.3 m that credited noise cannot exceed (ratio 0.91), against a state-blind time-kernel
baseline that removes none of the between-episode dispersion. Long-horizon credit through
this estimator is noise-dominated *by measurement*, not by conjecture. So "the deficit is the
reward/horizon pair" is right, but the minimal fix is not the gamma knob — gamma alone walks
into the credit channel this project has already shown is swamped.

**Also wrong as a completeness claim**: "a behaviour a competent SFT policy already has."
SFT's *successes* are greedy descents, but SFT spent 175 steps and 31.8 m failing episodes
ck391 solves in 22.4 m — the +0.05/+0.09 held-out gain is real headroom that this reward did
pay for. The pair is not useless; it is *spent*.

### 4. What follows

The falsification run comes first, and it is one knob: **same config, same pool, gamma
0.95 → 0.99** (horizon ~100, reaching the argmin-at-78 events). My analysis predicts it is
flat-to-worse on far episodes because the long-horizon advantage inherits the measured
variance floor. If it instead moves far-episode success, the diagnosis "horizon was binding
and the variance was manageable" wins and the cheap fix was real. If it is flat — the
predicted outcome — that run is the license and the control for the actual fix, which the
evidence chain now points at from three independent directions: a **state-dependent value
baseline** (the shelved SFT value co-train). It is the only single object that (i) removes
the episode fixed effect that biased small-pool credit and noised large-pool credit, (ii) is
the prerequisite for any horizon long enough to see search consequences, and (iii) can in
principle pay search *prospectively* (high value on entering unexplored space) rather than
punishing it at entry. Reward-shape surgery (coverage bonuses etc.) is the alternative but is
a bigger design change with its own reward-hacking surface; the falsifier for the whole
diagnosis, before any of it, is the gamma run.

One number to keep: **failures churn 12.8 m forward and 11.8 m back per episode, and their
best moment is 78 steps — gamma^78 = 0.018 — before the end.** That is the whole case in one
line: the signal is loud everywhere, and silent exactly where the episode is decided.

---

## Coordinator's note

One sharpening the report leaves implicit. During search, `r_t` is computed against the
**true goal position, which the agent cannot observe**. So the per-step credit there is not
merely cancelling — it is rewarding *luck*: the mechanical credit path (noise → executed
displacement → geodesic delta) is near-perfect, but what it credits is uncorrelated with any
decision the agent could have made better given what it saw. That is why the only gain this
reward has ever purchased is local motion quality, and why it is now spent.

This also converges independently on a conclusion already reached and shelved on this project
(`critic-is-last-resort`): the SFT value co-train is the real fix. Three separate evidence
chains now point at it.
