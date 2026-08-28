# RL with the code-conditioned policy: what is pending

Status: **plan.** Nothing built. The design was settled in `CODE_CONDITIONED_POLICY.md` §6;
this is the gap between that design and the code as it stands, in dependency order, plus the
decisions that must be taken before any of it starts.

Companions: `CODE_CONDITIONED_POLICY.md` (§6 RL, §3.2 why the code shrinks the z₀ threat),
`FLOW_SDE_RL.md` (machinery), `FLOW_RL_SAMPLE_EFFICIENCY.md` (why this architecture exists
at all), `RTC_RL.md` (the prefix contract this must not break).

---

## 0. Inventory — what exists today

| piece | state |
|---|---|
| frozen dual-FSQ tokenizer | **done** |
| `CodePolicyHead`, `CodeContextMixer`, code SFT | **done**, `run_code_v4_mlp_warm` running |
| rollout seam (`denormalize` samples `c`, decodes) | **done**, exercised by the viz harness |
| eval / mode visualisation | **done** |
| **everything RL** | **nothing** |

Measured directly: `code_head|code_mixer|last_code` appears **0 times** in
`flow_sde_policy.py`, `rl_core.py`, `vlm_worker.py`, `rollout_core.py`. The flow-SDE RL path
was written for a policy whose action is the chain alone and has no notion of a discrete
factor.

## 1. Decide this first — is the chain in the action at all?

Everything below scales with this answer, so it is not a detail to settle later.

**Option A — code only.** The action is `c`. The flow decode is the actuator: ODE, pinned
`z₀`, deterministic. `log π = log p(c|h)` exactly, entropy exact, no chain, no
`sde_positions`, no two-factor ratio, no density plumbing.

* This is the maximal Rao-Blackwellisation and it **eliminates the z₀ mis-credit problem
  outright** rather than quarantining it — the pathology `FLOW_RL_SAMPLE_EFFICIENCY.md` §6
  identifies (all learning flowing through `μ_θ` at 3 sampled points while `z₀` selects the
  trajectory shape and receives zero credit) simply does not arise.
* It makes the RL problem a **1600-way contextual bandit per step** — the regime the theory
  section says is well-behaved (regret in `log|F|`, not in context count).
* Cost: the continuous detail channel never improves under RL. Whatever `r(h)` and the
  decoder learned in SFT is frozen policy.
* Work: roughly §2 + §3 + a much smaller §4, and none of §5's second factor.

**Option B — hierarchical, both factors** (what `CODE_CONDITIONED_POLICY.md` §6 specifies).
Action is `(c, chain)`, `log π = log p(c|h) + Σ_k log N(transitions)`, an exact
factorisation. Flow head frozen. Detail channel stays learnable.

* Cost: all of §5, including the mandatory ratio split, and the z₀ pathology persists in the
  continuous factor (bounded, per §3.2, but present).

**Recommendation: A first, B as the follow-up.** A is a strictly smaller build, tests the
core claim of the whole architecture (that a discrete code is a tractable RL action where the
continuous chain was not), and its result determines whether B's extra machinery is worth
paying for. B cannot be evaluated cleanly without A as its baseline anyway.

## 2. P0 — the loading blocker

`load_flow_stack` (`flow_sde_policy.py:528`) constructs `FlowActionCodec` from
`fm_decoder_kwargs` and then calls `codec.load_state_dict(blob["normalizer"], strict=True)`.
A code-conditioned checkpoint's blob carries `code_head.*` and `code_mixer.*`, which the
freshly-built codec does not have, so this **raises**. It fails loudly rather than silently,
which is the good case, but nothing downstream can load a checkpoint until it is fixed.

Fix: call `restore_code_slot` (already exists in `code_conditioned_head.py`, already used by
the SFT/eval path) before the strict load. Small, and it gates everything else.

## 3. P1 — sampling `c` in the RL sampler

`sample_chain_np` takes `h` and conditions the decoder on it directly. It must instead draw
`c ~ p(·|h)`, mix `(c, r(h))` through `CodeContextMixer`, condition on that, and return `c`
and `log p(c|h)` alongside what it already returns.

The rollout seam in `FlowActionCodec.denormalize` already does exactly this and is exercised
by the eval harness — but it is the *eval* path. **The two must not become independent
implementations of the same sampling rule.** The existing file invariant (sampler and scorer
go through one `_sde_transition`) is the precedent: the code draw needs the same treatment,
one function used by both.

## 4. P2 — trajectory plumbing

New per-step keys, through the whole chain: `rollout_core` (store) → `collate_trajectories`
→ `postprocess_episode`'s old-log-prob recompute → `vlm_worker.rl_loss`.

* `code` — int64, the executed code. Store the **factored** `(c_xy, c_θ)` rather than the
  joint index; the head is factored and the joint is derivable, not vice versa.
* `code_logprob` — float32, `log p(c|h)` at action time.

The scorer must recondition on the **stored** code, never re-draw and never re-derive it
from the executed chunk. This is the same invariant as the RTC prefix
(`chain_log_prob_batch` already takes `prefix_actions` for exactly this reason) and breaking
it silently changes the estimator.

## 5. P3 — the loss (Option B only, and it is not optional there)

`CODE_CONDITIONED_POLICY.md` §6 is emphatic and the reasoning is quantitative: `log p(c|h)`
moves in **tenths of nats**; the chain term sits at **~0.0005–0.0006 nats** in the RTC run.
Summed into one ratio, the code term is three orders of magnitude larger, the clip binds
entirely on code changes, and **when it binds the gradient is zeroed for both factors** — the
continuous channel switches off in precisely the cycles where the code policy is learning
fastest.

So: each factor takes its own ratio, its own clip range, its own gauge.
`chain/abs_log_ratio_mean`'s band must be re-established per factor rather than inherited.

## 6. P4 — entropy is now load-bearing

`entropy_bonus: 0.0` was safe for a continuous policy because entropy came from the action
distribution itself. It is **not safe** for a categorical head: with a peaked SFT
initialisation, PPO can drive it deterministic within tens of cycles, and **a code that stops
being sampled receives no gradient and does not come back.** This is absorbing, not
recoverable.

Needs a code-entropy term with its own coefficient, and a floor gauge (how many distinct
codes were sampled per cycle — the SFT logs already track `code_pred_used`, currently 52–59
of 1600, which is worth watching from the start).

## 7. P5 — the reference policy will silently lie

The current RL run gets an exact SFT reference from `disable_adapter()`, which is valid only
because the SFT LoRA is merged into the base and the trainable delta starts at zero. **A code
head added as a plain trainable module is not covered by that trick**: disabling the adapter
leaves its RL-updated weights in place, so `ref_kl` reports a plausible, wrong number.

Options: keep a frozen reference copy of `code_head`/`code_mixer` (they are small), or put
them under an adapter. Whatever is chosen must also cover `proj`, which incidentally opens
the option of training `proj` during RL.

## 8. P6 — instruments, before the first cycle

* **Obedience by MISS DISTANCE, never strict equality.** Encode the generated chunk with the
  frozen tokenizer and log how far `g(A')` lands from `c` in grid steps. Headline the
  within-one-step rate; treat ≥2 steps as the real failure rate. On the c-only prototype:
  strict 0.840 xy / 0.795 θ / 0.672 joint, but **94.0% within one grid step**, gross misses
  ~3%. FSQ indices are ordinal, so an adjacent code is a neighbouring mode, not a wrong one.
* **`sde_noise_a` needs re-deriving with a two-sided bound** (Option B only): large enough to
  explore detail, small enough that the sample stays inside its code cell. Chain noise that
  moves `g(A')` randomises the mode RL just chose. The shipped 0.9 is inherited and must not
  be assumed.
* `chain/h_drift_from_init` and `ref_kl` keep their meanings.

## 9. The gate — RUN, AND PASSED (checkpoint-2000, 2026-08-28)

`data_scripts/measure_obedience.py`, ObjectNav validation, 250 held-out episodes,
11,822 valid turns, 4 z₀ draws each = 47,288 decodes. Miss is **L∞ on the FSQ lattice**
(see the metric note below).

| condition | strict xy | strict θ | strict joint | **within-1 joint** | gross ≥2 (xy / θ) |
|---|---|---|---|---|---|
| teacher `c* = g(A*)` | 0.836 | 0.757 | **0.637** | **0.966** | 2.3% / 1.2% |
| policy argmax | 0.948 | 0.800 | **0.754** | **0.996** | 0.3% / 0.1% |
| *c-only prototype* | *0.840* | *0.795* | *0.672* | *0.940 (not comparable)* | — |

**`c` still steers the decode. The RL lever is connected and the plan is not void.** Teacher
strict joint 0.637 against the prototype's 0.672 is a small drop, not a collapse, despite
`r(h)` having grown from nothing to `res/code = 0.56`.

Three readings worth carrying into the build:

* **Policy codes are obeyed BETTER than teacher codes** (0.754 vs 0.637 strict). Expected
  rather than surprising: the argmax code is the one the head is confident about, which
  correlates with `h` sitting where the decoder renders cleanly, and argmax codes are the
  frequent ones that training covered. It is also the favourable direction for RL, which
  conditions on policy codes, not teacher codes.
* **θ is the weaker channel on strict (0.757) but the stronger on within-1 (0.988)**, while
  xy carries the fat tail — xy's histogram has 110–182 counts out at misses of 5, 6 and 7,
  θ's is empty past 4. So xy fails rarely but badly; θ jitters by one step often. Different
  failure shapes, and only xy's produces a genuinely different trajectory.
* **z₀ moves the sample out of its cell often: only 0.551 (teacher) / 0.707 (policy) of
  extra draws land in the same cell as the first.** This is the §3.2 premise — that z₀'s
  influence is *sub-cell* — and measured, it is not: z₀ jitter is on the order of one
  lattice step, not less. Consistent with the obedience numbers (draws land within one step
  of `c` but not reliably in the same cell as each other), and bounded in behavioural terms
  because an adjacent code is a neighbouring mode. But it is **not** the clean separation
  §3.2 assumed.

  **This strengthens the case for Option A** (§1). With `z₀` pinned and the decode
  deterministic, that ~45% cross-cell jitter disappears entirely and the code is the whole
  action. Under Option B it compounds: `sde_noise_a` would add chain noise on top of a
  z₀ that already moves `g(A')` by about a cell, and the two-sided bound §8 asks for has to
  be derived against that, not against zero.

### Metric note: the first version of this measurement was wrong

Miss distance was initially computed as `|flat index difference|`. FSQ here is `levels
[8, 5]` per channel, so the flat index is mixed-radix `d0 + 8*d1`, and that metric

* counted a difference of **8** — one step in dim 1, an **adjacent cell** — as a gross
  8-step miss. It showed as a hard spike at exactly 8 in the histogram, which is what
  exposed it;
* counted a difference of **1** across a row boundary (7 → 8) as a near miss, when dim 0
  actually moves 7 steps.

Corrected to L∞ on the lattice, the gross-failure rate fell from ~11% to ~1.7% — the bug
inflated it more than 6×. Any earlier within-N obedience number computed on flat indices is
wrong in the same way.

**Consequence for the prototype comparison:** the recorded 94.0%-within-one-step figure
cannot be trusted as a target. The surviving diagnostic (`diagnose_code_flow_head.py`)
computes only strict equality, so how that 94% was derived is not recoverable from the
code; if it used flat indices it is wrong. **Only the strict columns compare.**

### The original gate rationale, for the record

The 94%-within-one-step figure is from the **c-only prototype**, a head with no backbone and
no competing flow objective. If obedience has degraded in the full model, then `c` does not
steer the decode, and RL over `c` is optimising a lever that is not connected. That would
invalidate the entire plan, and it costs one offline pass over held-out turns — no simulator,
no training.

Two further readings worth having at the same time, both cheap:

* **`p(c|h)` is underfit.** A linear probe on frozen `h` beats the in-run head by 0.32 nats
  (`ARCHITECTURAL_DEBT.md`). RL would start from a weaker prior than `h` supports. Not a
  blocker — RL improves the head — but it is a reason to prefer starting from a later
  checkpoint over an earlier one.
* **`res/code` trend.** `ctx_res_over_code` rose monotonically 0.075 → 0.561 over 1200 steps
  with `code_rms` flat. If `r(h)` keeps taking share, `c`'s authority over the decode weakens
  over training — the same thing the obedience gauge measures, from the input side. Watch it
  to the end of the SFT run.

## 10. Ordering

```
GATE   obedience on the trained checkpoint          DONE 2026-08-28, PASSED (section 9)
  │
P0     restore_code_slot in load_flow_stack          (blocker, small)
  │
DECIDE Option A or B                                 (§1)
  │
P1     sample c in the RL sampler, one shared rule   (§3)
P2     trajectory plumbing: code, code_logprob       (§4)
P5     reference policy that covers the code head    (§7)
P4     code entropy term + sampled-code-count gauge  (§6)
P6     obedience + noise gauges wired into the run   (§8)
  │
P3     two-factor ratio/clip                         (§5, Option B only)
  │
SMOKE  one cycle on the dummy env, then a real short run
```

## 11. Not claimed

* No estimate of how long any of this takes. The pieces are scoped, not costed.
* Option A's advantage is an argument from the estimator, not a measurement. The claim that a
  discrete action is more sample-efficient here is the hypothesis the build exists to test.
* Nothing has been run. The blocker in §2 was found by reading, and the rest of the gap
  analysis is a `grep` plus the design doc.
