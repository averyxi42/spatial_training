# RL for the code-conditioned head — revised plan (2026-08-30)

Status: **plan, nothing built.** Supersedes the ordering in `CODE_RL_PLAN.md` (whose §9 gate
result and §1 option analysis still stand); folds in what 2026-08-30 established. Companions:
`CODE_SOFT_LABELS.md` (the SFT-side prior), `FLOW_SDE_RL.md` (machinery + best recipe),
`RTC_RL.md` (prefix contract), `LIVE_ROLLOUT_CADENCE.md` (why the earlier live reads were wrong).

---

## 0. What changed since `CODE_RL_PLAN.md` was written

| then | now |
|---|---|
| live rollouts "spin, useless" | harness ran at 2x the trained replan rate; at gap 10 the hybrid's motion statistics match the expert (`LIVE_ROLLOUT_CADENCE.md`) |
| no closed-loop number | **sample101, matched instrument**: hybrid ck12000 greedy 0.257 succ / 0.584 oracle / 0.298 oSPL; T=0.5 0.267 / 0.535 / 0.321; vanilla 0.545 / 0.663 / 0.434. Deficit is settling + path efficiency, not search. Greedy and T=0.5 are a statistical tie |
| decode unstated | `mean` (p-weighted mean of top-K decodes) is built and is the one decode that is bad by construction — mode-averaging paralyses. Not a candidate |
| one-hot CE | soft metric labels built; `run_code_v6_soft010` training. Output-row geometry of the hard-CE head is only weakly metric (ρ +0.24 at ck12000); decoder-side embeddings are metric (+0.67/+0.76). `CODE_SOFT_LABELS.md` §"Experiment plan" |
| `pin_flow_noise` assumed available for Option A | it is a **no-op on the code path** (`denormalize` returns before the pin branch, `flow_matching_head.py:1096` vs `:1117`). Fix or, better, do not depend on it (§2.2) |
| RL sampler assumed to reuse the rollout seam | it does not: `sample_chain_np`/`chain_log_prob_batch` build `ctx` from raw `h` (`flow_sde_policy.py:342, :452`) and call the decoder directly. P1 is a real change to the sampler, not a wiring |

## 1. The RL policy is the tempered categorical, and the temperature is a policy-class choice

RL samples from what it scores. Define the policy as `π_T(c|h) ∝ p(c|h)^{1/T}` with `T` a
fixed constant folded into the logits (`logits / T` inside `CodePolicyHead.logits` under a
`policy_temperature` attribute). Then rollout sampling, `log π` in the ratio, entropy and the
reference KL are all exact for the same distribution, and the eval decode **is** the trained
policy — no train/eval mismatch, unlike "train at T=1, deploy greedy".

* `T = 1` is the SFT distribution: closed-loop it dithers (sign-flip 0.31 vs expert 0.21 on
  the 4-episode set) and was not run on sample101.
* `T = 0.5`: expert-like coherence, fewest collisions, sample101 tie with greedy on oracle,
  better SPL. **Start here.** Sweep 0.7 if entropy collapses too fast, 0.35 if it dithers.
* Greedy is not a policy RL can improve (no density); keep it as a diagnostic decode only.

The entropy floor (§2.5) is measured on `π_T`, not on `p`.

## 2. Path A — discrete-only RL (build first)

### 2.1 The MDP
Action `a_k = c_k ∈ [0,1600)`, one per 0.4 s decision. The flow decode is the **actuator**:
`chunk = decode(mixer(h, c), z0)`. `log π = log π_T(c|h)` exactly; entropy exact; no chain,
no `sde_positions`, no density mask. 1600-arm contextual bandit per state, the regime
`FLOW_RL_SAMPLE_EFFICIENCY.md` §6 says is well-posed.

### 2.2 z0 is environment noise, not a variable to pin
With the code as the whole action, `z0` is drawn after the policy acts and θ has no influence
on it; treating it as part of the transition kernel keeps the estimator unbiased with **no
pinning**. Pinning is variance reduction only. Given the pin is currently inert on the code
path and the measured z0 jitter is one lattice step (z0 same-cell 0.55), do not build the
plan on it: run unpinned, log the actuator's `mdist(chunk, centroid(c))` as a noise gauge,
and revisit pinning only if that gauge explains a variance problem. (Fixing the pin is a
5-line change if wanted: move the `pinned_flow_noise` branch above the code branch.)

### 2.3 Where the pieces go (file:line hooks)

| piece | hook | change |
|---|---|---|
| **A0 loader** | `flow_sde_policy.py:577` strict `load_state_dict` | call `restore_code_slot(model_shell, meta["fm_code"])` before it, as `flow_matching_head.py:1814` does. Blocker for everything |
| **A1 one code-sampling rule** | `flow_matching_head.py:1039-1055` (rollout seam) + `flow_sde_policy.py:317` (`sample_chain_np`) | factor the seam into `FlowActionCodec.sample_code(h, T, generator) -> (c_xy, c_theta, logp, logits)` and `code_logprob(h, c_xy, c_theta, T)`; `denormalize` AND the RL sampler call the same function. Sampler then sets `ctx = code_mixer(h, c_xy, c_theta)` (`:342`) and runs the ODE (`force_ode`), returning `c`, `logp_c` alongside the chunk |
| **A2 columns** | `rollout_core.py:231-260, 372-379` (store); `rl_core.py:41, :57, :88` (collate) | add `code_xy`, `code_theta` (int64, factored — joint is derivable) and `code_logprob` (float32). Collate: keep the two code columns out of the float-cast list (`:57`) and rank-safe under the `(B,S,1)` squeeze (`:88`) |
| **A3 old-log-prob + ref** | `rollout_core.py:548-580` (recompute), `:581-609` (ref pass) | recompute `log π_T(c_stored|h)` in-context via `code_logprob`; ref pass scores the **stored** code with the reference head (§2.4). Never re-draw, never re-derive `c` from the executed chunk |
| **A4 loss** | `vlm_worker.py:1234-1277` branch select; `:1276` discrete gather; `:1286` ratio | new branch `_is_code`: `log_prob = code_logprob(hh, code_xy, code_theta)`; single `policy_loss_fn` call as today (one factor). Reuse HAPO `clip_cov` / clip ranges; re-establish the `abs_log_ratio` band for the code factor (tenths of nats, not 1e-3) |
| **A5 entropy** | `vlm_worker.py:1376-1392` | `:1376-1384` raises on nonzero `entropy_bonus` for chain heads — do not reuse that key. Add `rl_config.code_entropy_coeff` with the discrete-entropy form at `:1385-1392` on `π_T` |
| **A6 reference policy** | `rollout_core.py:583-606` | `disable_adapter()` covers the LoRA only. Keep a **frozen deep copy** of `code_head` + `code_mixer` (+ `head`/readout if trainable) at RL start; ref pass uses the copy. Small modules; no adapter gymnastics. Reference checkpoint = the SFT checkpoint chosen in §5 |
| **A7 head config** | `vlm_configs.py` registry; new yaml | register `code_flow_head` (FlowSDEHead subclass: `force_ode=True`, code sampling on, `policy_temperature`); `code_rl_held128.yaml` cloned from `flow_sde_freezehead_a09_held128.yaml` (HAPO estimator, γ 0.95, `action_head_learning_rate: 0.0` frozen decoder, `learning_rate: 2e-6`, `ref_kl: true`, `ref_kl_coeff` — see §5) minus the RTC block |
| **A8 gauges** | `vlm_worker.py:1313-1341` panel | §2.5 |

What is trainable in A: the LoRA (→ `h`), `code_head`, `code_mixer` (embeddings + `r`).
The decoder stays frozen (`action_head_learning_rate: 0.0`, the a09 recipe). Whether
`code_mixer` should train is a switch: freezing it keeps the actuator exactly the SFT one
(cleanest attribution); training it lets `r(h)` improve detail. Start frozen.

### 2.4 Estimator
HAPO as shipped (`reinforce_plus_plus_time_kernel`, γ 0.95, `time_kernel_sigma` 40, `n_rollout`
16, `n_adv` 256, `clip_cov`). Per-step `log π` is now a real per-decision density, so nothing
about the estimator needs the chain-specific treatment. Value: the existing state-probe
distance head / time-kernel baseline, unchanged.

### 2.5 Gauges — before the first cycle
* `code/entropy_piT` (mean entropy of the sampled policy) with a **floor alarm**: a code that
  stops being sampled receives no gradient and does not come back. `code/n_distinct_sampled`
  per cycle (SFT sees ~160 of 1600 on ObjectNav eval at ck12000).
* `code/abs_log_ratio_mean`, clip fraction — its own band.
* `code/ref_kl` on the frozen copy (§2.3 A6).
* **Obedience by metric**, not grid: `mdist(g(chunk), c)` in per-tick action-scale units
  (`code_distance_matrix`) on the executed rows; `mass_near` analogue. Grid within-1 stays
  as the legacy number.
* **Closed-loop behaviour from the rollouts**: stationary share vs corpus 0.46, in-place-turn
  run length and `runs > 360°` (greedy's failure signature: 10% vs expert 0.6%), commanded
  |xy| per step. These are the offline precursors of the two known failure modes and cost
  nothing.
* `chain/h_drift_from_init` keeps its meaning.

### 2.6a Names, since the actuator is what distinguishes the variants

| name | policy | actuator | `h` reaches the actuator? | decoder in the loop? |
|---|---|---|---|---|
| **A0 — table** | `π_T(c\|h)` | precomputed `decoder(emb(c), r=0, z0=0)`, 1,600 (or fewer, §2.7) chunks | no | no |
| **A1 — completer** (A0 + RTC) | same head, prefix-aware (§4 RTC-2) | `decoder(emb(c), r=0, z0=0; prefix pinned)` | no | yes — the table is infinite in the prefix |
| B | `π_T(c\|h)` × chain density | SDE chain from `mixer(h, c)` | via `r(h)`, credited by the chain term | yes |
| B′ | `π_T(c\|h)` + critic `Q(s, A)` on the fixed actuator | `F_θ(c, h)` differentiable | via `r(h)`, credited pathwise | yes |

A0 and A1 share the estimator exactly; only the actuator (and, for A1, the head's prefix
input) changes.

### 2.7 Vocabulary pruning for A0: merge prefix-equivalent codes by marginalisation

In A0 only ticks 0–9 execute, so codes whose centroids agree on the prefix are
reward-indistinguishable (§9.7). Merge them **without touching the SFT head**: define the RL
action as the class, `π(class | h) = Σ_{c ∈ class} π_T(c | h)` — `log π(class)` is the
`logsumexp` of member logits — and execute one representative's table entry (the
heaviest-prior member, or the prefix medoid). Exact (a marginal of the trained policy, no
retraining), credit is shared across members by construction (the gradient of the
`logsumexp` spreads over members in proportion to their mass), and **reversible**: A1 drops
the merge and the same head is back to 1,600 because the tail is binding there.

Measured on the centroids (greedy classes by prior mass; the full-chunk codebook's own
nearest-neighbour spacing is 0.19, so radii below it merge only what the tokenizer itself
does not resolve on the prefix):

Units: the radius is in per-tick `action_scales` units (RMS over the 10 executed ticks and
3 dims of differential ÷ scale; 1 unit = 3 cm/tick of translation or 0.05 rad/tick of
rotation). 0.15 ≈ 4.5 mm/tick or 0.43°/tick — at most ~4.5 cm or ~4.3° over the executed
half-chunk if systematic, under the tokenizer's own reconstruction error. Not metres.

| merge radius (prefix metric) | classes | largest class | classes holding 90% of prior mass | prior entropy over classes (eff. #) |
|---|---|---|---|---|
| 0.10 | 1,400 | 4 | 135 | — |
| **0.15** | **1,016** | 11 | 97 | 4.08 nats (59) vs 4.84 (126) over codes |
| 0.20 | 667 | 20 | 54 | — |
| 0.25 | 430 | 43 | 38 | — |
| 0.30 | 283 | 76 | 23 | — |

Default 0.15: the vocabulary drops to ~1,000 classes and the *effective* action set (prior
entropy) halves, 126 → 59, while staying under the codebook's own resolution. 0.20–0.25 are
the aggressive sweep points. The class map is a (1600,) int array computed once from the
tokenizer + `action_scales`; store it in `meta["fm_code"]` for provenance.

### 2.6 Smoke → run
Dummy env one cycle (A0–A7 wired, gauges present) → `code_rl_held128` twin of
`flow_sde_rtc_held128` on the same 128 uids, interleaved eval26 → read against the sample101
baseline (§0) with the same instrument. Success criterion for continuing to B: held-out oSPL
and auto-stop success moving, entropy not collapsing.

## 3. Path B — hybrid: code + flow-SDE chain (only after A)

Action `(c, chain)`, `log π = log π_T(c|h) + Σ_k log N(transitions)`, exact factorisation
(`CODE_CONDITIONED_POLICY.md` §6). Adds to A:

* **B1** `chain_log_prob_batch` takes `codes` and builds `ctx = code_mixer(h, c)` (`:452`);
  sampler keeps the SDE path (`force_ode=False`) after the code draw (A1).
* **B2 two-factor ratio** — required, not optional: the code term moves in tenths of nats,
  the chain term in ~5e-4; one ratio clips on the code and zeroes the chain gradient. Two
  `policy_loss_fn` calls at `vlm_worker.py:1286`, own clip ranges, own gauges.
* **B3** `sde_noise_a` re-derived with the two-sided bound: chain noise must not move
  `g(chunk)` out of the cell RL just chose. Measured z0 jitter already sits at one lattice
  step, so the bound is against that, not zero; use the `mdist` gauge as a function of `a`.
  The shipped 0.9 is not inherited.
* What B buys: the continuous detail channel becomes learnable under RL. What it costs: the
  z0 mis-credit pathology of `FLOW_RL_SAMPLE_EFFICIENCY.md` §6 returns for the chain factor,
  bounded to sub-cell effects.
* Decision rule: build B only if A saturates **and** the obedience/`mdist` gauges say the
  actuator's detail (not the code choice) is what limits return.

### 3.1 What Path B is, plainly (and its successor)

With `z0` fixed, the flow-SDE construction is "denoising as MDP" (DPPO / Flow-GRPO): the K
solver steps are the actions, noise at `n` of them manufactures a density, PPO credits the
chain for the terminal reward — structurally a flow-matching-initialised *embodied chain of
thought* sampled at 3 of 10 tokens and tuned by RL. The analogy breaks where it matters: CoT
tokens are read and have semantics; the solver's intermediate states are numerics nobody
reads, and the stochasticity exists only to have something to take a ratio of.

Consequences:
* **B's exploration is weak by construction** — three Gaussian perturbations of a numeric
  trajectory; the exploration that matters is the code's. B ≈ A + a small local refinement
  of the actuator learned by REINFORCE on three numbers, paying the two-factor ratio and the
  noise bound for it. The strongest argument for A-first.
* **The honest successor to B is A + a critic on the fixed actuator**: with `z0` fixed the
  chunk is a differentiable function `A = F_θ(c, h)`, so `r(h)`/mixer get exact pathwise
  credit from `Q(s, A)` (`FLOW_RL_SAMPLE_EFFICIENCY.md` §6) with no manufactured density,
  while the code keeps its score-function credit. Record this as "B′"; build it instead of
  B if the gauges ever say the actuator's detail limits return.
* What the CoT lens adds constructively is process supervision: obedience (does the chain
  stay in the code's cell) is the natural intermediate reward — the content of B3's
  two-sided noise bound.

## 4. RTC integration

**Composition is clean at the decoder**: the mixer replaces `ctx`, the prefix pin is a row
mask, they are orthogonal (`pin_prefix`, `prefix_time`, `_sde_transition` all take a generic
`(N,T)` mask). Credit is already right at the reward level: `retime_commit_rewards`
(`train_loop.py:275-284`) bills the fresh rows to decision `k` and the committed rows to
`k-1`, so `log π_T(c_k|h_k)` pairs with the re-timed reward unchanged.

**The real issue is semantic**: `c_k` describes the whole 20-tick chunk, including the `d`
committed rows the policy cannot influence, and `c_k` is drawn independently of `c_{k-1}`
whose tail became the commitment. A code that disagrees with its own prefix is executed as a
pinned prefix + a postfix that obeys `c_k` — a discontinuity at the splice, the within-chunk
version of the dithering seen at T=1. This is not RTC-specific in principle (any replan
splices), but RTC makes the disagreement mandatory rather than incidental.

Ladder, cheapest first:
1. **Measure on the fresh rows only.** Obedience/`mdist` restricted to `[d, H)`; log the
   splice discontinuity (jump in commanded velocity at row `d`). Establish how large the
   effect is before designing around it.
2. **Prefix-aware code head.** Give `CodePolicyHead` the commitment as input: the `(H−gap,3)`
   zero-padded scaled prefix differentials plus `d`, through a small MLP concatenated to
   `LN(h)`. Train it in the RTC SFT fine-tune (the shipped recipe: normal run, then RTC
   conditioning at ~75% of compute), where the teacher prefix is the expert's own previous
   rows, so `p(c|h, prefix)` learns consistency. The tokenizer is untouched; this is ~50
   lines and one fine-tune.
3. **Postfix-only tokenizer** (retrain the FSQ on rows `[gap, H)` or on the fresh span) so
   the code never re-describes committed motion. Cleanest, most expensive; only if 2 is not
   enough.

Ordering: A without RTC first (the env's `rtc_delay_max: 0`), then A + RTC with step 2,
then B. RTC on B adds nothing new beyond B1–B3 composing with the mask.

## 5. Where the metric prior lives (decided; see `CODE_RL_PLAN.md` addendum)

* **Representation**: start RL from the soft-label checkpoint (`run_code_v6_soft010`) if it
  beats v4 on sample101 (T=0.5, gap 10) and its output-row geometry moved
  (`dump/audits/code_head_geometry.py`); otherwise from v4 ck12000. Either way the choice
  is made on the same instrument as §0.
* **Reference**: the same checkpoint as `π_ref`; `ref_kl: true` and start with
  `ref_kl_coeff: 0.0` (measure) as the a09 recipe does, raising it only if the code entropy
  floor or `ref_kl` says the policy is leaving the smooth prior faster than return justifies.
* **Not the estimator**: no kernel on the advantage.

## 6. Decisions to take before writing code

1. `T` for `π_T` (default 0.5).
2. Starting checkpoint: v4 ck12000 vs v6 (wait for v6's sample101 + geometry; ~1 day).
3. `code_mixer` frozen or trainable in A (default frozen).
4. Reference mechanism: frozen copy (default) vs adapter.
5. `code_flow_head` as a new registered head vs flags on `flow_sde_head` (default: new head;
   it keeps `flow_sde_*` yamls byte-identical, per the repo's stated preference).

## 6.5 Build state (2026-08-30, evening)

A0 is BUILT, by capability rather than by editing any dispatch site — the discrete,
Gaussian, latent and chain paths are untouched (verified: full `tests/rl_math` 36/36 and
`tests/rollout` 13/13 pass unchanged):

* `longnav/utils/code_flow_rl.py` — `CodeFlowHead(FlowSDEHead)`: presents the chain head's
  exact surface; the "chain" is the two code factors as float32 (exact < 2^24), so storage,
  collate, the old-log-prob recompute, the disable_adapter reference pass, `rl_loss`'s
  chain branch and the seam-gap gauge all carry it with ZERO edits. Table actuator
  `decoder(mixer(c, r=0), z0=0)` precomputed per device; `force_ode` (eval) = argmax;
  RTC prefix refused loudly (A1 unbuilt); `merge_radius > 0` refused (unbuilt).
* `load_flow_stack` gained one branch GUARDED on `meta["fm_code"]` (non-code checkpoints
  byte-identical), calling `build_code_slot_shell` — factored out of `restore_code_slot`
  with identical construction.
* `code_flow_head` registered in `vlm_configs.py` (additive).
* `code_rl_held128.yaml` — the a09 held128 twin, differing only in the head block;
  checkpoint v4 ck12000 until v6 decides (two lines to swap, `checkpoint_dir` and
  `merge_adapter_dir` together).
* Tests: `tests/rl_math/test_code_flow_rl.py` (8) — sampler/scorer parity at the stored
  code, differentiability through `h`, temperature in both, argmax determinism, table ==
  direct decode, codes exact through collate, RTC refusal, and BIT-PARITY of the sampled
  codes with the `denormalize` eval seam under a shared seed (the "one sampling rule"
  invariant, enforced by test rather than by refactoring the eval path).
* End-to-end on the real ck12000: loads, table (1600, 20, 3) finite, sampler/scorer agree
  to 1e-6; the vanilla ck12000 still loads with no code slot.

Compose-validated under real Hydra (`tests/compose_demos/cases/train_code_rl_a0.yaml`):
the resolved config matches the a09 held128 control on every field but the head block
(estimator `reinforce_plus_plus_time_kernel`, `action_head_learning_rate 0.0`,
`ref_kl true`, sim gap 10 / dt 0.04; head `CodeFlowHead`, T 0.5, ckpt v4 ck12000).

Compose case `train_code_rl_a0` PASSES in the demo suite. **Launch hazard, found and
recorded in the yaml header:** the longnav_vlm editable install resolves `longnav` to
`/Projects/spatial_training/src` (flowsde), and `train_rl.py` does not munge `sys.path` --
a launch without `PYTHONPATH=/Projects/spatial_training_tok/src` silently runs flowsde
code with this yaml's fields. The Ray actors' runtime_env must carry the same path.

Launch: `data_scripts/launch_code_rl_a0.sh` (defaults GPUS=4,5,6,7, quad resources,
refuses busy devices, exports the worktree PYTHONPATH, pins conda envs).

First-launch lessons (2026-08-30/31), each fixed and committed: seeded generator built on
CPU before the head moves to CUDA (draw on the generator's device); the env's
r_commit/r_fresh emissions arrive float64 and the retime identity rebuilt rewards as
Double (collate-cast + retime cast); DDP kills training when requires-grad parameters
never receive grad -- the code-only loss never touches the decoder/mixer, so the A0
actuator is now frozen at the PARAMETER level, which is also what 9.1's exactness wants
(the a09 lr-0 freeze sufficed for the chain head only because its density backprops
through the decoder); a zero-turn (DOA) episode crashed postprocess_episode's empty
torch.cat 23 cycles in -- guarded to the failed-episode drop path. Also: reusing
task.run_name across relaunches RESUMES the crashed wandb run and its step watermark
silently drops early rows (launch script now timestamps the name).

Deferred, refused loudly if requested: A1 completer, class merging, the code-entropy term
(first runs are the a09 freeze + measure-only ref-KL), frozen-copy reference (unneeded
under the freeze).

## 7. Order of work

```
A0  restore_code_slot in load_flow_stack                    (small; unblocks loading)
A1  one code-sampling rule shared by denormalize + sampler  (the design invariant)
A2  code_xy / code_theta / code_logprob columns              (store → collate → dispatch)
A3  in-context recompute + frozen-copy reference             (A6)
A4  rl_loss code branch, single ratio                        
A5  code_entropy_coeff + floor alarm                         
A7  code_flow_head registration + code_rl_held128.yaml       
A8  gauges (§2.5), incl. rollout behaviour stats             
SMOKE dummy env → held128 twin → sample101 read              
── decide on B and on RTC step 2 from the gauges ──
RTC-1  fresh-row obedience + splice gauge                    (measurement only)
RTC-2  prefix-aware code head + RTC SFT fine-tune            
B1-B3  chain factor, two-factor ratio, noise bound           
```

## 8. Not claimed
* No time estimates; pieces are scoped against file:line hooks, not costed.
* Path A's sample-efficiency advantage is the hypothesis the build tests; the sample101
  baseline in §0 is what it must beat, on the same instrument, paired.
* Nothing here has been run. The hooks were verified by reading
  (`flow_sde_policy.py`, `rollout_core.py`, `rl_core.py`, `vlm_worker.py`, `train_loop.py`)
  on 2026-08-30; `code_head|code_mixer|last_code|restore_code_slot` occur 0 times in them.

---

## 9. Theory notes (2026-08-30, while v6 trains)

### 9.1 `h` serves two masters in Path A, and the second one is uncredited

Write the chunk the environment executes as

    A = f( c, r(h_θ), z0 ),      c ~ π_T(· | h_θ),      h_θ = readout(LoRA_θ(o)).

The code-only score-function estimator differentiates `log π_T(c|h_θ)` and treats everything
downstream of `c` as the environment. That is exact **only if `f`'s other inputs do not
depend on θ**. `r(h_θ)` does: every LoRA step that improves the code choice also moves the
residual tokens the frozen decoder conditions on, so the meaning of every code drifts, and
the estimator has no term for it. Three consequences:

* the MDP the code policy is solving is **non-stationary in θ** — a bandit whose arms'
  payoffs move as you pull them, with drift proportional to the LoRA learning rate;
* the true policy gradient has a second, **pathwise** term `∂R/∂A · ∂f/∂r · ∂r/∂h · ∂h/∂θ`
  that the code-only estimator drops. It is not a variance issue; it is a missing term;
* the a09 recipe has the same structure (frozen decoder, LoRA moves `h`), but there the
  chain density credits `h` on the path the decoder actually uses. Here the code density
  credits `h` only for *which cell*, never for *what the cell renders*.

The uncredited path has to be made θ-independent, i.e. the actuator must be **fixed**.
The options, ranked by cleanliness:

| option | actuator | unbiased? | keeps SFT detail? | cost |
|---|---|---|---|---|
| (i) `r(h)` live, LoRA trains | `f(c, r(h_θ), z0)` | no (drift) | yes | none; gauge `h_drift`, `mdist(f(c,r(h_t)), f(c,r(h_0)))` |
| (ii) **zero `r(h)`** | `f(c, 0, z0)` | yes | only if the decoder is competent at `r = 0` — measure (§9.2) | none at RL time |
| (iii) `r(h_ref)` from the frozen reference forward | `f(c, r(h_ref), z0)` | yes | yes | a second backbone forward at **rollout**, not just at scoring (~2x) |
| (iv) freeze the LoRA too; train only `code_head` (+`code_mixer` off) | `f(c, r(h_0), z0)` | yes | yes | RL cannot improve `h`; the code head is a 1600-way linear-ish readout of a fixed feature |
| (v) credit the path (differentiable actuator + critic `Q(o, A)`) | — | yes | yes | that is Path B / the reparameterised route, not Path A |

(ii) is the simplest fixed actuator and has a strong precedent: the decoder was warm-started
from the c-only prototype, whose tokens 4–7 were exactly zero, and that prototype's teacher
obedience (0.672 strict) was essentially the full model's (0.637). Whether 12k steps of live
`r(h)` made the decoder dependent on it is an empirical question answered in §9.2.

**The clean SFT-side fix, if (ii) costs accuracy: `r`-dropout.** Zero tokens 4–7 for a random
half of the SFT batch (per turn), so the decoder is trained to be a valid actuator both with
and without the residual. Then RL under (ii) has no distribution shift, Path B can re-enable
`r(h)` later without retraining, and the SFT eval reports both regimes. This is the
ControlNet "zero-convolution" idea used at training time rather than only at init. One flag,
one more run; it belongs in the v7 recipe alongside soft labels.

(iv) is worth keeping as the **control**: it isolates "can the code head alone learn from
return on a fixed `h`", which is the purest form of the discrete-action hypothesis, and it
is the cheapest possible RL run (no backbone gradients at all).

### 9.2 Measurement: is the decoder competent with `r(h) = 0`? (ck12000, 200 val rows)

Per-tick composed pose RMSE against the expert chunk, 9,623 held-out ObjectNav turns, real
decoder (`dump/audits/policy_pose_error.py`, `pose_err_code_v4_ck12000_r0.json`):

| condition | dθ (rad) | dx (m) | dy (m) | final dθ | final xy |
|---|---|---|---|---|---|
| teacher code, `r(h)` live | 0.081 | 0.060 | 0.053 | 0.149 | 0.141 |
| teacher code, **`r = 0`** | 0.097 | 0.079 | 0.067 | 0.174 | 0.175 |
| argmax code, `r(h)` live | 0.553 | 0.250 | 0.149 | 0.945 | 0.511 |
| argmax code, **`r = 0`** | 0.562 | 0.253 | 0.151 | 0.966 | 0.518 |

Reading: `r(h)` carries real within-cell detail — with the *true* code, zeroing it costs
+20% dθ / +32% dx per tick (~0.02 rad, ~2 cm) — but at the *policy's* codes that loss is
1-2%, six times smaller than the error the code choice itself contributes. So for Path A,
zeroing `r(h)` is a fixed actuator at negligible cost today, and the `r`-dropout SFT (§9.1)
would make it exactly free while keeping the detail channel available for Path B. Default
for Path A: `r = 0`, pinned `z0` (§9.11).

### 9.3 What the code-only policy gradient is, exactly

With a fixed actuator the objective is `J(θ) = E[ Σ_k γ^k r_k ]` over `c_k ~ π_T(·|h_θ(o_k))`
and everything else environment (sim, `z0`). The estimator

    ∇J = E[ Σ_k ∇θ log π_T(c_k | h_θ) · Â_k ]

is the standard one; `z0` sits in the transition kernel, so its variance is paid for by
`Â_k`'s variance, not by any credit mis-assignment (`FLOW_RL_SAMPLE_EFFICIENCY.md` §6's
pathology is gone rather than bounded). Regret in the contextual-bandit reading scales with
`log|F|` for the head's function class, not with 1600 — *if* the class generalises across
codes, which is the output-row geometry question (`CODE_SOFT_LABELS.md` §"Experiment plan").

Temperature: `π_T ∝ p^{1/T}` is a reparameterisation of the logits, so `∇θ log π_T` is
`(1/T)`-scaled and the PPO ratio is on `π_T`. The KL to the reference should be
`KL(π_T ‖ π_T,ref)` — same `T` on both sides — or it measures the temperature, not the drift.

### 9.4 Exploration and collapse, with numbers

* At `T = 1` the SFT head's entropy is 3.4 nats (~30 effective codes); at `T = 0.5` it is
  lower by construction. The absorbing failure — a code that stops being sampled gets no
  gradient — is reached faster at low `T`. The floor gauge is on `π_T`'s entropy and on the
  per-cycle distinct-code count; the entropy coefficient is a real hyperparameter here, not
  the 0.0 the continuous recipe used.
* The soft-label prior acts on exploration through the reference KL: mass leaking to
  metric neighbours is cheap under `KL(π‖π_ref)` when `π_ref` is smooth, expensive when it
  is one-hot-shaped. With the v6 reference, "explore nearby motions" is the default direction
  of drift; with the v4 reference it is not. This is the practical content of "the prior
  lives in the reference".
* Effective horizon: γ = 0.95 at 2.5 Hz is ~8 s of credit; `time_kernel_sigma` 40 steps
  is ~16 s. Search decisions ("stop scanning, go through that door") pay off on that scale
  or longer, so the settling/stopping deficit the sample101 baseline shows is within reach of
  the estimator, the exploration-of-unseen-rooms deficit is at its edge.

### 9.5 Ratio scale

`log π_T(c|h)` moves in tenths of nats per update for ordinary probability shifts, three
orders of magnitude above the chain term the clip ranges were tuned on. Two consequences for
Path A even without a second factor: the clip will bind on a large fraction of tokens at the
shipped `learning_rate: 2e-6` unless the code head's effective step is smaller (its own lr,
or a wider clip for the code factor), and `abs_log_ratio_mean` needs a band established on
the first cycles rather than inherited from `chain/`. Path B inherits the same problem and
adds the factor split (§3 B2).

### 9.6 Entropy in cell space is not neutral in motion space
A uniform-over-1600 entropy bonus pushes mass toward the ~10 drive cells collectively and away
from the single stationary cell: it is a bias toward driving relative to a uniform-over-motions
prior — the "metric vs bins" trap in the regulariser. Use `KL(π_T ‖ π_ref)` (shaped by the
soft-labelled prior) as the anti-collapse term; keep absolute entropy as a floor alarm only.

### 9.7 The reward sees the executed prefix; the code describes the chunk
Under non-RTC replan ticks 10–19 never execute, so codes differing only there are
reward-indistinguishable. On the 1,600 centroids at radius 0.1 (per-tick metric): 1,568
distinct classes on the full chunk, **1,367 on the executed prefix**, 1,479 on the tail —
~15% of the vocabulary is invisible to return. The gradient cannot separate those codes; the
prior resolves the ties (`ref_kl_coeff > 0`). Under RTC the tail becomes binding, so
mass that drifted among tail-variants during non-RTC RL gets executed: start RTC-RL from its
own SFT fine-tune (§4 RTC-2) or keep the leash, never from an unleashed non-RTC RL result.

### 9.8 STOP is native to the discrete action
The stationary cell family is already a stop action. A rule "k consecutive stationary codes
= stop" makes the settling deficit (sample101: 0.27 vs 0.55 success at near-equal oracle)
learnable in Path A from a stop-time reward, with no stop head and no extra density. Worth
building as part of A8's gauges first (how often does the policy already satisfy the rule
inside the goal window?) and as the stop mechanism second.

### 9.9 One degree of freedom
`T` and the entropy coefficient regulate the same thing: fix `T` (policy class), tune the
KL/entropy coefficient. Score-function variance for a rare code scales as `1/π_T(c)`: lower
`T` thins the tail, the clip bounds it, the metric prior shares rare-code credit with
neighbours. Complementary, not redundant.

### 9.10 RTC makes the code MDP a POMDP unless the head sees the commitment
The commitment is state (`RTC_RL.md` §1). A code head that reads only `h` acts on a partial
state; the prefix-aware head (§4 RTC-2) is the theoretical requirement, not an empirical patch.

### 9.11 `z0`: pin, zero, or leave random — revised

Three different things, with very different risk:

* **Random `z0`** (SFT-typical). Within-cell jitter of ~one lattice step per decision
  (z0 same-cell 0.55), averaging out across steps. Unbiased for Path A; the cost is variance
  in `Â`, not credit.
* **Pinned typical draw** (one `z0 ~ N(0,I)` reused). Deterministic actuator per `(h, c)`,
  evaluated where the field was trained (`||z0|| ≈ 7.7`). Risk: the SAME within-cell
  realisation bias for every code and state ("always veers slightly left inside every
  cell") that the code policy cannot correct in Path A. Mitigation: pin **per decision**
  (seeded from episode/step) — deterministic given the state, uncorrelated across states.
  The Wen–Van Roy determinism argument is theoretical comfort here: KV-context states never
  repeat anyway, so it must not drive the choice.
* **`z0 = 0`**. The mode of the base density, a measure-zero point the velocity field never
  saw (every training start has norm ~7.7). The ODE from the origin tends toward the
  conditional's centre — the `mean`-decode failure in a milder nonlinear form. **High risk
  for Path A**: if the origin maps to a between-modes chunk for some cells, those cells are
  silently broken and the code lever cannot repair them. Measured in §9.12.

**Path B with a fixed `z0`** (pinned or zero) is the appealing version: the SDE transitions
at the `n` stochastic positions still form a valid Gaussian policy with an exact density, the
credit that used to be lost to `z0` lands on `μ_θ` and — through `h` — on `r(h)`, and `r(h)`
is taught by return to recover whatever expressiveness the fixed start removed. The
atypical-start risk applies to the first Euler steps only, so a pinned typical draw is the
safer form of the same idea.

Default for Path A (settled in §9.12): `r = 0` (§9.2) and **`z0 = 0`** — the actuator is a
fixed table; random `z0` is the ablation. The pin no-op in `denormalize` still needs fixing
so the eval harness can run the same actuator the policy was trained with.

### 9.12 Measurement: decode quality by `z0` regime (ck12000, teacher codes, `r(h)` live)

`dump/audits/z0_modes.py`, `z0_modes_ck12000.json`; per-tick composed pose RMSE vs expert and
metric distance to the code's own centroid (per-tick action-scale units):

| `z0` | dθ | dx | dy | final dθ | final xy | stationary | mdist to centroid |
|---|---|---|---|---|---|---|---|
| random (SFT-typical) | 0.083 | 0.060 | 0.052 | 0.152 | 0.138 | 0.437 | 0.335 |
| pinned typical draw (‖z0‖ 7.3) | 0.074 | 0.057 | 0.052 | 0.130 | 0.132 | 0.437 | 0.328 |
| half-scaled draw | 0.070 | 0.051 | 0.044 | 0.129 | 0.119 | 0.438 | 0.313 |
| **zero** | **0.070** | **0.048** | **0.044** | 0.131 | **0.116** | 0.441 | **0.312** |

`z0 = 0` is the best decode on every column (~15% below random) and the closest to the
centroid. The between-modes fear of §9.11 does not apply *inside* a cell: the code has
already chosen the mode, the within-cell conditional is unimodal, and the origin maps to its
centre — the RMSE-optimal point. This is the opposite of the `mean` failure, which averaged
*across* cells. (Teacher codes, `r(h)` live; §9.2 says `r = 0` costs 1–2% at policy codes.)

**Path A default, settled by measurement: `r = 0`, `z0 = 0`.** The actuator is then a
precomputed 1,600-entry table `decoder(emb(c), 0, z0=0)` — Path A is discrete RL over 1,600
learned motion primitives, and the decoder is not in the rollout loop at all. RTC-A runs the
same decoder as a prefix-conditioned completer with `z0 = 0` (the table becomes infinite in
the prefix; the flow head is needed exactly there). Random `z0` is the ablation.

## 10. Feature analysis: top-K code-path overlay in RL training videos (2026-08-30)

**Requested**: the eval harness's top-K mode overlay, in the videos the RL run logs.

**Videos are logged**: `code_rl_held128.yaml` sets `minimal_logging: false`, so the env
actor writes one MP4 per episode (`vid/episode_video` to wandb) from frames captured every
`video_tick_stride: 2` physics ticks.

**A corpse exists and its cause of death is repaired by A0.** Commit dc4f711 built exactly
this (env `step(mode_chunks=)` kwarg -> `_mode_chunks`/`_mode_anchor` -> `_overlay_modes`
in `_video_on_tick` via `continuous_demos.viewport_overlay.draw_commanded_chunk`; rollout
pass-through reading `codec.last_mode_chunks`); 09b30b6 removed it because the Ray path
never fills `last_mode_chunks` -- RL heads bypass `denormalize`, so the code slot never
fired. `CodeFlowHead` repairs precisely that: the top-K modes are the K highest tempered
logits' TABLE ROWS, available inside `sample_chain_np` with no decoder pass and no RNG.
Resurrect the 56 removed lines, fill `mode_chunks` from the head (K rows + probs), done.

**Cost, bounded**:
* GPU: **zero** for A0 (table lookups; the old denormalize version decoded K x M chunks and
  needed RNG snapshot/restore -- none of that exists here).
* Ray transfer: K=5 x (20,3) float32 + probs ~= 1.3 KB per step, riding the existing
  `step.remote` call.
* Sim-actor CPU: `draw_commanded_chunk` is "one 4x4 inverse, an (L+1)-point projection and
  one cv2.polylines" (its own docstring, quoted against ~3 ms for the render). K=5 draws on
  each captured frame, ~5 frames per decision (gap 10 / stride 2): ~1-2 ms/frame, so
  **~5-10 ms per env step**, against a 150-500 ms policy step. Worst case ~3%, typical ~1%
  -- far inside the 15% budget. If it ever matters, `video_tick_stride` or a
  draw-every-Nth-frame knob halves it.

**Contamination analysis (the real requirement)**:
* The policy's observation and the video frames are SEPARATE renders: obs rgb comes from
  `_render()` at the step/reset boundary (`objectnav_continuous.py:541, :778`); video
  frames from their own `_render()` calls inside `_video_on_tick`, JPEG-encoded
  immediately. The overlay lives only in the video path, one function deep.
* Residual risk: `_render()` returns a VIEW of habitat's sensor buffer and cv2 draws in
  place; habitat re-renders the buffer on every `get_obs`, so annotation cannot persist
  into the next obs render -- but the guarantee should not rest on habitat's buffer
  semantics. **Rule for the resurrection: draw on `rgb.copy()`** (one line, ~1 ms for
  640x480, inside the budget above).
* `mode_chunks` stays a SEPARATE step kwarg, never a `supplementary_logs` key (that dict
  feeds a scalar aggregator, and a (K,T,3) array would break it -- the removed code's own
  note) and is never written into `obs`; the dynamics never read it.
* Eval-cycle determinism: unaffected -- the modes are argsorted logits already in hand and
  the drawing consumes no RNG (unlike the denormalize overlay, which had to snapshot four
  RNG streams).

**BUILT (2026-08-30, same day):** 09b30b6 reverted with the `.copy()` hardening and a
reset-time stale-mode clear; the rollout pass-through reads-and-clears the ACTION HEAD's
`last_mode_chunks` (kwarg emitted only when non-empty, so every other head/env keeps its
exact signature); `CodeFlowHead.overlay_modes_k` fills the top-K tempered-logit table rows
(no decoder pass, no RNG); `overlay_modes_k: 5` in `code_rl_held128.yaml`. Row 0 of
`mode_chunks` is the SELECTED (executed) code by contract, drawn thick white on top; rows
1.. are the top-K alternatives by rank colour, selected excluded -- under sampling the
executed code is not the top-probability one, so without the convention the chosen path is
anonymous. Also built: `rl_config.code_entropy_coeff`, the discrete-code analogue of
`entropy_bonus` (which the chain guard rejects for this head); pi_T entropy is logged as
`code/entropy_piT` always, applied to the loss only at nonzero coeff (default 0.0 --
watch, don't leash). Tests: the
head-side top-K fill, and a real-run_episode pass-through test with a recording env actor
(kwarg present iff filled; historical signature byte-identical when off).

**Verdict: build it with the A0 head** (~70 lines: revert 09b30b6 + `.copy()` + the head
filling `last_mode_chunks` from the table + an `overlay_modes_k` env/rollout knob,
default 5, 0 = off). Cost is ~1-3%, well under the bar.
