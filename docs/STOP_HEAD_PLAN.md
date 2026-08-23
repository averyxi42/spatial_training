# Stop head: preliminary plan

> **RETRACTION (2026-08-23). Every on-policy number in this document is invalid.**
>
> The observations were recovered by decoding rollout `video.mp4` files
> (`data_scripts/mine_rollout_frames.py`). **Those videos carry a rendered HUD printing
> `distance_to_goal: <full float>`, `goal: <name>` and `step: <n>`.** A distance or stop
> head fit on them reads text; it does not see. The tell was a run reaching precision
> 1.000 and distance MAE 0.207 m on nominally held-out scenes.
>
> Independently: the "held-out" split was drawn from the on-policy rollout archive, i.e.
> HM3D **train** scenes, 11 of whose 12 scenes are in the PointNav corpus these
> checkpoints trained on. **Evaluation must use HM3D val** — the instruction was given and
> not followed.
>
> Deleted 2026-08-23: the mined frame stores, the `*_series.npz` hidden caches, the
> `refit_value_head_*.pt` heads, the `onpolicy_distance` corpora, and the
> `run_cotrain_v6_onpolicy` / `run_cotrain_v7_stop` runs. Summaries preserved for the
> record only in `dump/eval_system/QUARANTINE_contaminated/`.
>
> The one clean distance measurement for this lineage is live rollouts on HM3D val
> (`dump/probe_eval/probe_eval_v6ck2400_hm3dval/`): **MAE 3.32 m against a 3.19 m
> constant-prediction baseline — no skill on unseen scenes**, AUC 0.856 at 1 m, and the
> clock shortcut (partial corr with step at fixed true distance) unchanged at -0.130.
>
> Mechanics also repudiated, independent of the leak: do not run the RL trainer
> (`train_eval_rl`, `lr 0` or otherwise) to collect evaluation rollouts — use
> `scripts/eval_objectnav_policy.py` or `longnav.scripts.eval`; and do not re-run forward
> passes offline over saved frames — compute distance/p_stop/value INLINE in the rollout's
> own forward pass and return them with the action.
>
> What survives: the *structural* findings that were measured on live rollouts or on the
> demonstration corpora — the clock shortcut, the train/rollout distribution mismatch, and
> the stop-rule ceiling arithmetic.


**2026-08-18.** Prompted by the navverse cross-benchmark results: our continuous arms run
with oracle auto-stop (a fairness asterisk on every published number), and the discrete
family's measured stop gaps -- reaching-vs-stopping -- are worth +8 to +16 pp SR on that
benchmark. At home, ck791's sample400 shows 0.733 oracle vs 0.589 success under auto-stop:
~57 of 397 episodes touched the success radius and still failed. Stopping is the largest
quantified performance pool the program has never worked.

## What already exists (the gap is head quality, not plumbing)

| piece | state |
|---|---|
| `longnav/utils/stop_head.py` | classifier on pooled per-turn context; `stop_grad=True` detaches it from backbone/motion head (provably free) |
| `vector_sft.episode_stop_labels` | labels = "final observation of the episode is THE stop" -- ONE positive per episode, position-derived, no distance |
| `objectnav_eval.bridge._StopHeadControl` | finds `model.stop_head`, runtime `--stop-threshold/--stop-temperature/--stop-inference`, refuses `--auto-stop` combo |
| distance sidecar annotation | built+tested 2026-08-17 (`annotate_distance_to_goal.py`): per-frame geodesic d_t for any pose2d corpus, navmesh-matched |
| on-policy state supply | ~25k RL rollout episodes on disk with per-step d_t (`sequence.json`) and frames recoverable from `video.mp4` at policy-step indices |

## Design principle: predict distance, not "stopness"

Replace the binary end-of-episode label with **log-distance regression**: d_hat(h) trained
on dense, physically defined targets from the distance sidecar (demos) and sequence.json
(rollouts). Stopping is then `d_hat <= r` + hysteresis, with r a deploy-time knob:

* one head serves every radius (0.4 in-training / 1.0 sample400 / 1.6 navverse) -- the
  binary head would need retraining per benchmark;
* dense supervision (every frame) instead of one positive per episode;
* the same head is a **non-privileged distance feature**: it legitimizes the
  distance-kernel advantage baseline without oracle d_t, and is a concrete stepping stone
  to V^pi (predicting d is most of predicting cost-to-go). Three programs, one probe.

## Track A -- inference-time-only head (RECOMMENDED FIRST)

Probe on `h` (the pooled context the policy already computes every step): **zero rollout
slowdown by construction** -- `_StopHeadControl` already reads exactly this. Training is
fully offline on the idle eval GPUs (5-7), never touching the training fleet.

1. **Phase 0 (no GPU, ~half day).** Quantify the recoverable pool from s400 logs: of the
   57 oracle-not-success episodes, how many had a *sustained* (not transient) approach --
   the honest ceiling for any stop policy. Audit current stop-head AP as baseline.
2. **Phase 1 (~2-3 days, GPUs 5-7).** Run the sidecar annotation over the demo corpus
   (CPU). Mine on-policy frames: decode rollout MP4s at policy-step indices, pair with
   sequence.json d_t -- this is the distribution the head will actually run on, and its
   absence is the standard failure of offline stop heads. Extract h for both sets with the
   frozen ck791 backbone. Train log-distance MLP; report AP/AUROC at each radius +
   calibration curves, held out by scene.
3. **Phase 2 (~1 day).** Tune threshold + hysteresis (k-consecutive-fires, mirroring
   auto-stop-delay) on a DEV episode set drawn fresh from val -- never sample101/400 --
   maximizing SR directly. One confirming sample400 run `--stop-head` vs the existing
   `--auto-stop` ck791 result. **Gate: within ~2 pp of auto-stop SR.** Then hand the
   checkpoint+threshold to the navverse harness (it already supports
   `longnav_stop_prob_threshold`).
4. **Refresh rule (approximates joint training for free).** Stop credit is one-step: the
   decision's consequence is immediate. So on-policy adaptation needs no policy-gradient
   machinery -- periodically retrain the head on the newest rollouts (DAgger-style). This
   captures most of what joint training would buy, at probe cost.

Pros: cheap; provably cannot degrade the policy (detached); zero rollout cost; reversible;
feeds the critic program. Cons: head lags policy drift (mitigated by the refresh rule);
cannot condition on the policy's *intent*, only its state representation.

## The discrete precedent, and why its elegance does not port

In the discrete stack, stop is literally a regular action in the softmax -- credited by
the same policy gradient as motion, no extra machinery. Elegant, and it works: rl248's
stop behavior is RL-learned end to end. But two observations bound its performance:
(1) even so, rl248 leaves a **+7.9 pp** gap between reaching and stopping on navverse --
RL-action stop is decent, not optimized (disc-original, whose stop was never RL-tuned,
leaves +16.2 pp); (2) the elegance rides on the discrete action space -- stop competes in
the same distribution as motion. Our action is a 20-pose chunk through a frozen flow
head; there is no natural slot for a stop symbol, so "stop as an action" here means
grafting a separate binary head into the RL loss anyway -- i.e., Track B's costs without
the discrete case's structural elegance. This sets Track A's quantitative target: an
optimized classifier should beat the RL-action stop gap, i.e. land **within <8 pp of
oracle, ideally ~2 pp** (the auto-stop parity gate below).

## Track B -- jointly RL-trained stop head (DEFER)

Stop as a policy action with terminal success credit requires: env stops terminating at
reach, success bonus reintroduced, episode semantics change. Pros: calibrated to the
policy's own distribution and optimizing SR directly; could learn risk-sensitive stopping.
Cons, decisive today: (1) stop credit is sparse/terminal -- the exact channel this
program has measured weakest, and the reward-structure change (success_reward != 0)
re-opens questions the reward-horizon analysis settled; (2) changing episode semantics
breaks comparability of every running series and instrument mid-program; (3) classic
failure modes (never-stop or stop-collapse) need a supervised anchor to prevent -- which
is Track A anyway; (4) costs the training fleet. **Revisit only if Track A plateaus >3 pp
below auto-stop**, and then as a supervised-anchored auxiliary inside the next planned run
(curriculum or critic), never standalone.

## Consolidation with the critic track (2026-08-18 revision), and revised sequencing

The stop head and the critic are one track approached from two ends: under the telescoped
reward, V^pi(s) ~= d(s) - E[d_final | s, pi]. The stop head's distance regression IS the
first (oracle-geometry) term; everything the critic wants beyond the distance kernel IS
the second (policy) term. Build ONE state probe with two outputs, (log-d_hat, R_hat):
stop = threshold on the first; the candidate advantage baseline = the second; the
residual R_t - f(d_hat) is exactly the V*-vs-V^pi gap, and its explained variance from h
answers the critic program's central unknown ("can the backbone see where the policy will
fail"). Shared data pipeline (sidecar + rollout mining supplies BOTH labels per frame),
shared k-fold validation harness. Distinctions kept: d(s) is policy-independent (lazy
refresh, eval-gated); V^pi is policy-conditional (staleness = bias; strict offline gate,
lambda=1 entry).

**REVISED SEQUENCING: the training lands inside the critic-compatible SFT co-train, not
as a standalone probe run.** Two reasons beyond GPU cost: (1) the detached-value-probe
route was already tried and shelved (`critic-is-last-resort`: probe detached first) --
frozen-h probing risks failing for representation reasons, and the fix in that case is
letting the auxiliary gradients shape the backbone, which is precisely what co-training
does; (2) the backbone forwards are the expensive part, and the next SFT co-train run
(cotrain-v2 on the long-goal PointNav corpus, already queued) pays them anyway --
auxiliary (d, R) heads ride at ~zero marginal cost, with `stop_grad=False` arms so the
backbone becomes proximity- and outcome-aware rather than merely probed.

What proceeds NOW (cheap, CPU, no fleet):
* Phase 0 prize quantification from existing s400 logs (no GPU);
* the full distance-sidecar annotation run over the demo corpora (CPU-parallel);
* the rollout-mining pipeline (mp4 frame decode at policy-step indices + d_t/R_t label
  join) -- so the co-train run receives ready-made label columns, not a data project.

What waits for the co-train: trunk/head training, threshold calibration, the sample400
`--stop-head` confirmation, the R_hat-vs-kernel gate. Until then, published numbers keep
the explicit oracle-stop asterisk, on our benchmarks and navverse alike.

## Constraints honored

* Rollout speed: h-probe adds an MLP on an existing activation -- microseconds against a
  ~400 ms policy step. The separate-vision-encoder fallback (if h proves insufficient:
  frozen SigLIP + MLP) costs ~5-10 ms/step, still <3% -- but h is tried first.
* Compute: everything runs on GPUs 5-7 between evals; the gamma097 fleet is untouched.
* Comparability: all existing numbers stay valid -- `--auto-stop` results are labeled
  oracle-stop; `--stop-head` results are a new, honestly-labeled column.

## Build status (2026-08-18)

Built, tested, committed:
* habitat e457430 -- `--distance-sidecar` on the chunk builder: episode rows gain
  `obs_distances` (anchor-frame geodesic, NaN-preserving), hard error on a sidecar miss,
  omission keeps today's exact schema. 4 new tests.
* spatial 1cff19e -- `longnav/utils/state_probe.py`: `LogDistanceHead` (64 uniform
  log1p bins over [0,40 m], HL-Gauss, `p_within(logits, r)` as the calibrated stop rule,
  meter-space expectation), `ValueDistHead` (linear support [-8,24] per the audit),
  `StateProbe` (shared hidden, per-head weights, straight-through grad_scale),
  `distance_return_targets` (returns from the d-series under the RL reward; NaN-safe),
  save/load (`state_probe.pt` + config json). 11 tests.
  `format_action_chunk_dataset.py`: `--distance-column` + `--return-gamma` (required
  together, no gamma default) emit `distance_targets` / `return_targets` /
  per-row `return_gamma` at format time -- returns are corpus columns, never derived in
  the collator (decision 2026-08-18).

Remaining for the derived trainer (`train_flow_matching_sft_value.py`, to be copied from
`train_flow_matching_sft.py` per the preserve-the-original rule):
1. Collator: slice `distance_targets`/`return_targets` with the SAME window the collator
   already applies to action targets (the `start`/`n_turns` variables at the
   `stop_targets` emission site are exactly the needed ones).
2. Probe-hidden gather -- carries an OPEN DESIGN DECISION: train the probe on the pooled
   sandwich context (offset 0, trivially available at the `TurnEncoding.pooled` seam,
   maximally shapes h) or at the RL contract offset -2 (requires a dense->sparse index
   remap via `turn_vectors.to_sparse_indices` for positions outside the turn span; matches
   `ValueHeadConfig.readout_offset` so the RL handoff reads the SAME activation the head
   was trained on). The offset is checkpoint contract either way; whichever is chosen, the
   RL config must agree.
3. Save hook: `save_state_probe` next to each checkpoint; RL-side load path in
   `setup_training` (currently always fresh-inits).

## Derived trainer: BUILT AND SMOKE-TESTED (2026-08-18, spatial 1ce91f2)

`data_scripts/train_flow_matching_sft_value.py` -- verbatim copy of the original +
fenced `# --- PROBE ---` additions. Verified end to end on a real mini-pipeline
(2-scene corpus slice -> builder --distance-sidecar -> format --return-gamma 0.97 ->
4 training steps on GPU):

* probe CE losses start at their near-uniform theoretical values (distance ln64=4.16,
  value ln51=3.93) and DECLINE (4.21->4.04, 3.90->3.47 in 4 steps); probe grad norm ~0.9;
  `state_probe.pt`+config written into every checkpoint and `final/`.
* token-identity pin HELD at offset -2 across all turns -- the offset lands on a
  constant text token in the real template (closes the audit's open item).
* probe-off parity: the original trainer is NOT run-to-run deterministic (same seed,
  losses differ across identical invocations -- flash-attn class nondeterminism), so
  parity is guaranteed structurally: probe-off delegates to the parent forward
  immediately, uses the original collator class, and the original trainer class.
* three bugs found and pinned by tests: (1) pre-parser flags must be copied onto the
  merged namespace (the original's own warning); (2) `nn.Module` class-attribute
  default shadowed the registered probe submodule -> silent zero-grad co-training
  (test: test_no_state_probe_class_attribute_shadow); (3) `find_turn_spans` returns
  per-batch-row nested lists.
* suite state: 5 spatial test failures reproduce identically at the pre-change commit
  (pre-existing golden-fixture drift, forward/component + model_metrics families);
  my earlier audit fix broke `_compute_value` as a subclass extension point and is now
  restored with the refusal kept for the base case (trajectory-shapes tests pass).

Launch-blocking remainder: NONE on the trainer. The full-corpus data build (builder
--distance-sidecar + format --return-gamma over v2_25hz, CPU hours) and the three
launch decisions (mixture w/ or w/o the PointNav corpus, gamma column(s), fleet order)
are what stand between here and the co-train run.

## Status 2026-08-18 — probe integration proven; first counterfactual read (ck-200)

The full chain is validated end to end: SFT co-train (cotrain-v4, ddp4 GPUs 0-3,
meters-scale metrics per dataset) → checkpoint persists the worker-frame readout
convention (`worker_value_readout_offset`, `probe_token_id`; the span-shift
off-by-one this caught is documented in the parity script) →
`tests/parity_probe_paths.py` (token pin + packed-vs-incremental seams ≤0.05 m /
0.03) → 20-cycle mini RL run (`flow_sde_probe_mini`, GPU 4) with the probe frozen
and observational. Two integration bugs found and fixed on the way: peft
`modules_to_save` wrapping the frozen adapter (cache landed on peft's deep copy;
now gated on a trainable critic) and `StateProbeConfig.from_dict` on the new pins.

**Counterfactual verdict at checkpoint-200 (20 cycles, paired on the same
buffer):** adv-variance kernel 4.17 ≤ naive 4.29 < probe 4.98; corr(v̂, G) 0.071;
probe value MSE 18.8 vs kernel 4.2. The step-200 probe is not a baseline — as
pre-registered, this is the lower bound, not the answer. Distance head already
discriminates on-policy: p_stop(near) 0.16-0.39 vs p_stop(far) 0.016-0.052
(~10x) with regression-to-the-mean bias (far episodes under-predicted ~7 m).

Next read: `flow_sde_probe_mini_ck1000` auto-launches when checkpoint-1000 lands
(same instrument, deeper probe). Decision rule: probe adv-var crossing below the
kernel's makes the value-baseline case on-policy; still above at ck-2000+ routes
through calibration on the 294k mined on-policy frames
(`data/mined_rollout_frames/gamma097_g097`) before any ceiling conclusion.

## 2026-08-19 — the on-policy critic exists: post-hoc head refit beats the kernel

> **INVALID (retracted 2026-08-23).** Every figure in this section is computed from
> hidden states cached over HUD-bearing mined frames. The "on-policy" arms were fitting a
> printed number. The 23%-below-kernel claim, the three-arm table, and the post-hoc refit
> recipe are all withdrawn; the saved heads have been deleted.

Offline instrument (`data_scripts/eval_value_heads_offline.py`, pinned 64-episode
on-policy eval set + 64-episode expert set, per-step series + readout hiddens
cached as npz): the co-trained value head is representation-limited at ck-200 but
the REPRESENTATION keeps improving (achievable expert corr 0.404@ck200 ->
0.507@ck1000 by fresh-head refit) while the co-trained head lags (0.432 -> 0.391).
grad_scale does not throttle head gradients (straight-through), so raising it was
rejected: the backbone side already works at 0.1.

Scaled refit (512 disjoint on-policy episodes, 26,362 cached hiddens at ck-1000;
three arms, inner-val early stopping, episode-bootstrap CIs on the held-out
eval-64):

| arm            | eval64 corr           | eval64 Var(G-v)        | expert-test corr |
|----------------|-----------------------|------------------------|------------------|
| expert-only    | 0.120 [0.030, 0.212]  | 8.65                   | 0.460            |
| on-policy-only | 0.555 [0.456, 0.643]  | 5.08 [3.80, 6.53]      | -0.012           |
| joint          | 0.556 [0.454, 0.652]  | 5.12 [3.79, 6.58]      | 0.391            |

Reference on the same set: kernel 6.63, naive 7.05. The JOINT head cuts advantage
variance 23% below the time kernel (28% below naive) while keeping expert-side
fit -- no tradeoff. Heads saved: `dump/eval_system/value_offline/
refit_value_head_{expert_only,on_policy_only,joint}.pt`.

Recipe now: co-train SFT for the representation; fit the head POST-HOC on cached
hiddens (expert + mined on-policy); refresh per checkpoint in minutes. Remaining
to package: distance-head refit (stop side) the same way, then a composed
state_probe dir the RL `state_probe:` config can point at.

## Operational note: offline passes are sharded (2026-08-21)

Every offline GPU pass in this workflow -- `data_scripts/eval_value_heads_offline.py`
(hidden caching, checkpoint scoring, distance-band analysis) -- runs as **4+ parallel
shards**, one process per GPU, via `split -n l/4` on the episode-paths file. Two
incidents in one session made this a rule rather than a preference: a 5-checkpoint
scoring loop run sequentially while 4 GPUs idled, and a 60-episode band analysis run
on ONE GPU while both SFT arms saturated the fleet (~4x slower than the earlier
4-shard passes, which had also had an idle fleet).

When training owns the fleet, spread the shards over different GPU *pairs* (0,2,4,6)
so no single DDP rank absorbs the entire slowdown; each GPU normally has 50+ GB free.
The ~2 min model reload per shard is dwarfed by the parallel win, so never serialize
to avoid reloads.
