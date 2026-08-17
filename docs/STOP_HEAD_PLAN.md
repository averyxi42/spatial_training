# Stop head: preliminary plan

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
