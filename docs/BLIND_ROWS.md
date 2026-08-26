# Blind rows: what they are, what causes them, what we do about them

A **blind row** is a policy step whose `distance_to_goal` is non-finite. The
reward's progress term is guarded by `isfinite`, so a blind step pays
`-slack_penalty` and nothing else; the geodesic is held at its last finite value
and the *recovery* step is paid the whole accumulated delta (verified: 5.52 m ->
9 blind steps -> 5.29 m paid 0.238 at recovery). **No reward vanishes.**

Measured 2026-08-25 over the full training corpus (74,692 RL episodes, 6.89 M
steps, all on `navmesh: dataset` -- verified across 45 saved run configs, no run
has ever used the robot mesh).

## 1. Rates and distribution

| | value |
| --- | --- |
| blind steps | 75,946 (**1.10%** of all steps) |
| episodes with >= 1 blind row | 3,430 (**4.59%**) |
| of those, never recover (episode ends blind) | 2,841 (**82.8%**) |

Blind rows per affected episode: median 10, p75 25, p90 39, p95 142, max 173.
As a share of the episode: median 12%, p75 41%, p90 69%.

Histogram over all episodes: 0 -> 95.41% | 1-2 -> 1.87% | 3-5 -> 0.23% |
6-10 -> 0.24% | 11-20 -> 0.40% | 21-50 -> 1.47% | 51-100 -> 0.08% | 101+ ->
0.30%. The 21-50 spike is `reward_lost_steps = 25` ending doomed episodes; the
1-2 bucket is mostly terminal blindness at the very end of an episode.
(Chart: `dump/rtc_sft/blind_rows.png`; raw per-episode data:
`dump/rtc_sft/blind_row_stats.json`.)

**Scene-concentrated**, which is the important part: 62 of 83 scenes (>= 50
episodes) see some blindness, but the median scene sits at **0.7%** while the
worst run 43% / 28% / 28% / 26% (`77mMEyxhs44`, `PPTLa8SkUfo`, `xWvSkKiWQpC`,
`Jfyvj3xn2aJ`). This is a property of each scene's navmesh topology (section 3),
not random noise -- so untreated it is a scene-correlated reward bias, not
uniform variance.

## 2. Escapes and blindness are nearly the same population

Joining 37,848 episodes (blind pattern x outcome):

* **99.1% of escaped episodes are blind** (779 of 786);
* 45.6% of blind episodes are escapes, against **0.0%** of clean episodes;
* escape blindness is a *short tail*: median **2** blind rows (p90 3), 4% of the
  episode -- against median 25 for non-escape blind episodes.

The robot goes blind **because** it is falling out of the world. That is a
correctly-labelled terminal event carrying `escape_penalty`, not a measurement
gap -- hence the carve-out in section 4.

## 3. The cause is ISLAND CROSSING, not snap failure

Traced end to end on `77mMEyxhs44:18#3` (rejected sample #1, 95 steps, blind
from step 71). The scene's dataset navmesh has **3 disconnected islands**.

| step | planar (x, y) | snap distance | island | `is_navigable` | distance |
| --- | --- | --- | --- | --- | --- |
| 67-69 | 5.9 -> 7.4 | 0.06 m | **1** | yes | 5.13 -> 4.86 |
| 70 | 8.106 | 0.15 m | 1 | **no** | 5.33 |
| **71** | 8.818 | 0.06 m | **2** | yes | **blind** |
| 72+ | 9.5 -> 10.1 | 0.06 m | 2 | yes | blind |

`snap_point` **succeeds every time** (0.06 m -- textbook). What changes is which
island it resolves to. `geodesic(island 1 <-> island 2) = inf` in both
directions; the episode starts on island 1 with the goal reachable at 6.09 m,
and island 2 is a 3.9 m-radius pocket holding ~18% of the mesh. Step 70 is the
tell: already off-mesh (`is_navigable` False, snap 0.15 m away) -- the robot was
standing *in the gap* -- and one 0.71 m step later it was fully on island 2.

**The value is `inf`, not NaN.** NaN arises only with no pathfinder or an empty
goal set; `_finite()` collapses both to the same blind state, which is why it
reads as NaN downstream.

Mechanism, stated plainly: **the physics robot can cross island boundaries that
the point-agent navmesh treats as disconnected** -- a door sill it drives over, a
threshold the mesh generator left as a gap. The measurement is not broken; the
episode is genuinely unmeasurable from that moment on.

### Snapping code: audited, no bug found

* `LiveHeightNavAdapter._height()` correctly adds `joint_z` to the spawn height,
  and the env hardcodes `live_height=True`, so queries follow the robot
  vertically. (Its base `NavSimAdapter._height()` does **not** add `joint_z`;
  using that class directly would freeze queries at spawn height -- a real bug in
  a multi-storey scene. Nothing does.)
* `_NavAdapterBase.get_agent_state` falls back to the **raw, unsnapped** position
  when a snap fails, so the geodesic then fails from an off-mesh point. A silent
  degradation path by design -- not what happens here.
* `snap_point` is island-unconstrained (`island_index = -1`), which is what lets
  a boundary query resolve onto another island. Here it produces an honest `inf`
  rather than a wrong-but-finite number.

## 4. What we do: worker-side rejection (2026-08-25)

`rollout.reject_blind_episodes` (default **off** -- pure extension). When on, an
episode with any blind row is DISCARDED and the collector runs a replacement, so
no blind row ever reaches collation and the caller still gets exactly
`n_rollout` clean episodes. No padding, no fault tolerance: a crashed episode
still takes the loud `EPISODE_FAILED` path.

* **Worker decides, collector acts.** `EpisodeRolloutMixin._episode_rejected`
  owns the criterion (reads the env's cumulative `blind_total`);
  `collect_rollouts(respect_rejections=...)` decides per call. Training passes
  True, `run_eval_cycle` passes **False** -- a pinned eval set with a variable
  denominator stops being comparable, and blind episodes are part of what is
  being measured.
* **Escape carve-out, `reject_blind_keep_physical_terminal`, default on.**
  Rejecting physically-terminated episodes would delete ~99% of the escape
  population and with it every `escape_penalty` -- silently, while the knob still
  sat in the YAML. See section 2.
* **Isolated logs.** A rejected episode's text logs (and its video) go to
  `<run>/rollout/_rejected/`, and its wandb scalars route under `rejected/`. A
  `rejected: true` field in the normal stream would be correct and would still be
  forgotten by the next analysis; a separate root cannot be mixed in by omission.
* **Retry cap** (`max_dispatch_factor`, default 3x) raises with the observed
  rejection rate rather than spinning on a pathological shard.
* Observed cost in `flow_sde_rtc_a09_held128_r3`: ~1 rejection per cycle
  (~7.7% of dispatches in the cycles where it fires), in exactly the scenes
  section 1 predicts.

### Known trade-off

Rejection is a survivorship filter on *behaviour*, not only on geometry: every
sampled rejected episode was a failure, so the filter systematically removes
failures and the **training** success rate drifts up as an artifact. The eval
curve never rejects and remains the honest reference. Watch
`rollout/rejected_rate` -- if the policy learns to earn rejection, it climbs.

## 5. Not done, deliberately

* **Island-aware screening at reset** -- reject an episode whose reachable region
  neighbours islands the robot can physically cross into.
* **Per-step `island_id` logging** -- would make this failure mode a labelled
  event rather than "the distance went blind".
* **Truncate-at-first-blind** instead of discarding: keeps the ~88% of steps
  before the blind window, but needs `bootstrap_truncated` + a value head to
  avoid under-crediting the cut tail. Attractive once a critic is standard.
