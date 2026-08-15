# The in-training eval set, and its measured SFT baseline

An RL run's `eval/*` numbers mean nothing on their own: the set is drawn from the training
pool at startup, so a change to the pool, the category filter, `eval_set_size` or
`eval_seed` silently produces a *different* set, and every historical number on it becomes
incomparable without saying so. This file records the sets we have measured a baseline on,
so a curve can be judged against a distribution instead of against its own cycle-0 point --
which is one draw from a distribution whose sd is ~0.045.

## How the set is chosen (`train_loop.build_eval_partition`)

`sorted(sims[0].list_episode_uids())`, then `np.random.default_rng(eval_seed)` draws
`eval_set_size` uids without replacement, once, at startup; the same set runs every
`eval_every` cycles, partitioned round-robin across sims. `list_episode_uids` returns the
pool **after** `exclude_categories`, so the filter applies to the eval set too.

Two consequences worth stating out loud:

* the set is drawn FROM the training pool, so its episodes are trained on -- about once per
  `len(pool)/n_rollout` cycles. It is **quasi**-held-out, not held out. On the 8000-episode
  pool that is ~once per 500 cycles; on a 24-episode pool it is *every other cycle*, i.e.
  the "eval" there is a training-set metric.
* screening removes some episodes at serve time, so the effective n is below
  `eval_set_size` and should be quoted as the servable count.

## `fullpool_evalset` (2026-08-15)

**Definition.** `v1_train100x80` minus `exclude_categories: [plant]` (7429 of 8000),
`eval_set_size: 32`, `eval_seed: 0`. **32 drawn, 26 servable, 23 scenes.**

Uid lists (in `dump/`, which is gitignored -- these paths are the record):

    dump/eval_system/episode_sets/fullpool_evalset_drawn32.txt
    dump/eval_system/episode_sets/fullpool_evalset_servable26.txt
    dump/eval_system/episode_sets/fullpool_evalset_baseline.json

**SFT baseline** -- cotrain-v3 ck12000, which is also the cycle-0 model of any merged-base
run, so a run's own first eval point should land in the ODE row:

| arm | passes | success | oSPL | episodes flipping |
|---|---|---|---|---|
| ODE (matches `eval_ode: true`) | 8 | **0.6635 ± 0.0448**, range [0.615, 0.731] | **0.3823 ± 0.0516** | 10/26 |
| a = 0.9 (the training rollout distribution) | 6 | 0.6106 ± 0.0433, range [0.538, 0.692] | 0.3405 ± 0.0419 | **17/26** |

Measured by `scripts/probe_credited_channel.py` with `PROBE_M=8 PROBE_A=0.9` on the
`flow_sde_freezehead_a09_fullpool` config (run `sft_baseline_fullpool_evalset`), which
reuses `build_eval_partition` + `run_eval_cycle` and therefore reproduces the TRAINING
eval's semantics exactly: ODE decode, env-side termination, `success_distance` 0.4,
`max_steps` 175, dt 0.04, gap 10, navmesh dataset. **The offline harness is a different
measurement** (`success_distance` 1.0, `--auto-stop`); its numbers are not comparable to
these.

**How to use it.** 10 of 26 episodes flip between identical ODE passes, so a single eval
point carries sd ~0.045 in success and ~0.05 in oSPL. Judge a run against the whole
distribution over its whole series. On 2026-08-15 the same run was read as "declining" and
then as "recovered" within an hour, both times from a four-point window; neither reading
survived contact with the eight-pass baseline.

**Note on the a=0.9 row.** On THIS pool the credited noise nearly doubles the number of
outcome-uncertain episodes (17/26 vs 10/26) and raises per-episode outcome variance 56%
over the ODE floor. On the 24-episode/4-scene pool the same statistic was flat -- see the
2026-08-15 correction in `FLOW_SDE_RL.md`. Baselines and channel diagnostics are per-pool.

## Holding the set out of training (implemented 2026-08-15)

Two settings, and a run needs BOTH -- one without the other silently gives you the old
quasi-held-out arrangement under a new name.

    sim.train_uids       restrict the TRAINING stream to these uids (list, or a file path)
    task.eval_uids_file  pin the eval set to these uids verbatim, instead of drawing it

`train_uids` filters in `_reshuffle` and **only when no explicit shard is assigned**. That
is what keeps the two compatible: training runs on the trivial (`None`) shard, while
`run_eval_cycle` assigns explicit eval shards and restores `assign_shard(None)` afterwards,
so eval still reaches episodes training never serves. The driver **refuses to start** if
`train_uids` is set with `task.shard_size > 0`, because an explicitly sharded run would
bypass the filter and quietly train on the whole pool.

`eval_uids_file` makes `build_eval_partition` use the file's uids in order and raise on any
uid absent from the pool. The pool must therefore still CONTAIN the eval episodes -- which
is why the eval set is held out by restricting the training stream rather than by shrinking
the dataset. Uids are `scene:episode_id#occurrence` and the occurrence counter is assigned
per shard FILE, so a derived subset renumbers them and invalidates every pinned uid.

First run to use them: `flow_sde_freezehead_a09_held128` (128 balanced training episodes,
this file's pinned 26 as a genuinely held-out eval). Pure-logic tests, no simulator, in
`tests/test_held_out_eval.py`.

## `trainset128` (2026-08-15)

128 training uids drawn from `v1_train100x80` minus `plant` minus the 32 drawn eval uids
(7397 remaining), by `dump/eval_system/episode_sets/mk_trainset128.py` (seed 0,
deterministic; the file records the draw's parameters). **Stratified, not uniform** -- at
n=128 chance alone swings a category between 15 and 35, and a training pool whose
composition differs from the eval set's turns a delta into an artifact. Cells are goal
category x geodesic tercile; within a cell the least-picked scene wins.

| | trainset128 | pinned eval26 |
|---|---|---|
| categories | chair 27 / bed 26 / toilet 26 / sofa 25 / tv_monitor 24 | tv_monitor 10 / bed 6 / sofa 5 / chair 4 / toilet 1 |
| scenes | all 80 (32 x1, 48 x2) | 23 |
| geodesic | mean 7.49 m, 42/43/43 across terciles cut at 4.85 / 8.27 m | mean 8.10 m |

The eval set is skewed because it was a uniform draw of 32 and is now pinned to stay
comparable with the 8-pass baseline above; that is the cost of pinning, and it is the
reason the training set is balanced instead. Screening removes ~19% at serve time, so
expect ~104 of the 128 to be served -- the first cycle's `[env] train_uids:` line reports
the real number.
