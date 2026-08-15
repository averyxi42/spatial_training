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

## Fixing the set at training time (not implemented)

The env serves episodes from the whole pool, including the eval uids. Excluding
`fullpool_evalset_servable26.txt` from the training stream -- a filter in
`ContinuousObjectNavEnvActor._load_episodes`, alongside the existing `exclude_categories`
pass -- would make the in-training curve a clean generalisation signal rather than a
quasi-held-out one, at the cost of 26 episodes of training data out of 7429. The uid files
above exist so that change has something stable to reference.
