# Live rollout cadence: `--gap` must match the corpus stride

Status: **finding, 2026-08-30.** Retracts the behavioural reading of every `dump/code_viz/live*`
rollout of the code-conditioned head, and the origin of `ACTION_HISTORY.md`.

## What happened

All qualitative rollouts of `run_code_v4_mlp_warm` (`live800` … `live12000`, 2026-08-28..30),
and the first vanilla control, were launched through
`habitat_physical_nav_rtc/scripts/run_code_modes_viz.sh` at `--gap 5 --dt 0.05`. The
corpora the checkpoints trained on (`v2_25hz_obs2.5hz`, `pointnav_long_full`) are
`obs_stride_frames 10, dt 0.04`. The harness's own `SAMPLE101_EVALS.md` calls this "Silently
wrong #1": a wrong gap executes a fraction of each chunk and discards the rest, at the wrong
replan rate, and "presents as a slow, weak policy". It does not error.

Confirm from a run's own record, never from its name: `episodes.partial.jsonl` →
`ticks / steps` must be 10 and `motion.dt_s` must be 0.04.

## What it did to the conclusions

Same checkpoint (`ck12000`), same episode (s175, start 14.26 m), 175 steps:

| condition | stationary share | turn sign-flip | lag-1 autocorr | drive_frac | path |
|---|---|---|---|---|---|
| corpus expert | 0.46 | 0.21 | 0.70 | — | — |
| gap 5 / dt 0.05, `sample` | 0.78 | 0.45 | 0.24 | 0.21 | 7 m |
| gap 5 / dt 0.05, `argmax` | 0.95 | — | — | 0.06 | 2 m |
| **gap 10 / dt 0.04, `sample`** | **0.45** | **0.25** | **0.51** | 0.48 | 44 m |
| **gap 10 / dt 0.04, `argmax`** | **0.43** | **0.19** | **0.67** | 0.55 | 54 m |
| gap 10 / dt 0.04, vanilla `run_cotrain_v3_nopose_mix` | 0.67 | 0.33 | 0.51 | 0.21 | 17 m |

The "spinning in place", the argmax "stationary collapse", and the `mean`-decode paralysis
were all observed under gap 5. At the trained cadence the code head's motion statistics are
expert-like under both decodes, and the vanilla SFT checkpoint on the same episode is more
stationary with 10x the collision rate. None of the three succeeds on this one episode;
these are single-episode existence proofs, not a ranking. A multi-episode eval at gap 10 is
what would rank the heads.

PointNav (seed-0 goal chain, TEEsavR23oF): at gap 10 both the code head and vanilla reach
goal 0 (SPL 0.90 / 0.89); both fail goal 1 (code head to 0.59 m of a 0.5 m threshold, vanilla
never below 2.15 m, stationary 90% of ticks). The second-goal failure is shared and is not a
code-head property.

## What stays retracted

* `ACTION_HISTORY.md` "Origin" (rotation, "34 m buys 2.3 m" = `live1400_s350`, a gap-5 run).
* Every behavioural statement about `argmax` vs `sample` made from `live*` runs.
* `code_mode="mean"` was implemented and is the one decode that is bad by construction
  (mode-averaging paralyses); it also had the best open-loop pose RMSE, so open-loop RMSE
  is not a decode-selection metric.

## What was changed

* `run_code_modes_viz.sh` now derives `--gap` from the checkpoint's `fm_n_ticks`
  (10 → 5, 20 → 10, 30 → 25) and pins `--dt 0.04`; the tag carries `_g<gap>`.
* Full record: `dump/audits/code_sft_conditioning_audit_2026-08-30.md` §8–§9,
  `dump/code_viz/g10_*`.
