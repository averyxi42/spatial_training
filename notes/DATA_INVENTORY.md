# Data inventory — continuous navigation track

*Written 2026-08-23 from a read-only survey of `/Projects/data` (the `data/` symlink in this
repo) plus the adjacent data locations it depends on. Nothing was modified. Directory trees
were walked at most 2–3 levels deep; per-episode image directories were sampled, not
enumerated.*

---

## 1. The shape of it, in one paragraph

There are **two unrelated data axes** and confusing them is the main hazard:

| axis | what it is | where it lives | consumed by |
|---|---|---|---|
| **Demonstration corpora** | recorded expert trajectories, packed into HuggingFace Arrow datasets | `/Projects/data/<corpus>/{chunks,formatted*}` | the **SFT** trainers in `data_scripts/train_*_sft*.py` |
| **Habitat episode datasets** | goal/start specs for the simulator; no pixels | `/Projects/data/datasets/objectnav/hm3d/*` (an NFS symlink) | the **RL** trainer / eval (`sim.episodes:` in every experiment yaml) |

Everything under `pointnav_*`, `v2_25hz*`, `continuous_sft*` and `annotations/` is the first
axis. `datasets/`, `scene_datasets/`, `svln_scene_datasets/`, `StreamVLN/`, `R2R_VLNCE_v1-3.zip`
are the second axis or older/unrelated work.

Within the first axis every corpus flows through the same three stages:

```
 recordings                chunks/                    formatted*/
 (jpeg frames +   ──────►  one row per episode  ─────► same rows + a `messages`
  metadata jsonl)          images=paths,               conversation column
                           action_chunks=(n_obs,L,3)   (+ optional target columns)
   generator:              builder:                    formatter:
   habitat_physical_nav    habitat_physical_nav/       data_scripts/
   scripts/generate_*      scripts/build_action_       format_action_chunk_dataset.py
                           chunk_episode_dataset.py    data_scripts/add_distance_targets.py
                                                       data_scripts/add_stop_targets.py
```

`formatted*` never re-reads pixels — it stores **image paths**. That is why the raw recording
trees must not be moved: every corpus references them by absolute path.

---

## 2. Top-level folders in `/Projects/data`

Dates below are directory mtimes (build dates). Sizes for Arrow datasets are exact
(`data-*.arrow` bytes); sizes for jpeg trees are estimates from a sampled episode
(~84 MB per 2000-frame episode).

### 2.1 ObjectNav demonstration lineage (MP3D scenes)

| folder | built | what it is | size |
|---|---|---|---|
| `continuous_sft/` + `continuous_sft_formatted/` | 2026-07-30 | **Oldest, superseded.** Built from the 20 fps recordings (`output/`), `obs_hz 4.0`, chunk len 10. Column set is the pre-v2 one (no `obs_goals`, no `segment_indices`, no `recording_schema_version`). | 2.9 GB / 3.6 GB |
| `v2_25hz/` | 2026-08-07 | The 25 fps rebuild. `chunks/` = 39,061 train + 695 val episode rows; `formatted/` and `formatted_pose/` are two formatting passes over it (the `_pose` one injects `<pose>` / `obs_poses`). | 1.9 GB chunks + 2.4/2.4 GB formats |
| `v2_25hz_obs1hz/` | 2026-08-07 | Same recordings re-chunked at **1 Hz observations** (stride 25, chunk len 30). The slow end of the observation-rate sweep. Only `chunks/` + `formatted_pose/`. | 0.82 GB chunks + 0.93 GB formatted |
| `v2_25hz_obs2.5hz/` | 2026-08-07 → **2026-08-21** | **The live ObjectNav corpus.** 2.5 Hz (stride 10, chunk len 20). Six formatted variants, see §3. | 1.5 GB chunks + ~1.7 GB × 6 formats |

All three `v2_25hz*` share one recording source and were produced by
`data_scripts/build_slow_corpora.sh` (which carries the rate-sweep design table in its header:
the hard constraint is `chunk_duration >= obs_interval`, and 1 Hz breaks `overlap_fraction`
parity deliberately to keep the action dimension bounded).

**Recording source:** `/Projects/DEPRECATED_DO_NOT_USE_continuous_demos/output_25hz/`
— 56 **MP3D** scenes, 39,756 episodes. The *code* fork there is retired (merged into
`/Projects/habitat_physical_nav`, commit `5f56a9a`), but the *data* is still live and every
`chunk_config.json` in the `v2_25hz*` family points at it by absolute path. `output/` next to it
is the older 20 fps recording behind `continuous_sft`.

### 2.2 PointNav demonstration lineage (HM3D scenes)

Raw jpeg corpora (`corpus_params.json` + `metadata/*.jsonl` + `images/<scene>/<episode>/frame_*.jpeg`):

| folder | built | scenes | episodes | est. size | notes |
|---|---|---|---|---|---|
| `pointnav_hm3d/` | 2026-08-07 | 800 (762 with images) | 11,668 | ~1.0 TB | First HM3D PointNav corpus. Goals 1.5–8 m, unstratified. Has `batch.log` + `worker_logs/` + `failed_episodes.log`. |
| `pointnav_sweep/` | 2026-08-08 | 4 per condition | 26–29 per condition | ~12 GB | **A 5-condition pilot, not training data.** A_1p5_8 / B_2_30 / C_2_15 / D_2_30_strat / E_2_15_strat = goal geodesic range × `--stratified-pool 24`. Decided the long-goal sampling; the measured table is in `habitat_physical_nav/docs/POINTNAV_GENERATION.md` (D wins: median segment geodesic 10.59 m vs 4.03 m for production). |
| `pointnav_hm3d_long/` | 2026-08-09 | 800 (741 with images) | 7,998 | ~670 GB | The corpus that pilot chose: goals 2–30 m, `--stratified-pool 24`. **This is the live PointNav source.** |
| `pointnav_long_pilot_src/` | 2026-08-09 | 481 | 4,727 | 0 (symlink) | Not a corpus — a *view*: `images` symlinks into `pointnav_hm3d_long`, `metadata/` is the partial set that existed while generation was still running. Exists only so a chunk build could start early. |

Packed corpora built from those:

| folder | built | source | rows (train/val) | notes |
|---|---|---|---|---|
| `pointnav_hm3d_pilot/` | 2026-08-07 | `pointnav_hm3d` | 2,290 / 31 | 5 Hz first pass. `formatted/` only. |
| `pointnav_hm3d_2p5hz/` | 2026-08-07 | `pointnav_hm3d` | 3,008 / 59 | `--boundary-policy drop` — **known-bad**: chunks crossing a goal change were discarded, so the corpus contained essentially no arrivals (6 of 4,780 observations within 1 m). Trained `pointnav-v1`, which unlearned stopping. |
| `pointnav_hm3d_2p5hz_clamp/` | 2026-08-07 | `pointnav_hm3d` | 5,062 / 41 | `--boundary-policy clamp` — **also bad**: restored arrivals by holding the last pose, fabricating dead time in 11.2% of ticks. Trained `pointnav-v2`, which learned to stand still. |
| `pointnav_hm3d_2p5hz_keep/` | 2026-08-07 | `pointnav_hm3d` | 6,158 / 46 | `--boundary-policy keep` — **the one that worked**. Trained `pointnav-v3` and then `cotrain-v1`. Superseded by the long-goal corpora but historically the good short-goal one. |
| `pointnav_hm3d_full/` | 2026-08-07 | `pointnav_hm3d` | 11,512 / 156 | Full short-goal corpus. **Not referenced by any launch script found** — built and apparently passed over in favour of the long-goal rebuild two days later. |
| `pointnav_long_pilot/` | 2026-08-09 | `pointnav_long_pilot_src` | 4,695 / 32 | The early-start build. Trained `cotrain-v2` initially. |
| `pointnav_long_full/` | 2026-08-09 → **2026-08-21** | `pointnav_hm3d_long` | 7,971 / 27 | **The live PointNav corpus.** Three formatted variants, see §3. |

The `_2p5hz` / `_clamp` / `_keep` triple is a **boundary-policy ablation**, fully written up in
the headers of `dump/pose_injection/pointnav_v2_launch.sh` and `pointnav_v3_launch.sh`. Keep
those two files: they are the only record of why the first two are unusable.

### 2.3 `annotations/` — the distance sidecars (2026-08-17/18)

Per-frame ground-truth geodesic distance-to-goal, one JSONL per scene, joined into existing
formatted datasets by `episode_id` + `frame_indices` (no rebuild) via
`data_scripts/add_distance_targets.py`.

| folder | scenes | notes |
|---|---|---|
| `v2_25hz_distance_to_goal/` | 56 JSONL (~0.9 GB) | For the ObjectNav corpora. **100% coverage.** Goal source is the MP3D THDA ObjectNav episode files inside the deprecated fork. |
| `pointnav_hm3d_long_distance_to_goal/` | 730 JSONL | For the PointNav corpora. **~80% frame coverage** — gaps are NaN-filled and masked in the head (`--allow-missing`, refuses >50% missing). |

Both carry `annotation_manifest.json` (`distance_to: VIEW_POINTS`, `height_mode: chained`,
`navmesh: dataset`, `snap_agent_position: true`) and a `run.log`.

### 2.4 Second axis and older material (present but not part of this track)

`datasets/` → NFS symlink to the Habitat episode datasets. The RL-relevant ones are
`objectnav/hm3d/{v1, v1_train8x4, v1_train25x80, v1_train100x80, v1_memtest_4scenes}` (the
`v1_train*` ones built 2026-08-14/15). `scene_datasets/` → the shared HM3D/MP3D meshes.
`StreamVLN/`, `svln_scene_datasets/`, `R2R_VLNCE_v1-3.zip` are from the earlier VLN work
(2026-01) and are untouched by the continuous track.

**Note the scene mismatch:** the ObjectNav *demonstrations* are MP3D; RL *rollouts and eval*
run on HM3D ObjectNav; the PointNav demonstrations are HM3D **train** scenes. That last overlap
is what invalidated the on-policy stop-head work (§5).

---

## 3. The `formatted*` variants, and what each suffix means

Both live corpora carry a stack of formatted variants built by re-formatting or by joining
sidecar columns. Naming is compositional:

| suffix | meaning |
|---|---|
| `_pose` | `<pose>` modality marker + `obs_poses` injected into the conversation |
| `_nopose` | pose dropped entirely (the `obs_poses` **column is absent**) |
| `_dist097` | `distance_targets` + `return_targets` + `return_gamma` columns joined in at γ=0.97 |
| `_nostep` | turns read `"Observation:"` instead of `"Observation 42:"` — the step index removed |

**`v2_25hz_obs2.5hz/`** (7 variants):

| variant | built | live? |
|---|---|---|
| `formatted_pose` | 08-07 (touched 08-18) | historical — cotrain-v2/v3 |
| `formatted_nopose` | 08-09 (touched 08-18) | historical — cotrain-v3 |
| `formatted_nopose_raw` | 08-09 | **orphan** — keeps the `obs_poses` column but is referenced by no script or doc found |
| `formatted_pose_dist097` | 08-18 | cotrain-v4 |
| `formatted_nopose_dist097` | 08-18 | cotrain-v4 |
| `formatted_nopose_dist097_nostep` | **08-21 (rows touched 08-23)** | **current** — cotrain-v5 |

**`pointnav_long_full/`** (3 variants): `formatted_pose` (08-09, cotrain-v2/v3) →
`formatted_pose_dist097` (08-18, cotrain-v4) → `formatted_pose_dist097_nostep` (**08-21, current**).

The `_nostep` restripe was a deliberate response to a measured failure: the RL critic was keying
on the step token (`corr(V, step) = -0.407`). Per `dump/pose_injection/cotrain_v5_launch.sh`,
every non-`messages` column is byte-identical to the originals, and **checkpoints from the
`_nostep` lineage are not prompt-compatible with earlier ones** — an RL run on that base must
drop `$step` from its rollout templates.

---

## 4. Which corpus fed which run

Straight from the launch script headers in `dump/pose_injection/`:

| run | date | mixture |
|---|---|---|
| `pointnav-v1` | 08-07 | `pointnav_hm3d_2p5hz/formatted_pose` (drop) |
| `pointnav-v2` | 08-07 | `pointnav_hm3d_2p5hz_clamp/formatted_pose` |
| `pointnav-v3` | 08-07 | `pointnav_hm3d_2p5hz_keep/formatted_pose` |
| `cotrain-v1` | 08-07 | objectnav + `pointnav_hm3d_2p5hz_keep` |
| `cotrain-v2` | 08-09 | `v2_25hz_obs2.5hz/formatted_pose` : `pointnav_long_full/formatted_pose` = 1:1 |
| **`cotrain-v3`** | 08-09 | `formatted_nopose` : `formatted_pose` : `pointnav_long_full/formatted_pose` = **1:1:2** |
| `cotrain-v4` | 08-18 | the three `_dist097` variants, same 1:1:2 |
| **`cotrain-v5`** | 08-21 | `formatted_nopose_dist097_nostep` : `pointnav_long_full/formatted_pose_dist097_nostep` = 1:1 |
| `cotrain-v6/v7` | 08-22 | **deleted** — see §5 |

`cotrain-v3 ck12000` is the published SFT policy and the cycle-0 model of every RL run
(`docs/PUBLISHING_CHECKPOINTS.md`, `docs/TRAINING_EVAL_SET.md`). Note the 1:1 in v5 is *by row*;
PointNav rows carry ~198 observations against ObjectNav's ~51, so by supervised step it is ~76%
PointNav — roughly what v4's 1:1:2 gave.

---

## 5. What happened on 2026-08-23 (the newest thing, and it is a deletion)

Two folders that a survey a week ago would have listed are **gone on purpose**:
`data/onpolicy_distance/` and `data/mined_rollout_frames/`.

The on-policy distance/stop-head track built training corpora from frames recovered by decoding
rollout `video.mp4` files. Those videos carry a **rendered HUD printing `distance_to_goal`,
`goal` and `step`** — so a distance or stop head fit on them was reading text, not seeing. The
tell was precision 1.000 and MAE 0.207 m on nominally held-out scenes. Independently, that
"held-out" split came from HM3D **train** scenes, 11 of whose 12 are in the PointNav corpus the
same checkpoints trained on.

Purged: the mined frame stores, the `*_series.npz` hidden caches, the `refit_value_head_*.pt`
heads, the `onpolicy_distance` corpora, and the `run_cotrain_v6_onpolicy` / `run_cotrain_v7_stop`
runs. Summaries kept for the record only in `dump/eval_system/QUARANTINE_contaminated/`.
`data_scripts/mine_rollout_frames.py` is deleted; `data_scripts/build_onpolicy_distance_dataset.py`
still exists but its first statement is a `raise SystemExit("RETIRED 2026-08-23…")`.

The retraction banner at the top of `docs/STOP_HEAD_PLAN.md` is the authoritative account.
What survives it: the clock-shortcut finding, the train/rollout distribution mismatch, and the
stop-rule ceiling arithmetic — all measured on live rollouts or on the demonstration corpora.
The one clean number for this lineage is HM3D-val live rollouts: **MAE 3.32 m against a 3.19 m
constant-prediction baseline — no skill on unseen scenes.**

The working files touched today (`data_scripts/add_stop_targets.py`, new 08-23;
`eval_value_heads_offline.py`; `tests/test_gradient_isolation.py`) are the clean restart of that
same track: stop labels derived as `distance_targets <= radius` from the *sidecar*, with NaN
where the distance is unknown, on the demonstration corpora. No corpus has been built with
`stop_target` columns yet.

---

## 6. Practical notes

* **Absolute paths are load-bearing.** `chunk_config.json` records `image_root`, and for the
  whole `v2_25hz*` family it points into `/Projects/DEPRECATED_DO_NOT_USE_continuous_demos/`.
  The folder is named "do not use" about its *code*; moving its `output_25hz/` breaks every
  ObjectNav corpus.
* **`.mc/` and `.mapcache/` are HF map caches**, not data — 5–15 GB each under the
  `pointnav_hm3d_2p5hz*` and `pointnav_hm3d_pilot` folders. Safe to delete; they are the largest
  cheap reclaim on the packed side.
* **Candidates for reclaim** if space is ever needed: `pointnav_hm3d_2p5hz/` and
  `pointnav_hm3d_2p5hz_clamp/` (known-bad corpora, kept only as the ablation record — the
  *writeups* live in the launch scripts, not in the data), `pointnav_hm3d_full/` (built, never
  used), `formatted_nopose_raw/` (orphan), `continuous_sft*` (superseded 20 fps lineage,
  6.5 GB). The ~1.7 TB of raw jpegs in `pointnav_hm3d/` + `pointnav_hm3d_long/` dwarfs
  everything else, but `pointnav_hm3d/` is the source for four packed corpora and
  `pointnav_hm3d_long/` is live.
* **Row counts live in `chunks/chunk_config.json`** under `num_rows` — cheaper than opening the
  Arrow files.
* **The `_shards/episodes.txt` and `eval_episodes/` in this repo** are RL-axis artifacts (episode
  uid lists), unrelated to the demonstration corpora. Episode-set records for the RL eval sets
  are under `dump/eval_system/episode_sets/`.
