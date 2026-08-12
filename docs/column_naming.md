# `obs_poses` is a misnomer — a queued rename, deliberately not done yet

**Status:** known defect, not scheduled. Do it at the next corpus rebuild, not before.
**Raised:** 2026-08-07, while adding PointNav ratio mixing.

## What is wrong

The modality value column is called `obs_poses`, but it does not hold observation poses.
It holds **poses** — a mixed, occurrence-ordered stream of whatever `<pose>` markers the
conversation contains. For an ObjectNav row that happens to be one agent pose per turn. For
a PointNav row it is agent poses *and* goal poses interleaved:

```
segment placement : [agent_0, goal_0, pose_0, pose_1, ..., agent_k, goal_k, pose_k, ...]
observation placement : [pose_0, goal_0, pose_1, goal_1, ...]
```

Whether a given row is "an observation" or "a goal" is a property of **where its marker
sits in the text**, not of the value. The mechanism is deliberately blind to it — that
blindness is what let PointNav reuse the ObjectNav pose spec unchanged, with no `<goal>`
token and no change to `modality_embed.py`.

The parallel is `images`. That column is not called `obs_images`, and it would be wrong if
it were: an image-goal task (standard, not implemented here) would put a goal image in the
same column, bound by the same occurrence order, and nothing about the model would change.
Poses are the same. The `obs_` prefix encodes an assumption that the mechanism does not
make and that PointNav already violates.

## Why the rename is cheap

**The column name is not baked into any checkpoint.** Verified: `modality_specs` does not
appear in `turn_vector_head_config.json` for `run_v9_flow_planar_pose_2p5hz/checkpoint-11400`.
Specs are supplied at run time through `--modality-specs <file>`, and
`ModalityEmbedSpec.source_column` resolves as `column or key`. So no trained weight, and no
saved config, refers to the name — only the spec file passed on the command line does.

Renaming an arrow column is `Dataset.rename_column`, which is a schema operation, not a
data rewrite.

## What to change

**1. Pick the name.** Two candidates:

* `poses` — parallels `images`, reads naturally as a list column.
* `pose` — matches the spec's `key` (the token `<pose>` minus its brackets), which means
  `source_column = column or key` resolves correctly **with the `column` field deleted
  entirely**. Strictly less configuration, at the cost of a singular name for a list.

I lean `poses`, but `pose` is the one that removes a field rather than renaming one.

**2. Spec files** (`dump/pose_injection/`): `pose_spec_planar.json`, `pose_spec.json`,
`pose_spec_capped.json` — one `"column"` value each, or delete the key if `pose` is chosen.
`pose_spec_planar.json` is the one live runs use.

**3. Source references**, `spatial_training`:

| file | count |
|---|---|
| `tests/test_pose_injection.py` | 21 |
| `tests/test_flow_pose_injection.py` | 3 |
| `src/longnav/utils/vector_rollout.py` | 3 |
| `tests/test_mixture.py` | 2 |
| `data_scripts/format_action_chunk_dataset.py` | 2 (the `--modality-column` default) |
| `tests/test_ar_pose_injection.py`, `src/longnav/utils/pose_frame.py`, `src/longnav/utils/mixture.py`, `data_scripts/run_vector_policy_habitat.py`, `data_scripts/build_slow_corpora.sh` | 1 each |

**4. Source references**, `habitat_physical_nav`: the *writing* side is
`scripts/build_action_chunk_episode_dataset.py` (the row key, plus its docstring). Readers:
`scripts/render_demo_videos.py`, `scripts/verify_pose_injection.py`,
`src/objectnav_eval/bridge.py` (`_PoseFeed`'s `column` default).

**5. Corpora on disk** — `rename_column` + `save_to_disk`, or rebuild:

```
data/v2_25hz/formatted_pose
data/v2_25hz_obs2.5hz/formatted_pose          <- v9 and pointnav-v2 train on this lineage
data/v2_25hz_obs1hz/formatted_pose
/Projects/data/pointnav_hm3d_2p5hz_clamp/formatted_pose
/Projects/data/pointnav_hm3d_2p5hz/formatted_pose   (superseded, see below)
```

The `chunks/` tables upstream of each also carry it.

## Why it is not being done now

1. A live run (`run_pointnav_v2_flow_planar_2p5hz_clamp`) is reading `obs_poses`. Renaming
   the corpus under a running job breaks it on the next dataloader worker restart.
2. The corpora are due to be rebuilt anyway when PointNav generation finishes — the rename
   is free at that point and costs a full re-format if done separately.
3. It is cosmetic. Nothing computes a wrong number because of it; the cost is a reader
   inferring a constraint the mechanism does not have.

## The trap to avoid when doing it

A half-rename is worse than no rename: a spec naming `poses` against a corpus holding
`obs_poses` fails with `KeyError` inside the collator on the first batch — attributable, but
only after model load. Rename the corpora and the spec files in the same change, and grep
both repos, including `.sh` launch scripts, which are not covered by any test.

Related: `docs/placeholder_tokens.md` (the other naming decision that looks cosmetic and
is not), and `/Projects/habitat_physical_nav/docs/POINTNAV_GENERATION.md` for why the goal
rides the `<pose>` token at all.
