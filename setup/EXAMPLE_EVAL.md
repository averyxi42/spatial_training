# Running the eval on the example scene

The full pipeline — Ray, the VLM policy, habitat simulation, video and metric logging —
runs on the single downloadable MP3D example scene, so it can be reproduced without the
~15 GB MP3D download and without a wandb account.

These are the stock MP3D ObjectNav episodes. Nothing is generated or synthesised; the
restriction to one scene is done with habitat's own `content_scenes`. See
[DATASET_SETUP.md](DATASET_SETUP.md).

## Reproduce

Prerequisites: the container from [../docker/README.md](../docker/README.md), plus

```bash
bash setup/download_example_data.sh      # example scene, ~93 MB
bash setup/download_objectnav_mp3d.sh    # ObjectNav episodes, ~173 MB
```

Then, from inside the container (`docker compose exec longnav bash`):

```bash
# 6 episodes, ~65 s — checks the pipeline end to end, ~5 of 6 should succeed
python -m longnav.scripts.eval \
    +checkpoint=sft +dataset=mp3d_example_smoke \
    +experiment=eval_fast +resources=single \
    task.run_name=smoke

# 256 episodes, ~22 min — real numbers
python -m longnav.scripts.eval \
    +checkpoint=sft +dataset=mp3d_example \
    +experiment=eval +resources=single \
    task.run_name=mp3d_example_eval
```

There is also a habitat-only check that needs no GPU policy and no Ray:

```bash
conda run --no-capture-output -n vln python tests/mp3d_example_env.py
```

## Configs added for this

| config | group | what it does |
| --- | --- | --- |
| `dataset/mp3d_example` | dataset | Full eval on the example scene. Points `sim.config_path` at `habitat_configs/objectnav_mp3d_test.yaml`, `split: train`, and uses the trivial shard (`shard_size: 0`) so habitat picks episodes itself. |
| `dataset/mp3d_example_smoke` | dataset | Same, bounded to exactly 6 episodes via `episode_json: eval_episodes/mp3d_example_smoke.json`, chosen to be short enough to finish quickly but long enough to be real navigation. |
| `experiment/eval_fast` | experiment | `max_steps: 150` (from 500), top-down map off, `wandb_project: null`. |
| `resources/single` | resources | One GPU: `num_vlms: 1`, `num_sims: 2`, fractions `0.7 + 2 x 0.14 = 0.98`. |

`habitat_configs/objectnav_mp3d_test.yaml` carries the habitat side: `content_scenes:
["17DRP5sb8fy"]` plus the RGBD sensor setup.

`+checkpoint=sft` uses the SFT model with no separate checkpoint download;
`+checkpoint=longnav` pulls the released RL checkpoint from HuggingFace.

`17DRP5sb8fy` is a **train**-split scene — it does not appear in val, so `split: train`.

## Output

Per episode, under `<task.output_dir>/<run_name>/rollout/<scene>/<episode>/`:

| file | contents |
| --- | --- |
| `video.mp4` | egocentric RGB, 640x480 at 4 fps (plus top-down map when `add_top_down_map: true`, i.e. `+experiment=eval` but not `eval_fast`) |
| `summary.json` | `success`, `spl`, `soft_spl`, `distance_to_goal`, `n_steps`, `goal`, throughput |
| `sequence.json` | full action history and per-step records |
| `thumbnail.jpg` | first frame |

A representative smoke run (SFT checkpoint, `eval_fast`, 64 s):

```
    ep  succ    spl  steps  final_d
 10018   1.0  0.632     40     0.03
 10473   1.0  0.462     79     0.05
 10734   1.0  0.701     41     0.03
 11137   1.0  0.408     99     0.02
  9430   0.0  0.000     28     3.70
  9569   1.0  0.442     72     0.00
success rate 0.833
```

Treat that as a health check, not a benchmark: it is six short episodes of one goal
category on one scene, truncated at 150 steps. A success rate far below this, or episodes
ending at `n_steps == max_steps` with a small `final_d`, means something is wrong —
the latter is the agent reaching the goal but never emitting STOP.

## Constraints worth knowing

These are properties of the data and the current code, not of this setup:

- **Episode labels are ambiguous on MP3D.** Labels are `{scene}_{episode_id}`, but MP3D
  `episode_id` is only unique *within* an object category: the 50,000 episodes of
  17DRP5sb8fy collapse into 11,364 distinct labels, so a typical label pulls in ~4
  unrelated episodes. `mp3d_example_smoke.json` deliberately lists only the ids that map
  1:1, which is what makes its episode count exactly 6 — at the cost of all six sharing
  the `cushion` goal. HM3D does not have this problem (ids are globally unique), so
  `subset_label` works normally there.
- **256 is the floor for the trivial shard.** With `shard_size <= 0`, `get_shard_iterator`
  returns `trivial_shard_iterator(n=256)`, and `n` is not exposed to config. Bounding the
  episode count therefore requires `episode_json` (or `subset_label`, where labels are
  unambiguous). This is why `eval_fast` alone does not make a run short — it only shortens
  each episode.
- **`wandb_project: null` needs the `factories.py` fix** that returns `None, set()` instead
  of a bare `None`; callers unpack two values. Note `WANDB_MODE=offline` is *not* a
  workaround — `wandb.Api()` verifies the key against api.wandb.ai regardless of mode, so
  running without an account requires the null path to work.
- Rollouts log repeated `Failed to reduce user key sup/vlm_mem_GB: not enough values to
  unpack` warnings. Harmless to the run, but those supplementary metrics are not recorded.

## Folding into the install flow (not done yet)

When this becomes part of installation, the pieces that need to move:

1. The two downloads above, plus the `data/scene_datasets_example/` scene root that
   `download_example_data.sh` creates, become install steps. That root is deliberately
   separate from `data/scene_datasets/` so a later full-MP3D download cannot collide with
   it — see [DATASET_SETUP.md](DATASET_SETUP.md).
2. `python -m longnav.scripts.eval ... +dataset=mp3d_example_smoke` is a good final
   install check: ~50 s, exercises both conda envs through Ray, and fails loudly if
   habitat-sim, the policy, or the dataset wiring is wrong.
3. `eval_episodes/mp3d_example_smoke.json` is checked in and scene-specific. If the
   example scene ever changes, regenerate it by picking `episode_id`s whose count in the
   scene's content file is 1.
