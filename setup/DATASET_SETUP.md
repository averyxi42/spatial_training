# Dataset setup

Upstream references:

- [MP3D scenes (habitat-sim)](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#matterport3d-mp3d-dataset)
- [Episode datasets (habitat-lab)](https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md)

## Minimal setup for the habitat smoke test

You do **not** need the full ~15 GB MP3D scene set to instantiate and step a
`HabitatWorker`. One scene is enough:

```bash
bash setup/download_example_data.sh      # 17DRP5sb8fy, ~93 MB
bash setup/download_objectnav_mp3d.sh    # ObjectNav episodes, ~173 MB
conda run --no-capture-output -n vln python tests/mp3d_example_env.py
```

No dataset generation step: the episodes are the stock ones, restricted to the one
available scene by habitat's own `content_scenes` mechanism in
`habitat_configs/objectnav_mp3d_test.yaml`.

### How it works

Two pieces, both in the config or the download script — nothing rewrites the data.

1. **`content_scenes: ["17DRP5sb8fy"]`.** The ObjectNav dataset ships per-scene episode
   files under `<split>/content/`. Listing scenes explicitly makes habitat read only
   those, so no episode ever references a scene that was not downloaded. Left at the
   default `["*"]` it loads all 90 MP3D scenes and dies on the first missing one:

   ```
   ESP_CHECK failed: No Stage Attributes exists for requested scene
   'data/scene_datasets/mp3d/2azQ1b91cZZ/2azQ1b91cZZ.glb'
   ```

   Widen the list as you download more scenes, or set it back to `["*"]` for full MP3D.

2. **A separate scene root, `data/scene_datasets_example/`**, created by
   `setup/download_example_data.sh`. Episodes store `scene_id` as
   `mp3d/<scene>/<scene>.glb`, resolved against `habitat.dataset.scenes_dir`, so
   something must answer to the name `mp3d` — here a symlink to the example scene set.

   It lives outside `data/scene_datasets/` on purpose. The full MP3D download writes to
   `data/scene_datasets/mp3d/`, and if that name were already a symlink to the example
   set, the download would **follow it** and dump 90 scenes into
   `versioned_data/mp3d_example_scene_1.1/`. Keeping the demo in its own root means the
   two never interact.

`17DRP5sb8fy` is a **train** scene — it does not appear in val. `HabitatWorker` defaults
its `split` argument to `"val"` and overrides the config, so callers must pass
`split="train"`.

## Expected layout

```
data/
  scene_datasets/                                  # left free for a full download
    mp3d_example -> ../versioned_data/mp3d_example_scene_1.1
    mp3d/                                          # only if you download full MP3D
  scene_datasets_example/                          # the demo's scene root
    mp3d -> ../versioned_data/mp3d_example_scene_1.1
  versioned_data/
    mp3d_example_scene_1.1/17DRP5sb8fy/...
  datasets/
    objectnav/mp3d/v1/{train,val,val_mini}/
```

All symlinks are **relative** on purpose. The habitat downloader writes `mp3d_example`
as an absolute path, which breaks inside the docker container where the repo is
bind-mounted at `/workspace`; `setup/download_example_data.sh` rewrites it. Nothing here
is committed — `data/` is gitignored.

## Moving to full MP3D later

Nothing needs undoing. Download MP3D normally (it lands in `data/scene_datasets/mp3d/`,
which the example setup never touches), then in
`habitat_configs/objectnav_mp3d_test.yaml` set `scenes_dir: data/scene_datasets` and
widen or remove `content_scenes`. The example config keeps working alongside it, which is
useful for fast iteration.
