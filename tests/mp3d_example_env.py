"""Smoke test: instantiate a HabitatWorker and step it, using only the example scene.

Runs against the single downloadable MP3D example scene (17DRP5sb8fy, ~93 MB)
instead of the full ~15 GB MP3D set, using the stock ObjectNav episodes as-is.
No dataset is generated or modified: habitat's own `content_scenes` in
habitat_configs/objectnav_mp3d_test.yaml restricts loading to the one scene
present. See setup/DATASET_SETUP.md.

    bash setup/download_example_data.sh
    bash setup/download_objectnav_mp3d.sh
    conda run --no-capture-output -n vln python tests/mp3d_example_env.py
"""

import numpy as np

from longnav.env.habitat import HabitatWorker

# split="train" because 17DRP5sb8fy is a train scene and HabitatWorker otherwise
# forces the config's split to "val".
worker = HabitatWorker(
    config_path="habitat_configs/objectnav_mp3d_test.yaml",
    split="train",
)
worker.assign_shard()

print(f"\nepisodes in shard: {worker.total_episodes()}")

step = worker.reset()
obs = step["obs"]
print(f"observation keys : {sorted(obs.keys())}")
for key in sorted(obs):
    value = obs[key]
    if isinstance(value, np.ndarray):
        print(f"  {key:<16} {value.shape} {value.dtype}")
    else:
        print(f"  {key:<16} {value!r}")

print(f"\ngoal category    : {obs['instr_or_goal']}")
print(f"start distance   : {step['info']['distance_to_goal']:.2f} m")

# Follow the oracle rather than acting randomly: a random walk tells you nothing
# about whether the scene, the navmesh and the goal annotations line up, whereas
# the oracle's distance_to_goal should fall monotonically if they do, and the
# episode should end in success.
print("\nstepping with the shortest-path oracle:")
print(f"  {'step':>4}  {'action':>6}  {'reward':>7}  {'dist':>6}  done")
for i in range(500):
    action = worker.nav_oracle.get_best_action()
    step = worker.step(action)
    print(
        f"  {i:>4}  {action:>6}  {step['reward']:>7.3f}  "
        f"{step['info']['distance_to_goal']:>6.2f}  {step['done']}"
    )
    if step["done"]:
        break

# Reported on the final step rather than via get_metrics(), which returns {} once
# the episode is over.
final = step["info"]
print("\nfinal episode metrics:")
for key in ("success", "spl", "distance_to_goal", "softspl"):
    if key in final and np.isscalar(final[key]):
        print(f"  {key:<18} {final[key]:.3f}")

worker.close()
print("\nOK: habitat worker instantiated, reset, and stepped.")
