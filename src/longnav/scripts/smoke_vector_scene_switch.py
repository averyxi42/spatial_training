import json
import os
from pathlib import Path

from longnav.env.navverse import NavverseEnv


def main():
    batches = json.loads(Path(os.environ["SCENE_SWITCH_PROBE_BATCHES"]).read_text())
    if any(len(batch) != 8 for batch in batches):
        raise ValueError("Every scene-switch probe batch must contain eight episodes")

    probe = NavverseEnv()
    observations = []
    try:
        for batch in batches:
            probe.assign_shard(batch)
            rgb, _ = probe.reset_batch()
            episode = probe.vector_episodes[0]
            scene_type = episode["scene_type"]
            config = probe.vln_sim.env._terrain_collision_config(scene_type)
            observations.append(
                {
                    "scene_id": episode["scene_id"],
                    "scene_type": scene_type,
                    "episode_path": episode["path"],
                    "loaded_path": probe.vln_sim.env.usd_path,
                    "rgb_shape": list(rgb.shape),
                    "filtered_scene_types": list(
                        (config or {}).get("filtered_scene_types", [])
                    ),
                }
            )
            if probe.vln_sim.env.usd_path != episode["path"]:
                raise RuntimeError(
                    f"Loaded {probe.vln_sim.env.usd_path}, expected {episode['path']}"
                )
            if rgb.shape != (8, 480, 640, 3):
                raise RuntimeError(f"Unexpected vector RGB shape: {rgb.shape}")
        print(json.dumps(observations, indent=2))
    finally:
        probe.simulation_app.close()


if __name__ == "__main__":
    main()
