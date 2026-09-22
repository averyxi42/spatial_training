import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation

from longnav.env.navverse import NavverseEnv


@torch.inference_mode()
def _render_rgb(probe):
    env = probe.vln_sim.env
    zero_commands = torch.zeros(
        (probe.vector_envs, probe.vln_sim.command_dim),
        device=probe.vln_sim.device,
        dtype=probe.vln_sim.commands.dtype,
    )
    raw_obs = env.obs
    for _ in range(4):
        action = env._command_to_env_action(zero_commands)
        raw_obs, _, _, _ = env._step_low_level_env(action)
        env.obs = raw_obs
    return probe._vector_rgb(env._get_high_level_obs(raw_obs))


@torch.inference_mode()
def _place_other_robots_in_front(probe):
    robot = probe.vln_sim.env.scene["robot"]
    root_state = robot.data.root_state_w.clone()
    positions, quaternions = probe._vector_robot_poses()
    yaw = Rotation.from_quat(quaternions[0]).as_euler("ZYX")[0]
    forward = np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=np.float32)
    left = np.array([-np.sin(yaw), np.cos(yaw), 0.0], dtype=np.float32)
    offsets = [
        2.0 * forward,
        3.0 * forward + 0.7 * left,
        3.0 * forward - 0.7 * left,
        4.0 * forward + 1.4 * left,
        4.0 * forward - 1.4 * left,
        5.0 * forward + 2.0 * left,
        5.0 * forward - 2.0 * left,
    ]
    for index, offset in enumerate(offsets, start=1):
        root_state[index, :3] = torch.as_tensor(
            positions[0] + offset,
            device=robot.device,
            dtype=root_state.dtype,
        )
        root_state[index, 7:] = 0.0
    robot.write_root_state_to_sim(root_state)
    robot.write_data_to_sim()


def _assert_hidden_robot_geometry(probe):
    from pxr import UsdGeom

    stage = probe.vln_sim.manager_env.scene.stage
    checked = 0
    visible = []
    for env_index in range(probe.vector_envs):
        robot_prim = stage.GetPrimAtPath(f"/World/envs/env_{env_index}/Robot")
        checked += 1
        if (
            UsdGeom.Imageable(robot_prim).ComputeVisibility()
            != UsdGeom.Tokens.invisible
        ):
            visible.append(str(robot_prim.GetPath()))
    if visible:
        raise RuntimeError(f"Visible vector robot roots remain: {visible}")
    return checked


def main():
    episode_json = Path(os.environ["VISIBILITY_PROBE_EPISODES"])
    output_dir = Path(os.environ["VISIBILITY_PROBE_OUTPUT"])
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = json.loads(episode_json.read_text())[:8]

    probe = NavverseEnv()
    try:
        probe.assign_shard(labels)
        initial_rgb, _ = probe.reset_batch()
        hidden_prims = _assert_hidden_robot_geometry(probe)
        _place_other_robots_in_front(probe)
        near_rgb = _render_rgb(probe)
        Image.fromarray(initial_rgb[0]).save(output_dir / "initial_env0.png")
        Image.fromarray(near_rgb[0]).save(output_dir / "robots_in_front_hidden_env0.png")

        metrics = {
            "hidden_robot_roots": hidden_prims,
            "rgb_shape": list(near_rgb.shape),
            "env0_rgb_mean": float(near_rgb[0].mean()),
            "env0_rgb_std": float(near_rgb[0].std()),
        }
        if near_rgb.shape != (8, 480, 640, 3):
            raise RuntimeError(f"Unexpected vector RGB shape: {near_rgb.shape}")
        if metrics["env0_rgb_std"] < 5.0:
            raise RuntimeError(f"Camera output is degenerate: {metrics}")
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print(json.dumps(metrics, indent=2))
    finally:
        probe.simulation_app.close()


if __name__ == "__main__":
    main()
