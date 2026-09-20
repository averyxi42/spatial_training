"""Ray-actor bridge exposing MS-HAB (ManiSkill mobile manipulation) as a LongNav env.

Mirrors `DummyContinuousEnvActor`'s Ray-actor contract (`reset()`/`step(action)`
returning `(rgb, state_dict)`, plus `assign_shard`/`is_exhausted`/
`flush_logs_to_disk`) so the rest of LongNav (rollout_core, a continuous policy
head such as `gaussian_head`, or the flowsde head) works unchanged. MS-HAB's
native continuous Fetch action space (`single_action_space`, see
`ManiSkill/mani_skill/agents/robots/fetch/fetch.py::Fetch._controller_configs`,
`pd_joint_delta_pos`: 7 arm + 1 gripper + 3 body[head_pan, head_tilt,
torso_lift] + 2 base[forward_vel, angular_vel] = 13 dims) is exposed unchanged;
no discretization or remapping is done here.

Only the Fetch robot's head camera (`fetch_head`) is surfaced as `rgb` for this
first pass -- the wrist camera (`fetch_hand`) is intentionally dropped.
"""
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


class ManiskillHabEnvActor:
    """One MS-HAB env instance (num_envs=1) behind LongNav's per-actor env interface."""

    def __init__(
        self,
        logging_output_dir: Optional[str] = None,
        logger_actor: Any = None,
        mshab_root: Optional[str] = None,
        ms_asset_dir: Optional[str] = None,
        env_id: str = "NavigateSubtaskTrain-v0",
        task_plan_fp: Optional[str] = None,
        spawn_data_fp: Optional[str] = None,
        max_episode_steps: int = 1000,
        sim_backend: str = "gpu",
        camera: str = "fetch_head",
        instruction: str = "navigate to the goal",
        env_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.logging_output_dir = logging_output_dir
        self.logger_actor = logger_actor
        self.camera = camera
        self.instruction = instruction

        mshab_root = mshab_root or os.environ.get("MSHAB_ROOT")
        if not mshab_root:
            raise ValueError(
                "ManiskillHabEnvActor needs `mshab_root` (or env var MSHAB_ROOT) "
                "pointing at the maniskill-hab checkout."
            )
        if mshab_root not in sys.path:
            sys.path.insert(0, mshab_root)

        asset_dir = ms_asset_dir or os.environ.get(
            "MS_ASSET_DIR", os.path.join(mshab_root, "maniskill_data")
        )
        # `mani_skill`'s own ASSET_DIR (used by its scene builders to locate
        # ReplicaCAD scenes/objects, independent of task_plan_fp/spawn_data_fp)
        # is computed once at import time from this env var -- it must be set
        # before `mani_skill` is first imported in this process.
        os.environ.setdefault("MS_ASSET_DIR", asset_dir)
        rearr = os.path.join(
            asset_dir, "data/scene_datasets/replica_cad_dataset/rearrange"
        )
        task_plan_fp = task_plan_fp or os.path.join(
            rearr, "task_plans/set_table/navigate/train/all.json"
        )
        spawn_data_fp = spawn_data_fp or os.path.join(
            rearr, "spawn_data/set_table/navigate/train/spawn_data.pt"
        )

        from mshab.envs.make import EnvConfig, make_env

        # The stock task-plan JSONs cover far more build configs (scenes) than
        # a single-env LongNav actor has room for; without this flag ManiSkill
        # asserts that num_envs divide evenly across every build config.
        merged_env_kwargs = {"require_build_configs_repeated_equally_across_envs": False}
        merged_env_kwargs.update(env_kwargs or {})

        cfg = EnvConfig(
            env_id=env_id,
            num_envs=1,
            max_episode_steps=max_episode_steps,
            obs_mode="rgbd",
            render_mode="all",
            sim_backend=sim_backend,
            continuous_task=True,
            frame_stack=None,
            record_video=False,
            task_plan_fp=task_plan_fp,
            spawn_data_fp=spawn_data_fp,
            env_kwargs=merged_env_kwargs,
        )
        self.venv = make_env(cfg)
        self.base = self.venv.unwrapped
        self.action_space = self.base.single_action_space

    # -- LongNav Ray-actor contract -------------------------------------

    def assign_shard(self, episodes: Optional[List[str]] = None):
        """MS-HAB samples its own task-plan/spawn distribution; nothing to shard."""
        pass

    def is_exhausted(self) -> bool:
        return False

    def flush_logs_to_disk(self):
        return None

    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        self.venv.reset()
        rgb = self._head_rgb()
        state = {
            "obs": {"instr_or_goal": self.instruction},
            "done": False,
            "reward": 0.0,
            "is_exhausted": self.is_exhausted(),
            "info": {},
        }
        return rgb, state

    def step(
        self, action, supplementary_logs: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        action_t = torch.as_tensor(
            np.asarray(action, dtype=np.float32)
        ).reshape(1, -1)
        _, reward, term, trunc, info = self.venv.step(action_t)
        rgb = self._head_rgb()
        done = bool(term[0] or trunc[0])
        state = {
            "obs": {"instr_or_goal": self.instruction},
            "done": done,
            "reward": float(reward[0]),
            "is_exhausted": self.is_exhausted(),
            "info": {"success": bool(info["success"][0])} if "success" in info else {},
        }
        return rgb, state

    def close(self):
        self.venv.close()

    # -- internals --------------------------------------------------------

    def _head_rgb(self) -> np.ndarray:
        raw = self.base.get_obs()
        rgb = raw["sensor_data"][self.camera]["rgb"][0]
        return rgb.to(torch.uint8).cpu().numpy()
