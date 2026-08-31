"""ManiSkill-HAB (Home Assistant Benchmark on SAPIEN) sim actor.

Implements the five-method sim contract from ``env_base.py`` / ``objectnav_continuous.py``
(``reset``, ``step``, ``assign_shard``, ``flush_logs_to_disk``, ``is_exhausted``) on top of
the MS-HAB envs (``PickSubtaskTrain-v0``, ``NavigateSubtaskTrain-v0``, ... registered by
``mshab.envs``). See docs/MSHAB_INTEGRATION.md.

Runs inside its own conda env (``mshab``): SAPIEN needs Vulkan and its own torch; the
habitat env pins numpy to habitat-sim's ABI. Everything ManiSkill is imported lazily
inside the actor process -- the driver never imports sapien.

Two action modes:

* ``joint``      -- the raw Fetch ``pd_joint_delta_pos`` vector (13-d: arm 7, gripper 1,
                    body 3 [head pan, head tilt, torso], base 2 [v, w]); one sim step per
                    row of a ``(gap, 13)`` chunk.
* ``base_chunk`` -- the continuous ObjectNav policy's ``(gap, 3)`` chunk of poses
                    ``[dx, dy, dtheta]`` relative to the robot's pose at the start of the
                    step (x forward, y left, CCW yaw -- ``continuous_demos.action_chunking``
                    convention, identical to SAPIEN's planar base frame). Ticks are
                    ``dt`` apart (0.04 s for the SFT corpus). The base is Fetch's
                    forward+yaw velocity controller, so the lateral component is dropped
                    (non-holonomic) and the chunk is tracked by a P + feedforward
                    controller at the sim's 20 Hz control rate. Arm/gripper/torso/head hold
                    still (zero deltas) after being tucked at reset.

Contract decisions (mirrors the continuous ObjectNav actor):
  * rgb  : uint8 (H, W, 3) from the Fetch head camera.
  * obs  : {"instr_or_goal": str}. Anything else added here becomes a $template variable.
  * info : identical key set on EVERY step (rollout_core packs columns from step 0).
"""
from __future__ import annotations

import json
import math
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np


_TASK_TO_ENV_ID = {
    "pick": "PickSubtaskTrain-v0",
    "place": "PlaceSubtaskTrain-v0",
    "open": "OpenSubtaskTrain-v0",
    "close": "CloseSubtaskTrain-v0",
    "navigate": "NavigateSubtaskTrain-v0",
    "sequential": "SequentialTask-v0",
}

# Fetch arm joint order in ManiSkill: shoulder_pan, shoulder_lift, upperarm_roll,
# elbow_flex, forearm_roll, wrist_flex, wrist_roll. This is the standard Fetch "tuck".
_ARM_JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "upperarm_roll_joint",
               "elbow_flex_joint", "forearm_roll_joint", "wrist_flex_joint", "wrist_roll_joint"]
_TUCK_QPOS = [1.32, 1.40, -0.20, 1.72, 0.0, 1.66, 0.0]
_BASE_JOINTS = ["root_x_axis_joint", "root_y_axis_joint", "root_z_rotation_joint"]


def _to_np(x) -> np.ndarray:
    import torch

    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _scalar(x) -> float:
    a = _to_np(x).reshape(-1)
    return float(a[0]) if a.size else 0.0


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _relative_to_pose(anchor, rel):
    x0, y0, th0 = anchor
    lx, ly, lth = rel
    c, s = math.cos(th0), math.sin(th0)
    return np.array([x0 + c * lx - s * ly, y0 + s * lx + c * ly, _wrap(th0 + lth)])


def _pose_to_relative(anchor, target):
    x0, y0, th0 = anchor
    dx, dy = target[0] - x0, target[1] - y0
    c, s = math.cos(th0), math.sin(th0)
    return np.array([c * dx + s * dy, -s * dx + c * dy, _wrap(target[2] - th0)])


def _patch_fetch_holonomic() -> None:
    """Swap Fetch's forward-only base controller for the (vx, vy, w) one in the
    pd_joint_delta_pos mode. The base is three virtual planar joints (no wheels are
    simulated), so this is a controller choice, not a physics change -- the same
    holonomic model the SFT corpus was collected with."""
    from mani_skill.agents.controllers import PDBaseVelControllerConfig
    from mani_skill.agents.robots.fetch import fetch as fetch_mod

    if getattr(fetch_mod.Fetch, "_holonomic_patched", False):
        return
    orig = fetch_mod.Fetch._controller_configs.fget

    def patched(self):
        cfgs = orig(self)
        cfgs["pd_joint_delta_pos"]["base"] = PDBaseVelControllerConfig(
            self.base_joint_names, lower=[-2, -2, -3.14], upper=[2, 2, 3.14],   # corpus PID v_max 2.0, w_max pi
            damping=1000, force_limit=500,
            # RAW physical velocities, clipped at the bounds. The default (True) maps a
            # normalized [-1,1] action onto the bounds -- with +-2 bounds that silently
            # DOUBLED every command the tracker sent (measured: cmd 1.0 -> steady 2.0),
            # wrecking chunk-scale control. The stock +-1 bounds masked this.
            normalize_action=False)
        return cfgs

    fetch_mod.Fetch._controller_configs = property(patched)
    fetch_mod.Fetch._holonomic_patched = True


# HM3D ObjectNav category -> ReplicaCAD template-name substrings (frl_apartment_*).
_CATEGORY_TEMPLATES = {
    "chair": ["chair"],
    "sofa": ["sofa"],
    "plant": ["indoor_plant"],
    "tv_monitor": ["tv_screen"],
    "bed": ["bed"],
    "table": ["table"],
    "refrigerator": ["fridge"],
}


class MSHabEnvActor:
    def __init__(
        self,
        task: str = "tidy_house",
        subtask: str = "pick",
        split: str = "train",
        asset_dir: Optional[str] = None,
        task_plan_fp: Optional[str] = None,
        spawn_data_fp: Optional[str] = None,
        max_episode_steps: int = 200,
        gap: int = 1,
        dt: float = 0.04,
        action_mode: str = "joint",
        obs_mode: str = "rgb",
        sim_backend: str = "gpu",
        shader_dir: str = "minimal",
        width: int = 128,
        height: int = 128,
        fov: Optional[float] = None,
        camera: str = "fetch_head",
        tuck_arm: bool = False,
        head_tilt: float = 0.0,
        torso_lift: Optional[float] = None,
        instruction: Optional[str] = None,
        goal_categories: Optional[List[str]] = None,
        match_scene_instance: Optional[str] = None,
        match_episodes_json: Optional[str] = None,
        success_distance: float = 1.0,
        end_on_success: bool = True,
        seed: Optional[int] = None,
        max_plans: int = 64,
        scene_index: int = 0,
        success_bonus: float = 0.0,
        fail_penalty: float = 0.0,
        end_on_fail: bool = True,
        record_video: bool = False,
        video_every: int = 1,
        chase_camera: bool = False,
        video_layout: str = "full",
        holonomic_base: bool = False,
        episode_budget: int = 0,
        minimal_logging: bool = True,
        logging_output_dir: Optional[str] = None,
        logger_actor=None,
        **kwargs,
    ):
        self.task, self.subtask, self.split = task, subtask, split
        self.asset_dir = asset_dir
        self.task_plan_fp, self.spawn_data_fp = task_plan_fp, spawn_data_fp
        self.max_episode_steps = int(max_episode_steps)
        self.gap = int(gap)
        self.dt = float(dt)
        self.action_mode = action_mode
        self.obs_mode, self.sim_backend, self.shader_dir = obs_mode, sim_backend, shader_dir
        self.width, self.height, self.camera, self.fov = int(width), int(height), camera, fov
        self.tuck_arm, self.head_tilt = bool(tuck_arm), float(head_tilt)
        self.torso_lift = None if torso_lift is None else float(torso_lift)
        self.instruction = instruction
        self.goal_categories = list(goal_categories) if goal_categories else None
        self.match_scene_instance = match_scene_instance
        self.match_episodes_json = match_episodes_json
        self._match_eps = None
        self.success_distance = float(success_distance)
        self.end_on_success = bool(end_on_success)
        self._goal = None
        self.seed = seed
        self.max_plans = int(max_plans)
        self.scene_index = int(scene_index)
        self.success_bonus, self.fail_penalty = float(success_bonus), float(fail_penalty)
        self.end_on_fail = bool(end_on_fail)
        self.record_video, self.video_every = bool(record_video), int(video_every)
        self.chase_camera = bool(chase_camera)
        self.video_layout = str(video_layout)
        self.holonomic_base = bool(holonomic_base)
        self.episode_budget = int(episode_budget)
        self.minimal_logging = minimal_logging
        self.logging_output_dir = logging_output_dir
        self.logger_actor = logger_actor
        self.extra_kwargs = kwargs

        self._env = None
        self._shard: Optional[List[str]] = None
        self._shard_pos = 0
        self._episode_counter = 0
        self._steps = 0          # policy steps
        self._sim_steps = 0
        self._label = "unassigned"
        self._episodes_log: List[Dict[str, Any]] = []
        self._log_prefix = ""
        self._writer = None
        self._trail: List[np.ndarray] = []
        self._ep_return = 0.0

    # -- contract ---------------------------------------------------------------------------
    def assign_shard(self, episodes: Optional[List[str]] = None) -> None:
        """None = sample episodes forever (MS-HAB randomises plan + spawn per reset).
        A list of strings is treated as a budget of labels; each reset consumes one."""
        self._shard = list(episodes) if episodes is not None else None
        self._shard_pos = 0

    def is_exhausted(self) -> bool:
        if self.episode_budget and self._episode_counter >= self.episode_budget:
            return True   # hard cap for deploy/eval runs under the trivial (None) shard
        return self._shard is not None and self._shard_pos >= len(self._shard)

    def set_log_prefix(self, prefix: str) -> None:
        self._log_prefix = prefix

    def list_episode_uids(self) -> List[str]:
        return list(self._shard) if self._shard is not None else []

    def reset(self):
        self._close_video()
        if self.is_exhausted():
            rgb = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            return rgb, {"obs": {"instr_or_goal": ""}, "reward": 0.0, "done": True,
                         "is_exhausted": True, "info": self._info(None, 0.0),
                         "exhausted_sentinel": True}
        self._build()
        seed = None if self.seed is None else int(self.seed) + self._episode_counter
        obs, info = self._env.reset(seed=seed)
        self._settle_robot()
        self._apply_match()
        self._setup_goal()
        obs = self._env.unwrapped.get_obs()
        self._steps, self._sim_steps, self._ep_return = 0, 0, 0.0
        self._trail = [self._base_pose()]
        if self._shard is not None:
            self._label = self._shard[self._shard_pos]
            self._shard_pos += 1
        else:
            self._label = f"{self.task}/{self.subtask}/{self.split}#{self._episode_counter}"
        if self._goal is not None:
            self._label += f":{self._goal['category']}"
        self._episode_counter += 1
        if self.record_video and (self._episode_counter - 1) % self.video_every == 0:
            self._open_video()
        rgb = self._rgb(obs)
        self._record_frame(rgb, None, None)
        return rgb, {
            "obs": {"instr_or_goal": self._instruction()},
            "reward": 0.0,
            "done": False,
            "is_exhausted": self.is_exhausted(),
            "info": self._info(info, 0.0),
        }

    def step(self, action, supplementary_logs: Optional[Dict[str, Any]] = None):
        if self.action_mode == "base_chunk":
            obs, reward, done, info = self._step_base_chunk(action)
        else:
            obs, reward, done, info = self._step_joint(action)
        self._steps += 1
        if self._steps >= self.max_episode_steps:
            done = True
        if self._goal is not None:
            base = self._base_pose()[:2]
            eu = float(np.min(np.linalg.norm(np.asarray(self._goal["pts"])[:, :2] - base, axis=1)))
            self._goal["min_eu"] = min(self._goal.get("min_eu", eu), eu)
            d = self._goal_distance()
            if d < self._goal["min_d"]:
                self._goal["min_d"] = d
                self._goal["path_at_min"] = self._path_length()
            if d <= self.success_distance:
                self._goal["reached"] = True
                if self.end_on_success:
                    done = True
        self._ep_return += reward
        rgb = self._rgb(obs)
        out_info = self._info(info, reward)
        if done:
            self._episodes_log.append({"label": self._label, **{k: v for k, v in out_info.items()
                                                                 if np.isscalar(v)}})
            self._close_video()
        return rgb, {
            "obs": {"instr_or_goal": self._instruction()},
            "reward": float(reward),
            "done": bool(done),
            "is_exhausted": self.is_exhausted(),
            "info": out_info,
        }

    def flush_logs_to_disk(self, clear_steps: bool = True):
        if self.logging_output_dir is None or not self._episodes_log:
            return None
        os.makedirs(self.logging_output_dir, exist_ok=True)
        path = os.path.join(self.logging_output_dir,
                            f"{self._log_prefix}mshab_{os.getpid()}_{int(time.time())}.json")
        with open(path, "w") as f:
            json.dump({"episodes": self._episodes_log}, f)
        if clear_steps:
            self._episodes_log = []
        return path

    # -- stepping ------------------------------------------------------------------------
    def _sim_step(self, vec: np.ndarray):
        import torch

        a = torch.as_tensor(np.asarray(vec, np.float32), device=self._env.unwrapped.device)[None]
        obs, r, terminated, truncated, info = self._env.step(a)
        self._sim_steps += 1
        return obs, _scalar(r), info

    def _terminal(self, info) -> (bool, float):
        success = bool(_scalar(info.get("success", 0)))
        fail = bool(_scalar(info.get("fail", 0)))
        if success:
            return True, self.success_bonus
        if fail and self.end_on_fail:
            return True, -self.fail_penalty
        return False, 0.0

    def _step_joint(self, action):
        chunk = np.asarray(action, dtype=np.float32).reshape(-1, self._act_dim)
        if len(chunk) != self.gap:
            raise ValueError(f"expected an action chunk of {self.gap} rows, got {len(chunk)}")
        reward, done, info, obs = 0.0, False, None, None
        for row in chunk:
            obs, r, info = self._sim_step(row)
            reward += r
            self._record_frame(self._rgb(obs), None, None)
            done, bonus = self._terminal(info)
            reward += bonus
            if done:
                break
        return obs, reward, done, info

    def _step_base_chunk(self, action):
        """Track a (gap, 3) anchor-relative SE(2) chunk with the forward/yaw base controller."""
        chunk = np.asarray(action, dtype=np.float64).reshape(-1, 3)
        if len(chunk) != self.gap:
            raise ValueError(
                f"env gap={self.gap} but received a {len(chunk)}-row chunk; the head and the "
                "env must agree on ticks-per-step (dt=0.04, gap=10 for the SFT policies).")
        anchor = self._base_pose()
        world = np.stack([_relative_to_pose(anchor, row) for row in chunk])   # (gap, 3)
        prev = np.vstack([anchor[None], world[:-1]])
        # Feedforward per tick: forward speed along the previous heading, yaw rate.
        d = world[:, :2] - prev[:, :2]
        v_ff = (d[:, 0] * np.cos(prev[:, 2]) + d[:, 1] * np.sin(prev[:, 2])) / self.dt
        vy_ff = (-d[:, 0] * np.sin(prev[:, 2]) + d[:, 1] * np.cos(prev[:, 2])) / self.dt
        w_ff = np.array([_wrap(a - b) for a, b in zip(world[:, 2], prev[:, 2])]) / self.dt
        ctrl_dt = 1.0 / self._control_freq
        n_sim = max(1, int(round(self.gap * self.dt * self._control_freq)))
        k_v, k_y, k_w = 3.0, 2.0, 4.0
        reward, done, info, obs = 0.0, False, None, None
        for k in range(n_sim):
            t = (k + 1) * ctrl_dt
            i = min(self.gap - 1, max(0, int(math.ceil(t / self.dt - 1e-9)) - 1))
            err = _pose_to_relative(self._base_pose(), world[i])       # in current body frame
            vmax = 2.0 if self.holonomic_base else 1.0
            v = float(np.clip(k_v * err[0] + v_ff[i], -vmax, vmax))
            vec = np.zeros(self._act_dim, np.float32)
            if self.holonomic_base:
                # (vx, vy, w): lateral tracked directly, yaw no longer steers toward ey
                vy = float(np.clip(k_v * err[1] + vy_ff[i], -2.0, 2.0))
                w = float(np.clip(k_w * err[2] + w_ff[i], -3.14, 3.14))
                vec[self._base_slice] = [v, vy, w]
            else:
                w = float(np.clip(k_w * err[2] + k_y * err[1] + w_ff[i], -3.14, 3.14))
                vec[self._base_slice] = [v, w]
            obs, r, info = self._sim_step(vec)
            reward += r
            self._trail.append(self._base_pose())
            self._record_frame(self._rgb(obs), world, anchor)
            done, bonus = self._terminal(info)
            reward += bonus
            if done:
                break
        return obs, reward, done, info

    # -- construction, lazily inside the actor process ---------------------------------------
    def _build(self) -> None:
        if self._env is not None:
            return
        if self.asset_dir:
            os.environ["MS_ASSET_DIR"] = self.asset_dir
        # A VNC X server on DISPLAY makes SAPIEN's Vulkan init block forever (headless
        # rendering does not need X). Drop it before sapien is imported.
        os.environ.pop("DISPLAY", None)
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401  (registers envs)
        import mshab.envs  # noqa: F401  (registers MS-HAB envs)
        from mani_skill import ASSET_DIR
        from mshab.envs.planner import plan_data_from_file

        if self.holonomic_base:
            _patch_fetch_holonomic()
        rearrange = ASSET_DIR / "scene_datasets/replica_cad_dataset/rearrange"
        tp_fp = self.task_plan_fp or str(
            rearrange / "task_plans" / self.task / self.subtask / self.split / "all.json")
        sd_fp = self.spawn_data_fp or str(
            rearrange / "spawn_data" / self.task / self.subtask / self.split / "spawn_data.pt")
        plan_data = plan_data_from_file(tp_fp)
        plans = plan_data.plans
        # 100k plans per split over ~40 scenes (build configs). With num_envs=1 MS-HAB
        # requires the plan pool to cover exactly one build config, so pick the
        # `scene_index`-th scene (sorted by name) and cap its plans at `max_plans`.
        bcs = sorted({tp.build_config_name for tp in plans})
        bc = bcs[int(self.scene_index) % len(bcs)]
        plans = [tp for tp in plans if tp.build_config_name == bc]
        if self.max_plans and len(plans) > self.max_plans:
            rng = np.random.RandomState(0 if self.seed is None else int(self.seed))
            plans = [plans[i] for i in sorted(rng.choice(len(plans), self.max_plans, replace=False))]
        self._scene_name = bc
        cam_cfg: Dict[str, Any] = dict(width=self.width, height=self.height)
        if self.fov is not None:
            cam_cfg["fov"] = float(self.fov)
        env_kwargs: Dict[str, Any] = dict(
            task_plans=plans,
            scene_builder_cls=plan_data.dataset,
            sensor_configs={self.camera: cam_cfg},
        )
        if self.subtask != "sequential":
            env_kwargs["spawn_data_fp"] = sd_fp
        env_kwargs.update(self.extra_kwargs)
        self._env = gym.make(
            _TASK_TO_ENV_ID[self.subtask],
            max_episode_steps=self.max_episode_steps * max(1, self.gap) * 4,  # ours governs
            obs_mode=self.obs_mode,
            reward_mode="normalized_dense" if self.subtask != "sequential" else "sparse",
            control_mode="pd_joint_delta_pos",
            render_mode="rgb_array",
            shader_dir=self.shader_dir,
            robot_uids="fetch",
            num_envs=1,
            sim_backend=self.sim_backend,
            **env_kwargs,
        )
        u = self._env.unwrapped
        self._act_dim = int(self._env.action_space.shape[-1])
        self._control_freq = float(u.control_freq)
        # Slices of the flat action per sub-controller, in CombinedController order.
        self._slices = {}
        off = 0
        for name, c in u.agent.controller.controllers.items():
            n = int(np.prod(c.action_space.shape[-1:]))
            self._slices[name] = slice(off, off + n)
            off += n
        assert off == self._act_dim, (self._slices, self._act_dim)
        self._base_slice = self._slices["base"]
        names = [j.name for j in u.agent.robot.active_joints]
        self._base_idx = [names.index(n) for n in _BASE_JOINTS]
        self._arm_idx = [names.index(n) for n in _ARM_JOINTS]
        self._head_tilt_idx = names.index("head_tilt_joint")
        self._torso_idx = names.index("torso_lift_joint")
        self._plan_data = plan_data
        self._chase = None
        if self.record_video and self.chase_camera:
            self._add_chase_camera()

    def _settle_robot(self) -> None:
        """Tuck the arm / level the head so the head camera sees the room, not the gripper.
        The delta-position controllers hold whatever qpos they start from."""
        if not self.tuck_arm and self.head_tilt == 0.0 and self.torso_lift is None:
            return
        import torch

        u = self._env.unwrapped
        q = u.agent.robot.get_qpos()
        q = q.clone()
        if self.tuck_arm:
            q[:, self._arm_idx] = torch.tensor(_TUCK_QPOS, dtype=q.dtype, device=q.device)
        q[:, self._head_tilt_idx] = self.head_tilt
        if self.torso_lift is not None:
            q[:, self._torso_idx] = self.torso_lift
        u.agent.robot.set_qpos(q)
        u.agent.robot.set_qvel(torch.zeros_like(u.agent.robot.get_qvel()))
        if u.gpu_sim_enabled:
            u.scene._gpu_apply_all()
            u.scene.px.gpu_update_articulation_kinematics()
            u.scene._gpu_fetch_all()
        # a few zero-action steps so the PD targets latch onto the new pose
        for _ in range(3):
            self._sim_step(np.zeros(self._act_dim, np.float32))
        self._sim_steps = 0
        try:
            cam = u.scene.sensors[self.camera].camera
            z = float(_to_np(cam.get_model_matrix()).reshape(-1, 4, 4)[0][2, 3])
            print(f"[mshab] {self.camera} world height {z:.2f} m (corpus sensor: 0.88 m)")
        except Exception as exc:
            print(f"[mshab] camera height unavailable: {exc}")

    def _base_pose(self) -> np.ndarray:
        """WORLD planar pose of the base link (x, y, yaw). The root joints' qpos is relative
        to the articulation root, which MS-HAB places at the spawn -- not world."""
        pose = self._env.unwrapped.agent.base_link.pose
        p = _to_np(pose.p).reshape(-1, 3)[0]
        w, x, y, z = _to_np(pose.q).reshape(-1, 4)[0]
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return np.array([p[0], p[1], _wrap(yaw)])

    def _rgb(self, obs) -> np.ndarray:
        frame = _to_np(obs["sensor_data"][self.camera]["rgb"])
        if frame.ndim == 4:
            frame = frame[0]
        return np.ascontiguousarray(frame[..., :3], dtype=np.uint8)

    def _instruction(self) -> str:
        if self._goal is not None:
            return self._goal["category"]
        if self.instruction:
            return self.instruction
        env = self._env.unwrapped
        try:
            st = env.task_plan[0]
            obj = getattr(st, "obj_id", None)
            name = obj.rsplit("-", 1)[0].split("_", 1)[-1].replace("_", " ") if obj else None
        except Exception:
            name = None
        verbs = {"pick": "Pick up", "place": "Place", "open": "Open", "close": "Close",
                 "navigate": "Navigate to"}
        if self.subtask == "sequential":
            return f"Complete the {self.task.replace('_', ' ')} task."
        return f"{verbs.get(self.subtask, self.subtask)} the {name or 'target object'}."

    def _info(self, info, reward: float) -> Dict[str, Any]:
        g = (lambda k: _scalar(info.get(k, 0))) if info is not None else (lambda k: 0.0)
        pose = self._base_pose() if self._env is not None else np.zeros(3)
        # Uniform key set every step -- terminal-only keys break trajectory packing.
        return {
            "episode_label": self._label,
            "success": float(g("success")),
            "fail": float(g("fail")),
            "is_grasped": float(g("is_grasped")),
            "subtask_steps_left": float(g("subtask_steps_left")),
            "truncated": float(self._steps >= self.max_episode_steps),
            "steps_sim": float(self._sim_steps),
            "ep_return": float(self._ep_return),
            "base_x": float(pose[0]), "base_y": float(pose[1]), "base_yaw": float(pose[2]),
            "path_length_m": self._path_length(),
            "distance_to_goal": float(self._goal_distance()) if self._goal else float("nan"),
            "start_distance": float(self._goal["start_d"]) if self._goal else float("nan"),
            "min_distance_to_goal": float(self._goal["min_d"]) if self._goal else float("nan"),
            "oracle_success": float(self._goal["reached"]) if self._goal else 0.0,
            "min_euclid_to_goal": float(self._goal.get("min_eu", float("nan"))) if self._goal else float("nan"),
            "ospl_fix": (float(self._goal["reached"]) * self._goal["start_d"]
                         / max(self._goal["path_at_min"], self._goal["start_d"], 1e-6)) if self._goal else 0.0,
        }

    def _path_length(self) -> float:
        return float(sum(np.linalg.norm(b[:2] - a[:2])
                         for a, b in zip(self._trail[:-1], self._trail[1:])))

    # -- ObjectNav goals on top of the navigate env -------------------------------------
    def _setup_goal(self) -> None:
        """Pick this episode's category (cycled per episode), collect the matching scene
        instances, and initialise the distance bookkeeping. Requires the navigate env
        (its precomputed floor-map all-pairs table gives approximate geodesics)."""
        self._goal = None
        mg = getattr(self, "_match_goal", None)
        if self.match_episodes_json and mg is not None:
            self._label = mg["label"]      # the generic label path ran before this; override
            self._goal = {"category": mg["category"], "pts": mg["pts"], "reached": False}
            d0 = self._goal_distance()
            self._goal.update(start_d=float(d0), min_d=float(d0), path_at_min=0.0)
            return
        if not self.goal_categories:
            return
        u = self._env.unwrapped
        cat = self.goal_categories[(self._episode_counter) % len(self.goal_categories)]
        keys = _CATEGORY_TEMPLATES.get(cat, [cat])
        sb = u.scene_builder
        pts = []
        for reg in (getattr(sb, "scene_objects", {}), getattr(sb, "movable_objects", {}),
                    getattr(sb, "articulations", {})):
            for name, obj in reg.items():
                if name.startswith("env-0_") and any(k in name for k in keys):
                    pts.append(_to_np(obj.pose.p).reshape(-1, 3)[0][:2])
        if not pts:
            print(f"[mshab] no instances of '{cat}' in {self._scene_name}; goal disabled this episode")
            return
        self._goal = {"category": cat, "pts": np.array(pts), "reached": False}
        d0 = self._goal_distance()
        self._goal.update(start_d=float(d0), min_d=float(d0), path_at_min=0.0)

    @staticmethod
    def _hab_to_sapien_pose(t, q):
        """EXACTLY the ReplicaCAD SceneBuilder's conversion (scene_builder.py ~L160):
        all ReplicaCAD assets are y-up; left-multiply by the fixed +90deg-about-x
        quaternion, json rotation used raw (wxyz)."""
        import sapien
        import transforms3d

        q_conv = transforms3d.quaternions.axangle2quat(np.array([1.0, 0, 0]), theta=np.deg2rad(90))
        if q is None:
            q = [1.0, 0.0, 0.0, 0.0]
        return sapien.Pose(q=q_conv) * sapien.Pose(list(t), list(q))

    def _apply_match(self) -> None:
        """Episode-matched mode: restore the habitat staging furniture arrangement and
        serve the habitat-generated episode starts/goals (docs/MSHAB_INTEGRATION.md).
        Requires scipy (a mani_skill dependency)."""
        if not self.match_scene_instance:
            return
        import gzip
        import torch
        from mani_skill.utils.structs.pose import Pose as MsPose

        u = self._env.unwrapped
        sb = u.scene_builder
        inst = json.load(open(self.match_scene_instance))["object_instances"]
        regs = {}
        for reg in (getattr(sb, "scene_objects", {}), getattr(sb, "movable_objects", {})):
            for name, obj in reg.items():
                if name.startswith("env-0_"):
                    regs.setdefault(name.split("env-0_")[1].rsplit("-", 1)[0], []).append(
                        (int(name.rsplit("-", 1)[1]), obj))
        for v in regs.values():
            v.sort()
        used = {k: 0 for k in regs}
        parked = 0
        placed = set()
        park_z = -900.0
        for o in inst:
            tmpl = o["template_name"].split("/")[-1]
            pool = regs.get(tmpl)
            if not pool or used[tmpl] >= len(pool):
                continue
            idx, obj = pool[used[tmpl]]
            used[tmpl] += 1
            sp = self._hab_to_sapien_pose(o["translation"], o.get("rotation"))
            obj.set_pose(MsPose.create(sp))
            try:                                   # dynamic actors: kill any spawn velocity
                obj.set_linear_velocity(torch.zeros(1, 3))
                obj.set_angular_velocity(torch.zeros(1, 3))
            except Exception:
                pass
            placed.add(f"env-0_{tmpl}-{idx}")
        # Park every unmatched frl_* / YCB movable far away (visual parity with habitat).
        for reg in (getattr(sb, "scene_objects", {}), getattr(sb, "movable_objects", {})):
            for name, obj in reg.items():
                if not name.startswith("env-0_") or name in placed:
                    continue
                base = name.split("env-0_")[1].rsplit("-", 1)[0]
                if base.startswith("frl_apartment_") or base[:3].isdigit():
                    park_z -= 3.0
                    obj.set_pose(MsPose.create_from_pq(
                        p=torch.tensor([0.0, 0.0, park_z], dtype=torch.float32)[None]))
                    parked += 1
        if u.gpu_sim_enabled:
            u.scene._gpu_apply_all()
            u.scene.px.step()
            u.scene._gpu_fetch_all()
        # Serve the habitat episode: start pose + category + goal points.
        if self.match_episodes_json:
            if self._match_eps is None:
                self._match_eps = json.load(gzip.open(self.match_episodes_json, "rt"))["episodes"]
            # NOTE: _episode_counter has NOT been incremented yet at this point.
            ep = self._match_eps[self._episode_counter % len(self._match_eps)]
            sp = ep["start_position"]                      # habitat [x, y, z]
            X, Y = float(sp[0]), -float(sp[2])
            # yaw from the habitat quat (about +y): forward = R * (0, 0, -1)
            from scipy.spatial.transform import Rotation as _R
            f = _R.from_quat([ep["start_rotation"][0], ep["start_rotation"][1],
                              ep["start_rotation"][2], ep["start_rotation"][3]]).apply([0, 0, -1.0])
            th = math.atan2(-f[2], f[0])
            q = u.agent.robot.get_qpos().clone()
            root_p = _to_np(u.agent.robot.pose.p).reshape(-1)[:3]
            q[:, self._base_idx[0]] = X - root_p[0]
            q[:, self._base_idx[1]] = Y - root_p[1]
            q[:, self._base_idx[2]] = th
            u.agent.robot.set_qpos(q)
            if u.gpu_sim_enabled:
                u.scene._gpu_apply_all()
                u.scene.px.gpu_update_articulation_kinematics()
                u.scene._gpu_fetch_all()
            # THE habitat ObjectNav success surface, ported verbatim: the episode's own
            # view points (navigable poses around each goal instance), habitat frame
            # (x, y, z) -> planar (x, -z). Success = geodesic to nearest view point
            # <= success_distance, identical to the habitat eval's VIEW_POINTS rule
            # (floor-map geodesic here; view points are navigable, so the vert-snap
            # residual is small, unlike object centers).
            vps = [vp["agent_state"]["position"]
                   for g in ep["goals"] for vp in g.get("view_points", [])]
            if not vps:
                vps = [g["position"] for g in ep["goals"]]
            self._match_goal = {
                "label": f"matched:{ep['episode_id']}:{ep['object_category']}",
                "category": ep["object_category"],
                "pts": np.array([[v[0], -v[2]] for v in vps]),
            }
            self._trail = [self._base_pose()]

    def _goal_distance(self) -> float:
        """Min over instances of (floor-map geodesic agent->goal), MS-HAB's navigate metric."""
        import torch

        u = self._env.unwrapped
        g = self._goal
        try:
            verts = u.floor_map_verts[int(u.env_idx_to_floor_map[0])]
            dists = u.floor_map_all_pairs_dists[int(u.env_idx_to_floor_map[0])]
            a = torch.as_tensor(self._base_pose()[:2], device=verts.device, dtype=verts.dtype)
            a_idx = int(torch.argmin(torch.norm(verts - a, dim=1)))
            best = float("inf")
            for p in g["pts"]:
                gp = torch.as_tensor(p, device=verts.device, dtype=verts.dtype)
                dv = torch.norm(verts - gp, dim=1)
                g_idx = int(torch.argmin(dv))
                best = min(best, float(dists[a_idx, g_idx]) + float(dv[g_idx]))
            return best
        except Exception:
            base = self._base_pose()[:2]
            return float(np.min(np.linalg.norm(g["pts"] - base, axis=1)))

    # -- video: third-person view(s) with the 3-D action chunk projected in --------------
    def _add_chase_camera(self) -> None:
        """A raw SAPIEN camera mounted behind/above the base, added outside ManiSkill's
        camera registry so it never touches the policy's sensors."""
        try:
            import sapien

            u = self._env.unwrapped
            sub = u.scene.sub_scenes[0]
            base = u.agent.robot.links_map["base_link"]._objs[0]
            cam = sub.add_mounted_camera(
                name="chase", mount=base.entity, pose=sapien.Pose([-2.2, 0.0, 1.9], [1, 0, 0, 0]),
                width=512, height=512, fovy=1.1, near=0.05, far=30.0)
            # pitch the camera down ~28 deg (rotation about y): quaternion [w, x, y, z]
            ang = 0.45
            cam.set_local_pose(sapien.Pose([-2.2, 0.0, 1.9], [math.cos(ang / 2), 0, math.sin(ang / 2), 0]))
            self._chase = cam
        except Exception as exc:                        # never let the overlay kill a rollout
            print(f"[mshab] chase camera unavailable: {exc}")
            self._chase = None

    def _open_video(self) -> None:
        if self.logging_output_dir is None:
            return
        import imageio

        vdir = os.path.join(self.logging_output_dir, "videos")
        os.makedirs(vdir, exist_ok=True)
        safe = self._label.replace("/", "_").replace("#", "_")
        self._video_path = os.path.join(vdir, f"{self._log_prefix}{safe}.mp4")
        self._writer = imageio.get_writer(self._video_path, fps=int(self._control_freq),
                                          codec="libx264", quality=7, macro_block_size=None)

    def _close_video(self) -> None:
        if self._writer is not None:
            self._writer.close()
            print(f"[mshab] wrote {self._video_path}")
            self._writer = None

    def _project(self, cam_params, pts_world: np.ndarray) -> np.ndarray:
        """(N, 3) world -> (N, 3) [u, v, depth] with OpenCV extrinsic/intrinsic."""
        E = _to_np(cam_params["extrinsic_cv"]); K = _to_np(cam_params["intrinsic_cv"])
        E = E[0] if E.ndim == 3 else E
        K = K[0] if K.ndim == 3 else K
        P = np.concatenate([pts_world, np.ones((len(pts_world), 1))], 1) @ E[:3].T
        z = P[:, 2:3]
        uv = (P @ K.T)[:, :2] / np.where(np.abs(z) < 1e-6, 1e-6, z)
        return np.concatenate([uv, z], 1)

    def _draw_chunk(self, img: np.ndarray, cam_params, world_chunk, anchor) -> np.ndarray:
        from PIL import Image, ImageDraw

        im = Image.fromarray(img)
        dr = ImageDraw.Draw(im)
        z0 = 0.03
        # executed trail (grey) -- the 3-D path the base actually drove
        if len(self._trail) > 1:
            tr = np.array([[p[0], p[1], z0] for p in self._trail[-400:]])
            uvz = self._project(cam_params, tr)
            pts = [(float(u), float(v)) for u, v, z in uvz if z > 0.05]
            if len(pts) > 1:
                dr.line(pts, fill=(200, 200, 200), width=2)
        if world_chunk is not None:
            pts3 = np.concatenate([[[anchor[0], anchor[1], z0]],
                                   [[p[0], p[1], z0] for p in world_chunk]], 0)
            uvz = self._project(cam_params, pts3)
            pts = [(float(u), float(v)) for u, v, z in uvz if z > 0.05]
            if len(pts) > 1:
                dr.line(pts, fill=(255, 80, 0), width=4)
            # per-tick heading ticks + posts so the chunk reads as 3-D
            for (x, y, th), (u, v, z) in zip(world_chunk, uvz[1:]):
                if z <= 0.05:
                    continue
                head = np.array([[x + 0.12 * math.cos(th), y + 0.12 * math.sin(th), z0],
                                 [x, y, z0 + 0.25]])
                h = self._project(cam_params, head)
                if h[0, 2] > 0.05:
                    dr.line([(u, v), (h[0, 0], h[0, 1])], fill=(0, 200, 255), width=2)
                if h[1, 2] > 0.05:
                    dr.line([(u, v), (h[1, 0], h[1, 1])], fill=(255, 220, 0), width=1)
                dr.ellipse([u - 3, v - 3, u + 3, v + 3], fill=(255, 80, 0))
        return np.asarray(im)

    def _chunk_panel(self, world_chunk, anchor, size: int) -> np.ndarray:
        """Top-down body-frame plot of the chunk: x forward = up, y left = left."""
        from PIL import Image, ImageDraw

        im = Image.new("RGB", (size, size), (24, 24, 28))
        dr = ImageDraw.Draw(im)
        c = size // 2
        ppm = size / 2.0                                  # 1 m half-extent
        for r in (0.25, 0.5, 0.75, 1.0):
            dr.ellipse([c - r * ppm, c - r * ppm, c + r * ppm, c + r * ppm], outline=(60, 60, 70))
        dr.line([(c, 0), (c, size)], fill=(60, 60, 70)); dr.line([(0, c), (size, c)], fill=(60, 60, 70))
        dr.polygon([(c, c - 8), (c - 5, c + 6), (c + 5, c + 6)], fill=(120, 120, 130))
        if world_chunk is not None:
            rel = [_pose_to_relative(anchor, p) for p in world_chunk]
            pts = [(c, c)] + [(c - y * ppm, c - x * ppm) for x, y, th in rel]
            dr.line(pts, fill=(255, 80, 0), width=3)
            for (x, y, th), (u, v) in zip(rel, pts[1:]):
                dr.line([(u, v), (u - 12 * math.sin(th), v - 12 * math.cos(th))], fill=(0, 200, 255), width=2)
                dr.ellipse([u - 2, v - 2, u + 2, v + 2], fill=(255, 80, 0))
            dr.text((6, 6), "chunk (body frame) 1m ring", fill=(200, 200, 200))
            dr.text((6, size - 30), f"dx={rel[-1][0]:+.2f} dy={rel[-1][1]:+.2f}", fill=(255, 160, 80))
            dr.text((6, size - 16), f"dth={math.degrees(rel[-1][2]):+.1f}deg", fill=(255, 160, 80))
        return np.asarray(im)

    def _record_frame(self, head_rgb: np.ndarray, world_chunk, anchor) -> None:
        if self._writer is None:
            return
        from PIL import Image, ImageDraw

        if self.video_layout == "headcam":
            self._writer.append_data(np.ascontiguousarray(head_rgb))
            return
        u = self._env.unwrapped
        third = _to_np(u.render_rgb_array())
        third = third[0] if third.ndim == 4 else third
        third = np.ascontiguousarray(third[..., :3], dtype=np.uint8)
        H = third.shape[0]
        cam = u.scene.human_render_cameras["render_camera"]
        third = self._draw_chunk(third, cam.get_params(), world_chunk, anchor)
        panels = [third]
        if self._chase is not None:
            try:
                self._chase.take_picture()
                ch = self._chase.get_picture("Color")
                ch = np.clip(_to_np(ch)[..., :3] * 255, 0, 255).astype(np.uint8)
                params = dict(extrinsic_cv=self._chase.get_extrinsic_matrix(),
                              intrinsic_cv=self._chase.get_intrinsic_matrix())
                ch = self._draw_chunk(ch, params, world_chunk, anchor)
                if ch.shape[0] != H:
                    ch = np.asarray(Image.fromarray(ch).resize((int(ch.shape[1] * H / ch.shape[0]), H)))
                panels.append(ch)
            except Exception as exc:
                print(f"[mshab] chase render failed: {exc}")
                self._chase = None
        half = H // 2
        head = np.asarray(Image.fromarray(head_rgb).resize((half, half)))
        side = np.concatenate([head, self._chunk_panel(world_chunk, anchor, half)], 0)
        frame = np.concatenate(panels + [side], 1)
        im = Image.fromarray(frame)
        dr = ImageDraw.Draw(im)
        dr.text((8, 8), f"{self._label}  step {self._steps}  sim {self._sim_steps}  "
                        f"{self._instruction()}", fill=(255, 255, 255))
        self._writer.append_data(np.asarray(im))
