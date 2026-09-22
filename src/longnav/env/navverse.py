
import numbers
import os
import time

import numpy as np
import torch


class NavverseEnv:
    def __init__(self):

        def is_linux_headless():
            # Returns True if no display environment variable is set
            has_x11 = "DISPLAY" in os.environ
            has_wayland = "WAYLAND_DISPLAY" in os.environ
            return not (has_x11 or has_wayland)

        import argparse

        # start simulation
        from isaaclab.app import AppLauncher
        import navverse.utils.rsl_rl_cli_args as rsl_rl_cli_args
        import navverse.vln_args as vln_cli_args
        # Add command line arguments
        parser = argparse.ArgumentParser(description="Benchmark")
        rsl_rl_cli_args.add_rsl_rl_args(parser)
        vln_cli_args.add_vln_args(parser)
        AppLauncher.add_app_launcher_args(parser)
        import sys

        # 1. Back up the original arguments
        original_argv = sys.argv.copy()

        navverse_config = os.environ.get("NAVVERSE_CONFIG", "configs/default.yaml")
        navverse_args = [
            "--episode_folder",
            os.environ.get("NAVVERSE_EPISODE_FOLDER", "episodes"),
            "--scene_folder",
            os.environ.get("NAVVERSE_SCENE_FOLDER", "navverse_data"),
            "--task_type",
            os.environ.get("NAVVERSE_TASK_TYPE", "placenav"),
            "--on_demand_render",
            os.environ.get("NAVVERSE_ON_DEMAND_RENDER", "False"),
            "--disable_camera",
            os.environ.get("NAVVERSE_DISABLE_CAMERA", "False"),
            "--profile_step_timing",
            os.environ.get("NAVVERSE_PROFILE_STEP_TIMING", "False"),
            "--profile_step_interval",
            os.environ.get("NAVVERSE_PROFILE_STEP_INTERVAL", "100"),
        ]
        if navverse_config.lower() != "auto":
            navverse_args.extend(["--config", navverse_config])
        vector_envs = int(os.environ.get("NAVVERSE_VECTOR_ENVS", "1"))
        navverse_args.extend(["--num_envs", str(vector_envs)])
        trajectory_planner = os.environ.get("NAVVERSE_TRAJECTORY_PLANNER")
        if trajectory_planner:
            navverse_args.extend(["--trajectory_planner", trajectory_planner])

        # 2. Replace sys.argv with only NavVerse args while its parser runs.
        sys.argv = [sys.argv[0], *navverse_args]

        # -- Your code runs here with cleared arguments --
        args = vln_cli_args.parse_args(parser)

        # 3. Restore the original arguments when done
        sys.argv = original_argv
        args.disable_socket_server = True
        args.headless = is_linux_headless()
        # Launch Isaac Lab app
        app_launcher = AppLauncher(args)
        self.simulation_app = app_launcher.app

        # Enable Extension and setup settings
        import omni.kit.app
        # from isaacsim.core.utils.extensions import enable_extension
        # enable_extension("omni.anim.navigation.bundle")
        # settings.set("/renderer/multiGPU/enabled", False)
        # settings.set("/renderer/activeGpu", 0) 

        omni.kit.app.get_app().update()
        # Local imports
        from navverse.sim import VLNSim

        # setup simulation
        self.vector_envs = vector_envs
        self.vln_sim = VLNSim(args)
        self.hide_vector_robot_visuals = (
            vector_envs > 1
            and os.environ.get("NAVVERSE_HIDE_VECTOR_ROBOTS", "1").lower()
            in {"1", "true", "yes"}
        )
        self.vector_robot_visuals_hidden = False
        if vector_envs > 1:
            self.vln_sim.update_obs = lambda obs, info: None
        self.action_names = ["STOP", "FORWARD", "TURN_LEFT", "TURN_RIGHT"]
        self.action_timeout_sim_s = float(os.environ.get("NAVVERSE_ACTION_TIMEOUT_SIM_S", "2.0"))
        self.shaped_reward = os.environ.get("NAVVERSE_RL_SHAPED_REWARD", "0").lower() in {
            "1",
            "true",
            "yes",
        }
        self.progress_reward_scale = float(
            os.environ.get("NAVVERSE_RL_PROGRESS_REWARD_SCALE", "1.0")
        )
        self.slack_reward = float(os.environ.get("NAVVERSE_RL_SLACK_REWARD", "-0.01"))
        self.success_reward = float(os.environ.get("NAVVERSE_RL_SUCCESS_REWARD", "2.5"))
        self.exploration_bonus = float(
            os.environ.get("NAVVERSE_RL_EXPLORATION_BONUS", "0.13")
        )
        self.collision_penalty = float(
            os.environ.get("NAVVERSE_RL_COLLISION_PENALTY", "0.05")
        )
        self.false_stop_penalty = float(
            os.environ.get("NAVVERSE_RL_FALSE_STOP_PENALTY", "0.3")
        )
        self.previous_distance_to_goal = None
        self.previous_covered_area = None
        self.episode_ptr = 0
        self.episodes = []
        self.vector_episodes = []
        self.vector_done = None
        self.vector_previous_distance = None
        self.vector_previous_covered_cells = None
        self.vector_path_length = None
        self.vector_start_distance = None
        self.vector_previous_pose = None
        self.vector_followers = None
        self.vector_latest_obs = None
        self.benchmark_video_config = None
        self.benchmark_video_frames = None
        self.benchmark_video_trajectories = None
        self.vector_profile_stats = {}
        self.manager_profile_stats = {}
        self.manager_profile_hooks_installed = False

    def _install_manager_profile_hooks(self):
        if self.manager_profile_hooks_installed:
            return
        manager_env = self.vln_sim.manager_env
        hooks = (
            (manager_env.action_manager, "process_action", "action/process"),
            (manager_env.action_manager, "apply_action", "action/apply"),
            (manager_env.scene, "write_data_to_sim", "scene/write"),
            (manager_env.sim, "step", "sim/physics_step"),
            (manager_env.sim, "render", "sim/render"),
            (manager_env.scene, "update", "scene/update"),
            (manager_env.observation_manager, "compute", "manager/observation"),
            (manager_env.termination_manager, "compute", "manager/termination"),
            (manager_env.reward_manager, "compute", "manager/reward"),
        )
        for owner, method_name, profile_key in hooks:
            original = getattr(owner, method_name)

            def profiled(*args, _original=original, _key=profile_key, **kwargs):
                started = time.perf_counter()
                try:
                    return _original(*args, **kwargs)
                finally:
                    total_s, count = self.manager_profile_stats.get(
                        _key,
                        (0.0, 0),
                    )
                    self.manager_profile_stats[_key] = (
                        total_s + time.perf_counter() - started,
                        count + 1,
                    )

            setattr(owner, method_name, profiled)
        self.manager_profile_hooks_installed = True

    def _record_vector_profile(self, key, duration_s, count=1):
        if not self.vln_sim.args.profile_step_timing:
            return
        total_s, total_count = self.vector_profile_stats.get(key, (0.0, 0))
        self.vector_profile_stats[key] = (
            total_s + float(duration_s),
            total_count + int(count),
        )

    def get_vector_profile(self, reset=True):
        profile = {}
        sources = (
            ("vector", self.vector_profile_stats),
            ("manager", self.manager_profile_stats),
            ("sim", self.vln_sim.profile_stats),
            ("wrapper", self.vln_sim.env.profile_stats),
        )
        for prefix, stats in sources:
            for key, (total_s, count) in stats.items():
                profile[f"{prefix}/{key}"] = {
                    "total_seconds": float(total_s),
                    "count": int(count),
                    "mean_seconds": float(total_s) / max(int(count), 1),
                }
            if reset:
                stats.clear()
        return profile

    def _set_vector_robot_visuals_visible(self, visible, env_indices=None):
        from pxr import UsdGeom

        if self.vln_sim.manager_env is None:
            raise RuntimeError(
                "Cannot change vector robot visibility before the scene is initialized"
            )
        stage = self.vln_sim.manager_env.scene.stage
        indices = (
            range(self.vector_envs)
            if env_indices is None
            else tuple(int(index) for index in env_indices)
        )
        updated_roots = 0
        for env_index in indices:
            if env_index < 0 or env_index >= self.vector_envs:
                raise IndexError(f"Vector environment index is invalid: {env_index}")
            robot_prim = stage.GetPrimAtPath(f"/World/envs/env_{env_index}/Robot")
            if not robot_prim.IsValid():
                raise RuntimeError(
                    f"Vector robot prim is missing for env {env_index}"
                )
            imageable = UsdGeom.Imageable(robot_prim)
            if not imageable:
                raise RuntimeError(f"Vector robot prim is not imageable for env {env_index}")
            if visible:
                imageable.MakeVisible()
            else:
                imageable.MakeInvisible()
            updated_roots += 1
        if env_indices is None:
            self.vector_robot_visuals_hidden = not visible
        return updated_roots

    def _hide_vector_robot_visuals(self):
        hidden_roots = self._set_vector_robot_visuals_visible(False)
        print(
            f"[SIM] Hid {hidden_roots} vector robot render roots; "
            "physics and cameras remain active"
        )
   
    def assign_shard(self, episodes: list[str]|None = None):
        '''
        assign a list of episodes identified via strings to the actor.
        if None is passed, load all available episodes.
        '''
        self.episodes = list(episodes or [])
        self.episode_ptr = 0
        
    def flush_logs_to_disk(self):
        '''
        flush any internal logging. returns either None or a path pointing to a json file.
        '''
        pass
    
    def is_exhausted(self):
        '''
        returns True if the actor has exhausted its assigned episodes.
        '''
        return self.episode_ptr >= len(self.episodes)

    def _sim_time(self):
        manager_env = getattr(self.vln_sim, "manager_env", None)
        if manager_env is None:
            return 0.0
        return float(getattr(manager_env, "common_step_counter", 0) or 0) * float(
            getattr(manager_env, "step_dt", 0.0) or 0.0
        )

    def _latest_state_tuple(self, default_reward=0.0, default_done=False):
        obs = self.vln_sim.obs
        reward = self.vln_sim.reward
        done = self.vln_sim.done
        info = self.vln_sim.info or {}
        if reward is None:
            reward = default_reward
        if done is None:
            done = default_done
        return obs, reward, done, info

    def _step_until(self, predicate, timeout_sim_s=None):
        start_sim_time = self._sim_time()
        while True:
            self.vln_sim.step()
            if predicate():
                break
            if self.vln_sim.sim_state == "terminated":
                break
            if timeout_sim_s is not None and self._sim_time() - start_sim_time >= timeout_sim_s:
                self.vln_sim.clear_waypoints()
                break
        return self._latest_state_tuple()
    
    def reset(self):
        if self.is_exhausted():
            raise RuntimeError("NavverseEnv reset called after episode shard was exhausted")
        episode_label = self.episodes[self.episode_ptr]
        self.episode_ptr += 1
        self.vln_sim.load_episode(episode_label)
        obs, _, _, info = self._step_until(lambda: self.vln_sim.sim_state == "running")
        measurements = (info or {}).get("measurements", info or {})
        self.previous_distance_to_goal = measurements.get("distance_to_goal")
        self.previous_covered_area = measurements.get("covered_area")
        return self._convert((obs,0,False,info))

    @staticmethod
    def _distance_to_goal(episode, position):
        best_distance = float("inf")
        closest_goal_idx = 0
        for goal_idx, goal in enumerate(episode["goals"]):
            reference_path = goal.get("reference_path") or [goal["location"]]
            remaining = 0.0
            for idx in range(len(reference_path) - 1, -1, -1):
                if idx < len(reference_path) - 1:
                    remaining += float(
                        np.linalg.norm(
                            np.asarray(reference_path[idx], dtype=np.float64)
                            - np.asarray(reference_path[idx + 1], dtype=np.float64)
                        )
                    )
                candidate = float(
                    np.linalg.norm(
                        np.asarray(position, dtype=np.float64)
                        - np.asarray(reference_path[idx], dtype=np.float64)
                    )
                ) + remaining
                if candidate < best_distance:
                    best_distance = candidate
                    closest_goal_idx = goal_idx
        radius = float(episode["goals"][closest_goal_idx].get("radius", 0.0))
        return max(1e-5, best_distance - radius), closest_goal_idx

    def _vector_robot_poses(self):
        robot = self.vln_sim.env.scene["robot"]
        root = robot.data.root_state_w[:, :7].detach().cpu().numpy()
        positions = root[:, :3].astype(np.float32)
        quats_wxyz = root[:, 3:7].astype(np.float32)
        quats_xyzw = np.concatenate([quats_wxyz[:, 1:], quats_wxyz[:, :1]], axis=1)
        return positions, quats_xyzw

    def _vector_rgb(self, obs):
        if self.vln_sim.args.on_demand_render:
            self.vln_sim.update_on_demand_sensor_data()
            pov_rgb = self.vln_sim.manager_env.scene.sensors[
                "pov_camera"
            ].data.output["rgb"]
            return pov_rgb[..., :3].detach().cpu().numpy().astype(np.uint8)
        pov_rgb = self._find_obs_value(obs, "pov_rgb")
        if pov_rgb is None:
            raise KeyError(f"pov_rgb not found in NavVerse obs keys: {self._obs_keys(obs)}")
        return pov_rgb[..., :3].detach().cpu().numpy().astype(np.uint8)

    def configure_benchmark_video_capture(self, config=None):
        self.benchmark_video_config = dict(config) if config else None
        if self.vln_sim.args.on_demand_render:
            camera_names = (
                ("pov_camera", "third_person_camera")
                if self.benchmark_video_config is not None
                else ("pov_camera",)
            )
            self.vln_sim.set_on_demand_camera_names(camera_names)
        if self.benchmark_video_config is None:
            self.benchmark_video_frames = None
            self.benchmark_video_trajectories = None
            return
        self.benchmark_video_frames = [[] for _ in range(self.vector_envs)]
        self.benchmark_video_trajectories = [dict() for _ in range(self.vector_envs)]

    def _vector_camera_rgb(self, obs, key):
        if self.vln_sim.args.on_demand_render:
            camera_name = (
                "pov_camera" if key == "pov_rgb" else "third_person_camera"
            )
            value = self.vln_sim.manager_env.scene.sensors[
                camera_name
            ].data.output["rgb"]
            return value[..., :3].detach().cpu().numpy().astype(np.uint8)
        value = self._find_obs_value(obs, key)
        if value is None:
            raise KeyError(f"{key} not found in NavVerse obs keys: {self._obs_keys(obs)}")
        return value[..., :3].detach().cpu().numpy().astype(np.uint8)

    @staticmethod
    def _pose_matrix(position, quat_xyzw):
        from scipy.spatial.transform import Rotation

        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = Rotation.from_quat(quat_xyzw).as_matrix()
        matrix[:3, 3] = position
        return matrix

    def _capture_benchmark_video_step(self, actions):
        if self.benchmark_video_config is None or self.vector_latest_obs is None:
            return
        if self.vln_sim.args.on_demand_render and self.vector_robot_visuals_hidden:
            self.vln_sim.update_on_demand_sensor_data(("pov_camera",))
            pov_batch = self._vector_camera_rgb(self.vector_latest_obs, "pov_rgb")
            third_person_batch = None
            for index in range(self.vector_envs):
                if self.vector_done[index]:
                    continue
                self._set_vector_robot_visuals_visible(True, (index,))
                try:
                    self.vln_sim.update_on_demand_sensor_data(
                        ("third_person_camera",)
                    )
                    rendered_batch = self._vector_camera_rgb(
                        self.vector_latest_obs,
                        "third_person_rgb",
                    )
                    if third_person_batch is None:
                        third_person_batch = np.empty_like(rendered_batch)
                    third_person_batch[index] = rendered_batch[index]
                finally:
                    self._set_vector_robot_visuals_visible(False, (index,))
            if third_person_batch is None:
                third_person_batch = np.empty_like(pov_batch)
        else:
            pov_batch = self._vector_camera_rgb(self.vector_latest_obs, "pov_rgb")
            third_person_batch = self._vector_camera_rgb(
                self.vector_latest_obs,
                "third_person_rgb",
            )
        positions, quats = self._vector_robot_poses()
        for index, action in enumerate(actions):
            if self.vector_done[index]:
                continue
            step = len(self.benchmark_video_frames[index])
            self.benchmark_video_frames[index].append(
                {
                    "pov": pov_batch[index].copy(),
                    "third_person": third_person_batch[index].copy(),
                    "action": self.action_names[int(action)],
                    "step": step,
                }
            )
            self.benchmark_video_trajectories[index][step] = self._pose_matrix(
                positions[index],
                quats[index],
            ).tolist()

    @staticmethod
    def _load_benchmark_video_helpers():
        import importlib.util
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[6]

        def load(name, path):
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module

        vis = load("navverse_benchmark_vis", repo_root / "baselines/utils/vis.py")
        bev = load(
            "navverse_benchmark_bev_video",
            repo_root / "baselines/utils/bev_video.py",
        )
        return vis, bev

    def finalize_benchmark_videos(self):
        import math
        import time

        if self.benchmark_video_config is None:
            return {"videos": {}, "video_encode_seconds": 0.0}
        started = time.perf_counter()
        config = self.benchmark_video_config
        output_dir = config["video_output_dir"]
        os.makedirs(output_dir, exist_ok=True)
        vis, bev = self._load_benchmark_video_helpers()
        renderer = bev.BEVVideoRenderer(
            episode_folder=os.environ.get("NAVVERSE_EPISODE_FOLDER", "episodes"),
        )
        video_paths = {}
        for index, episode in enumerate(self.vector_episodes):
            captured = self.benchmark_video_frames[index]
            if not captured:
                continue
            instruction = (
                episode.get("placenav_goal")
                or episode.get("instruction")
                or episode.get("objnav")
                or ""
            )
            instruction_lines = ["Instruction: "]
            for word in instruction.split():
                if len(instruction_lines[-1]) + len(word) > 30:
                    instruction_lines.append(word)
                else:
                    instruction_lines[-1] += f" {word}"
            base_frames = []
            for frame in captured:
                grid = [
                    {
                        "type": "text",
                        "args": vis.text_args,
                        "text": f"Episode: {episode.episode_label}",
                    },
                    *[
                        {"type": "text", "args": vis.text_args, "text": line}
                        for line in instruction_lines
                    ],
                    {
                        "type": "text",
                        "args": vis.text_args,
                        "text": f"Step: {frame['step']}, action: {frame['action']}",
                    },
                    {
                        "type": "image_row",
                        "height": int(frame["pov"].shape[0]),
                        "image": [
                            ["First person view", frame["pov"]],
                            ["Third person view", frame["third_person"]],
                        ],
                    },
                ]
                base_frames.append(vis.render_grid(grid))
            panel_width = 760
            margin = 8
            renderer.output_size = (
                panel_width - 2 * margin,
                int(base_frames[0].shape[0]) - 2 * margin,
            )
            label = episode.episode_label
            scene_id, episode_id = label.rsplit("_", 1)
            frame_steps = [frame["step"] for frame in captured]
            bev_frames = renderer.render_sequence(
                scene_id,
                episode_id,
                self.benchmark_video_trajectories[index],
                frame_steps,
            )
            if bev_frames:
                base_frames = bev.append_bev_panels_to_frames(
                    base_frames,
                    bev_frames,
                    panel_width=panel_width,
                    margin=margin,
                )
            frames = vis.pad_frames_to_same_size(base_frames)
            fps = 2
            if len(frames) / fps > 60:
                fps = max(fps, math.ceil(len(frames) / 60))
            output_path = os.path.join(
                output_dir,
                f"{label}_obs_action_LongNav.mp4",
            )
            import imageio

            imageio.mimsave(output_path, frames, fps=fps)
            video_paths[label] = output_path
        elapsed = time.perf_counter() - started
        self.configure_benchmark_video_capture(None)
        return {
            "videos": video_paths,
            "video_encode_seconds": elapsed,
        }

    def export_benchmark_video_capture(self):
        import io
        import time
        from PIL import Image

        if self.benchmark_video_config is None:
            return {"jobs": [], "export_seconds": 0.0}
        started = time.perf_counter()
        config = self.benchmark_video_config
        jobs = []
        for index, episode in enumerate(self.vector_episodes):
            frames = []
            for frame in self.benchmark_video_frames[index]:
                encoded = {}
                for camera_name in ("pov", "third_person"):
                    buffer = io.BytesIO()
                    Image.fromarray(frame[camera_name]).save(
                        buffer,
                        format="JPEG",
                        quality=90,
                    )
                    encoded[camera_name] = buffer.getvalue()
                frames.append(
                    {
                        **encoded,
                        "action": frame["action"],
                        "step": frame["step"],
                    }
                )
            label = episode.episode_label
            jobs.append(
                {
                    "episode_label": label,
                    "instruction": (
                        episode.get("placenav_goal")
                        or episode.get("instruction")
                        or episode.get("objnav")
                        or ""
                    ),
                    "episode_folder": os.environ.get(
                        "NAVVERSE_EPISODE_FOLDER",
                        "episodes",
                    ),
                    "output_path": os.path.join(
                        config["video_output_dir"],
                        f"{label}_obs_action_LongNav.mp4",
                    ),
                    "frames": frames,
                    "trajectory": self.benchmark_video_trajectories[index],
                }
            )
        self.configure_benchmark_video_capture(None)
        return {
            "jobs": jobs,
            "export_seconds": time.perf_counter() - started,
        }

    def _vector_state(self, obs, rewards, actions=None, termination_reasons=None):
        self.vector_latest_obs = obs
        positions, quats = self._vector_robot_poses()
        states = []
        termination_reasons = termination_reasons or [None] * self.vector_envs
        for index, episode in enumerate(self.vector_episodes):
            distance, closest_goal_idx = self._distance_to_goal(episode, positions[index])
            previous_position, previous_quat = self.vector_previous_pose[index]
            pose_delta = float(np.linalg.norm(positions[index] - previous_position))
            quat_delta = float(1.0 - abs(np.dot(quats[index], previous_quat)))
            self.vector_path_length[index] += pose_delta
            collision = float(
                actions is not None
                and int(actions[index]) != 0
                and pose_delta + quat_delta < 0.015
            )
            stop_called = actions is not None and int(actions[index]) == 0
            success = float(stop_called and distance < 1.6)
            if stop_called:
                self.vector_done[index] = True
                termination_reasons[index] = "stop_called"
            if termination_reasons[index] is not None:
                self.vector_done[index] = True

            progress = float(self.vector_previous_distance[index] - distance)
            reward = float(rewards[index])
            if self.shaped_reward:
                reward = self.progress_reward_scale * progress + self.slack_reward
                if success:
                    reward += self.success_reward
                elif stop_called:
                    reward -= self.false_stop_penalty
                cell = (int(np.floor(positions[index, 0])), int(np.floor(positions[index, 1])))
                if cell not in self.vector_previous_covered_cells[index]:
                    self.vector_previous_covered_cells[index].add(cell)
                    reward += self.exploration_bonus
                if collision:
                    reward -= self.collision_penalty

            spl = success * self.vector_start_distance[index] / max(
                self.vector_start_distance[index],
                float(self.vector_path_length[index]),
                1e-5,
            )
            info = {
                "episode_label": episode.episode_label,
                "distance_to_goal": distance,
                "start_distance": float(self.vector_start_distance[index]),
                "closest_goal": closest_goal_idx,
                "success": success,
                "spl": spl,
                "path_length": float(self.vector_path_length[index]),
                "collision": collision,
                "distance_progress": progress,
                "termination_reason": termination_reasons[index],
            }
            states.append(
                {
                    "obs": {
                        "instr_or_goal": (
                            episode.get("placenav_goal")
                            or episode.get("instruction")
                            or episode.get("objnav")
                            or ""
                        )
                    },
                    "reward": reward,
                    "done": bool(self.vector_done[index]),
                    "info": info,
                    "is_exhausted": self.is_exhausted(),
                }
            )
            self.vector_previous_distance[index] = distance
            self.vector_previous_pose[index] = (positions[index].copy(), quats[index].copy())
        return self._vector_rgb(obs), states

    def reset_batch(self):
        if self.vector_envs <= 1:
            raise RuntimeError("reset_batch requires NAVVERSE_VECTOR_ENVS > 1")
        remaining = len(self.episodes) - self.episode_ptr
        if remaining < self.vector_envs:
            raise RuntimeError(
                f"reset_batch needs {self.vector_envs} episode labels, only {remaining} remain"
            )
        labels = self.episodes[self.episode_ptr : self.episode_ptr + self.vector_envs]
        self.episode_ptr += self.vector_envs
        episode_map = {episode.episode_label: episode for episode in self.vln_sim.episode_list}
        self.vector_episodes = [episode_map[label] for label in labels]
        scene_paths = {episode["path"] for episode in self.vector_episodes}
        if len(scene_paths) != 1:
            raise ValueError("Vector NavVerse rollout requires all episodes to share one scene")

        self.vln_sim.load_episode(labels[0])
        obs, _, _, _ = self._step_until(lambda: self.vln_sim.sim_state == "running")
        print(f"[SIM] Vector scene ready for {labels[0]}")
        if self.vln_sim.args.profile_step_timing:
            self._install_manager_profile_hooks()
        if self.hide_vector_robot_visuals:
            self._hide_vector_robot_visuals()
        env = self.vln_sim.env
        robot = env.scene["robot"]
        with torch.inference_mode():
            root_state = robot.data.root_state_w.clone()
            for index, episode in enumerate(self.vector_episodes):
                position = list(episode["start_position"])
                position[2] += float(
                    episode.get(
                        "robot_spawn_height",
                        getattr(env.manager_env.cfg, "robot_spawn_height", 0.6),
                    )
                )
                root_state[index, :3] = torch.tensor(
                    position, device=robot.device, dtype=root_state.dtype
                )
                root_state[index, 3:7] = torch.tensor(
                    episode["start_rotation"], device=robot.device, dtype=root_state.dtype
                )
                root_state[index, 7:] = 0.0
            robot.write_root_state_to_sim(root_state)
            robot.write_data_to_sim()
            env._reset_low_level_policy_state()
            zero_commands = torch.zeros(
                (self.vector_envs, self.vln_sim.command_dim),
                device=self.vln_sim.device,
                dtype=self.vln_sim.commands.dtype,
            )
            for _ in range(2):
                low_level_action = env._command_to_env_action(zero_commands)
                raw_obs, _, _, _ = env._step_low_level_env(low_level_action)
                env.obs = raw_obs
            obs = env._get_high_level_obs(raw_obs)
        print(f"[SIM] Reset {self.vector_envs} vector robot poses")

        self.vln_sim.configure_benchmark_waypoint_follower()
        print("[SIM] Configured vector waypoint followers")
        import copy

        self.vector_followers = [
            copy.deepcopy(self.vln_sim.waypoint_follower) for _ in range(self.vector_envs)
        ]
        self.vector_done = np.zeros(self.vector_envs, dtype=bool)
        positions, quats = self._vector_robot_poses()
        distances = [
            self._distance_to_goal(episode, positions[index])[0]
            for index, episode in enumerate(self.vector_episodes)
        ]
        self.vector_previous_distance = np.asarray(distances, dtype=np.float64)
        self.vector_start_distance = np.asarray(distances, dtype=np.float64)
        self.vector_path_length = np.zeros(self.vector_envs, dtype=np.float64)
        self.vector_previous_pose = [
            (positions[index].copy(), quats[index].copy()) for index in range(self.vector_envs)
        ]
        self.vector_previous_covered_cells = [
            {(int(np.floor(position[0])), int(np.floor(position[1])))}
            for position in positions
        ]
        return self._vector_state(obs, np.zeros(self.vector_envs, dtype=np.float32))

    @torch.inference_mode()
    def step_batch(self, actions, supplementary_logs=None):
        batch_started = time.perf_counter()
        if len(actions) != self.vector_envs:
            raise ValueError(f"Expected {self.vector_envs} actions, got {len(actions)}")
        from scipy.spatial.transform import Rotation

        setup_started = time.perf_counter()
        self._capture_benchmark_video_step(actions)
        env = self.vln_sim.env
        positions, quats = self._vector_robot_poses()
        waypoints = [None] * self.vector_envs
        for index, action in enumerate(actions):
            if self.vector_done[index] or int(action) == 0:
                continue
            yaw = Rotation.from_quat(quats[index]).as_euler("ZYX")[0]
            if int(action) == 1:
                target = [
                    positions[index, 0] + np.cos(yaw),
                    positions[index, 1] + np.sin(yaw),
                    yaw,
                ]
            elif int(action) == 2:
                target = [positions[index, 0], positions[index, 1], yaw + np.deg2rad(30.0)]
            elif int(action) == 3:
                target = [positions[index, 0], positions[index, 1], yaw - np.deg2rad(30.0)]
            else:
                raise ValueError(f"Unsupported action id: {action}")
            target[2] = float(np.arctan2(np.sin(target[2]), np.cos(target[2])))
            waypoints[index] = [target]
            self.vector_followers[index].reset()
        self._record_vector_profile(
            "step_batch/setup",
            time.perf_counter() - setup_started,
        )

        start_sim_time = self._sim_time()
        raw_reward = np.zeros(self.vector_envs, dtype=np.float32)
        termination_reasons = [None] * self.vector_envs
        raw_obs = env.obs
        low_level_iterations = 0
        while any(
            waypoints[index] is not None and not self.vector_done[index]
            for index in range(self.vector_envs)
        ):
            follower_started = time.perf_counter()
            positions, quats = self._vector_robot_poses()
            commands = []
            for index in range(self.vector_envs):
                if waypoints[index] is None or self.vector_done[index]:
                    commands.append(torch.zeros((1, 3), device=self.vln_sim.device))
                    continue
                command = self.vector_followers[index].update(
                    positions[index], quats[index], waypoints[index], verbose=False
                )
                commands.append(command)
                if self.vector_followers[index].arrived_at_goal:
                    waypoints[index] = None
            command_batch = torch.cat(commands, dim=0)
            self._record_vector_profile(
                "step_batch/follower",
                time.perf_counter() - follower_started,
            )
            command_started = time.perf_counter()
            low_level_action = env._command_to_env_action(command_batch)
            self._record_vector_profile(
                "step_batch/command_to_action",
                time.perf_counter() - command_started,
            )
            env_step_started = time.perf_counter()
            raw_obs, reward, low_level_done, _ = env._step_low_level_env(low_level_action)
            self._record_vector_profile(
                "step_batch/raw_env_step",
                time.perf_counter() - env_step_started,
            )
            low_level_iterations += 1
            env.obs = raw_obs
            if torch.is_tensor(reward):
                reward_array = reward.detach().float().cpu().numpy().reshape(-1)
                if reward_array.size == 1:
                    raw_reward += float(reward_array[0])
                else:
                    raw_reward += reward_array[: self.vector_envs]
            termination_started = time.perf_counter()
            reset_buf = env.termination_manager.compute()
            for index in range(self.vector_envs):
                if bool(reset_buf[index]):
                    terms = env.termination_manager.get_active_iterable_terms(index)
                    termination_reasons[index] = (
                        max(terms, key=lambda item: item[1])[0] if terms else "low_level_done"
                    )
                    self.vector_done[index] = True
                    waypoints[index] = None
                elif (
                    torch.is_tensor(low_level_done)
                    and low_level_done.numel() > index
                    and bool(low_level_done.reshape(-1)[index])
                ):
                    termination_reasons[index] = "low_level_done"
                    self.vector_done[index] = True
                    waypoints[index] = None
            self._record_vector_profile(
                "step_batch/termination",
                time.perf_counter() - termination_started,
            )
            if self._sim_time() - start_sim_time >= self.action_timeout_sim_s:
                break

        obs = env._get_high_level_obs(raw_obs)
        state_started = time.perf_counter()
        output = self._vector_state(obs, raw_reward, actions, termination_reasons)
        self._record_vector_profile(
            "step_batch/state_and_render",
            time.perf_counter() - state_started,
        )
        self._record_vector_profile(
            "step_batch/total",
            time.perf_counter() - batch_started,
        )
        self._record_vector_profile(
            "step_batch/low_level_iterations",
            0.0,
            count=low_level_iterations,
        )
        return output

    def _apply_shaped_reward(self, state, action):
        if not self.shaped_reward:
            return state

        measurements = state["info"]
        distance = measurements.get("distance_to_goal")
        progress = 0.0
        if distance is not None and self.previous_distance_to_goal is not None:
            progress = float(self.previous_distance_to_goal) - float(distance)
        if distance is not None:
            self.previous_distance_to_goal = float(distance)

        reward = self.progress_reward_scale * progress + self.slack_reward
        success = float(measurements.get("success", 0.0) or 0.0)
        if success > 0.0:
            reward += self.success_reward * success
        elif self.action_names[int(action)] == "STOP":
            reward -= self.false_stop_penalty

        covered_area = measurements.get("covered_area")
        exploration_delta = 0.0
        if covered_area is not None and self.previous_covered_area is not None:
            exploration_delta = float(covered_area) - float(self.previous_covered_area)
        if covered_area is not None:
            self.previous_covered_area = float(covered_area)
        if exploration_delta > 0.0:
            reward += self.exploration_bonus

        collision = float(measurements.get("collision", 0.0) or 0.0)
        if collision > 0.0:
            reward -= self.collision_penalty

        state["info"]["raw_env_reward"] = state["reward"]
        state["info"]["distance_progress"] = progress
        state["info"]["exploration_delta"] = exploration_delta
        state["reward"] = reward
        return state
    
    def _step(self,action):
        action_name = self.action_names[int(action)]
        with torch.inference_mode():
            self.vln_sim.set_discrete_waypoint_action(action_name)
            if action_name == "STOP":
                self.vln_sim.step()
                return self._latest_state_tuple(default_done=True)
            return self._step_until(
                lambda: self.vln_sim.waypoints is None,
                timeout_sim_s=self.action_timeout_sim_s,
            )

    def _instruction(self):
        episode = self.vln_sim.current_episode or {}
        return (
            episode.get("placenav_goal")
            or episode.get("instruction")
            or episode.get("objnav")
            or ""
        )

    def _find_obs_value(self, obs, key):
        if not hasattr(obs, "keys"):
            return None
        keys = list(obs.keys())
        if key in keys:
            return obs[key]
        for obs_key in keys:
            value = obs[obs_key]
            found = self._find_obs_value(value, key)
            if found is not None:
                return found
        return None

    def _obs_keys(self, obs):
        if not hasattr(obs, "keys"):
            return type(obs).__name__
        keys = {}
        for key in obs.keys():
            value = obs[key]
            keys[key] = self._obs_keys(value) if hasattr(value, "keys") else type(value).__name__
        return keys
    
    def _convert(self,state_tuple):
        obs,reward,done,info = state_tuple
        pov_rgb = self._find_obs_value(obs, "pov_rgb")
        if pov_rgb is None:
            raise KeyError(f"pov_rgb not found in NavVerse obs keys: {self._obs_keys(obs)}")
        rgb = pov_rgb.squeeze().cpu().numpy()
        measurements = info.get("measurements", info)
        state = {
            "obs":{
                "instr_or_goal": self._instruction()
            },
            "reward":reward.squeeze().item() if not isinstance(reward, numbers.Number) else reward,
            "done":done.item() if not isinstance(done,bool) else done,
            "info": measurements,
            "is_exhausted":self.is_exhausted()
        }
        return rgb,state
    
    def step(self,action, supplementary_logs):
        rgb, state = self._convert(self._step(action))
        return rgb, self._apply_shaped_reward(state, action)
