"""Run pinned NavVerse batches with zero SE(2) commands as a simulator control."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import ray

from longnav.env.navverse import NavVerseHostActor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--slots", type=int, default=8)
    parser.add_argument("--task-type", default="placenav")
    parser.add_argument("--robot-name", default="cylinder")
    parser.add_argument("--tidybot-embodiment", default="full")
    parser.add_argument("--hide-robot-visuals", action="store_true")
    parser.add_argument("--forward-step-m", type=float, default=0.0)
    parser.add_argument("--lateral-step-m", type=float, default=0.0)
    parser.add_argument("--yaw-step-deg", type=float, default=0.0)
    parser.add_argument(
        "--trajectory-pattern",
        choices=("line", "arc-left", "arc-right", "zigzag", "rotate-then-forward"),
        default="line",
    )
    parser.add_argument("--host-settle-s", type=float, default=0.0)
    parser.add_argument("--on-demand-render", action="store_true")
    parser.add_argument("--camera-rgb-only", action="store_true")
    parser.add_argument("--policy-camera-only", action="store_true")
    parser.add_argument("--minimal-logging", action="store_true")
    parser.add_argument("--profile-step-timing", action="store_true")
    parser.add_argument("--save-rgb-arrays", action="store_true")
    return parser.parse_args()


def action_chunk(args, step_index: int) -> np.ndarray:
    fraction = np.linspace(0.1, 1.0, 10, dtype=np.float32)
    distance = float(args.forward_step_m)
    lateral = float(args.lateral_step_m)
    yaw = np.deg2rad(float(args.yaw_step_deg))

    if args.trajectory_pattern == "arc-left":
        yaw = abs(yaw)
    elif args.trajectory_pattern == "arc-right":
        yaw = -abs(yaw)
    elif args.trajectory_pattern == "zigzag":
        yaw = abs(yaw) * (1.0 if (step_index // 2) % 2 == 0 else -1.0)
    elif args.trajectory_pattern == "rotate-then-forward":
        if step_index % 6 < 2:
            distance = 0.0
            lateral = 0.0
            yaw = abs(yaw)
        else:
            yaw = 0.0

    action = np.zeros((10, 3), dtype=np.float32)
    theta = yaw * fraction
    if args.trajectory_pattern in {"arc-left", "arc-right", "zigzag"} and abs(yaw) > 1e-8:
        radius = distance / yaw
        action[:, 0] = radius * np.sin(theta)
        action[:, 1] = radius * (1.0 - np.cos(theta))
    else:
        action[:, 0] = distance * fraction
        action[:, 1] = lateral * fraction
    action[:, 2] = theta
    return action


def main():
    args = parse_args()
    labels = [line.strip() for line in Path(args.episodes_file).read_text().splitlines() if line.strip()]
    if not labels or len(labels) % args.slots:
        raise ValueError(f"episode count must be a positive multiple of {args.slots}")
    groups = [labels[index:index + args.slots] for index in range(0, len(labels), args.slots)]
    for group in groups:
        scenes = {label.rsplit("_", 1)[0] for label in group}
        if len(scenes) != 1:
            raise ValueError(f"each group must contain one scene, got {group}")

    os.makedirs(args.output_dir, exist_ok=True)
    thread_env = {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    ray.init(
        resources={"navverse_control": len(groups)},
        object_store_memory=64 * 1024**3,
        object_spilling_directory="/tmp/navverse_longnav_probe_spilling",
    )
    remote_host = ray.remote(NavVerseHostActor)
    hosts = []
    states = []
    initial_rgb = []
    final_rgb = []
    for group in groups:
        host = remote_host.options(
            resources={"navverse_control": 1},
            num_cpus=4,
            num_gpus=1,
            runtime_env={"conda": "navverse", "env_vars": thread_env},
        ).remote(
            slots_per_host=args.slots,
            navverse_repo_root="/home/ubuntu/Projects/NavVerse-Benchmark",
            config_path="configs/default.yaml",
            episode_folder="/home/ubuntu/Projects/navverse_data/episodes/",
            scene_folder="/home/ubuntu/Projects/navverse_data/",
            episodes_path="/home/ubuntu/Projects/NavVerse-Benchmark/episodes/train_set.txt",
            task_type=args.task_type,
            robot_name=args.robot_name,
            tidybot_embodiment=args.tidybot_embodiment,
            hide_robot_visuals=args.hide_robot_visuals,
            on_demand_render=args.on_demand_render,
            camera_rgb_only=args.camera_rgb_only,
            policy_camera_only=args.policy_camera_only,
            profile_step_timing=args.profile_step_timing,
            gap=10,
            dt=0.04,
            max_steps=args.max_steps,
            success_distance=1.6,
            slack_penalty=0.01,
            collision_penalty=0.0,
            failure_penalty=2.5,
            progress_reward_clip=0.75,
            success_reward=0.0,
            minimal_logging=args.minimal_logging,
            video_tick_stride=1,
            video_realtime_factor=1.0,
            logging_output_dir=args.output_dir,
        )
        # Isaac Kit startup is not safe when all eight processes initialize at once.
        # Waiting on a host method makes constructor completion the gate for the next GPU.
        print("HOST_READY", json.dumps(ray.get(host.worker_placement.remote())), flush=True)
        ray.get(host.assign_shard.remote(group))
        reset_rgb, reset_states = ray.get(host.reset_vector.remote())
        initial_rgb.append(reset_rgb)
        final_rgb.append(reset_rgb)
        states.append(reset_states)
        print("HOST_RESET", json.dumps(group), flush=True)
        hosts.append(host)
        if len(hosts) < len(groups) and args.host_settle_s > 0:
            time.sleep(args.host_settle_s)
    try:
        step_started = time.perf_counter()
        executed_steps = 0
        for step_index in range(args.max_steps):
            action = action_chunk(args, step_index)
            refs = []
            active_indexes = []
            for index, (host, worker_states) in enumerate(zip(hosts, states)):
                if all(state["done"] for state in worker_states):
                    continue
                actions = [None if state["done"] else action for state in worker_states]
                refs.append(host.step_vector.remote(actions))
                active_indexes.append(index)
            if not refs:
                break
            for index, result in zip(active_indexes, ray.get(refs)):
                final_rgb[index] = result[0]
                states[index] = result[1]
            executed_steps += 1
            print(
                "STEP_COMPLETE",
                json.dumps(
                    {
                        "step": executed_steps,
                        "elapsed_seconds": time.perf_counter() - step_started,
                        "active_episodes": sum(
                            not state["done"]
                            for worker_states in states
                            for state in worker_states
                        ),
                        "states": [
                            {
                                "episode_label": state["info"].get("episode_label"),
                                "root_z": state["info"].get("root_z"),
                                "tilt_deg": state["info"].get("tilt_deg"),
                                "contact_force": state["info"].get("physics_contact_force"),
                                "termination_reason": state["info"].get("termination_reason"),
                            }
                            for worker_states in states
                            for state in worker_states
                        ],
                    }
                ),
                flush=True,
            )
        step_wall_seconds = time.perf_counter() - step_started
        paths = ray.get([host.flush_logs_to_disk.remote() for host in hosts])
        profiles = ray.get(
            [host.get_vector_profile.remote(reset=False) for host in hosts]
        )
        final = [state["info"] for worker_states in states for state in worker_states]
        def hashes(batches):
            return [
                hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest()
                for batch in batches
                for frame in batch
            ]

        if args.save_rgb_arrays:
            np.save(Path(args.output_dir, "initial_rgb.npy"), np.concatenate(initial_rgb))
            np.save(Path(args.output_dir, "final_rgb.npy"), np.concatenate(final_rgb))

        Path(args.output_dir, "control_results.json").write_text(
            json.dumps(
                {
                    "episodes": final,
                    "summaries": paths,
                    "step_wall_seconds": step_wall_seconds,
                    "executed_steps": executed_steps,
                    "initial_rgb_sha256": hashes(initial_rgb),
                    "final_rgb_sha256": hashes(final_rgb),
                    "profiles": profiles,
                },
                indent=2,
            )
        )
        print("CONTROL_COMPLETE", json.dumps({
            "n": len(final),
            "bad_orientation": sum(x.get("termination_reason") == "bad_orientation" for x in final),
            "terrain_out_of_bounds": sum(x.get("termination_reason") == "terrain_out_of_bounds" for x in final),
        }), flush=True)
    finally:
        for host in hosts:
            ray.kill(host)
        ray.shutdown()


if __name__ == "__main__":
    main()
