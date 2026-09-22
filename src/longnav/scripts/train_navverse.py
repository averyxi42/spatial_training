'''
🚀 [run experiment]:
python3 -m longnav.training_scripts.train_rl.py +experiment=<experiment_name>
NOTE: experiment_name must be a config that exists in src/conf/experiment.

⚙️ [add experiment config]:
add new yaml to src/conf/experiment. see config_schema.py for requirements or reference existing yaml.
NOTE: need to have "# @package _global_" at the start of your config.

👾 [see hydra help]:
python3 -m longnav.training_scripts.train_rl.py --help
https://hydra.cc/docs/intro/

🔧 [install tab completion]:
eval "$(python3 -m longnav.training_scripts.train_rl.py -sc install=bash)"
NOTE: tab completion only works if your command uses python not python3. somehow.
'''
import os
import time

# NUCLEAR THREAD CAP: Must be set before importing numpy/torch/ray
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_ENABLE_PARALLEL_LOADING"] = "false"

import hydra
from longnav.conf.register_configs import register_configs
from longnav.config_schema import RLConfig
from longnav.env.navverse import NavverseEnv
import os 

DEBUG_FLAG = False
FREEZE_DATA = False # for debugging only
env_actor_cls = NavverseEnv
# 1. Register our command variants
register_configs()
@hydra.main(version_base=None, config_name="rl_config",config_path='../config')
def main(cfg: RLConfig):
    # keep heavy imports here so hydra tab complete is snappier?
    import ray
    import numpy as np

    from longnav.utils.factories import ExpBootstrapper,get_shard_iterator,get_console_logger
    from longnav.utils.rl_core import collate_trajectories
    from longnav.utils.rollout_core import collect_rollouts, collect_vector_rollouts
    from longnav.utils.scene_sampler import (
        SceneBalancedShardSampler,
        load_episode_labels,
    )
    from verl.trainer.ppo.core_algos import get_adv_estimator_fn
    import json
    
    cfg.resources.vlm_conda_env = os.environ.get("LONGNAV_VLM_CONDA_ENV", cfg.resources.vlm_conda_env)
    cfg.resources.habitat_conda_env = os.environ.get("LONGNAV_HABITAT_CONDA_ENV", cfg.resources.habitat_conda_env)
    cfg.resources.num_sims = int(os.environ.get("LONGNAV_NUM_SIMS", cfg.resources.num_sims))
    cfg.resources.num_vlms = int(os.environ.get("LONGNAV_NUM_VLMS", cfg.resources.num_vlms))
    cfg.resources.osm_gb = int(os.environ.get("LONGNAV_OSM_GB", cfg.resources.osm_gb))
    cfg.rollout.max_steps = int(os.environ.get("LONGNAV_ROLLOUT_MAX_STEPS", cfg.rollout.max_steps))
    cfg.resources.sim_gpu_fraction = float(os.environ.get("LONGNAV_SIM_GPU_FRACTION", cfg.resources.sim_gpu_fraction))
    cfg.resources.vlm_gpu_fraction = float(os.environ.get("LONGNAV_VLM_GPU_FRACTION", cfg.resources.vlm_gpu_fraction))
    cfg.training.rl_config.n_rollout = int(os.environ.get("LONGNAV_N_ROLLOUT", cfg.training.rl_config.n_rollout))
    cfg.training.rl_config.n_adv = int(
        os.environ.get(
            "LONGNAV_N_ADV",
            max(cfg.training.rl_config.n_adv, cfg.training.rl_config.n_rollout),
        )
    )
    eval_interval = int(os.environ.get("LONGNAV_EVAL_INTERVAL", "0"))
    test_episode_path = os.environ.get(
        "LONGNAV_EVAL_EPISODE_SET",
        os.environ.get("LONGNAV_TEST_EPISODE_JSON"),
    )
    train_episode_set = os.environ.get("LONGNAV_TRAIN_EPISODE_SET")
    sampling_seed = int(os.environ.get("LONGNAV_EPISODE_SAMPLING_SEED", "42"))
    eval_at_start = os.environ.get("LONGNAV_EVAL_AT_START", "0").lower() in {
        "1",
        "true",
        "yes",
    }
    eval_capture_video = os.environ.get("LONGNAV_EVAL_CAPTURE_VIDEO", "1").lower() in {
        "1",
        "true",
        "yes",
    }
    eval_video_width = int(os.environ.get("LONGNAV_EVAL_VIDEO_WIDTH", "480"))
    eval_video_fps = int(os.environ.get("LONGNAV_EVAL_VIDEO_FPS", "4"))
    eval_video_style = os.environ.get(
        "LONGNAV_EVAL_VIDEO_STYLE",
        "benchmark",
    )
    log_rollout_actions = os.environ.get(
        "LONGNAV_LOG_ROLLOUT_ACTIONS",
        "0",
    ).lower() in {"1", "true", "yes"}
    vector_rollout = os.environ.get("LONGNAV_VECTOR_ROLLOUT", "0").lower() in {
        "1",
        "true",
        "yes",
    }
    if vector_rollout:
        vector_envs_per_worker = int(
            os.environ.get(
                "LONGNAV_VECTOR_ENVS_PER_WORKER",
                cfg.training.rl_config.n_rollout // cfg.resources.num_sims,
            )
        )
        if cfg.resources.num_sims != cfg.resources.num_vlms:
            raise ValueError(
                "Vector rollout requires LONGNAV_NUM_SIMS == LONGNAV_NUM_VLMS"
            )
        if vector_envs_per_worker <= 1:
            raise ValueError("Vector rollout requires at least two environments per worker")
        rollouts_per_wave = cfg.resources.num_sims * vector_envs_per_worker
        if cfg.training.rl_config.n_rollout % rollouts_per_wave:
            raise ValueError(
                "LONGNAV_N_ROLLOUT must be a multiple of LONGNAV_NUM_SIMS * "
                f"LONGNAV_VECTOR_ENVS_PER_WORKER; got "
                f"{cfg.training.rl_config.n_rollout} and wave size "
                f"{rollouts_per_wave}"
            )
        os.environ["NAVVERSE_VECTOR_ENVS"] = str(vector_envs_per_worker)
    if eval_interval < 0:
        raise ValueError("LONGNAV_EVAL_INTERVAL cannot be negative")
    if eval_interval and not vector_rollout:
        raise ValueError("Periodic NavVerse evaluation requires vector rollout mode")
    if eval_interval and not test_episode_path:
        raise ValueError(
            "LONGNAV_EVAL_EPISODE_SET is required when LONGNAV_EVAL_INTERVAL is set"
        )
    stop_on_success = os.environ.get("LONGNAV_STOP_ON_SUCCESS", "0").lower() in {
        "1",
        "true",
        "yes",
    }
    cfg.vlm.save_outputs = True
    cfg.vlm.attn_impl = "flash_attention_2"

    advantage_estimator_fn = get_adv_estimator_fn(cfg.training.rl_config.advantage_estimator)
    print(f"Model ID: {cfg.vlm.model_id}")
    bootstrapper = ExpBootstrapper(cfg)
    logger = get_console_logger()

    bootstrapper.setup_cluster()
    trainers = bootstrapper.bootstrap_vlms_rl() #allocate vlms first to prevent out of room issues
    wandb_objs = bootstrapper.bootstrap_logger()
    if wandb_objs is not None:
        wandb_actor,excluded_episodes = wandb_objs
    else:
        excluded_episodes = None
        wandb_actor = None
    #TODO: clean up this mess with factory
    res_cfg = cfg.resources
    navverse_env_vars = {
        key: value
        for key, value in os.environ.items()
        if key.startswith("NAVVERSE_")
        or key in {"PYTHONPATH", "CUDA_VISIBLE_DEVICES", "WANDB_MODE"}
    }
    env_dict = {"conda": res_cfg.habitat_conda_env, "env_vars": navverse_env_vars}

    RemoteSim = ray.remote(env_actor_cls).options(
        resources={res_cfg.sim_resource_tag: 1},
        num_cpus=res_cfg.sim_cpus,
        num_gpus=res_cfg.sim_gpu_fraction,
        runtime_env=env_dict ,
        max_restarts=0,        # <--- CRITICAL: Do not restart on crash.
        max_task_retries=-1,
    )
    sims = [RemoteSim.remote() for _ in range(res_cfg.num_sims)]
    shard_iter = None
    dynamic_sampler = None
    if train_episode_set:
        if not vector_rollout:
            raise ValueError("LONGNAV_TRAIN_EPISODE_SET requires vector rollout mode")
        train_labels = load_episode_labels(train_episode_set)
        dynamic_sampler = SceneBalancedShardSampler(
            train_labels,
            episodes_per_scene=vector_envs_per_worker,
            scenes_per_update=(
                cfg.training.rl_config.n_rollout // vector_envs_per_worker
            ),
            seed=sampling_seed,
        )
    else:
        shard_iter = get_shard_iterator(
            subset_label="" if cfg.task.episode_json else cfg.task.subset_label,
            episode_json=cfg.task.episode_json,
            shard_size=cfg.task.shard_size,
            logger=logger,
            excluded_episodes=excluded_episodes,
        )
    training_shards = None
    test_shards = None
    if vector_rollout:
        import itertools

        if dynamic_sampler is None:
            training_shards = list(shard_iter)
            if not training_shards:
                raise ValueError("Vector rollout requires at least one episode shard")
            for labels in training_shards:
                if len(labels) != vector_envs_per_worker:
                    raise ValueError(
                        "Every vector training shard must contain exactly "
                        f"{vector_envs_per_worker} episodes"
                    )
                if len({label.rsplit("_", 1)[0] for label in labels}) != 1:
                    raise ValueError(
                        f"Vector training shard spans multiple scenes: {labels}"
                    )
            shard_iter = itertools.cycle(training_shards)
        if eval_interval:
            if test_episode_path.endswith(".json"):
                with open(test_episode_path) as file:
                    test_labels = json.load(file)
            else:
                test_labels = load_episode_labels(test_episode_path)
            expected_test_episodes = cfg.resources.num_sims * vector_envs_per_worker
            if len(test_labels) != expected_test_episodes:
                raise ValueError(
                    "Periodic test set must contain exactly one 8-episode scene "
                    f"per worker; got {len(test_labels)} labels, expected "
                    f"{expected_test_episodes}"
                )
            test_shards = [
                test_labels[index : index + vector_envs_per_worker]
                for index in range(0, len(test_labels), vector_envs_per_worker)
            ]
            for labels in test_shards:
                if len({label.rsplit("_", 1)[0] for label in labels}) != 1:
                    raise ValueError(f"Vector test shard spans multiple scenes: {labels}")
            training_scenes = (
                dynamic_sampler.scenes
                if dynamic_sampler is not None
                else {
                    label.rsplit("_", 1)[0]
                    for labels in training_shards
                    for label in labels
                }
            )
            test_scenes = {
                label.rsplit("_", 1)[0]
                for labels in test_shards
                for label in labels
            }
            overlap = training_scenes & test_scenes
            if overlap:
                raise ValueError(
                    f"Training and test scenes must be disjoint; overlap={sorted(overlap)}"
                )
            def physical_scene(scene_id):
                if scene_id.startswith("vcp_"):
                    return scene_id.removeprefix("vcp_")
                if scene_id.startswith("innout_"):
                    return scene_id.removeprefix("innout_")
                return scene_id

            physical_overlap = {
                physical_scene(scene) for scene in training_scenes
            } & {physical_scene(scene) for scene in test_scenes}
            if physical_overlap:
                raise ValueError(
                    "Training and test physical scenes must be disjoint; "
                    f"overlap={sorted(physical_overlap)}"
                )
            logger.info(
                "Periodic test configured: "
                f"interval={eval_interval} policy updates, "
                f"scenes={sorted(test_scenes)}, episodes={len(test_labels)}"
            )
    trajectory_list = []

    def finite_mean(values):
        numeric = [float(value) for value in values if np.isfinite(float(value))]
        return float(np.mean(numeric)) if numeric else float("nan")

    def summarize_episodes(prefix, rollouts, results):
        rows = []
        action_counts = np.zeros(4, dtype=np.int64)
        termination_counts = {}
        for rollout, result in zip(rollouts, results):
            trajectory = rollout[0]
            actions = np.asarray(trajectory.get("actions", []), dtype=np.int64)
            rewards = np.asarray(trajectory.get("rewards", []), dtype=np.float64)
            distances = np.asarray(
                trajectory.get("distance_to_goal", []),
                dtype=np.float64,
            )
            collisions = np.asarray(
                trajectory.get("collision", []),
                dtype=np.float64,
            )
            if actions.size:
                action_counts += np.bincount(actions, minlength=4)[:4]
            termination = result.get("termination_reason") or (
                "max_steps" if len(actions) >= cfg.rollout.max_steps else "unknown"
            )
            termination_counts[termination] = termination_counts.get(termination, 0) + 1
            label = result.get("episode_label", "unknown")
            start_distance = float(
                result.get(
                    "start_distance",
                    (
                        distances[0]
                        + np.asarray(
                            trajectory.get("distance_progress", [0.0]),
                            dtype=np.float64,
                        )[0]
                    )
                    if distances.size
                    else float("nan"),
                )
            )
            min_distance = (
                float(np.min(distances)) if distances.size else float("nan")
            )
            progress_fraction = (
                float(np.clip((start_distance - min_distance) / start_distance, 0.0, 1.0))
                if np.isfinite(start_distance) and start_distance > 0.0
                else float("nan")
            )
            rows.append(
                {
                    "episode_label": label,
                    "scene_id": label.rsplit("_", 1)[0],
                    "goal": result.get("instr_or_goal", ""),
                    "success": float(result.get("success", 0.0) or 0.0),
                    "spl": float(result.get("spl", 0.0) or 0.0),
                    "reward": float(rewards.sum()) if rewards.size else 0.0,
                    "steps": int(len(actions)),
                    "start_distance": start_distance,
                    "min_distance": min_distance,
                    "final_distance": (
                        float(distances[-1]) if distances.size else float("nan")
                    ),
                    "progress_fraction": progress_fraction,
                    "collision_rate": (
                        float(np.mean(collisions)) if collisions.size else 0.0
                    ),
                    "termination_reason": termination,
                    "video_path": result.get("video_path"),
                    "video_fps": eval_video_fps,
                }
            )
        total_actions = max(int(action_counts.sum()), 1)
        metrics = {
            f"{prefix}/success_rate": finite_mean(row["success"] for row in rows),
            f"{prefix}/spl_mean": finite_mean(row["spl"] for row in rows),
            f"{prefix}/reward_mean": finite_mean(row["reward"] for row in rows),
            f"{prefix}/episode_length_mean": finite_mean(row["steps"] for row in rows),
            f"{prefix}/episode_length_p50": float(
                np.percentile([row["steps"] for row in rows], 50)
            ),
            f"{prefix}/episode_length_p90": float(
                np.percentile([row["steps"] for row in rows], 90)
            ),
            f"{prefix}/min_distance_mean": finite_mean(
                row["min_distance"] for row in rows
            ),
            f"{prefix}/final_distance_mean": finite_mean(
                row["final_distance"] for row in rows
            ),
            f"{prefix}/collision_rate": finite_mean(
                row["collision_rate"] for row in rows
            ),
            f"{prefix}/progress_fraction_mean": finite_mean(
                row["progress_fraction"] for row in rows
            ),
            f"{prefix}/action_stop_fraction": float(action_counts[0] / total_actions),
            f"{prefix}/action_forward_fraction": float(action_counts[1] / total_actions),
            f"{prefix}/action_left_fraction": float(action_counts[2] / total_actions),
            f"{prefix}/action_right_fraction": float(action_counts[3] / total_actions),
        }
        for termination, count in termination_counts.items():
            safe_name = "".join(
                character if character.isalnum() else "_"
                for character in termination.lower()
            )
            metrics[f"{prefix}/termination_{safe_name}_fraction"] = count / len(rows)
        if prefix == "test":
            scenes = sorted({row["scene_id"] for row in rows})
            for scene in scenes:
                scene_rows = [row for row in rows if row["scene_id"] == scene]
                metrics[f"test_scene/{scene}/success_rate"] = finite_mean(
                    row["success"] for row in scene_rows
                )
                metrics[f"test_scene/{scene}/spl_mean"] = finite_mean(
                    row["spl"] for row in scene_rows
                )
        return metrics, rows

    def summarize_collection_runtime(prefix, timings):
        worker_runtimes = timings.get("worker_runtimes", [])
        metrics = {
            f"runtime/{prefix}_total_seconds": timings["total_seconds"],
            f"runtime/{prefix}_episode_seconds": timings["episode_seconds"],
            f"runtime/{prefix}_postprocess_seconds": timings["postprocess_seconds"],
            f"runtime/{prefix}_assign_seconds": timings["assign_seconds"],
        }
        for key in (
            "wall_seconds",
            "vlm_inference_seconds",
            "environment_step_seconds",
            "video_encode_seconds",
            "episode_steps_total",
            "episode_steps_max",
        ):
            values = [runtime.get(key, 0.0) for runtime in worker_runtimes]
            if values:
                metrics[f"runtime/{prefix}_worker_{key}_mean"] = float(np.mean(values))
                metrics[f"runtime/{prefix}_worker_{key}_max"] = float(np.max(values))
        profile_keys = sorted(
            {
                profile_key
                for runtime in worker_runtimes
                for profile_key in runtime.get("environment_profile", {})
            }
        )
        for profile_key in profile_keys:
            safe_key = "".join(
                character if character.isalnum() else "_"
                for character in profile_key
            ).strip("_")
            for field in ("total_seconds", "count"):
                values = [
                    runtime.get("environment_profile", {})
                    .get(profile_key, {})
                    .get(field, 0.0)
                    for runtime in worker_runtimes
                ]
                metrics[
                    f"runtime/{prefix}_env_{safe_key}_{field}_mean"
                ] = float(np.mean(values))
                metrics[
                    f"runtime/{prefix}_env_{safe_key}_{field}_max"
                ] = float(np.max(values))
        return metrics

    pending_eval_videos = []

    def cleanup():
        drain_pending_evaluations(wait=True)
        if wandb_actor is not None:
            try:
                ray.get(wandb_actor.finish.remote(), timeout=120)
            except Exception as exc:
                logger.warning(f"W&B finish failed during cleanup: {exc}")
        for trainer in trainers:
            ray.kill(trainer)
        for sim in sims:
            ray.kill(sim)
        if wandb_actor is not None:
            ray.kill(wandb_actor)
        ray.shutdown()

    def debug():
        global DEBUG_FLAG
        DEBUG_FLAG = False # consume the flag
        import ipdb
        ipdb.set_trace()

    # convenience functions for ipdb abuse
    def save_checkpoint(name):
        ray.get(trainers[0].save_checkpoint_unsafe.remote(os.path.join(bootstrapper.typed_cfg.task.output_dir,bootstrapper.typed_cfg.task.run_name,"checkpoints",f"manual_checkpoint_{name}")))

    def pickle_obj(obj,filename):
        import pickle
        dirname = os.path.join(bootstrapper.typed_cfg.task.output_dir,bootstrapper.typed_cfg.task.run_name,"dbg")
        os.makedirs(dirname,exist_ok=True)
        filepath = os.path.join(dirname,f"{filename}.pkl")
        with open(filepath,'wb') as f:
            pickle.dump(obj,f)

    def finalize_evaluation_videos(record, outputs):
        render_seconds = [float(output.get("render_seconds", 0.0)) for output in outputs]
        export_seconds = [float(output.get("export_seconds", 0.0)) for output in outputs]
        expected_paths = [
            row["video_path"] for row in record["episode_rows"] if row.get("video_path")
        ]
        missing_paths = [path for path in expected_paths if not os.path.isfile(path)]
        if missing_paths:
            raise RuntimeError(
                f"Async test video generation missed {len(missing_paths)} files: "
                f"{missing_paths[:3]}"
            )
        if render_seconds:
            record["metrics"]["runtime/test_video_render_seconds_mean"] = float(
                np.mean(render_seconds)
            )
            record["metrics"]["runtime/test_video_render_seconds_max"] = float(
                np.max(render_seconds)
            )
        if export_seconds:
            record["metrics"]["runtime/test_video_export_seconds_mean"] = float(
                np.mean(export_seconds)
            )
            record["metrics"]["runtime/test_video_export_seconds_max"] = float(
                np.max(export_seconds)
            )
        with open(record["result_path"], "r") as file:
            payload = json.load(file)
        payload["metrics"] = record["metrics"]
        temporary_result_path = f"{record['result_path']}.tmp"
        with open(temporary_result_path, "w") as file:
            json.dump(payload, file, indent=2, allow_nan=False)
        os.replace(temporary_result_path, record["result_path"])
        if wandb_actor is not None:
            ray.get(
                wandb_actor.log_eval_batch.remote(
                    record["policy_update"],
                    record["metrics"],
                    record["episode_rows"],
                )
            )
        with open(record["complete_path"], "w") as file:
            json.dump(
                {
                    "policy_update": record["policy_update"],
                    "episodes": len(record["episode_rows"]),
                    "result_path": record["result_path"],
                    "wandb_logged": wandb_actor is not None,
                    "videos_complete": True,
                },
                file,
                indent=2,
            )
        logger.info(
            "Async test videos complete "
            f"policy_update={record['policy_update']} files={len(expected_paths)}"
        )

    def drain_pending_evaluations(wait=False):
        remaining = []
        for record in pending_eval_videos:
            futures = record["futures"]
            if wait:
                outputs = ray.get(futures)
            else:
                ready, _ = ray.wait(
                    futures,
                    num_returns=len(futures),
                    timeout=0.0,
                )
                if len(ready) != len(futures):
                    remaining.append(record)
                    continue
                outputs = ray.get(futures)
            finalize_evaluation_videos(record, outputs)
        pending_eval_videos[:] = remaining

    def queue_eval_video_backfill():
        if wandb_actor is None:
            return
        enabled = os.environ.get(
            "LONGNAV_WANDB_BACKFILL_EVAL_VIDEOS",
            "0",
        ).lower() in {"1", "true", "yes"}
        if not enabled:
            return
        eval_root = os.path.join(
            bootstrapper.typed_cfg.task.output_dir,
            bootstrapper.typed_cfg.task.run_name,
            "eval",
        )
        if not os.path.isdir(eval_root):
            return
        queued = 0
        for dirname in sorted(os.listdir(eval_root)):
            iteration_dir = os.path.join(eval_root, dirname)
            result_path = os.path.join(iteration_dir, "result.json")
            complete_path = os.path.join(iteration_dir, "complete.json")
            marker_path = os.path.join(
                iteration_dir,
                "wandb_direct_videos_v3.complete",
            )
            if (
                not os.path.isfile(result_path)
                or not os.path.isfile(complete_path)
                or os.path.isfile(marker_path)
            ):
                continue
            with open(result_path) as file:
                payload = json.load(file)
            episode_rows = payload.get("episodes", [])
            missing_videos = [
                row.get("video_path")
                for row in episode_rows
                if not row.get("video_path")
                or not os.path.isfile(row["video_path"])
            ]
            if missing_videos:
                logger.warning(
                    f"Skipping W&B video backfill for {dirname}: "
                    f"{len(missing_videos)} videos missing"
                )
                continue
            wandb_actor.log_eval_batch.remote(
                int(payload["policy_update"]),
                payload.get("metrics", {}),
                episode_rows,
                marker_path,
            )
            queued += 1
        if queued:
            logger.info(
                f"Queued {queued} completed test iterations for W&B video backfill"
            )

    def run_evaluation(policy_update):
        if not eval_interval:
            return {}
        eval_root = os.path.join(
            bootstrapper.typed_cfg.task.output_dir,
            bootstrapper.typed_cfg.task.run_name,
            "eval",
            f"iter_{policy_update:06d}",
        )
        result_path = os.path.join(eval_root, "result.json")
        complete_path = os.path.join(eval_root, "complete.json")
        skip_completed = os.environ.get(
            "LONGNAV_EVAL_SKIP_COMPLETED",
            "1",
        ).lower() in {"1", "true", "yes"}
        if skip_completed and os.path.isfile(complete_path):
            logger.info(
                f"Skipping completed test evaluation at policy_update={policy_update}"
            )
            return {}
        os.makedirs(eval_root, exist_ok=True)
        logger.info(
            f"Starting periodic test evaluation at policy_update={policy_update}"
        )
        eval_started = time.perf_counter()
        eval_rollouts, eval_results, _, eval_timings = collect_vector_rollouts(
            sims,
            trainers,
            iter(test_shards),
            cfg.resources.num_sims * vector_envs_per_worker,
            vector_envs_per_worker,
            postprocess_kwargs={"return_inputs": False, "eval": True},
            eval_config={
                "enabled": True,
                "capture_video": eval_capture_video,
                "video_output_dir": os.path.join(eval_root, "videos"),
                "video_width": eval_video_width,
                "video_fps": eval_video_fps,
                "video_style": eval_video_style,
                "policy_update": policy_update,
            },
            return_timings=True,
        )
        eval_metrics, episode_rows = summarize_episodes(
            "test",
            eval_rollouts,
            eval_results,
        )
        video_futures = eval_timings.pop("video_futures", [])
        eval_metrics.update(summarize_collection_runtime("test", eval_timings))
        eval_metrics["runtime/test_full_seconds"] = time.perf_counter() - eval_started
        serializable_rows = [
            {
                key: value
                for key, value in row.items()
                if key != "video_fps"
            }
            for row in episode_rows
        ]
        temporary_result_path = f"{result_path}.tmp"
        with open(temporary_result_path, "w") as file:
            json.dump(
                {
                    "policy_update": policy_update,
                    "metrics": eval_metrics,
                    "episodes": serializable_rows,
                },
                file,
                indent=2,
                allow_nan=False,
            )
        os.replace(temporary_result_path, result_path)
        if video_futures:
            if wandb_actor is not None:
                ray.get(
                    wandb_actor.log_policy_update.remote(
                        policy_update,
                        eval_metrics,
                    )
                )
            pending_eval_videos.append(
                {
                    "policy_update": policy_update,
                    "metrics": eval_metrics,
                    "episode_rows": episode_rows,
                    "result_path": result_path,
                    "complete_path": complete_path,
                    "futures": video_futures,
                }
            )
        elif wandb_actor is not None:
            ray.get(
                wandb_actor.log_eval_batch.remote(
                    policy_update,
                    eval_metrics,
                    episode_rows,
                )
            )
        if not video_futures:
            with open(complete_path, "w") as file:
                json.dump(
                    {
                        "policy_update": policy_update,
                        "episodes": len(episode_rows),
                        "result_path": result_path,
                        "wandb_logged": wandb_actor is not None,
                        "videos_complete": True,
                    },
                    file,
                    indent=2,
                )
        logger.info(
            "Periodic test complete "
            f"policy_update={policy_update} "
            f"success_rate={eval_metrics['test/success_rate']:.6f} "
            f"spl={eval_metrics['test/spl_mean']:.6f} "
            f"seconds={eval_metrics['runtime/test_full_seconds']:.1f}"
        )
        return eval_metrics

    queue_eval_video_backfill()
    try:
        requested_policy_updates = os.environ.get("LONGNAV_POLICY_UPDATES")
        if requested_policy_updates is None:
            total_rollout_cycles = (
                bootstrapper.typed_cfg.training.total_optimization_steps
                * bootstrapper.typed_cfg.training.grad_accum_steps
                // bootstrapper.typed_cfg.training.rl_config.n_rollout
            )
        else:
            total_rollout_cycles = int(requested_policy_updates)
            if total_rollout_cycles <= 0:
                raise ValueError("LONGNAV_POLICY_UPDATES must be positive")
        cycle_offset = int(os.environ.get("LONGNAV_GLOBAL_CYCLE_OFFSET", "0"))
        if cycle_offset < 0 or cycle_offset > total_rollout_cycles:
            raise ValueError(
                f"LONGNAV_GLOBAL_CYCLE_OFFSET={cycle_offset} is outside "
                f"[0, {total_rollout_cycles}]"
            )
        if eval_at_start and eval_interval and cycle_offset % eval_interval == 0:
            run_evaluation(cycle_offset)
        for global_cycle in range(cycle_offset, total_rollout_cycles):
            cycle_started = time.perf_counter()
            policy_update = global_cycle + 1
            drain_pending_evaluations(wait=False)
            # ------------------------------------------- rollouts ------------------------------------------
            logger.info("Starting rollout collection!")

            if dynamic_sampler is not None:
                sampled_shards, sample_digest = dynamic_sampler.sample(policy_update)
                shard_iter = iter(sampled_shards)
                sampled_scenes = [labels[0].rsplit("_", 1)[0] for labels in sampled_shards]
                logger.info(
                    "Dynamic rollout sample "
                    f"policy_update={policy_update} scenes={len(sampled_scenes)} "
                    f"episodes={sum(len(labels) for labels in sampled_shards)} "
                    f"sha256={sample_digest} selected_scenes={sampled_scenes}"
                )

            # rollout_list = collect_rollouts(sims,trainers,shard_iter,target_episodes=bootstrapper.typed_cfg.training.rl_config.n_rollout) #
            if vector_rollout:
                (
                    rollout_list,
                    result_list,
                    log_list,
                    rollout_timings,
                ) = collect_vector_rollouts(
                    sims,
                    trainers,
                    shard_iter,
                    bootstrapper.typed_cfg.training.rl_config.n_rollout,
                    vector_envs_per_worker,
                    return_timings=True,
                )
            else:
                rollout_list,result_list,log_list = collect_rollouts(sims,trainers,shard_iter,bootstrapper.typed_cfg.training.rl_config.n_rollout) #
                rollout_timings = {
                    "total_seconds": time.perf_counter() - cycle_started,
                    "episode_seconds": float("nan"),
                    "postprocess_seconds": float("nan"),
                    "assign_seconds": float("nan"),
                    "worker_runtimes": [],
                }

            print("done collecting")
            if len(rollout_list) != bootstrapper.typed_cfg.training.rl_config.n_rollout:
                raise RuntimeError(
                    f"Policy update {global_cycle} collected {len(rollout_list)} "
                    f"rollouts; expected "
                    f"{bootstrapper.typed_cfg.training.rl_config.n_rollout}"
                )
            cycle_succeeded = any(
                float(result.get("success", 0.0) or 0.0) > 0.0
                for result in result_list
            )
            cycle_successes = sum(
                float(result.get("success", 0.0) or 0.0) > 0.0
                for result in result_list
            )
            cycle_mean_spl = float(
                np.mean(
                    [float(result.get("spl", 0.0) or 0.0) for result in result_list]
                )
            )
            print(
                "Policy update rollout batch "
                f"cycle={global_cycle} rollouts={len(rollout_list)} "
                f"workers={len(sims)} "
                f"episodes_per_worker={vector_envs_per_worker if vector_rollout else 1} "
                f"successes={cycle_successes} mean_spl={cycle_mean_spl:.6f}"
            )
            train_metrics, _ = summarize_episodes(
                "train",
                rollout_list,
                result_list,
            )
            train_metrics.update(
                summarize_collection_runtime("train", rollout_timings)
            )
            wave_runtimes = rollout_timings.get("wave_runtimes", [])
            if wave_runtimes:
                critical_runtimes = [
                    max(
                        wave["worker_runtimes"],
                        key=lambda runtime: float(runtime.get("wall_seconds", 0.0)),
                    )
                    for wave in wave_runtimes
                    if wave["worker_runtimes"]
                ]
                train_metrics["train/agent_inference_seconds"] = float(
                    sum(
                        runtime.get("vlm_inference_seconds", 0.0)
                        for runtime in critical_runtimes
                    )
                )
                train_metrics["train/sim_rollout_seconds"] = float(
                    sum(
                        runtime.get("environment_step_seconds", 0.0)
                        for runtime in critical_runtimes
                    )
                )
            for rollout_idx, (rollout, result) in enumerate(zip(rollout_list, result_list)):
                trajectory = rollout[0]
                distances = trajectory.get("distance_to_goal", [])
                rollout_probs = trajectory.get("rollout_probs", [])
                min_distance = float(distances.min()) if len(distances) else float("nan")
                final_distance = float(distances[-1]) if len(distances) else float("nan")
                max_stop_prob = (
                    float(rollout_probs[:, 0].max())
                    if getattr(rollout_probs, "ndim", 0) == 2 and rollout_probs.shape[1]
                    else float("nan")
                )
                actions_text = (
                    f"actions={trajectory.get('actions', []).tolist()} "
                    if log_rollout_actions
                    else f"steps={len(trajectory.get('actions', []))} "
                )
                print(
                    "Rollout summary "
                    f"cycle={global_cycle} index={rollout_idx} "
                    f"{actions_text}"
                    f"reward={float(trajectory.get('rewards', []).sum()):.4f} "
                    f"success={float(result.get('success', 0.0) or 0.0):.1f} "
                    f"min_distance={min_distance:.3f} final_distance={final_distance:.3f} "
                    f"max_stop_prob={max_stop_prob:.4f} "
                    f"termination={result.get('termination_reason', 'unknown')}"
                )
            num_vlms = len(trainers)
            # -------------------------------------------unpack and collate the trajectories

            driver_postprocess_started = time.perf_counter()
            trajectory_list += [tup[0] for tup in rollout_list]
            trajectory_list = trajectory_list[-bootstrapper.typed_cfg.training.rl_config.n_adv:]
            traj_batch = collate_trajectories(trajectory_list)
            model_inputs = [(tup[1],tup[2]) for tup in rollout_list]
            
            values = traj_batch.get("values",None)
            distances = traj_batch.get('distance_to_goal',None)
            # ---------------------------------- compute gae ----------------------------------------------
            print("Computing Advantages")
            # config = bootstrapper.resolved_dict['training']['rl_config']
            # Note: compute_gae expects (B, T) inputs and returns (B, T)
            adv_tuple = advantage_estimator_fn(
                token_level_rewards=traj_batch['rewards'],
                values=values,
                # distances = distances,
                response_mask=traj_batch['response_mask'],
                config = cfg.training.rl_config,
                # gamma=config.get('gamma', 0.99), # Fallback defaults if not in config
                # lam=config.get('lam', 0.95)
            )
            advantages, returns = adv_tuple[0],adv_tuple[1]
            if len(adv_tuple)>2:
                traj_batch['baseline'] = adv_tuple[2]
                print("DEBUG: computing variances")
                print(f"Rtn Var: {(returns[traj_batch['response_mask']==1]).var().item():.4f}")
                print(f"MSE Error: {((traj_batch['baseline'][traj_batch['response_mask']==1]-returns[traj_batch['response_mask']==1])**2).mean().item():.4f}")
            traj_batch['advantages'] = advantages
            traj_batch['returns'] = returns
            global_return_mean = returns[traj_batch['response_mask']==1].mean().item()
            traj_batch = traj_batch[-bootstrapper.typed_cfg.training.rl_config.n_rollout:] # only train on most recent.
            print(f"Advantage Mean: {advantages.mean().item():.4f}, Std: {advantages.std().item():.4f}")
            train_metrics["runtime/driver_collate_gae_seconds"] = (
                time.perf_counter() - driver_postprocess_started
            )

            if DEBUG_FLAG:
                debug() # great spot to intercept the trajectories for saving etc
                
            # ------------------------------ non logged training (extra epochs) ----------
            logger.info("Starting training")
            training_started = time.perf_counter()

            for i in range(bootstrapper.typed_cfg.training.rl_config.n_epoch-1):
                print(f"epoch {i}")
                training_futures = []
                perm_indices = np.random.permutation(bootstrapper.typed_cfg.training.rl_config.n_rollout) # shuffle
                for batch_start in range(0, bootstrapper.typed_cfg.training.rl_config.n_rollout, num_vlms):
                    # Create futures for this specific "global step"
                    # We map workers 0..N to data indices batch_start..batch_start+N
                    step_futures = []
                    for worker_idx, trainer in enumerate(trainers):
                        global_idx = batch_start + worker_idx
                        global_idx = perm_indices[global_idx]
                        # Access the specific inputs and the sliced TensorDict for this index
                        # We use global_idx to ensure we pull the correct corresponding data
                        ref = trainer.train_rl_step.remote(
                                *model_inputs[global_idx], 
                                traj_batch[global_idx : global_idx + 1, traj_batch['response_mask'][global_idx].bool()]
                            )
                        step_futures.append(ref)
                    training_futures.extend(step_futures)
                ray.get(training_futures)
            # ------------------------------ dispatch logged training ----------------------------
            future_metadata = {}
            training_futures = []
            perm_indices = np.random.permutation(bootstrapper.typed_cfg.training.rl_config.n_rollout) # shuffle
            for batch_start in range(0, bootstrapper.typed_cfg.training.rl_config.n_rollout, num_vlms):
                # Create futures for this specific "global step"
                # We map workers 0..N to data indices batch_start..batch_start+N
                step_futures = []
                for worker_idx, trainer in enumerate(trainers):
                    global_idx = batch_start + worker_idx
                    global_idx = perm_indices[global_idx]
                    # Access the specific inputs and the sliced TensorDict for this index
                    # We use global_idx to ensure we pull the correct corresponding data
                    ref = trainer.train_rl_step.remote(
                            *model_inputs[global_idx], 
                            traj_batch[global_idx : global_idx + 1, traj_batch['response_mask'][global_idx].bool()]
                        )
                    step_futures.append(ref)
                    future_metadata[ref] = global_idx
                training_futures.extend(step_futures)

            print(f"Dispatched {len(training_futures)} training tasks for final epoch")
            # ------------------------------------- monitor training live ---------------------------------
            pending_futures = training_futures
            total_tasks = len(pending_futures)
            completed_count = 0
            failed_training_tasks = 0
            training_result_rows = []

            while pending_futures:
                # 1. Block until at least one future is ready
                ready_refs, pending_futures = ray.wait(pending_futures, num_returns=1)
                
                # 2. Process the ready future(s)
                for ref in ready_refs:
                    # We catch exceptions here to prevent one failed batch from crashing the loop
                    try:
                        result = ray.get(ref)
                        rollout_idx = future_metadata[ref]

                        batch_row = traj_batch[rollout_idx] 
                        valid_mask = batch_row['response_mask'].bool()
                        traj_stats = batch_row[valid_mask] 


                        # 4. Log
                        rollout_stats = {
                            "rollout/ep_rew": traj_stats['rewards'].sum().item(),
                            "rollout/ep_len": valid_mask.sum().item(),
                            "rollout/success": traj_stats['success'].max().item(),
                            "rollout/spl": traj_stats['spl'].max().item(),
                            "rollout/ep_rtn": traj_stats['returns'].mean().item(),
                            "rollout/rtn_var": traj_stats['returns'].var(unbiased=False).item(),
                            "rollout/global_cycle": global_cycle
                        }
                        try:
                            critic_mse = ((traj_stats['baseline']-traj_stats['returns'])**2).mean()
                            naive_mse = ((traj_stats['returns'] - global_return_mean)**2).mean().item()
                            rollout_stats |= {
                                "rollout/baseline_mse":critic_mse,
                                "rollout/naive_mse":naive_mse
                            }
                        except:
                            print("cannot compute baseline metric")
                        result |= rollout_stats
                        training_result_rows.append(result)
                        log_ref = log_list[rollout_idx]
                        try:
                            # 1. Try to get the path with a short timeout (e.g., 0.1s)
                            # If the Sim Worker is done, this is instant.
                            log_path = ray.get(log_ref, timeout=30.0)
                            if log_path is not None:
                                with open(log_path, 'r') as f:
                                    vlm_log_dict = json.load(f)
                                result |= vlm_log_dict
                            
                        except ray.exceptions.GetTimeoutError:
                            # 2. If Sim Worker is stuck, log a warning but DO NOT FREEZE training
                            logger.warning(f"Log file for rollout {rollout_idx} not ready (Sim Worker I/O Lag). Skipping detailed logs.")
                        except Exception as e:
                            logger.warning(f"Failed to read log file: {e}")
                        completed_count += 1
                     
                        print(f"[{completed_count}/{total_tasks}] Complete. {result}")
                        
                    except Exception as e:
                        failed_training_tasks += 1
                        logger.error(f"[{completed_count}/{total_tasks}] Task failed: {e}")
            if failed_training_tasks:
                raise RuntimeError(
                    f"Policy update {global_cycle} had "
                    f"{failed_training_tasks}/{total_tasks} failed training tasks"
                )
            print(
                "Policy update training complete "
                f"cycle={global_cycle} trajectories={len(training_futures)} "
                f"ddp_workers={num_vlms} "
                f"optimizer_steps_per_worker="
                f"{len(training_futures) // max(num_vlms, 1)}"
            )
            train_metrics["runtime/policy_optimization_seconds"] = (
                time.perf_counter() - training_started
            )
            train_metrics["train/policy_update_seconds"] = train_metrics[
                "runtime/policy_optimization_seconds"
            ]
            metric_key_map = {
                "loss/pg_loss": "train/pg_loss",
                "actor/ppo_kl": "train/ppo_kl",
                "actor/pg_clipfrac": "train/pg_clip_fraction",
                "train/entropy": "train/policy_entropy",
                "train/rollout_kl_divergence": "train/rollout_kl",
                "train/grad_norm": "optimizer/grad_norm",
                "train/lr": "optimizer/lr",
                "return": "train/return",
            }
            for source_key, destination_key in metric_key_map.items():
                values = [
                    float(row[source_key])
                    for row in training_result_rows
                    if source_key in row
                    and isinstance(row[source_key], (int, float, np.number))
                ]
                if values:
                    train_metrics[destination_key] = float(np.mean(values))
            #------------------------------------ save checkpoint ------------------------------------
            checkpoint_started = time.perf_counter()
            steps_until_save = (global_cycle+1) % bootstrapper.typed_cfg.training.save_step
            if steps_until_save == 0 or (stop_on_success and cycle_succeeded):
                print("saving checkpoint")
                ray.get(trainers[0].save_checkpoint_unsafe.remote(os.path.join(bootstrapper.typed_cfg.task.output_dir,bootstrapper.typed_cfg.task.run_name,"checkpoints",f"checkpoint_{global_cycle}")))
            else:
                print(f"T-{steps_until_save} steps until checkpoint!")
            train_metrics["runtime/checkpoint_seconds"] = (
                time.perf_counter() - checkpoint_started
            )

            del model_inputs
            train_metrics["runtime/policy_update_total_seconds"] = (
                time.perf_counter() - cycle_started
            )
            train_metrics["train/iter_seconds"] = train_metrics[
                "runtime/policy_update_total_seconds"
            ]
            accounted_seconds = sum(
                train_metrics.get(key, 0.0)
                for key in (
                    "train/agent_inference_seconds",
                    "train/sim_rollout_seconds",
                    "train/policy_update_seconds",
                )
            )
            train_metrics["train/iter_overhead_seconds"] = max(
                train_metrics["train/iter_seconds"] - accounted_seconds,
                0.0,
            )
            if eval_interval and policy_update % eval_interval == 0:
                run_evaluation(policy_update)
            train_metrics["runtime/cycle_with_test_seconds"] = (
                time.perf_counter() - cycle_started
            )
            train_metrics["train/policy_update"] = policy_update
            if wandb_actor is not None:
                ray.get(
                    wandb_actor.log_policy_update.remote(
                        policy_update,
                        train_metrics,
                    )
                )
            if stop_on_success and cycle_succeeded:
                print(f"Success observed at rollout cycle {global_cycle}; stopping training.")
                break
    finally:

        cleanup()

if __name__ == "__main__":
    main()
