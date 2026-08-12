"""Shared orchestration loop extracted from scripts/train_rl.py and scripts/eval.py.

Both scripts inlined the same bootstrap -> rollout -> advantage -> train ->
log -> checkpoint sequence independently, and the four smoke tests
(tests/rl_smoke.py, rl_single.py, rl_continuous_single.py, eval_smoke.py)
each hand-copied it again with observable drift between copies. This module
is the single implementation all of those now call.

Two deviations from a pure 1:1 split, both because the data is structurally
required downstream and there's no other place to carry it:
- `run_rollout_cycle` also returns `log_list` (needed by `stream_results_and_log`
  to look up each rollout's on-disk VLM log).
- `compute_advantages_and_returns` also returns `global_return_mean` (needed by
  `stream_results_and_log`'s naive-baseline MSE metric).
"""
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import ray

from longnav.config_schema import RLConfig
from longnav.utils.factories import ExpBootstrapper, get_console_logger, get_shard_iterator
from longnav.utils.rl_core import collate_trajectories
from longnav.utils.rollout_core import collect_rollouts


@dataclass
class BootstrapContext:
    bootstrapper: ExpBootstrapper
    trainers: list
    sims: list
    wandb_actor: Any
    excluded_episodes: Any
    shard_iter: Iterator
    logger: Any
    num_rollouts: int


def bootstrap_all(cfg: RLConfig, training: bool) -> BootstrapContext:
    """Cluster + VLM + logger + sim + shard-iterator setup shared by train_rl.py
    and eval.py.

    Generalizes the previously-orphaned `ExpBootstrapper.bootstrap_eval()`,
    which wired `bootstrap_vlms_infer` (an inference-only worker with no
    training/checkpoint-loading path) instead of `bootstrap_vlms_rl` -- the
    factory both train_rl.py and eval.py actually need. That wiring is fixed
    here: `training` selects whether the VLM workers set up optimizer/DDP
    state (`bootstrap_vlms_rl(training=True)`, train_rl.py) or just load a
    checkpoint for inference (`bootstrap_vlms_rl(training=False)`, eval.py).
    """
    logger = get_console_logger()
    bootstrapper = ExpBootstrapper(cfg)
    bootstrapper.setup_cluster()

    trainers = bootstrapper.bootstrap_vlms_rl(training=training)

    wandb_objs = bootstrapper.bootstrap_logger()
    if wandb_objs is not None:
        wandb_actor, excluded_episodes = wandb_objs
    else:
        wandb_actor = None
        excluded_episodes = None

    sims = bootstrapper.bootstrap_sims(wandb_actor)

    shard_iter = get_shard_iterator(
        subset_label=bootstrapper.typed_cfg.task.subset_label,
        episode_json=bootstrapper.typed_cfg.task.episode_json,
        shard_size=bootstrapper.typed_cfg.task.shard_size,
        logger=logger,
        excluded_episodes=excluded_episodes,
    )

    num_rollouts = (
        bootstrapper.typed_cfg.training.total_optimization_steps
        * bootstrapper.typed_cfg.training.grad_accum_steps
        // bootstrapper.typed_cfg.training.rl_config.n_rollout
    )

    return BootstrapContext(
        bootstrapper=bootstrapper,
        trainers=trainers,
        sims=sims,
        wandb_actor=wandb_actor,
        excluded_episodes=excluded_episodes,
        shard_iter=shard_iter,
        logger=logger,
        num_rollouts=num_rollouts,
    )


def run_rollout_cycle(
    sims,
    trainers,
    shard_iter: Iterator,
    trajectory_list: list,
    n_rollout: int,
    n_adv: int,
) -> Tuple[Any, list, Any, Any, list]:
    """Collect one rollout cycle and collate it into a trajectory batch.

    Matches train_rl.py L140-156 (rollout collection through the
    values/distances lookups). Also returns `log_list` -- structurally
    required by `stream_results_and_log` further down the pipeline, and
    produced by the same `collect_rollouts` call, so there's nowhere else
    to source it from without a second call.

    Returns: (traj_batch, model_inputs, values, distances, log_list)
    """
    rollout_list, result_list, log_list = collect_rollouts(sims, trainers, shard_iter, n_rollout)

    trajectory_list += [tup[0] for tup in rollout_list]
    del trajectory_list[: max(0, len(trajectory_list) - n_adv)]
    traj_batch = collate_trajectories(trajectory_list)
    model_inputs = [(tup[1], tup[2]) for tup in rollout_list]

    values = traj_batch.get("values", None)
    distances = traj_batch.get("distance_to_goal", None)

    return traj_batch, model_inputs, values, distances, log_list


def compute_advantages_and_returns(
    traj_batch,
    advantage_estimator_fn,
    cfg: RLConfig,
) -> Tuple[Any, float]:
    """Pure advantage/return math. Matches train_rl.py L157-178.

    Does NOT fold in the batch-truncation-to-n_rollout step (L179) -- that's
    the caller's job, since it's a training-loop concern, not an advantage-math
    one.

    Returns: (traj_batch, global_return_mean) -- global_return_mean is needed
    by `stream_results_and_log`'s naive-baseline MSE metric and is nowhere
    else to compute it from once this function returns.
    """
    values = traj_batch.get("values", None)
    adv_tuple = advantage_estimator_fn(
        token_level_rewards=traj_batch["rewards"],
        values=values,
        response_mask=traj_batch["response_mask"],
        config=cfg.training.rl_config,
    )
    advantages, returns = adv_tuple[0], adv_tuple[1]
    if len(adv_tuple) > 2:
        traj_batch["baseline"] = adv_tuple[2]
        print("DEBUG: computing variances")
        print(f"Rtn Var: {(returns[traj_batch['response_mask']==1]).var().item():.4f}")
        print(
            "MSE Error: "
            f"{((traj_batch['baseline'][traj_batch['response_mask']==1]-returns[traj_batch['response_mask']==1])**2).mean().item():.4f}"
        )
    traj_batch["advantages"] = advantages
    traj_batch["returns"] = returns
    global_return_mean = returns[traj_batch["response_mask"] == 1].mean().item()

    print(f"Advantage Mean: {advantages.mean().item():.4f}, Std: {advantages.std().item():.4f}")

    return traj_batch, global_return_mean


def run_training_epochs(
    trainers,
    model_inputs,
    traj_batch,
    n_epoch: int,
    n_rollout: int,
    num_vlms: int,
) -> Tuple[list, dict]:
    """Dispatch training steps for `n_epoch` epochs over the current batch.

    Collapses train_rl.py's two nearly-identical nested loops (the blocking
    "extra epochs" loop, L186-207, and the non-blocking "final epoch" dispatch,
    L208-229) into one parameterized loop: every epoch except the last blocks
    on `ray.get` immediately, the last epoch's futures/metadata are returned
    for the caller to monitor via `stream_results_and_log`.

    Preserves the original edge-case behavior for n_epoch <= 0: the original
    code's "extra epochs" loop was `range(n_epoch - 1)` (empty for n_epoch<=1)
    but the final-epoch dispatch ran unconditionally afterward -- i.e. at
    least one (non-blocking) training epoch always runs. `max(n_epoch, 1)`
    reproduces that floor.

    Returns: (training_futures, future_metadata) for the final epoch only.
    """
    total_epochs = max(n_epoch, 1)
    training_futures: list = []
    future_metadata: dict = {}

    for epoch in range(total_epochs):
        is_final_epoch = epoch == total_epochs - 1
        epoch_futures = []
        epoch_metadata = {}
        perm_indices = np.random.permutation(n_rollout)
        for batch_start in range(0, n_rollout, num_vlms):
            for worker_idx, trainer in enumerate(trainers):
                global_idx = batch_start + worker_idx
                global_idx = perm_indices[global_idx]
                ref = trainer.train_rl_step.remote(
                    *model_inputs[global_idx],
                    traj_batch[global_idx : global_idx + 1, traj_batch["response_mask"][global_idx].bool()],
                )
                epoch_futures.append(ref)
                epoch_metadata[ref] = global_idx

        if is_final_epoch:
            training_futures = epoch_futures
            future_metadata = epoch_metadata
        else:
            ray.get(epoch_futures)

    return training_futures, future_metadata


def stream_results_and_log(
    training_futures: list,
    future_metadata: dict,
    traj_batch,
    wandb_actor,
    global_cycle: int,
    global_return_mean: float,
    logger,
    log_list: Optional[list] = None,
) -> None:
    """Async ray.wait monitor loop. Matches train_rl.py L230-293.

    `log_list` is an added parameter (not in the metaplan's literal signature)
    -- the original code reads `log_list[rollout_idx]` inside this loop, and
    since `log_list` is only produced by `run_rollout_cycle`'s `collect_rollouts`
    call, it has to be threaded through as a parameter here. Pass `None` (e.g.
    from eval-only callers that never enter this loop) to skip per-rollout log
    file reads.
    """
    pending_futures = training_futures
    total_tasks = len(pending_futures)
    completed_count = 0

    while pending_futures:
        ready_refs, pending_futures = ray.wait(pending_futures, num_returns=1)

        for ref in ready_refs:
            try:
                result = ray.get(ref)
                rollout_idx = future_metadata[ref]

                batch_row = traj_batch[rollout_idx]
                valid_mask = batch_row["response_mask"].bool()
                traj_stats = batch_row[valid_mask]

                rollout_stats = {
                    "rollout/ep_rew": traj_stats["rewards"].sum().item(),
                    "rollout/ep_len": valid_mask.sum().item(),
                    "rollout/ep_rtn": traj_stats["returns"].mean().item(),
                    "rollout/rtn_var": traj_stats["returns"].var().item(),
                    "rollout/global_cycle": global_cycle,
                }
                try:
                    critic_mse = ((traj_stats["baseline"] - traj_stats["returns"]) ** 2).mean()
                    naive_mse = ((traj_stats["returns"] - global_return_mean) ** 2).mean().item()
                    rollout_stats |= {
                        "rollout/baseline_mse": critic_mse,
                        "rollout/naive_mse": naive_mse,
                    }
                except Exception:
                    print("cannot compute baseline metric")
                result |= rollout_stats

                if log_list is not None:
                    log_ref = log_list[rollout_idx]
                    try:
                        log_path = ray.get(log_ref, timeout=30.0)
                        with open(log_path, "r") as f:
                            vlm_log_dict = json.load(f)
                        result |= vlm_log_dict
                    except ray.exceptions.GetTimeoutError:
                        logger.warning(
                            f"Log file for rollout {rollout_idx} not ready (Sim Worker I/O Lag). Skipping detailed logs."
                        )
                    except Exception as e:
                        logger.warning(f"Failed to read log file: {e}")

                if wandb_actor is not None:
                    wandb_actor.log_row.remote(result)
                completed_count += 1

                print(f"[{completed_count}/{total_tasks}] Complete. {result}")

            except Exception as e:
                logger.error(f"[{completed_count}/{total_tasks}] Task failed: {e}")


def maybe_checkpoint(
    trainers,
    global_cycle: int,
    save_step: int,
    output_dir: str,
    run_name: str,
) -> None:
    """Matches train_rl.py L294-300. Gated independently from
    `stream_results_and_log` since checkpointing happens every `save_step`
    cycles, not every cycle."""
    steps_until_save = (global_cycle + 1) % save_step
    if steps_until_save == 0:
        print("saving checkpoiutsnt")
        ray.get(
            trainers[0].save_checkpoint_unsafe.remote(
                os.path.join(output_dir, run_name, "checkpoints", f"checkpoint_{global_cycle}")
            )
        )
    else:
        print(f"T-{steps_until_save} steps until checkpoint!")
