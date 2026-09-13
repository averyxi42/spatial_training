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
import math
import os
import shutil
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
import ray
import torch

from longnav.config_schema import RLConfig
from longnav.utils.factories import ExpBootstrapper, get_console_logger, get_shard_iterator
from longnav.utils.rl_core import collate_trajectories
from longnav.utils.rollout_core import collect_rollouts, collect_vector_rollouts

_LAST_PROBE_COUNTERFACTUAL = None  # see compute_advantages_and_returns


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
    vector_envs_per_sim: int = 1
    placement_groups: Optional[list] = None
    sim_rebuilder: Optional[Callable[[int], Any]] = None
    sim_rebuild_validator: Optional[Callable[[int, Any], None]] = None
    episode_soft_timeout_seconds: float = 600.0
    episode_hard_timeout_seconds: float = 900.0
    sim_restart_limit: int = 1
    sim_recycle_every_cycles: int = 0


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
    paired = bool(bootstrapper.typed_cfg.resources.paired_gpu_workers)
    placement_groups = None
    scheduling_strategies = None
    if paired:
        placement_groups, scheduling_strategies = (
            bootstrapper.create_paired_gpu_placement_groups()
        )

    trainers = bootstrapper.bootstrap_vlms_rl(
        training=training, scheduling_strategies=scheduling_strategies
    )

    wandb_objs = bootstrapper.bootstrap_logger()
    if wandb_objs is not None:
        wandb_actor, excluded_episodes = wandb_objs
    else:
        wandb_actor = None
        excluded_episodes = None

    sims = bootstrapper.bootstrap_sims(
        wandb_actor, scheduling_strategies=scheduling_strategies
    )
    if paired:
        bootstrapper.verify_paired_gpu_placement(trainers, sims)

    vector_envs_per_sim = int(getattr(bootstrapper.typed_cfg.sim, "slots_per_host", 1))
    if paired:
        wave_size = len(sims) * vector_envs_per_sim
        n_rollout = bootstrapper.typed_cfg.training.rl_config.n_rollout
        if n_rollout % wave_size:
            raise ValueError(
                f"Vector n_rollout must be a multiple of {wave_size}; got {n_rollout}"
            )
        episode_path = bootstrapper.typed_cfg.sim.episodes_path
        if not episode_path:
            raise ValueError("Paired NavVerse vector rollout requires sim.episodes_path")
        from longnav.utils.scene_sampler import SceneGroupedBatchIterator

        eval_labels = None
        eval_uids_file = getattr(bootstrapper.typed_cfg.task, "eval_uids_file", None)
        if eval_uids_file:
            with open(eval_uids_file) as eval_file:
                eval_labels = [
                    label.strip()
                    for label in eval_file.read().replace("\n", ",").split(",")
                    if label.strip()
                ]

        shard_iter = SceneGroupedBatchIterator(
            episode_path,
            vector_envs_per_sim,
            groups_per_wave=len(sims),
            seed=42,
            # Evaluation never consumes this iterator.  Inference-only evaluation may
            # intentionally pin every episode in the pool, which would otherwise make
            # an unused training iterator empty during bootstrap.
            excluded_labels=eval_labels if training else None,
        )
    else:
        vector_envs_per_sim = 1
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

    def sim_rebuilder(worker_index):
        strategy = (
            scheduling_strategies[worker_index]
            if scheduling_strategies is not None
            else None
        )
        return bootstrapper.bootstrap_sim(
            worker_index,
            logger=wandb_actor,
            scheduling_strategy=strategy,
        )

    def sim_rebuild_validator(worker_index, sim_handle):
        if not paired:
            return
        trainer_info, sim_info = ray.get(
            [
                trainers[worker_index].worker_placement.remote(),
                sim_handle.worker_placement.remote(),
            ]
        )
        trainer_gpus = tuple(trainer_info["ray_gpu_ids"])
        sim_gpus = tuple(sim_info["ray_gpu_ids"])
        if len(sim_gpus) != 1 or sim_gpus != trainer_gpus:
            raise RuntimeError(
                f"Replacement sim {worker_index} is not co-located with its VLM: "
                f"vlm={trainer_info}, sim={sim_info}"
            )
        print(
            f"Verified replacement sim {worker_index} on paired GPU {sim_gpus[0]}",
            flush=True,
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
        vector_envs_per_sim=vector_envs_per_sim,
        placement_groups=placement_groups,
        sim_rebuilder=sim_rebuilder,
        sim_rebuild_validator=sim_rebuild_validator,
        episode_soft_timeout_seconds=float(
            bootstrapper.typed_cfg.rollout.episode_soft_timeout_seconds
        ),
        episode_hard_timeout_seconds=float(
            bootstrapper.typed_cfg.rollout.episode_hard_timeout_seconds
        ),
        sim_restart_limit=int(bootstrapper.typed_cfg.rollout.sim_restart_limit),
        sim_recycle_every_cycles=int(
            bootstrapper.typed_cfg.rollout.sim_recycle_every_cycles
        ),
    )


def recycle_vector_sims(sims, sim_rebuilder, sim_rebuild_validator=None) -> None:
    """Replace idle vector simulators sequentially at a safe cycle boundary."""
    if sim_rebuilder is None:
        raise RuntimeError("sim_rebuilder is required for simulator recycling")
    for worker_index, old_sim in enumerate(list(sims)):
        print(
            f"[sim_recycle] replacing idle simulator {worker_index + 1}/{len(sims)}",
            flush=True,
        )
        ray.kill(old_sim, no_restart=True)
        new_sim = sim_rebuilder(worker_index)
        if new_sim is None:
            raise RuntimeError(
                f"sim_rebuilder returned no actor for worker {worker_index}"
            )
        sims[worker_index] = new_sim
        if sim_rebuild_validator is not None:
            sim_rebuild_validator(worker_index, new_sim)


def run_rollout_cycle(
    sims,
    trainers,
    shard_iter: Iterator,
    trajectory_list: list,
    n_rollout: int,
    n_adv: int,
    vector_envs_per_sim: int = 1,
    sim_rebuilder=None,
    sim_rebuild_validator=None,
    episode_soft_timeout_seconds: float = 600.0,
    episode_hard_timeout_seconds: float = 900.0,
    sim_restart_limit: int = 1,
    runtime_metrics: Optional[Dict[str, float]] = None,
) -> Tuple[Any, list, Any, Any, list]:
    """Collect one rollout cycle and collate it into a trajectory batch.

    Matches train_rl.py L140-156 (rollout collection through the
    values/distances lookups). Also returns `log_list` -- structurally
    required by `stream_results_and_log` further down the pipeline, and
    produced by the same `collect_rollouts` call, so there's nowhere else
    to source it from without a second call.

    Returns: (traj_batch, model_inputs, values, distances, log_list)
    """
    if vector_envs_per_sim > 1:
        rollout_list, result_list, log_list = collect_vector_rollouts(
            sims,
            trainers,
            shard_iter,
            n_rollout,
            vector_envs_per_sim,
            episode_soft_timeout_seconds=episode_soft_timeout_seconds,
            episode_hard_timeout_seconds=episode_hard_timeout_seconds,
            sim_restart_limit=sim_restart_limit,
            sim_rebuilder=sim_rebuilder,
            sim_rebuild_validator=sim_rebuild_validator,
            timing_out=runtime_metrics,
        )
    else:
        rollout_list, result_list, log_list = collect_rollouts(
            sims, trainers, shard_iter, n_rollout
        )

    # ALIGNMENT INVARIANT: traj_batch row i and model_inputs[i] must describe the SAME
    # episode -- stored actions/old_log_prob are scored against those cached embeds. A
    # failed episode returns trajectory=None; letting collate_trajectories drop it
    # internally while model_inputs keeps all entries shifts every subsequent pairing and
    # trains episode X's actions against episode Y's embeds (a one-sided ppo_kl blowup
    # indistinguishable from real off-policy drift). Drop the failure as a UNIT and pad
    # back to size by duplicating kept episodes: every dispatch round must occupy all
    # num_vlms DDP ranks or the allreduce hangs, and grad-accum counters must stay
    # cycle-aligned.
    kept = [i for i, tup in enumerate(rollout_list) if tup[0] is not None]
    if not kept:
        raise RuntimeError(
            "every episode in this rollout cycle failed; check actor stdout for 'Episode failed'")
    if len(kept) < len(rollout_list):
        print(f"WARNING: {len(rollout_list) - len(kept)} episode(s) failed this cycle; "
              "padding the batch with duplicates of kept episodes")
        order = kept + [kept[i % len(kept)] for i in range(len(rollout_list) - len(kept))]
        rollout_list = [rollout_list[i] for i in order]
        result_list = [result_list[i] for i in order]
        log_list = [log_list[i] for i in order]

    trajectory_list += [tup[0] for tup in rollout_list]
    del trajectory_list[: max(0, len(trajectory_list) - n_adv)]
    traj_batch = collate_trajectories(trajectory_list)
    model_inputs = [(tup[1], tup[2]) for tup in rollout_list]

    values = traj_batch.get("values", None)
    distances = traj_batch.get("distance_to_goal", None)

    return traj_batch, model_inputs, values, distances, log_list


def build_eval_partition(sims, set_size: int, seed: int, uids: Optional[List[str]] = None):
    """Draw the FIXED eval set once (seeded, from the pool the sims already parsed) and
    partition it round-robin across sims. Fixed set => consecutive eval points are PAIRED
    on identical episodes; a fresh random sample each cycle would bury real movement
    under episode variance (measured: block-50 sd 0.063 at p=0.71)."""
    pool = sorted(ray.get(sims[0].list_episode_uids.remote()))
    if uids:
        # PINNED set: use it verbatim, in the given order. A redrawn set is a different
        # set -- change the pool, the filter, the size or the seed and every historical
        # number on it becomes incomparable in silence. Pinning is how a run's eval
        # survives those changes, and how an eval set can be HELD OUT of training
        # (see the env's `train_uids`).
        eval_uids = [u for u in uids if u]
        missing = [u for u in eval_uids if u not in set(pool)]
        if missing:
            raise KeyError(
                f"{len(missing)} pinned eval uid(s) are not in this pool "
                f"(e.g. {missing[:3]}). The pool must CONTAIN the eval episodes even "
                "when training never serves them.")
    else:
        rng = np.random.default_rng(seed)
        k = min(set_size, len(pool))
        chosen = sorted(rng.choice(len(pool), size=k, replace=False).tolist())
        eval_uids = [pool[i] for i in chosen]
    parts = [eval_uids[i::len(sims)] for i in range(len(sims))]
    return eval_uids, parts


def build_vector_eval_groups(
    sims,
    set_size: int,
    seed: int,
    slots_per_sim: int,
    uids: Optional[List[str]] = None,
):
    """Build fixed same-scene groups for batched NavVerse evaluation."""
    pool = sorted(ray.get(sims[0].list_episode_uids.remote()))
    pool_set = set(pool)
    wave_size = len(sims) * slots_per_sim

    if uids:
        eval_uids = [uid for uid in uids if uid]
        missing = [uid for uid in eval_uids if uid not in pool_set]
        if missing:
            raise KeyError(
                f"{len(missing)} pinned eval label(s) are not in this pool "
                f"(e.g. {missing[:3]})"
            )
        if len(eval_uids) != len(set(eval_uids)):
            raise ValueError("Pinned vector eval set contains duplicate labels")
    else:
        by_scene: Dict[str, List[str]] = {}
        for label in pool:
            by_scene.setdefault(label.rsplit("_", 1)[0], []).append(label)
        eligible = sorted(
            scene for scene, labels in by_scene.items() if len(labels) >= slots_per_sim
        )
        n_groups = set_size // slots_per_sim
        if set_size % slots_per_sim or n_groups > len(eligible):
            raise ValueError(
                f"Vector eval_set_size={set_size} needs {n_groups} distinct same-scene "
                f"groups of {slots_per_sim}; eligible scenes={len(eligible)}"
            )
        rng = np.random.default_rng(seed)
        scenes = [eligible[index] for index in rng.choice(
            len(eligible), size=n_groups, replace=False
        ).tolist()]
        groups = [
            [by_scene[scene][index] for index in rng.choice(
                len(by_scene[scene]), size=slots_per_sim, replace=False
            ).tolist()]
            for scene in scenes
        ]
        eval_uids = [uid for group in groups for uid in group]

    if len(eval_uids) == 0 or len(eval_uids) % wave_size:
        raise ValueError(
            f"Vector eval set must contain a positive multiple of wave_size={wave_size}; "
            f"got {len(eval_uids)}"
        )
    groups = [
        eval_uids[start : start + slots_per_sim]
        for start in range(0, len(eval_uids), slots_per_sim)
    ]
    for group in groups:
        scenes = {label.rsplit("_", 1)[0] for label in group}
        if len(group) != slots_per_sim or len(scenes) != 1:
            raise ValueError(
                f"Each vector eval group must have {slots_per_sim} labels from one scene; "
                f"got {group}"
            )
    return eval_uids, groups


def run_eval_cycle(sims, trainers, eval_parts, total, wandb_actor, global_cycle,
                   out_dir, ode: bool = True, vector_envs_per_sim: int = 1,
                   sim_rebuilder=None, episode_soft_timeout_seconds: float = 600.0,
                   episode_hard_timeout_seconds: float = 900.0,
                   sim_restart_limit: int = 1, sim_rebuild_validator=None,
                   stop_success_radius: float = 1.0):
    """One interleaved eval pass: fixed slices to exhaustion, pure-ODE sampler, no
    training-buffer contamination, one scalar wandb row, per-episode jsonl for pairing.

    Runs on the SAME actors as training -- zero extra GPU residents, which is also the
    fix for the eval-vs-training OOM class (2026-08-14, v3 crash)."""
    if wandb_actor is not None:
        ray.get(wandb_actor.set_context.remote(global_cycle, "eval"))
    ray.get([sim.set_media_enabled.remote(True) for sim in sims])
    if ode:
        ray.get([t.set_ode_sampling.remote(True) for t in trainers])
    try:
        if vector_envs_per_sim > 1:
            _, result_list, _ = collect_vector_rollouts(
                sims,
                trainers,
                iter(eval_parts),
                total,
                vector_envs_per_sim,
                postprocess_kwargs={"return_inputs": False, "eval": True},
                episode_soft_timeout_seconds=episode_soft_timeout_seconds,
                episode_hard_timeout_seconds=episode_hard_timeout_seconds,
                sim_restart_limit=sim_restart_limit,
                sim_rebuilder=sim_rebuilder,
                sim_rebuild_validator=sim_rebuild_validator,
            )
        else:
            for sim, part in zip(sims, eval_parts):
                sim.set_log_prefix.remote("eval_env/")
                sim.assign_shard.remote(list(part))
            _, result_list, _ = collect_rollouts(
                sims, trainers, iter([]), total,
                postprocess_kwargs={"return_inputs": False, "eval": True})
    finally:
        ray.get([sim.set_media_enabled.remote(False) for sim in sims])
        if ode:
            ray.get([t.set_ode_sampling.remote(False) for t in trainers])
        if vector_envs_per_sim == 1:
            for sim in sims:
                sim.set_log_prefix.remote("")
                sim.assign_shard.remote(None)   # back to the full training pool
    res = [r for r in result_list if r and not r.get("exhausted_sentinel")]
    expected_uids = [uid for part in eval_parts for uid in part]
    covered_uids = [str(r.get("episode_label")) for r in res]
    if len(res) != total or set(covered_uids) != set(expected_uids):
        missing = sorted(set(expected_uids) - set(covered_uids))
        unexpected = sorted(set(covered_uids) - set(expected_uids))
        raise RuntimeError(
            "Fixed evaluation coverage failed: "
            f"expected={len(expected_uids)} returned={len(res)} "
            f"missing={missing[:8]} unexpected={unexpected[:8]}"
        )
    def _m(key, cast=float):
        v = [cast(r.get(key, 0) or 0) for r in res]
        return float(np.mean(v)) if v else float("nan")
    row = {
        "eval/success_rate": _m("success"),
        "eval/oracle_success_rate": _m("oracle_success"),
        "eval/spl": _m("spl_fix") if any("spl_fix" in r for r in res)
        else _m("spl"),
        "eval/ospl": _m("ospl_fix") if any("ospl_fix" in r for r in res)
        else _m("oracle_spl"),
        "eval/mean_path_length_m": _m("path_length_m")
        if any("path_length_m" in r for r in res) else _m("path_length"),
        "eval/mean_action_path_length_m": _m("mean_action_path_length_m"),
        "eval/mean_steps": _m("steps"),
        "eval/episodes": len(res),
    }
    def _within_stop_radius(result):
        try:
            return bool(float(result.get("distance_to_goal")) <= stop_success_radius)
        except (TypeError, ValueError):
            return False

    counterfactual_keys = (
        "stop_counterfactual_tp", "stop_counterfactual_fp",
        "stop_counterfactual_fn", "stop_counterfactual_tn",
    )
    if all(all(key in result for key in counterfactual_keys) for result in res):
        stop_tp, stop_fp, stop_fn, stop_tn = (
            int(sum(result[key] for result in res)) for key in counterfactual_keys
        )
        stop_threshold = float(res[0]["stop_counterfactual_threshold"])
    else:
        stop_targets = np.asarray([_within_stop_radius(r) for r in res], dtype=bool)
        stop_predictions = np.asarray([
            str(r.get("termination_reason") or "") == "policy_stop" for r in res
        ], dtype=bool)
        stop_tp = int(np.sum(stop_predictions & stop_targets))
        stop_fp = int(np.sum(stop_predictions & ~stop_targets))
        stop_fn = int(np.sum(~stop_predictions & stop_targets))
        stop_tn = int(np.sum(~stop_predictions & ~stop_targets))
        stop_threshold = float("nan")
    row.update({
        "eval/stop_tp": stop_tp,
        "eval/stop_fp": stop_fp,
        "eval/stop_fn": stop_fn,
        "eval/stop_tn": stop_tn,
        "eval/stop_precision": stop_tp / max(stop_tp + stop_fp, 1),
        "eval/stop_recall": stop_tp / max(stop_tp + stop_fn, 1),
        "eval/policy_stop_rate": (stop_tp + stop_fp) / max(stop_tp + stop_fp + stop_fn + stop_tn, 1),
        "eval/stop_metric_mode": "counterfactual_per_decision",
        "eval/stop_threshold": stop_threshold,
        "eval/mean_stop_probability": _m("policy_stop_probability"),
        "eval/stop_success_radius_m": float(stop_success_radius),
    })
    # Episode finalizers encode MP4s asynchronously on the simulator actors. This host-wide
    # barrier makes their paths safe to hand to W&B before the cycle-level gallery is logged.
    ray.get([sim.flush_logs_to_disk.remote() for sim in sims])
    if wandb_actor is not None:
        ray.get(wandb_actor.log_eval_metrics.remote(dict(row), global_cycle))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "eval_episodes.jsonl"), "a") as f:
        for r in res:
            f.write(json.dumps({"cycle": global_cycle,
                                "uid": r.get("episode_label"),
                                "success": int(bool(r.get("success"))),
                                "ospl_fix": float(r.get("ospl_fix") or 0.0),
                                "steps": r.get("steps"),
                                "termination_reason": r.get("termination_reason"),
                                "policy_stop_mode": r.get("policy_stop_mode"),
                                "mean_action_path_length_m": r.get(
                                    "mean_action_path_length_m")}) + "\n")
    print(f"[eval cycle @ {global_cycle}] n={len(res)} success={row['eval/success_rate']:.3f} "
          f"oracle={row['eval/oracle_success_rate']:.3f} ospl={row['eval/ospl']:.3f}")
    return row


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
    rewards = traj_batch["rewards"]
    if (getattr(cfg.training.rl_config, "bootstrap_truncated", False)
            and values is not None and "bootstrap_eligible" in traj_batch.keys()):
        # A budget-capped episode is TRUNCATED, not terminated: treating the cap as
        # absorbing zeroes the tail's future value and under-credits long episodes.
        # Standard fix: fold gamma*V onto the last reward. V(s_T) (the last observed
        # state's value) stands in for V(s_{T+1}) -- one step stale, the usual
        # approximation when the post-terminal observation is never forwarded.
        rewards = rewards.clone()   # ep_rew logging must keep the raw rewards
        mask = traj_batch["response_mask"]
        last_idx = mask.sum(-1).long().clamp(min=1) - 1
        for i in range(rewards.shape[0]):
            li = int(last_idx[i])
            if bool(traj_batch["bootstrap_eligible"][i, li]):
                rewards[i, li] = rewards[i, li] + cfg.training.rl_config.gamma * values[i, li]
    adv_tuple = advantage_estimator_fn(
        token_level_rewards=rewards,
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

    # ---- probe counterfactual (frozen state-probe values present, e.g. the mini
    # parity run). PAIRED on the same buffer: variance of the advantage under the
    # probe baseline vs the configured kernel baseline vs no baseline, plus each
    # baseline's MSE against the realized returns. Purely observational -- what
    # trains is unchanged. Handed to stream_results_and_log via the module-level
    # slot below because the public 2-tuple return is pinned by tests; the driver
    # is single-threaded between these two calls.
    global _LAST_PROBE_COUNTERFACTUAL
    _LAST_PROBE_COUNTERFACTUAL = None
    if values is not None:
        vm = traj_batch["response_mask"] == 1
        r = returns[vm].float()
        pv = values[vm].float()
        cf = {
            "probe/value_mse": ((pv - r) ** 2).mean().item(),
            "probe/adv_var_probe": (r - pv).var().item(),
            "probe/adv_var_naive": r.var().item(),
        }
        if "baseline" in traj_batch.keys():
            kb = traj_batch["baseline"][vm].float()
            cf["probe/kernel_mse"] = ((kb - r) ** 2).mean().item()
            cf["probe/adv_var_kernel"] = (r - kb).var().item()
        def _corr(a, b):
            a = a - a.mean()
            b = b - b.mean()
            return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-8))
        cf["probe/corr_value_return"] = _corr(pv, r)
        _LAST_PROBE_COUNTERFACTUAL = cf
        print(f"PROBE COUNTERFACTUAL: {cf}")

    print(f"Advantage Mean: {advantages.mean().item():.4f}, Std: {advantages.std().item():.4f}")

    return traj_batch, global_return_mean


def minibatch_token_scales(response_mask, n_rollout: int):
    """Per-episode loss scales T_i / mean(T) over the cycle's episodes.

    Each minibatch is ONE episode with token-mean loss inside and equal weight in
    gradient accumulation -- an EPISODE-weighted objective. Multiplying minibatch i's
    loss by T_i/mean(T) makes the accumulated gradient the global TOKEN mean:
    mean_i[(T_i/T_bar) * token_mean_i] == token_mean over all tokens. Measured without
    it: episode-weighted mean advantage +0.26 vs token-weighted -0.02, i.e. quick
    successes carried ~4x per-token influence."""
    lengths = response_mask[:n_rollout].float().sum(-1).clamp(min=1)
    return (lengths / lengths.mean()).cpu().numpy()


def run_training_epochs(
    trainers,
    model_inputs,
    traj_batch,
    n_epoch: int,
    n_rollout: int,
    num_vlms: int,
    token_weighted: bool = False,
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
    scales = (minibatch_token_scales(traj_batch["response_mask"], n_rollout)
              if token_weighted else None)

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
                    loss_scale=(float(scales[global_idx]) if scales is not None else 1.0),
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
    runtime_metrics: Optional[Dict[str, float]] = None,
    update_started: Optional[float] = None,
    cycle_started: Optional[float] = None,
) -> dict:
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
    cycle_rows = []

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
                    # float, not bool: the wandb logger excludes bools from define_metric,
                    # so a bool here would be table-only and never chart. Guarded: eval
                    # trajectories may lack these keys.
                    **({"rollout/success": float(traj_stats["success"].max().item())}
                       if "success" in traj_stats.keys() else {}),
                    **({"rollout/oracle_success": float(traj_stats["oracle_success"].max().item())}
                       if "oracle_success" in traj_stats.keys() else {}),
                }
                if "probe_distance_m" in traj_stats.keys() and \
                        "distance_to_goal" in traj_stats.keys():
                    pd = traj_stats["probe_distance_m"].float()
                    td = traj_stats["distance_to_goal"].float()
                    fin = torch.isfinite(td)
                    if bool(fin.any()):
                        rollout_stats["probe/dist_mae_m"] = (pd[fin] - td[fin]).abs().mean().item()
                        rollout_stats["probe/dist_bias_m"] = (pd[fin] - td[fin]).mean().item()
                        near = td <= 1.0
                        if bool(near.any()):
                            rollout_stats["probe/p_stop_near"] = \
                                traj_stats["probe_p_stop"][near].float().mean().item()
                        if bool((~near & fin).any()):
                            rollout_stats["probe/p_stop_far"] = \
                                traj_stats["probe_p_stop"][~near & fin].float().mean().item()
                if "probe_p_stop" in traj_stats.keys() and "stop_target" in traj_stats.keys():
                    probability = traj_stats["probe_p_stop"].float()
                    target = traj_stats["stop_target"].float()
                    valid = torch.isfinite(probability) & torch.isfinite(target)
                    if bool(valid.any()):
                        predicted = probability[valid] >= 0.95
                        positive = target[valid] > 0.5
                        tp = (predicted & positive).sum().item()
                        fp = (predicted & ~positive).sum().item()
                        fn = (~predicted & positive).sum().item()
                        rollout_stats |= {
                            "probe/stop_tp": tp,
                            "probe/stop_fp": fp,
                            "probe/stop_fn": fn,
                            "probe/stop_precision": tp / max(tp + fp, 1),
                            "probe/stop_recall": tp / max(tp + fn, 1),
                            "probe/p_stop_near": probability[valid & (target > 0.5)].mean().item()
                            if bool((valid & (target > 0.5)).any()) else float("nan"),
                            "probe/p_stop_far": probability[valid & (target <= 0.5)].mean().item()
                            if bool((valid & (target <= 0.5)).any()) else float("nan"),
                        }
                if ("shadow_stop_action" in traj_stats.keys()
                        and "shadow_stop_reward" in traj_stats.keys()):
                    shadow_action = traj_stats["shadow_stop_action"].float()
                    shadow_reward = traj_stats["shadow_stop_reward"].float()
                    valid_shadow = torch.isfinite(shadow_action) & torch.isfinite(shadow_reward)
                    if bool(valid_shadow.any()):
                        rollout_stats |= {
                            "probe/shadow_stop_sample_rate": shadow_action[valid_shadow].mean().item(),
                            "probe/shadow_stop_reward_mean": shadow_reward[valid_shadow].mean().item(),
                        }
                if "exploration_gain_raw_m2" in traj_stats.keys():
                    rollout_stats["exploration/raw_gain_m2"] = \
                        traj_stats["exploration_gain_raw_m2"].float().sum().item()
                    rollout_stats["exploration/reward"] = \
                        traj_stats["exploration_reward"].float().sum().item()
                if "action_path_length_m" in traj_stats.keys():
                    path_lengths = traj_stats["action_path_length_m"].float()
                    finite_path_lengths = path_lengths[torch.isfinite(path_lengths)]
                    if len(finite_path_lengths):
                        rollout_stats["rollout/mean_action_path_length_m"] = (
                            finite_path_lengths.mean().item()
                        )
                if "post_goal_stillness_reward" in traj_stats.keys():
                    rollout_stats["rollout/post_goal_stillness_reward"] = (
                        traj_stats["post_goal_stillness_reward"].float().sum().item()
                    )
                if "values" in traj_stats.keys():
                    rollout_stats["probe/value_mae_ep"] = \
                        (traj_stats["values"].float() - traj_stats["returns"].float()).abs().mean().item()
                global _LAST_PROBE_COUNTERFACTUAL
                if completed_count == 0 and _LAST_PROBE_COUNTERFACTUAL:
                    rollout_stats |= _LAST_PROBE_COUNTERFACTUAL
                    _LAST_PROBE_COUNTERFACTUAL = None
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

                cycle_rows.append(result)
                completed_count += 1

                print(f"[{completed_count}/{total_tasks}] Complete. {result}")

            except Exception as e:
                logger.error(f"[{completed_count}/{total_tasks}] Task failed: {e}")

    metrics = aggregate_cycle_metrics(cycle_rows) if cycle_rows else {}
    if metrics:
        if runtime_metrics is not None:
            if update_started is not None:
                runtime_metrics["runtime/update_wall_seconds"] = (
                    time.perf_counter() - update_started
                )
            if cycle_started is not None:
                runtime_metrics["runtime/train_cycle_wall_seconds"] = (
                    time.perf_counter() - cycle_started
                )
        metrics.update(runtime_metrics)
    if wandb_actor is not None and metrics:
        ray.get(wandb_actor.log_cycle_metrics.remote(metrics, global_cycle, "train"))
    return metrics


def aggregate_cycle_metrics(rows: list) -> dict:
    """Reduce one rollout cycle to the small set of W&B curves used for decisions."""
    def _scalar(value):
        if isinstance(value, torch.Tensor):
            return value.detach().float().mean().item()
        if isinstance(value, np.ndarray):
            return float(np.asarray(value).mean())
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _mean(key):
        values = [_scalar(row.get(key)) for row in rows]
        values = [value for value in values if value is not None and np.isfinite(value)]
        return float(np.mean(values)) if values else float("nan")

    def _reason_rate(reason):
        return float(np.mean([
            str(row.get("termination_reason") or "") == reason for row in rows
        ]))

    return {
        "rollout/success_rate": _mean("rollout/success"),
        "rollout/oracle_success_rate": _mean("rollout/oracle_success"),
        "rollout/mean_reward": _mean("rollout/ep_rew"),
        "rollout/mean_return": _mean("rollout/ep_rtn"),
        "rollout/mean_steps": _mean("rollout/ep_len"),
        "rollout/bad_orientation_rate": _reason_rate("bad_orientation"),
        "rollout/out_of_bounds_rate": _reason_rate("terrain_out_of_bounds"),
        "rollout/truncated_rate": _mean("truncated"),
        "rollout/policy_stop_rate": _reason_rate("policy_stop"),
        "rollout/mean_action_path_length_m": _mean(
            "rollout/mean_action_path_length_m"
        ),
        "rollout/post_goal_stillness_reward": _mean(
            "rollout/post_goal_stillness_reward"
        ),
        "probe/stop_precision": _mean("probe/stop_precision"),
        "probe/stop_recall": _mean("probe/stop_recall"),
        "exploration/raw_gain_m2": _mean("exploration/raw_gain_m2"),
        "exploration/reward": _mean("exploration/reward"),
        "train/policy_loss": _mean("loss/pg_loss_scaled"),
        "train/value_mse": _mean("rollout/baseline_mse"),
        "train/ppo_kl": _mean("actor/ppo_kl"),
        "train/clip_fraction": _mean("actor/pg_clipfrac"),
        "train/grad_norm": _mean("train/grad_norm"),
        "train/lr": _mean("train/lr"),
        "policy/ref_kl": _mean("ref/kl_k2"),
        "policy/ref_log_ratio_p95": _mean("ref/r_p95"),
        "policy/ref_log_ratio_absmax": _mean("ref/r_absmax"),
        "policy/log_ratio_abs_mean": _mean("chain/abs_log_ratio_mean"),
        "policy/hidden_drift": _mean("chain/h_drift_from_init"),
    }


def save_rl_state(path: str, global_cycle: int, trajectory_list: list) -> None:
    """The driver-side state a weights checkpoint does NOT contain.

    `save_checkpoint_unsafe` writes the adapter, the optimizer and the scheduler -- so a
    resume already restores the model and the optimizer moments. What it cannot see is
    state the DRIVER owns: the advantage buffer (`trajectory_list`, the last `n_adv`
    episodes the time-kernel baseline is fitted on) and the cycle counter. Restarting
    without the buffer refits the baseline from empty, which is a real discontinuity in
    the advantage scale exactly when a run is resumed -- i.e. exactly when someone is
    trying to compare before and after.

    Written next to the weights so the two cannot drift apart. Failure to write is
    reported and swallowed: losing the buffer must never cost the checkpoint."""
    import torch
    try:
        torch.save({"schema": 1, "global_cycle": int(global_cycle),
                    "trajectory_list": trajectory_list},
                   os.path.join(path, "rl_state.pt"))
    except Exception as exc:                       # noqa: BLE001 -- see docstring
        print(f"WARNING: could not write rl_state.pt to {path}: {exc}", flush=True)


def load_rl_state(path: Optional[str]):
    """`(next_cycle, trajectory_list)` from a checkpoint dir, or `(0, [])`.

    Absent file, unreadable file or no path all mean "start fresh", so an ordinary launch
    and a resume from a pre-2026-08-15 checkpoint take the identical path. Returns the
    NEXT cycle to run, not the one that was saved."""
    if not path:
        return 0, []
    import torch
    f = os.path.join(path, "rl_state.pt")
    if not os.path.exists(f):
        return 0, []
    try:
        d = torch.load(f, map_location="cpu", weights_only=False)
    except Exception as exc:                       # noqa: BLE001
        print(f"WARNING: rl_state.pt at {path} is unreadable ({exc}); starting fresh",
              flush=True)
        return 0, []
    traj = list(d.get("trajectory_list") or [])
    cyc = int(d.get("global_cycle", -1)) + 1
    print(f"resuming: cycle {cyc}, advantage buffer {len(traj)} episodes (from {f})",
          flush=True)
    return cyc, traj


def emergency_checkpoint(trainers, global_cycle: int, output_dir: str, run_name: str,
                        trajectory_list: Optional[list] = None,
                        timeout_s: float = 180.0) -> Optional[str]:
    """Save on the way DOWN, after a fault, before the actors are torn down.

    Ordering is the whole point. The advantage buffer lives in the DRIVER's memory and
    needs no actor, no GPU and no collective, so it is written FIRST -- a dead rank, a
    wedged NCCL group or an OOM'd worker cannot cost us the thing that is cheapest to
    keep. Only then do we ask an actor for the weights, under a timeout, because that
    request is exactly what hangs when a rank has died (observed 2026-08-15: an OOM took
    rank 4's process group with it and the driver sat on a ray.get that never returned).

    Everything here is best-effort and swallows its own failures: a crash handler that
    raises replaces the real traceback with its own."""
    path = os.path.join(output_dir, run_name, "checkpoints", f"checkpoint_{global_cycle}_crash")
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as exc:                                   # noqa: BLE001
        print(f"emergency checkpoint: cannot create {path}: {exc}", flush=True)
        return None
    if trajectory_list is not None:
        save_rl_state(path, global_cycle, trajectory_list)     # driver-side, no actors
        print(f"emergency: advantage buffer saved ({len(trajectory_list)} episodes)", flush=True)
    try:
        ray.get(trainers[0].save_checkpoint_unsafe.remote(path), timeout=timeout_s)
        print(f"emergency: weights saved -> {path}", flush=True)
    except Exception as exc:                                   # noqa: BLE001
        print(f"emergency: weights NOT saved ({type(exc).__name__}: {exc}); "
              f"the buffer at {path} still pairs with the last periodic checkpoint",
              flush=True)
    return path


def _save_checkpoint_snapshot(
    trainers,
    final_dir: str,
    global_cycle: int,
    trajectory_list: Optional[list] = None,
) -> str:
    """Write one complete immutable checkpoint directory before publishing it."""
    checkpoint_root = os.path.dirname(final_dir)
    os.makedirs(checkpoint_root, exist_ok=True)
    temp_dir = os.path.join(
        checkpoint_root,
        f".{os.path.basename(final_dir)}.tmp-{os.getpid()}",
    )
    if os.path.lexists(final_dir) or os.path.lexists(temp_dir):
        raise FileExistsError(
            f"refusing to overwrite an existing checkpoint path: {final_dir}"
        )
    try:
        ray.get(trainers[0].save_checkpoint_unsafe.remote(temp_dir))
        if trajectory_list is not None:
            save_rl_state(temp_dir, global_cycle, trajectory_list)
        required = ["optimizer.pt", "scheduler.pt"]
        required.extend(ray.get(trainers[0].checkpoint_required_files.remote()))
        required.append(
            "adapter_model.safetensors"
            if os.path.exists(os.path.join(temp_dir, "adapter_model.safetensors"))
            else "adapter_model.bin"
        )
        if trajectory_list is not None:
            required.append("rl_state.pt")
        missing = [
            name for name in required
            if not os.path.isfile(os.path.join(temp_dir, name))
            or os.path.getsize(os.path.join(temp_dir, name)) == 0
        ]
        if missing:
            raise RuntimeError(f"incomplete checkpoint {temp_dir}: missing {missing}")
        os.replace(temp_dir, final_dir)
    except BaseException:
        if os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir)
        raise
    return final_dir


def maybe_save_eval_bests(
    trainers,
    eval_row: Dict[str, float],
    global_cycle: int,
    output_dir: str,
    run_name: str,
    trajectory_list: Optional[list] = None,
) -> None:
    """Keep one immutable snapshot for each newly best fixed-eval metric."""
    slots = {
        "best_ospl": "eval/ospl",
        "best_spl": "eval/spl",
        "best_sr": "eval/success_rate",
    }
    checkpoint_root = os.path.join(output_dir, run_name, "checkpoints")
    metadata_path = os.path.join(checkpoint_root, "best_metrics.json")
    previous = {}
    if os.path.isfile(metadata_path):
        with open(metadata_path) as f:
            previous = json.load(f)
    improved = [
        (slot, metric, float(eval_row[metric]))
        for slot, metric in slots.items()
        if metric in eval_row and math.isfinite(float(eval_row[metric]))
        and float(eval_row[metric]) > float(previous.get(slot, {}).get("value", -math.inf))
    ]
    if not improved:
        return
    snapshot = _save_checkpoint_snapshot(
        trainers,
        os.path.join(checkpoint_root, f"checkpoint_eval_{global_cycle}"),
        global_cycle,
        trajectory_list,
    )
    for slot, metric, value in improved:
        _set_checkpoint_link(checkpoint_root, slot, snapshot)
        previous[slot] = {
            "metric": metric,
            "value": value,
            "cycle": global_cycle,
            "checkpoint": os.path.basename(snapshot),
        }
    temp_metadata = f"{metadata_path}.tmp-{os.getpid()}"
    with open(temp_metadata, "w") as f:
        json.dump(previous, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(temp_metadata, metadata_path)
    print(
        "fixed-eval best checkpoint saved: "
        + ", ".join(f"{slot}={value:.6f}" for slot, _, value in improved),
        flush=True,
    )


def maybe_checkpoint(
    trainers,
    global_cycle: int,
    save_step: int,
    output_dir: str,
    run_name: str,
    trajectory_list: Optional[list] = None,
) -> None:
    """Save every completed cycle and atomically advance ``checkpoints/latest``.

    Numbered checkpoints remain on the configured ``save_step`` cadence. Between
    them, exactly one hidden rolling directory is retained; the ``latest`` symlink
    changes only after weights, optimizer, scheduler, and driver state are complete.
    """
    steps_until_save = (global_cycle + 1) % save_step
    checkpoint_root = os.path.join(output_dir, run_name, "checkpoints")
    final_name = (
        f"checkpoint_{global_cycle}"
        if steps_until_save == 0
        else f".latest_cycle_{global_cycle}"
    )
    final_dir = os.path.join(checkpoint_root, final_name)
    _save_checkpoint_snapshot(trainers, final_dir, global_cycle, trajectory_list)
    _set_checkpoint_link(checkpoint_root, "latest", final_dir)

    if steps_until_save == 0:
        print(f"numbered checkpoint saved -> {final_dir}", flush=True)
    else:
        print(
            f"rolling latest saved at cycle {global_cycle}; "
            f"T-{save_step - steps_until_save} cycles until numbered checkpoint",
            flush=True,
        )


def _set_checkpoint_link(checkpoint_root: str, link_name: str, target_dir: str) -> None:
    checkpoint_root = os.path.abspath(checkpoint_root)
    target_dir = os.path.abspath(target_dir)
    if os.path.dirname(target_dir) != checkpoint_root:
        raise ValueError("latest checkpoint target must be inside checkpoint_root")

    latest = os.path.join(checkpoint_root, link_name)
    if os.path.lexists(latest) and not os.path.islink(latest):
        raise RuntimeError(f"checkpoint link path is not a symlink: {latest}")
    old_target = (
        os.path.abspath(os.path.join(checkpoint_root, os.readlink(latest)))
        if os.path.islink(latest) else None
    )

    temp_link = os.path.join(checkpoint_root, f".latest-link-{os.getpid()}")
    if os.path.lexists(temp_link):
        os.unlink(temp_link)
    os.symlink(os.path.basename(target_dir), temp_link)
    os.replace(temp_link, latest)

    if (
        link_name == "latest"
        and
        old_target
        and old_target != target_dir
        and os.path.dirname(old_target) == checkpoint_root
        and os.path.basename(old_target).startswith(".latest_cycle_")
        and os.path.isdir(old_target)
    ):
        shutil.rmtree(old_target)
