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
import torch

from longnav.config_schema import RLConfig
from longnav.utils.factories import ExpBootstrapper, get_console_logger, get_shard_iterator
from longnav.utils.rl_core import collate_trajectories
from longnav.utils.rollout_core import collect_rollouts

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

    # The two step caps must agree, and nothing else checks it (2026-08-24 audit, F4):
    # if the VLM loop's cap (rollout.max_steps) is SMALLER than the env's
    # (sim.max_steps), episodes end with done=False and truncated=False on their last
    # stored step, so bootstrap_truncated never corrects them and their returns are
    # computed as if the episode continued for free. Larger is harmless (the env
    # truncates first, flagged properly).
    _sim_cap = getattr(bootstrapper.typed_cfg.sim, "max_steps", None)
    _roll_cap = bootstrapper.typed_cfg.rollout.max_steps
    if _sim_cap is not None and _roll_cap < int(_sim_cap):
        raise ValueError(
            f"rollout.max_steps={_roll_cap} < sim.max_steps={_sim_cap}: the rollout "
            "loop would cut episodes the env never flags as truncated, and their "
            "returns would silently skip the truncation bootstrap. Set "
            "rollout.max_steps >= sim.max_steps."
        )

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
    # TRAINING honours the workers' rejection verdicts: a rejected episode is replaced,
    # so the batch is n_rollout clean episodes and no blind row reaches collation. Eval
    # (run_eval_cycle) passes False -- see there.
    rollout_list, result_list, log_list = collect_rollouts(
        sims, trainers, shard_iter, n_rollout, respect_rejections=True)

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


def run_eval_cycle(sims, trainers, eval_parts, total, wandb_actor, global_cycle,
                   out_dir, ode: bool = True):
    """One interleaved eval pass: fixed slices to exhaustion, pure-ODE sampler, no
    training-buffer contamination, one scalar wandb row, per-episode jsonl for pairing.

    Runs on the SAME actors as training -- zero extra GPU residents, which is also the
    fix for the eval-vs-training OOM class (2026-08-14, v3 crash)."""
    if ode:
        ray.get([t.set_ode_sampling.remote(True) for t in trainers])
    for sim, part in zip(sims, eval_parts):
        sim.set_log_prefix.remote("eval_env/")
        sim.assign_shard.remote(list(part))
    try:
        _, result_list, _ = collect_rollouts(
            sims, trainers, iter([]), total,
            postprocess_kwargs={"return_inputs": False, "eval": True},
            # NEVER on eval: the pinned set defines the denominator, and silently
            # replacing episodes would make the number describe a different (easier)
            # population than the one the set names. Blind episodes are part of what
            # is being measured.
            respect_rejections=False)
    finally:
        if ode:
            ray.get([t.set_ode_sampling.remote(False) for t in trainers])
        for sim in sims:
            sim.set_log_prefix.remote("")
            sim.assign_shard.remote(None)   # back to the full training pool
    res = [r for r in result_list if r and not r.get("exhausted_sentinel")]
    def _m(key, cast=float):
        v = [cast(r.get(key, 0) or 0) for r in res]
        return float(np.mean(v)) if v else float("nan")
    row = {
        "eval/success": _m("success"), "eval/oracle_success": _m("oracle_success"),
        "eval/ospl_fix": _m("ospl_fix"), "eval/path_length_m": _m("path_length_m"),
        "eval/steps": _m("steps"), "eval/n": len(res),
        "eval/global_cycle": global_cycle,
    }
    if wandb_actor is not None:
        wandb_actor.log_row.remote(dict(row))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "eval_episodes.jsonl"), "a") as f:
        for r in res:
            f.write(json.dumps({"cycle": global_cycle,
                                "uid": r.get("episode_label"),
                                "success": int(bool(r.get("success"))),
                                "ospl_fix": float(r.get("ospl_fix") or 0.0),
                                "steps": r.get("steps")}) + "\n")
    print(f"[eval cycle @ {global_cycle}] n={len(res)} success={row['eval/success']:.3f} "
          f"oracle={row['eval/oracle_success']:.3f} ospl={row['eval/ospl_fix']:.3f}")
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
    # RTC return re-timing (docs/RTC_RL.md section 4): move each interval's
    # COMMITTED-tick progress to the action that committed it,
    #   r~_k = r_fresh_k + gamma * r_commit_{k+1}  ==  R_k - r_commit_k  (the
    # subtraction identity; gamma is the unique weight for which it holds).
    # Unbiased, variance-reducing, objective-preserving. Applies only when the env
    # emitted the split (RTC runs); the flag exists for the ablation. Padding rows
    # carry r_commit = 0, so the shift needs no length bookkeeping, and the terminal
    # step correctly receives no next-commit term (its tail never executed).
    if (getattr(cfg.training.rl_config, "retime_commit_rewards", True)
            and "r_commit" in traj_batch.keys() and "r_fresh" in traj_batch.keys()):
        gamma = cfg.training.rl_config.gamma
        r_commit = traj_batch["r_commit"]
        next_commit = torch.zeros_like(r_commit)
        next_commit[:, :-1] = r_commit[:, 1:]
        rewards = (traj_batch["r_fresh"] + (rewards - traj_batch["r_fresh"]
                                            - r_commit) + gamma * next_commit).float()
        # .float(): r_commit/r_fresh arrive float64 when the env emits numpy scalars,
        # and a float64 rewards tensor propagates into Double returns/baseline while
        # the kernel critic predicts float32 -- the index_put dtype crash of the first
        # code_rl_a0 launch. Identity for float32 inputs.
        # (rewards - r_fresh - r_commit) preserves any non-split reward component
        # exactly; with the env's exhaustive split it is ~0 by construction.
        traj_batch["rewards"] = rewards
    if (getattr(cfg.training.rl_config, "bootstrap_truncated", False)
            and values is not None and "truncated" in traj_batch.keys()):
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
            if bool(traj_batch["truncated"][i, li]):
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
        # HONEST kernel reference: under a critic estimator, traj_batch["baseline"]
        # IS the critic, so reading it as "the kernel" compares the critic to itself
        # (observed: critic smoke 2026-08-19, kernel_mse == value_mse identically).
        # Fit the time kernel counterfactually on the same buffer instead, matching
        # the estimator's configuration.
        try:
            from longnav.utils.rl_core import BinnedKernelCritic
            _rlc = cfg.training.rl_config
            _t = torch.arange(returns.shape[1], device=returns.device).unsqueeze(0) \
                .expand(returns.shape[0], -1)
            if getattr(_rlc, "time_alignment", "start") == "end":
                _t = _t - traj_batch["response_mask"].sum(-1).long().unsqueeze(1)
            _tf = _t[vm]
            _kc = BinnedKernelCritic(n_bins=1024, device=returns.device)
            _kc.fit(_tf, r, sigma=getattr(_rlc, "time_kernel_sigma", 40.0))
            kb = _kc.predict(_tf).float()
            cf["probe/kernel_mse"] = ((kb - r) ** 2).mean().item()
            cf["probe/adv_var_kernel"] = (r - kb).var().item()
        except Exception as _e:   # the gauge must never kill training
            print(f"counterfactual kernel fit failed: {_e}")
        def _corr(a, b):
            a = a - a.mean(); b = b - b.mean()
            return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-8))
        cf["probe/corr_value_return"] = _corr(pv, r)
        # buffer-level explained variance, plus its between/within-episode split
        # (the per-minibatch critic/explained_variance is within ONE episode only).
        cf["probe/explained_var_pooled"] = float(1.0 - (r - pv).var() / r.var().clamp_min(1e-5))
        _rm = traj_batch["response_mask"]
        _ep_mean_g = (returns * _rm).sum(-1) / _rm.sum(-1).clamp_min(1)
        _ep_mean_v = (values * _rm).sum(-1) / _rm.sum(-1).clamp_min(1)
        cf["probe/explained_var_between"] = float(
            1.0 - (_ep_mean_g - _ep_mean_v).var() / _ep_mean_g.var().clamp_min(1e-5))
        _rw = (returns - _ep_mean_g.unsqueeze(1))[vm]
        _vw = (values - _ep_mean_v.unsqueeze(1))[vm]
        cf["probe/explained_var_within"] = float(1.0 - (_rw - _vw).var() / _rw.var().clamp_min(1e-5))
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

                if wandb_actor is not None:
                    wandb_actor.log_row.remote(result)
                completed_count += 1

                print(f"[{completed_count}/{total_tasks}] Complete. {result}")

            except Exception as e:
                logger.error(f"[{completed_count}/{total_tasks}] Task failed: {e}")


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


def maybe_checkpoint(
    trainers,
    global_cycle: int,
    save_step: int,
    output_dir: str,
    run_name: str,
    trajectory_list: Optional[list] = None,
) -> None:
    """Matches train_rl.py L294-300. Gated independently from
    `stream_results_and_log` since checkpointing happens every `save_step`
    cycles, not every cycle."""
    steps_until_save = (global_cycle + 1) % save_step
    if steps_until_save == 0:
        print("saving checkpoiutsnt")
        ckpt_dir = os.path.join(output_dir, run_name, "checkpoints",
                                f"checkpoint_{global_cycle}")
        ray.get(trainers[0].save_checkpoint_unsafe.remote(ckpt_dir))
        # None (the default) keeps the pre-2026-08-15 behaviour exactly: weights only.
        if trajectory_list is not None:
            save_rl_state(ckpt_dir, global_cycle, trajectory_list)
    else:
        print(f"T-{steps_until_save} steps until checkpoint!")
