"""Short, no-update diagnostics for the history-conditioned STOP policy.

Set ``STOP_DIAG_KIND=trace`` for a fixed-eval trace or ``gradient`` for a full-history
training-batch gradient reading.  Hydra still supplies the checkpoint, environment and
two-GPU resource configuration, so these probes exercise the production code path.
"""
import json
import os

import hydra
import numpy as np

from longnav.conf.register_configs import register_configs
from longnav.config_schema import RLConfig

register_configs()


def _cleanup(ctx):
    import ray

    for trainer in ctx.trainers:
        ray.kill(trainer)
    for sim in ctx.sims:
        ray.kill(sim)
    if ctx.wandb_actor is not None:
        try:
            ray.get(ctx.wandb_actor.close.remote(), timeout=30)
        except Exception as exc:  # noqa: BLE001 - teardown must preserve the diagnosis
            print(f"W&B close failed during diagnostic teardown: {exc}", flush=True)
        ray.kill(ctx.wandb_actor)
    ray.shutdown()


def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as file:
        json.dump(payload, file, indent=2, sort_keys=True)


def _trace_eval(cfg):
    from longnav.utils.train_loop import bootstrap_all, build_eval_partition, run_eval_cycle

    ctx = bootstrap_all(cfg, training=False)
    typed = ctx.bootstrapper.typed_cfg
    run_dir = os.path.join(typed.task.output_dir, typed.task.run_name)
    try:
        if not typed.task.eval_uids_file:
            raise ValueError("STOP trace diagnostic requires task.eval_uids_file")
        with open(typed.task.eval_uids_file) as file:
            uids = [line.strip() for line in file if line.strip()]
        _, parts = build_eval_partition(ctx.sims, len(uids), typed.task.eval_seed, uids=uids)
        row = run_eval_cycle(
            ctx.sims,
            ctx.trainers,
            parts,
            len(uids),
            None,
            0,
            run_dir,
            ode=True,
            vector_envs_per_sim=ctx.vector_envs_per_sim,
            sim_rebuilder=ctx.sim_rebuilder,
            sim_rebuild_validator=ctx.sim_rebuild_validator,
            episode_soft_timeout_seconds=ctx.episode_soft_timeout_seconds,
            episode_hard_timeout_seconds=ctx.episode_hard_timeout_seconds,
            sim_restart_limit=ctx.sim_restart_limit,
            stop_success_radius=float(typed.sim.success_distance),
            save_stop_traces=True,
        )
        _write_json(os.path.join(run_dir, "stop_trace_summary.json"), row)
        print(json.dumps(row, indent=2, sort_keys=True), flush=True)
    finally:
        _cleanup(ctx)


def _gradient_batch(cfg):
    import ray

    from longnav.utils.train_loop import bootstrap_all, run_rollout_cycle

    ctx = bootstrap_all(cfg, training=True)
    typed = ctx.bootstrapper.typed_cfg
    run_dir = os.path.join(typed.task.output_dir, typed.task.run_name)
    n_rollout = int(os.environ.get("STOP_DIAG_N_ROLLOUT", "4"))
    if n_rollout < len(ctx.trainers):
        raise ValueError("STOP_DIAG_N_ROLLOUT must cover every training worker")
    try:
        trajectories, model_inputs, _, _, _ = run_rollout_cycle(
            ctx.sims,
            ctx.trainers,
            ctx.shard_iter,
            [],
            n_rollout,
            int(typed.training.rl_config.n_adv),
            ctx.vector_envs_per_sim,
            sim_rebuilder=ctx.sim_rebuilder,
            sim_rebuild_validator=ctx.sim_rebuild_validator,
            episode_soft_timeout_seconds=ctx.episode_soft_timeout_seconds,
            episode_hard_timeout_seconds=ctx.episode_hard_timeout_seconds,
            sim_restart_limit=ctx.sim_restart_limit,
        )
        selected = []
        for index in range(n_rollout):
            targets = trajectories["stop_target"][index].cpu().numpy()
            if np.isfinite(targets).any() and np.any(targets > 0.5):
                selected.append(index)
            if len(selected) == len(ctx.trainers):
                break
        if len(selected) != len(ctx.trainers):
            raise RuntimeError(
                "gradient diagnostic batch contains too few reached-goal histories; "
                "increase STOP_DIAG_N_ROLLOUT"
            )
        futures = []
        for trainer, index in zip(ctx.trainers, selected):
            tensors, metadata = model_inputs[index]
            futures.append(trainer.diagnose_stop_loss_gradients.remote(
                tensors,
                metadata,
                trajectories["stop_target"][index:index + 1].cpu().numpy(),
                trajectories["shadow_stop_action"][index:index + 1].cpu().numpy(),
                trajectories["shadow_stop_reward"][index:index + 1].cpu().numpy(),
            ))
        readings = ray.get(futures)
        payload = {
            "n_rollout_collected": n_rollout,
            "selected_rollout_indices": selected,
            "selected_lengths": [
                int(trajectories["response_mask"][index].sum().item()) for index in selected
            ],
            "readings": readings,
        }
        path = os.path.join(run_dir, "stop_gradient_diagnostic.json")
        _write_json(path, payload)
        print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    finally:
        _cleanup(ctx)


@hydra.main(version_base=None, config_name="rl_config", config_path="../config")
def main(cfg: RLConfig):
    kind = os.environ.get("STOP_DIAG_KIND", "trace")
    if kind == "trace":
        _trace_eval(cfg)
    elif kind == "gradient":
        _gradient_batch(cfg)
    else:
        raise ValueError("STOP_DIAG_KIND must be 'trace' or 'gradient'")


if __name__ == "__main__":
    main()
