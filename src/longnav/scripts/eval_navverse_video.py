"""Run a pinned batched NavVerse checkpoint evaluation and wait for video encoding."""

import os
import time

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


register_configs()


@hydra.main(version_base=None, config_name="rl_config", config_path="../config")
def main(cfg: RLConfig):
    import ray

    from longnav.utils.train_loop import (
        bootstrap_all,
        build_vector_eval_groups,
        run_eval_cycle,
    )

    if not cfg.task.eval_uids_file:
        raise ValueError("task.eval_uids_file is required")
    with open(cfg.task.eval_uids_file) as file:
        pinned = [
            label.strip()
            for label in file.read().replace("\n", ",").split(",")
            if label.strip()
        ]

    ctx = bootstrap_all(cfg, training=False)
    try:
        slots = ctx.vector_envs_per_sim
        if slots > 1:
            eval_uids, eval_parts = build_vector_eval_groups(
                ctx.sims, len(pinned), cfg.task.eval_seed, slots, uids=pinned
            )
        else:
            # Scalar workers each need their own slice. Giving the full list to only
            # the first worker leaves the other workers on their default sampler.
            eval_uids = pinned
            eval_parts = [pinned[i::len(ctx.sims)] for i in range(len(ctx.sims))]
        run_dir = os.path.join(cfg.task.output_dir, cfg.task.run_name)
        row = run_eval_cycle(
            ctx.sims,
            ctx.trainers,
            eval_parts,
            len(eval_uids),
            ctx.wandb_actor,
            0,
            run_dir,
            ode=cfg.task.eval_ode,
            vector_envs_per_sim=slots,
            sim_rebuilder=ctx.sim_rebuilder,
            sim_rebuild_validator=ctx.sim_rebuild_validator,
            episode_soft_timeout_seconds=ctx.episode_soft_timeout_seconds,
            episode_hard_timeout_seconds=ctx.episode_hard_timeout_seconds,
            sim_restart_limit=ctx.sim_restart_limit,
        )
        # Per-slot flush calls are queued by collect_vector_rollouts. A host-wide flush
        # submitted afterward is a barrier that guarantees every MP4 has closed.
        ray.get([sim.flush_logs_to_disk.remote() for sim in ctx.sims])
        print(f"VIDEO_EVAL_COMPLETE {row}", flush=True)
    finally:
        for trainer in ctx.trainers:
            ray.kill(trainer)
        for sim in ctx.sims:
            ray.kill(sim)
        if ctx.wandb_actor is not None:
            try:
                ray.get(ctx.wandb_actor.close.remote(), timeout=30.0)
            except Exception as exc:
                print(f"W&B logger close failed: {exc}", flush=True)
            ray.kill(ctx.wandb_actor)
        if ctx.placement_groups:
            from ray.util.placement_group import remove_placement_group

            for group in ctx.placement_groups:
                remove_placement_group(group)
        ray.shutdown()
        time.sleep(1)


if __name__ == "__main__":
    main()
