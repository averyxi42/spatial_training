'''Interleaved train/eval RL driver.

train_rl.py plus a periodic in-loop evaluation pass, controlled by four task fields
(config_schema.RunConfig):

  task.eval_every      one eval pass every N training cycles (default 4)
  task.eval_set_size   FIXED eval set size; 0 = n_rollout (default 0)
  task.eval_seed       seed for drawing the eval set from the training pool (default 0)
  task.eval_ode        run the chain head as the pure ODE during eval (default true)
  task.eval_uids_file  PIN the eval set to a file of uids instead of drawing it (default
                       none). With sim.train_uids naming a disjoint set, this is what makes
                       eval genuinely held out rather than drawn from the training pool --
                       see docs/TRAINING_EVAL_SET.md.

Design decisions, each load-bearing:
  * The eval set is FIXED -- seeded and drawn once from the pool the sims already parsed,
    or pinned verbatim by task.eval_uids_file --
    and partitioned across sims to run to exhaustion -- consecutive eval points are
    PAIRED on identical episodes, which is what buys resolution at small n (a fresh
    random sample each time would bury real movement under episode variance).
  * Eval runs on the SAME actors as training: no second model, no policy server, zero
    extra GPU residents -- the memory-contention crash class of 2026-08-14 cannot occur.
  * Eval rollouts never enter the training buffer, and the chain head samples the pure
    ODE (the deploy arbiter) for their duration.
  * Absolute eval numbers are under the TRAINING env config (success_distance, budget,
    train scenes) -- they are the relative progress curve, not a sample101 replacement.

Launch exactly like train_rl.py:
  python3 -m longnav.scripts.train_eval_rl +training=hapo +resources=octo \
      +experiment=<exp> resources.vlm_conda_env=longnav_vlm \
      task.eval_every=8 task.eval_set_size=64
'''
import math
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


@hydra.main(version_base=None, config_name="rl_config", config_path='../config')
def main(cfg: RLConfig):
    cfg.vlm.save_outputs = True
    import ray

    from longnav.utils.train_loop import (
        bootstrap_all,
        build_eval_partition,
        build_vector_eval_groups,
        compute_advantages_and_returns,
        maybe_checkpoint,
        maybe_save_eval_bests,
        recycle_vector_sims,
        run_eval_cycle,
        run_rollout_cycle,
        run_training_epochs,
        stream_results_and_log,
     load_rl_state, emergency_checkpoint)
    from verl.trainer.ppo.core_algos import get_adv_estimator_fn

    advantage_estimator_fn = get_adv_estimator_fn(cfg.training.rl_config.advantage_estimator)
    print(f"Model ID: {cfg.vlm.model_id}")

    ctx = bootstrap_all(cfg, training=True)
    bootstrapper, trainers, sims = ctx.bootstrapper, ctx.trainers, ctx.sims
    wandb_actor, shard_iter, logger = ctx.wandb_actor, ctx.shard_iter, ctx.logger
    num_rollouts = ctx.num_rollouts
    tcfg = bootstrapper.typed_cfg
    n_rollout = tcfg.training.rl_config.n_rollout
    num_vlms = len(trainers)

    eval_every = int(getattr(tcfg.task, "eval_every", 4))
    eval_set_size = int(getattr(tcfg.task, "eval_set_size", 0)) or n_rollout
    eval_seed = int(getattr(tcfg.task, "eval_seed", 0))
    eval_ode = bool(getattr(tcfg.task, "eval_ode", True))
    run_dir = os.path.join(tcfg.task.output_dir, tcfg.task.run_name)

    # Draw the fixed eval set up front (the sims have parsed the pool during bootstrap's
    # first reset) and log its identity so any later analysis can reproduce it.
    if getattr(tcfg.sim, "train_uids", None) and int(getattr(tcfg.task, "shard_size", 0)) > 0:
        raise ValueError(
            "sim.train_uids is set but task.shard_size > 0. The uid filter applies only to "
            "the trivial (None) shard, so an explicitly sharded run would train on the WHOLE "
            "pool while the config says otherwise -- and the eval set would no longer be "
            "held out. Set shard_size: 0.")
    eval_uids_file = getattr(tcfg.task, "eval_uids_file", None)
    pinned = None
    if eval_uids_file:
        with open(eval_uids_file) as f:
            pinned = [u.strip() for u in f.read().replace("\n", ",").split(",") if u.strip()]
    if ctx.vector_envs_per_sim > 1:
        eval_uids, eval_parts = build_vector_eval_groups(
            sims,
            eval_set_size,
            eval_seed,
            ctx.vector_envs_per_sim,
            uids=pinned,
        )
    else:
        eval_uids, eval_parts = build_eval_partition(
            sims, eval_set_size, eval_seed, uids=pinned
        )
    if pinned:
        logger.info(f"eval set: PINNED from {eval_uids_file}")
    logger.info(f"eval set: {len(eval_uids)} fixed episodes (seed {eval_seed}), "
                f"every {eval_every} cycles, ode={eval_ode}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "eval_set_uids.txt"), "w") as f:
        f.write("\n".join(eval_uids) + "\n")

    # RESUME. `training.checkpoint` always restores weights (and optimizer/scheduler
    # when load_optim/load_sched are set).  Driver state is separate so a checkpoint can
    # also initialize an independent study without inheriting its cycle or baseline.
    if tcfg.training.resume_driver_state:
        start_cycle, trajectory_list = load_rl_state(tcfg.training.checkpoint)
    else:
        start_cycle, trajectory_list = 0, []
        logger.info("starting fresh driver state from checkpoint weights")
    if start_cycle > 0 and ctx.vector_envs_per_sim > 1:
        if n_rollout % ctx.vector_envs_per_sim:
            raise ValueError(
                "Vector resume requires n_rollout to be divisible by slots_per_host"
            )
        skipped_groups = start_cycle * n_rollout // ctx.vector_envs_per_sim
        shard_iter.skip_groups(skipped_groups)
        print(
            f"resuming: advanced scene sampler by {skipped_groups} groups "
            f"for {start_cycle} completed cycles",
            flush=True,
        )
    global_cycle = start_cycle          # defined before the loop so a fault during
                                        # setup still names a cycle in the crash dir

    def cleanup():
        for trainer in trainers:
            ray.kill(trainer)
        for sim in sims:
            ray.kill(sim)
        if wandb_actor is not None:
            try:
                ray.get(wandb_actor.close.remote(), timeout=30.0)
            except Exception as exc:
                logger.warning(f"W&B logger close failed: {exc}")
            ray.kill(wandb_actor)
        if ctx.placement_groups:
            from ray.util.placement_group import remove_placement_group

            for group in ctx.placement_groups:
                remove_placement_group(group)
        ray.shutdown()

    try:
        for global_cycle in range(start_cycle, num_rollouts):
            # ------------------------- interleaved eval pass -------------------------
            if eval_every > 0 and global_cycle % eval_every == 0:
                logger.info(f"Eval pass @ cycle {global_cycle}")
                eval_row = run_eval_cycle(
                    sims, trainers, eval_parts, len(eval_uids), wandb_actor,
                    global_cycle, run_dir, ode=eval_ode,
                    vector_envs_per_sim=ctx.vector_envs_per_sim,
                    sim_rebuilder=ctx.sim_rebuilder,
                    sim_rebuild_validator=ctx.sim_rebuild_validator,
                    episode_soft_timeout_seconds=ctx.episode_soft_timeout_seconds,
                    episode_hard_timeout_seconds=ctx.episode_hard_timeout_seconds,
                    sim_restart_limit=ctx.sim_restart_limit,
                    stop_success_radius=float(tcfg.sim.success_distance),
                )
                if bool(getattr(tcfg.task, "save_eval_bests", True)):
                    maybe_save_eval_bests(
                        trainers, eval_row, global_cycle, tcfg.task.output_dir,
                        tcfg.task.run_name, trajectory_list,
                    )

            # ------------------------------ rollouts ---------------------------------
            logger.info("Starting rollout collection!")
            cycle_started = time.perf_counter()
            runtime_metrics = {}
            if wandb_actor is not None:
                ray.get(wandb_actor.set_context.remote(global_cycle, "train"))
            if ctx.vector_envs_per_sim > 1:
                ray.get([sim.set_media_enabled.remote(False) for sim in sims])
            traj_batch, model_inputs, values, distances, log_list = run_rollout_cycle(
                sims, trainers, shard_iter, trajectory_list, n_rollout,
                tcfg.training.rl_config.n_adv,
                ctx.vector_envs_per_sim,
                sim_rebuilder=ctx.sim_rebuilder,
                sim_rebuild_validator=ctx.sim_rebuild_validator,
                episode_soft_timeout_seconds=ctx.episode_soft_timeout_seconds,
                episode_hard_timeout_seconds=ctx.episode_hard_timeout_seconds,
                sim_restart_limit=ctx.sim_restart_limit,
                runtime_metrics=runtime_metrics,
            )
            print("done collecting")

            # ------------------------------ advantages -------------------------------
            advantage_started = time.perf_counter()
            traj_batch, global_return_mean = compute_advantages_and_returns(
                traj_batch, advantage_estimator_fn, cfg)
            traj_batch = traj_batch[-n_rollout:]  # only train on most recent
            runtime_metrics["runtime/advantage_wall_seconds"] = (
                time.perf_counter() - advantage_started
            )

            # ------------------------------ training ---------------------------------
            logger.info("Starting training")
            update_started = time.perf_counter()
            training_futures, future_metadata = run_training_epochs(
                trainers, model_inputs, traj_batch,
                tcfg.training.rl_config.n_epoch, n_rollout, num_vlms,
                token_weighted=getattr(tcfg.training.rl_config,
                                       "token_weighted_loss", False),
            )
            cycle_metrics = stream_results_and_log(
                training_futures, future_metadata, traj_batch, wandb_actor,
                global_cycle, global_return_mean, logger, log_list, runtime_metrics,
                update_started, cycle_started,
            )
            max_ref_p95 = getattr(
                tcfg.training.rl_config, "max_ref_log_ratio_p95", None
            )
            observed_ref_p95 = cycle_metrics.get("policy/ref_log_ratio_p95")
            if (max_ref_p95 is not None and observed_ref_p95 is not None
                    and math.isfinite(observed_ref_p95)
                    and observed_ref_p95 > float(max_ref_p95)):
                raise RuntimeError(
                    "reference-density drift guard tripped: "
                    f"p95={observed_ref_p95:.6f} > {float(max_ref_p95):.6f}"
                )

            maybe_checkpoint(trainers, global_cycle, tcfg.training.save_step,
                             tcfg.task.output_dir, tcfg.task.run_name,
                             trajectory_list=trajectory_list)
            del model_inputs
            recycle_every = ctx.sim_recycle_every_cycles
            if (
                ctx.vector_envs_per_sim > 1
                and recycle_every > 0
                and (global_cycle + 1) % recycle_every == 0
            ):
                logger.info(
                    f"Recycling all simulators before cycle {global_cycle + 1}"
                )
                recycle_vector_sims(
                    sims,
                    ctx.sim_rebuilder,
                    ctx.sim_rebuild_validator,
                )
    except BaseException as exc:
        # FAULT PATH. Save before teardown: an OOM or a dead rank otherwise costs every
        # cycle since the last save_step boundary. Re-raises, so the real traceback is
        # what the operator sees.
        print(f"FAULT at cycle {global_cycle}: {type(exc).__name__}: {exc}", flush=True)
        try:
            emergency_checkpoint(trainers, global_cycle, tcfg.task.output_dir,
                                 tcfg.task.run_name, trajectory_list)
        except BaseException as exc2:                     # noqa: BLE001
            print(f"emergency checkpoint itself failed: {exc2}", flush=True)
        raise
    finally:
        cleanup()


if __name__ == "__main__":
    main()
