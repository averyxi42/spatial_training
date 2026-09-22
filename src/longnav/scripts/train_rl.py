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
# NUCLEAR THREAD CAP: Must be set before importing numpy/torch/ray
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_ENABLE_PARALLEL_LOADING"] = "false"
import sys

import hydra
from longnav.conf.register_configs import register_configs
from longnav.config_schema import RLConfig
import os

DEBUG_FLAG = False
FREEZE_DATA = False # for debugging only
# 1. Register our command variants
register_configs()
@hydra.main(version_base=None, config_name="rl_config",config_path='../config')
def main(cfg: RLConfig):
    cfg.vlm.save_outputs = True
    # keep heavy imports here so hydra tab complete is snappier?
    import ray

    from longnav.utils.factories import get_shard_iterator
    from longnav.utils.train_loop import (
        bootstrap_all,
        run_rollout_cycle,
        compute_advantages_and_returns,
        run_training_epochs,
        stream_results_and_log,
        maybe_checkpoint,
        load_rl_state,
    )
    from verl.trainer.ppo.core_algos import get_adv_estimator_fn

    def debug_signal_handler(sig, frame):
        # should allow us to interrupt the loop, save data etc, and resume
        global DEBUG_FLAG
        DEBUG_FLAG = True
        decision = input("debug: y, exit: n, wait: any other key")
        if decision == 'y':
            import ipdb
            ipdb.set_trace()
        if decision == 'n':
            try:
                cleanup()
            finally:
                sys.exit()
    # signal.signal(signal.SIGINT, debug_signal_handler)

    advantage_estimator_fn = get_adv_estimator_fn(cfg.training.rl_config.advantage_estimator)
    print(f"Model ID: {cfg.vlm.model_id}")

    ctx = bootstrap_all(cfg, training=True)
    bootstrapper = ctx.bootstrapper
    trainers = ctx.trainers
    sims = ctx.sims
    wandb_actor = ctx.wandb_actor
    # NOTE (behavior change, deliberate): the pre-extraction code built a second,
    # redundant get_shard_iterator() right after this one and used that one for
    # the training loop -- it dropped the `excluded_episodes` the first (dead)
    # call had wired in, so resumed runs silently retrained on already-completed
    # episodes. bootstrap_all's shard_iter carries excluded_episodes through
    # (matching what eval.py always did), so resumed runs now correctly skip
    # them -- a bug fix, not a regression, but flagging it since it changes
    # resumed-run behavior and isn't visible to the dummy-env smoke tests
    # (excluded_episodes is None there).
    shard_iter = ctx.shard_iter
    logger = ctx.logger
    num_rollouts = ctx.num_rollouts

    if bootstrapper.typed_cfg.training.resume_driver_state:
        start_cycle, trajectory_list = load_rl_state(
            bootstrapper.typed_cfg.training.checkpoint
        )
    else:
        start_cycle, trajectory_list = 0, []
        logger.info("starting fresh driver state from checkpoint weights")
    resume_cycle = bootstrapper.typed_cfg.training.resume_cycle
    if resume_cycle is not None:
        if resume_cycle < 0:
            raise ValueError("training.resume_cycle must be non-negative")
        start_cycle = int(resume_cycle)
        if not trajectory_list:
            logger.info(
                "resuming global cycle from explicit override without an advantage buffer"
            )

    def cleanup():
        for trainer in trainers:
            ray.kill(trainer)
        for sim in sims:
            ray.kill(sim)
        if wandb_actor is not None:
            ray.kill(wandb_actor)
        if ctx.placement_groups:
            from ray.util.placement_group import remove_placement_group

            for group in ctx.placement_groups:
                remove_placement_group(group)
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

    try:
        for global_cycle in range(start_cycle, num_rollouts):
            if FREEZE_DATA:
                # reset the dataset
                shard_iter = get_shard_iterator(
                    subset_label= cfg.task.subset_label,
                    episode_json= cfg.task.episode_json,
                    shard_size=cfg.task.shard_size,
                    logger=logger
                )
            # ------------------------------------------- rollouts ------------------------------------------
            logger.info("Starting rollout collection!")

            traj_batch, model_inputs, values, distances, log_list = run_rollout_cycle(
                sims,
                trainers,
                shard_iter,
                trajectory_list,
                bootstrapper.typed_cfg.training.rl_config.n_rollout,
                bootstrapper.typed_cfg.training.rl_config.n_adv,
                ctx.vector_envs_per_sim,
                sim_rebuilder=ctx.sim_rebuilder,
                sim_rebuild_validator=ctx.sim_rebuild_validator,
                episode_soft_timeout_seconds=ctx.episode_soft_timeout_seconds,
                episode_hard_timeout_seconds=ctx.episode_hard_timeout_seconds,
                sim_restart_limit=ctx.sim_restart_limit,
            )

            print("done collecting")
            num_vlms = len(trainers)

            # ---------------------------------- compute gae ----------------------------------------------
            print("Computing Advantages")
            traj_batch, global_return_mean = compute_advantages_and_returns(traj_batch, advantage_estimator_fn, cfg)
            traj_batch = traj_batch[-bootstrapper.typed_cfg.training.rl_config.n_rollout:] # only train on most recent.

            if DEBUG_FLAG:
                debug() # great spot to intercept the trajectories for saving etc

            # ------------------------------ training (extra + final logged epoch) ----------
            logger.info("Starting training")

            training_futures, future_metadata = run_training_epochs(
                trainers,
                model_inputs,
                traj_batch,
                bootstrapper.typed_cfg.training.rl_config.n_epoch,
                bootstrapper.typed_cfg.training.rl_config.n_rollout,
                num_vlms,
                token_weighted=getattr(
                    bootstrapper.typed_cfg.training.rl_config,
                    "token_weighted_loss", False),
            )

            print(f"Dispatched {len(training_futures)} training tasks for final epoch")
            # ------------------------------------- monitor training live ---------------------------------
            stream_results_and_log(
                training_futures,
                future_metadata,
                traj_batch,
                wandb_actor,
                global_cycle,
                global_return_mean,
                logger,
                log_list,
            )

            #------------------------------------ save checkpoint ------------------------------------
            maybe_checkpoint(
                trainers,
                global_cycle,
                bootstrapper.typed_cfg.training.save_step,
                bootstrapper.typed_cfg.task.output_dir,
                bootstrapper.typed_cfg.task.run_name,
                trajectory_list=trajectory_list,
            )

            del model_inputs
    finally:

        cleanup()

if __name__ == "__main__":
    main()
