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

from verl.single_controller import ray
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
    shard_iter = ctx.shard_iter
    logger = ctx.logger
    num_rollouts = ctx.num_rollouts

    trajectory_list = []

    def cleanup():
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

    try:
        for global_cycle in range(num_rollouts):
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
            )

            del model_inputs
    finally:

        cleanup()

if __name__ == "__main__":
    main()
