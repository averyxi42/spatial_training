'''
'''

import math
# _ROOT = Path(__file__).resolve().parents[1]
# if str(_ROOT) not in sys.path:
#     sys.path.insert(0, str(_ROOT))
import hydra
from longnav.conf.register_configs import register_configs
from longnav.config_schema import RLConfig
import os
import itertools
import json

DEBUG_FLAG = False

# 1. Register our command variants
register_configs()
@hydra.main(version_base=None, config_name="rl_config",config_path='../config')
def main(cfg: RLConfig):
    # keep heavy imports here so hydra tab complete is snappier?
    import ray
    import time

    from longnav.utils.rollout_core import collect_rollouts
    from longnav.utils.train_loop import (
        bootstrap_all,
        build_eval_partition,
        build_vector_eval_groups,
        run_eval_cycle,
    )

    print(f"Model ID: {cfg.vlm.model_id}")

    ctx = bootstrap_all(cfg, training=False)
    bootstrapper = ctx.bootstrapper
    trainers = ctx.trainers
    sims = ctx.sims
    wandb_actor = ctx.wandb_actor
    shard_iter = ctx.shard_iter
    logger = ctx.logger

    shard_iter,shard_iter_copy = itertools.tee(shard_iter)
    try:
        all_episodes = [s for shard in shard_iter_copy for s in shard]
    except:
        all_episodes = [None]*10000 #fallback

    def cleanup():
        for trainer in trainers:
            ray.kill(trainer)
        for sim in sims:
            ray.kill(sim)
        if wandb_actor is not None:
            ray.get(wandb_actor.close.remote())
            time.sleep(15)
            ray.kill(wandb_actor)
        ray.shutdown()

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

    eval_uids_file = getattr(bootstrapper.typed_cfg.task, "eval_uids_file", None)
    if eval_uids_file:
        with open(eval_uids_file) as f:
            uids = [uid.strip() for uid in f.read().replace("\n", ",").split(",")
                    if uid.strip()]
        vector_envs_per_sim = ctx.vector_envs_per_sim
        if vector_envs_per_sim > 1:
            _, eval_parts = build_vector_eval_groups(
                sims,
                len(uids),
                int(getattr(bootstrapper.typed_cfg.task, "eval_seed", 0)),
                vector_envs_per_sim,
                uids=uids,
            )
        else:
            _, eval_parts = build_eval_partition(
                sims,
                len(uids),
                int(getattr(bootstrapper.typed_cfg.task, "eval_seed", 0)),
                uids=uids,
            )
        run_dir = os.path.join(
            bootstrapper.typed_cfg.task.output_dir,
            bootstrapper.typed_cfg.task.run_name,
        )
        row = run_eval_cycle(
            sims,
            trainers,
            eval_parts,
            len(uids),
            wandb_actor,
            0,
            run_dir,
            ode=bool(getattr(bootstrapper.typed_cfg.task, "eval_ode", True)),
            vector_envs_per_sim=vector_envs_per_sim,
            sim_rebuilder=ctx.sim_rebuilder,
            sim_rebuild_validator=ctx.sim_rebuild_validator,
            episode_soft_timeout_seconds=ctx.episode_soft_timeout_seconds,
            episode_hard_timeout_seconds=ctx.episode_hard_timeout_seconds,
            sim_restart_limit=ctx.sim_restart_limit,
            stop_success_radius=float(bootstrapper.typed_cfg.sim.success_distance),
        )
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "eval_summary.json"), "w") as f:
            json.dump(row, f, indent=2, sort_keys=True)
        print(json.dumps(row, indent=2, sort_keys=True), flush=True)
        cleanup()
        return

    # ------------------------------------------- rollouts ------------------------------------------
    batch_size = 32 # fixed batch size decoupled from RL logic for eval
    for i in range(max(math.ceil(len(all_episodes)/batch_size),1)):
        logger.info("Starting rollout collection!")
        rollout_list,result_list,log_list = collect_rollouts(sims,trainers,shard_iter,batch_size,{"return_inputs":False,"eval":True}) #
        if len(rollout_list) == 0:
            print("rollout list empty, exiting")
            break
        # save for analysis
        # pickle_obj(rollout_list, f"rollout_{i}")
        # pickle_obj(result_list, f"result_{i}")
        # pickle_obj(log_list,f"logpaths_{i}")
    ray.get(log_list)
    cleanup()

if __name__ == "__main__":
    main()
