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

DEBUG_FLAG = False

# 1. Register our command variants
register_configs()
@hydra.main(version_base=None, config_name="rl_config",config_path='../config')
def main(cfg: RLConfig):
    # keep heavy imports here so hydra tab complete is snappier?
    import ray
    import time

    from longnav.utils.rollout_core import collect_rollouts
    from longnav.utils.train_loop import bootstrap_all
    from longnav.utils.probe_eval_report import (aggregate, format_summary,
                                                 write_episode_records)

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

    # ------------------------------------------- rollouts ------------------------------------------
    run_dir = os.path.join(bootstrapper.typed_cfg.task.output_dir,
                           bootstrapper.typed_cfg.task.run_name)
    os.makedirs(run_dir, exist_ok=True)
    written = 0
    batch_size = 32 # fixed batch size decoupled from RL logic for eval
    for i in range(max(math.ceil(len(all_episodes)/batch_size),1)):
        logger.info("Starting rollout collection!")
        rollout_list,result_list,log_list = collect_rollouts(sims,trainers,shard_iter,batch_size,{"return_inputs":False,"eval":True}) #
        if len(rollout_list) == 0:
            print("rollout list empty, exiting")
            break
        # Per-episode probe records, written as each BATCH lands so a long eval has an
        # early read and nothing accumulates in memory. Aggregation is pure
        # post-processing over these files (probe_eval_report.aggregate), so it can be
        # run here at the end or standalone against the finished run directory.
        if getattr(bootstrapper.typed_cfg.training.rl_config, "state_probe", None):
            n = write_episode_records(run_dir, rollout_list, result_list)
            written += n
            try:
                print(format_summary(aggregate(run_dir, write=False)), flush=True)
            except Exception as exc:               # never let reporting kill an eval
                print(f"[probe] interim summary unavailable ({exc})", flush=True)
        # save for analysis
        # pickle_obj(rollout_list, f"rollout_{i}")
        # pickle_obj(result_list, f"result_{i}")
        # pickle_obj(log_list,f"logpaths_{i}")
    ray.get(log_list)
    if written:
        summary = aggregate(run_dir)
        print("\n===== FINAL =====")
        print(format_summary(summary), flush=True)
        print(f"records: {os.path.join(run_dir, 'probe_records.jsonl')}")
    cleanup()

if __name__ == "__main__":
    main()
