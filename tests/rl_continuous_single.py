from longnav.config_schema import *
from longnav.conf.env_configs import DummyContinuousEnvConfig
from longnav.conf.vlm_configs import GaussianHeadConfig
from longnav.utils.factories import ExpBootstrapper, get_shard_iterator
from longnav.utils.rollout_core import collect_rollouts
from longnav.utils.rl_core import collate_trajectories
from verl.trainer.ppo.core_algos import get_adv_estimator_fn

import ray
import numpy as np
import json

cfg = RLConfig()
cfg.resources.osm_gb = 28
cfg.resources.num_vlms = 1
cfg.resources.vlm_gpu_fraction = 0.3
cfg.resources.num_sims = 2
cfg.resources.vlm_conda_env = None
cfg.resources.habitat_conda_env = None
cfg.sim = DummyContinuousEnvConfig()

cfg.vlm.attn_impl = "sdpa"
cfg.vlm.save_outputs = True
cfg.vlm.policy_head = GaussianHeadConfig(action_space_dim=2)

cfg.rollout.convo_start_template = [
    {"role": "user", "content": [{"type": "text", "text": "example substitution: $instr_or_goal"}]},
    {"role": "user", "content": [{"type": "image"}]},
    {"role": "assistant", "content": [{"type": "text", "text": "**forward**"}]},
]
cfg.rollout.max_steps = 16
cfg.training.rl_config.n_rollout = 1
advantage_estimator_fn = get_adv_estimator_fn("reinforce_plus_plus")

cfg.task.run_name = "rl_cont_step"
bootstrapper = ExpBootstrapper(cfg)

bootstrapper.setup_cluster()

trainers = bootstrapper.bootstrap_vlms_rl(training=True)
sims = bootstrapper.bootstrap_sims()
try:
    wandb_actor, _ = bootstrapper.bootstrap_logger()
except Exception as e:
    wandb_actor = None
    print(f"Logger setup failed with error: {e}. Continuing without logger.")

trajectory_list = []

rollout_list, result_list, log_list = collect_rollouts(
    sims,
    trainers,
    get_shard_iterator(0),
    bootstrapper.typed_cfg.training.rl_config.n_rollout,
)
trajectory_list += [tup[0] for tup in rollout_list]
trajectory_list = trajectory_list[-bootstrapper.typed_cfg.training.rl_config.n_adv :]
traj_batch = collate_trajectories(trajectory_list)
model_inputs = [(tup[1], tup[2]) for tup in rollout_list]

values = traj_batch.get("values", None)

print("Computing Advantages")
adv_tuple = advantage_estimator_fn(
    token_level_rewards=traj_batch["rewards"],
    values=values,
    response_mask=traj_batch["response_mask"],
    config=cfg.training.rl_config,
)
advantages, returns = adv_tuple[0], adv_tuple[1]
traj_batch["advantages"] = advantages
traj_batch["returns"] = returns

print(f"traj batch shape: {traj_batch.shape}")
traj_batch = traj_batch[-bootstrapper.typed_cfg.training.rl_config.n_rollout :]

print("training step:")

future_metadata = {}
training_futures = []
perm_indices = np.random.permutation(bootstrapper.typed_cfg.training.rl_config.n_rollout)
for batch_start in range(
    0,
    bootstrapper.typed_cfg.training.rl_config.n_rollout,
    bootstrapper.typed_cfg.resources.num_vlms,
):
    step_futures = []
    for worker_idx, trainer in enumerate(trainers):
        global_idx = batch_start + worker_idx
        global_idx = perm_indices[global_idx]
        ref = trainer.train_rl_step.remote(
            *model_inputs[global_idx],
            traj_batch[global_idx : global_idx + 1, traj_batch["response_mask"][global_idx].bool()],
        )
        step_futures.append(ref)
        future_metadata[ref] = global_idx
    training_futures.extend(step_futures)

print(f"Dispatched {len(training_futures)} training tasks for epoch")
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
            }
            result |= rollout_stats

            log_ref = log_list[rollout_idx]
            try:
                log_path = ray.get(log_ref, timeout=30.0)
                with open(log_path, "r") as f:
                    vlm_log_dict = json.load(f)
                result |= vlm_log_dict
            except ray.exceptions.GetTimeoutError:
                print(f"Log file for rollout {rollout_idx} not ready. Skipping detailed logs.")
            except Exception as e:
                print(f"Failed to read log file: {e}")

            if wandb_actor is not None:
                wandb_actor.log_row.remote(result)
            completed_count += 1

            print(f"[{completed_count}/{total_tasks}] Complete. {result}")
        except Exception as e:
            print(f"Error in training future: {e}")
            completed_count += 1
            print(f"[{completed_count}/{total_tasks}] Complete with error. Check logs for details.")

if wandb_actor is not None:
    ray.get(wandb_actor.close.remote())
