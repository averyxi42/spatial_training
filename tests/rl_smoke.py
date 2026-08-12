from longnav.config_schema import *
from longnav.conf.env_configs import DummyDiscreteEnvConfig
from longnav.conf.vlm_configs import LMHeadConfig
from longnav.conf.logging_configs import WandbLoggerConfig
from longnav.utils.factories import ExpBootstrapper, get_shard_iterator, get_console_logger
from longnav.utils.train_loop import (
    run_rollout_cycle,
    compute_advantages_and_returns,
    run_training_epochs,
    stream_results_and_log,
)
from verl.trainer.ppo.core_algos import get_adv_estimator_fn

import ray
cfg = RLConfig()
cfg.resources.osm_gb=8
cfg.resources.num_vlms=1
cfg.resources.vlm_gpu_fraction=0.3
cfg.resources.num_sims=2
cfg.resources.vlm_conda_env=None
cfg.resources.habitat_conda_env=None
cfg.sim = DummyDiscreteEnvConfig()
cfg.vlm.policy_head = LMHeadConfig()

cfg.vlm.attn_impl = "sdpa"
cfg.vlm.save_outputs=True
cfg.rollout.convo_start_template=[
        {"role": "user", "content": [{"type": "text", "text": "example substitution: $instr_or_goal"}]},
        {"role": "user", "content": [{"type": "image"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "**forward**"}]}
    ]
cfg.rollout.max_steps=16
cfg.training.rl_config.n_rollout=4
advantage_estimator_fn = get_adv_estimator_fn("reinforce_plus_plus")

cfg.task.run_name = "rl_step"
cfg.task.logger = WandbLoggerConfig(project="longnav_smoke_test")
bootstrapper = ExpBootstrapper(cfg)
logger = get_console_logger()

bootstrapper.setup_cluster()

trainers = bootstrapper.bootstrap_vlms_rl(training=True)
sims = bootstrapper.bootstrap_sims()
try:
    wandb_actor,_ = bootstrapper.bootstrap_logger()
except Exception as e:
    wandb_actor = None
    print(f"Logger setup failed with error: {e}. Continuing without logger.")


trajectory_list = []

for i in range(3):
    traj_batch, model_inputs, values, distances, log_list = run_rollout_cycle(
        sims,
        trainers,
        get_shard_iterator(0),
        trajectory_list,
        bootstrapper.typed_cfg.training.rl_config.n_rollout,
        bootstrapper.typed_cfg.training.rl_config.n_adv,
    )

    print("Computing Advantages")
    traj_batch, global_return_mean = compute_advantages_and_returns(traj_batch, advantage_estimator_fn, cfg)

    print(f"traj batch shape: {traj_batch.shape}")
    traj_batch = traj_batch[-bootstrapper.typed_cfg.training.rl_config.n_rollout:] # only train on most recent.

    print("training step:")

    training_futures, future_metadata = run_training_epochs(
        trainers,
        model_inputs,
        traj_batch,
        1, # one logged epoch, matching the original single-epoch smoke test
        bootstrapper.typed_cfg.training.rl_config.n_rollout,
        bootstrapper.typed_cfg.resources.num_vlms,
    )

    print(f"Dispatched {len(training_futures)} training tasks for epoch")
    stream_results_and_log(
        training_futures,
        future_metadata,
        traj_batch,
        wandb_actor,
        i,
        global_return_mean,
        logger,
        log_list,
    )
if wandb_actor is not None:
    ray.get(wandb_actor.close.remote())
