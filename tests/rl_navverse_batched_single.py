"""Single-cycle smoke test for the BATCHED NavVerse host/proxy machinery: one host running
slots_per_host=2 vectorized robots (the smallest configuration that actually exercises two
siblings waiting on each other in NavVerseHostActor's reset/step barrier), one rollout per
slot, one training step, then exit. See tests/rl_navverse_single.py for the single-actor
(unbatched) counterpart, which should be run and pass first.
"""

from longnav.config_schema import *
from longnav.conf.env_configs import NavVerseProxyBatchedEnvConfig
from longnav.conf.vlm_configs import FlowSDEHeadConfig
from longnav.utils.factories import ExpBootstrapper, get_shard_iterator, get_console_logger
from longnav.utils.train_loop import (
    run_rollout_cycle,
    compute_advantages_and_returns,
    run_training_epochs,
    stream_results_and_log,
)
from verl.trainer.ppo.core_algos import get_adv_estimator_fn

import ray

CHECKPOINT_DIR = (
    "/home/ubuntu/.cache/navverse_huggingface/hub/"
    "models--Aasdfip--longnav-objectnav-flow-nopose-cotrain-2p5hz/"
    "snapshots/5090ef4c9eeddec1230a5118303fafbdd4353155"
)

cfg = RLConfig()
cfg.resources.osm_gb = 64
cfg.resources.num_vlms = 2
# 1.0, not a fraction: 2 DDP ranks MUST land on distinct physical GPUs (NCCL refuses
# "Duplicate GPU detected" otherwise), and Ray's fractional-GPU scheduling doesn't guarantee
# that -- only a whole-GPU request per rank does, on a node with enough GPUs to spare.
cfg.resources.vlm_gpu_fraction = 1.0
cfg.resources.num_sims = 2  # must equal sim.num_hosts * sim.slots_per_host
cfg.resources.sim_gpu_fraction = 0.0  # proxies are near-zero resource; host claims its own GPU
cfg.resources.sim_cpus = 0
cfg.resources.vlm_conda_env = None
cfg.resources.habitat_conda_env = "navverse"

cfg.sim = NavVerseProxyBatchedEnvConfig()
cfg.sim.num_hosts = 1
cfg.sim.slots_per_host = 2
cfg.sim.host_num_gpus = 0.5
cfg.sim.episode_folder = "/home/ubuntu/Projects/navverse_data/episodes/"
cfg.sim.scene_folder = "/home/ubuntu/Projects/navverse_data/"
cfg.sim.episodes_path = "/home/ubuntu/Projects/NavVerse-Benchmark/episodes/rl_navverse_smoke128_16x8.txt"
cfg.sim.task_type = "placenav"
cfg.sim.gap = 10
cfg.sim.dt = 0.04
cfg.sim.max_steps = 20  # short: this is a wiring check, not a training signal
cfg.sim.success_distance = 1.6
cfg.sim.minimal_logging = True

cfg.vlm.attn_impl = "flash_attention_2"
cfg.vlm.save_outputs = True
cfg.vlm.policy_head = FlowSDEHeadConfig(
    checkpoint_dir=CHECKPOINT_DIR,
    gap=10,
    sde_n=1,
    sde_noise_a=0.15,
)

cfg.rollout.convo_start_template = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    "You are a robot navigating an indoor environment toward a goal.\n"
                    "Goal: $instr_or_goal\nAt each step you receive the current RGB "
                    "observation. Produce the next short trajectory of poses to follow, "
                    "relative to your current pose."
                ),
            }
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Observation 0:"},
            {"type": "image"},
            {"type": "text", "text": "Action:"},
        ],
    },
    {"role": "assistant", "content": [{"type": "text", "text": "**____**"}]},
]
cfg.rollout.convo_turn_template = [
    {"role": "assistant", "content": [{"type": "text", "text": "**____**"}]},
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Observation $step:"},
            {"type": "image"},
            {"type": "text", "text": "Action:"},
        ],
    },
    {"role": "assistant", "content": [{"type": "text", "text": "**____**"}]},
]
cfg.rollout.max_steps = 20

cfg.training.checkpoint = CHECKPOINT_DIR + "/adapter"
cfg.training.load_optim = False
cfg.training.load_sched = False
cfg.training.action_head_learning_rate = 1e-6
cfg.training.rl_config.n_rollout = 2
cfg.training.rl_config.n_adv = 2
cfg.training.rl_config.entropy_bonus = 0.0
cfg.training.rl_config.token_weighted_loss = True
cfg.training.rl_config.bootstrap_truncated = False
advantage_estimator_fn = get_adv_estimator_fn(cfg.training.rl_config.advantage_estimator)

cfg.task.run_name = "navverse_rl_batched_single_smoke"
cfg.task.output_dir = "/home/ubuntu/Projects/NavVerse-Benchmark/dump/navverse_flow_rl_smoke"
bootstrapper = ExpBootstrapper(cfg)
logger = get_console_logger()

bootstrapper.setup_cluster()

print("bootstrapping VLM workers...")
trainers = bootstrapper.bootstrap_vlms_rl(training=True)
print("bootstrapping sim actors (2 proxies -> 1 host, slots_per_host=2)...")
sims = bootstrapper.bootstrap_sims()
try:
    wandb_actor, _ = bootstrapper.bootstrap_logger()
except Exception as e:
    wandb_actor = None
    print(f"Logger setup failed with error: {e}. Continuing without logger.")

trajectory_list = []

print("collecting rollouts (both slots should reset/step in lockstep)...")
traj_batch, model_inputs, values, distances, log_list = run_rollout_cycle(
    sims,
    trainers,
    get_shard_iterator(0),
    trajectory_list,
    bootstrapper.typed_cfg.training.rl_config.n_rollout,
    bootstrapper.typed_cfg.training.rl_config.n_adv,
)

print("computing advantages...")
traj_batch, global_return_mean = compute_advantages_and_returns(traj_batch, advantage_estimator_fn, cfg)
print(f"traj batch shape: {traj_batch.shape}")
traj_batch = traj_batch[-bootstrapper.typed_cfg.training.rl_config.n_rollout :]

print("running one training step...")
training_futures, future_metadata = run_training_epochs(
    trainers,
    model_inputs,
    traj_batch,
    1,
    bootstrapper.typed_cfg.training.rl_config.n_rollout,
    bootstrapper.typed_cfg.resources.num_vlms,
)

print(f"dispatched {len(training_futures)} training tasks")
stream_results_and_log(
    training_futures,
    future_metadata,
    traj_batch,
    wandb_actor,
    0,
    global_return_mean,
    logger,
    log_list,
)

if wandb_actor is not None:
    ray.get(wandb_actor.close.remote())

print("BATCHED SMOKE TEST PASSED")
