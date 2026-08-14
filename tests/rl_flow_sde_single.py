"""One full RL cycle with the flow-SDE chain head: rollout -> advantages -> training step.

`tests/rl_continuous_single.py` with the Gaussian head swapped for `FlowSDEHeadConfig` on the
REAL cotrain-v3 checkpoint, against the dummy continuous env. What this proves that the unit
tests cannot: the `{"h"}` dict survives both passthroughs, the chain and `sde_positions` ride
the trajectory dict through collation, the recompute branch produces `old_log_prob` from
`chain_log_prob_batch`, and `rl_loss` computes a finite pg_loss from the chain ratio with the
discrete-standard estimator -- the whole loop, on a GPU, with the 2B backbone in place.

Run:  /workspace/conda/envs/longnav_vlm/bin/python tests/rl_flow_sde_single.py
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from longnav.config_schema import *                                  # noqa: E402,F403
from longnav.conf.env_configs import DummyContinuousEnvConfig        # noqa: E402
from longnav.conf.vlm_configs import FlowSDEHeadConfig               # noqa: E402
from longnav.utils.factories import (ExpBootstrapper,                # noqa: E402
                                     get_console_logger, get_shard_iterator)
from longnav.utils.train_loop import (compute_advantages_and_returns,  # noqa: E402
                                      run_rollout_cycle, run_training_epochs,
                                      stream_results_and_log)
from verl.trainer.ppo.core_algos import get_adv_estimator_fn         # noqa: E402

CKPT = "/Projects/spatial_training/dump/pose_injection/run_cotrain_v3_nopose_mix/checkpoint-12000"

cfg = RLConfig()                                                     # noqa: F405
cfg.resources.osm_gb = 28
cfg.resources.num_vlms = 1
cfg.resources.vlm_gpu_fraction = 0.3
cfg.resources.num_sims = 2
cfg.resources.vlm_conda_env = None
cfg.resources.habitat_conda_env = None
cfg.sim = DummyContinuousEnvConfig()

cfg.vlm.attn_impl = "flash_attention_2"
cfg.vlm.save_outputs = True
# n=3 / a=0.15: the launch defaults. The smoke does not care about the value of `a`,
# only that every seam holds at the real chain dimension (11 x 20 x 3 = 660).
cfg.vlm.policy_head = FlowSDEHeadConfig(checkpoint_dir=CKPT, sde_n=3, sde_noise_a=0.15)

cfg.rollout.convo_start_template = [
    {"role": "user", "content": [{"type": "text", "text": "example substitution: $instr_or_goal"}]},
    {"role": "user", "content": [{"type": "image"}]},
    {"role": "assistant", "content": [{"type": "text", "text": "**forward**"}]},
]
cfg.rollout.max_steps = 8
cfg.training.rl_config.n_rollout = 1
cfg.training.rl_config.entropy_bonus = 0.0        # nonzero raises for a chain head, by design
advantage_estimator_fn = get_adv_estimator_fn("reinforce_plus_plus")

cfg.task.run_name = "rl_flow_sde_step"
bootstrapper = ExpBootstrapper(cfg)
logger = get_console_logger()
bootstrapper.setup_cluster()
trainers = bootstrapper.bootstrap_vlms_rl(training=True)

sims = bootstrapper.bootstrap_sims()
try:
    wandb_actor, _ = bootstrapper.bootstrap_logger()
except Exception as e:
    wandb_actor = None
    print(f"Logger setup failed: {e}. Continuing without logger.")

import numpy as np                                                   # noqa: E402
import torch                                                         # noqa: E402

# The dummy env's reward is `1 - 0.1 * ||chunk||`: shrink motion, earn more. A handful of
# cycles is enough to see mean ||chunk|| FALL if -- and only if -- the whole loop transports
# gradient from reward to the velocity field through the chain ratio. That trend is the
# system-health check; a single cycle only proves the plumbing.
N_CYCLES = 3
norms, checked = [], False
for cycle in range(N_CYCLES):
    trajectory_list = []
    traj_batch, model_inputs, values, distances, log_list = run_rollout_cycle(
        sims, trainers, get_shard_iterator(0), trajectory_list,
        bootstrapper.typed_cfg.training.rl_config.n_rollout,
        bootstrapper.typed_cfg.training.rl_config.n_adv,
    )
    if not checked:   # the seams the unit tests cannot reach, once, before any update
        assert "actions_continuous" in traj_batch.keys(), "chain never reached the batch"
        assert "sde_positions" in traj_batch.keys(), "sde_positions dropped en route"
        ac = traj_batch["actions_continuous"]
        assert ac.shape[-1] == 11 * 20 * 3, f"chain flattened wrong: {tuple(ac.shape)}"
        assert torch.isfinite(traj_batch["old_log_prob"]).all(), "non-finite old_log_prob"
        print(f"[seams] chain {tuple(ac.shape)}  positions "
              f"{tuple(traj_batch['sde_positions'].shape)}  old_log_prob mean "
              f"{traj_batch['old_log_prob'].float().mean():.2f}")
        checked = True
    # decode mean executed-chunk norm this cycle from the stored chains' final block
    chains = traj_batch["actions_continuous"].reshape(-1, 660)
    z_k = chains[:, -60:]
    norms.append(float(z_k.norm(dim=1).mean()))

    traj_batch, global_return_mean = compute_advantages_and_returns(
        traj_batch, advantage_estimator_fn, cfg)
    traj_batch = traj_batch[-bootstrapper.typed_cfg.training.rl_config.n_rollout:]
    training_futures, future_metadata = run_training_epochs(
        trainers, model_inputs, traj_batch, 1,
        bootstrapper.typed_cfg.training.rl_config.n_rollout,
        bootstrapper.typed_cfg.resources.num_vlms,
    )
    stream_results_and_log(training_futures, future_metadata, traj_batch, wandb_actor,
                           cycle, global_return_mean, logger, log_list)
    print(f"[cycle {cycle}] mean final-block norm {norms[-1]:.3f}  "
          f"return_mean {global_return_mean:.3f}")

print(f"norm trajectory: {['%.3f' % n for n in norms]}")
print("FLOW-SDE RL LOOP SMOKE:", "LEARNING (norms fell)" if norms[-1] < norms[0]
      else "ran, but no norm decrease -- inspect before launching")
