"""Collect frozen-SFT predicted-visibility statistics before exploration RL."""

from __future__ import annotations

import json
import os
import time

import hydra
import numpy as np
import ray

from longnav.conf.register_configs import register_configs
from longnav.config_schema import RLConfig

register_configs()


@hydra.main(version_base=None, config_name="rl_config", config_path="../config")
def main(cfg: RLConfig):
    from longnav.utils.train_loop import bootstrap_all, run_rollout_cycle

    cfg.vlm.save_outputs = True
    if not cfg.sim.exploration_enabled:
        raise ValueError("calibration requires sim.exploration_enabled=true")
    if cfg.sim.exploration_reward_weight != 0.0:
        raise ValueError("calibration must collect raw gain with zero reward weight")
    ctx = bootstrap_all(cfg, training=True)
    values = {"gain": [], "progress": [], "ray_seconds": [], "policy_step_seconds": []}
    trajectory_list = []
    try:
        # Match the stochastic chain actions used by training rollouts.  Pure ODE
        # chains intentionally contain no stochastic positions, so they cannot be
        # scored by the PPO postprocessor even though this frozen probe has no update.
        ray.get([trainer.set_ode_sampling.remote(False) for trainer in ctx.trainers])
        episodes_target = 128
        n_rollout = int(cfg.training.rl_config.n_rollout)
        for _ in range((episodes_target + n_rollout - 1) // n_rollout):
            started = time.perf_counter()
            traj_batch, model_inputs, _, _, _ = run_rollout_cycle(
                ctx.sims, ctx.trainers, ctx.shard_iter, trajectory_list, n_rollout,
                cfg.training.rl_config.n_adv, ctx.vector_envs_per_sim,
            )
            elapsed = time.perf_counter() - started
            recent = traj_batch[-n_rollout:]
            mask = recent["response_mask"].bool()
            steps = int(mask.sum().item())
            for source, target in (
                ("exploration_gain_raw_m2", "gain"),
                ("distance_progress", "progress"),
                ("exploration_ray_seconds", "ray_seconds"),
            ):
                if source in recent.keys():
                    values[target].extend(recent[source][mask].float().cpu().tolist())
            if steps:
                values["policy_step_seconds"].append(elapsed * len(ctx.sims) / steps)
            del model_inputs
        positive_gain = np.asarray([x for x in values["gain"] if x > 0 and np.isfinite(x)])
        positive_progress = np.asarray([
            x for x in values["progress"] if x > 0 and np.isfinite(x)
        ])
        if not len(positive_gain) or not len(positive_progress):
            raise RuntimeError("calibration produced no positive visibility gain or progress")
        gain_p90 = float(np.percentile(positive_gain, 90))
        progress_p90 = float(np.percentile(positive_progress, 90))
        coefficient = progress_p90 / gain_p90
        mean_ray = float(np.mean(values["ray_seconds"]))
        mean_step = float(np.mean(values["policy_step_seconds"]))
        ray_fraction = mean_ray / max(mean_step, 1e-9)
        result = {
            "episodes_target": episodes_target,
            "sampling": "sde",
            "reference_coefficient": 0.05,
            "selected_coefficient": coefficient,
            "raw_nonzero_rate": float(np.mean(np.asarray(values["gain"]) > 0)),
            "raw_gain_percentiles_m2": {
                str(q): float(np.percentile(positive_gain, q)) for q in (50, 90, 99)
            },
            "positive_progress_p90": progress_p90,
            "mean_ray_seconds": mean_ray,
            "mean_policy_step_seconds": mean_step,
            "ray_latency_fraction": ray_fraction,
            "latency_gate_passed": bool(ray_fraction < 0.05),
            "finite": bool(all(np.isfinite(x) for group in values.values() for x in group)),
        }
        out_dir = os.path.join(cfg.task.output_dir, cfg.task.run_name)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "calibration.json"), "w") as output:
            json.dump(result, output, indent=2)
            output.write("\n")
        print(json.dumps(result, indent=2), flush=True)
        if not result["latency_gate_passed"]:
            raise RuntimeError("predicted-visibility ray tracing exceeded the 5% latency gate")
    finally:
        for trainer in ctx.trainers:
            ray.kill(trainer, no_restart=True)
        for sim in ctx.sims:
            ray.kill(sim, no_restart=True)
        if ctx.wandb_actor is not None:
            ray.kill(ctx.wandb_actor, no_restart=True)
        ray.shutdown()


if __name__ == "__main__":
    main()
