"""Short, no-update diagnostics for the history-conditioned STOP policy.

Set ``STOP_DIAG_KIND=trace`` for a fixed-eval trace or ``gradient`` for a full-history
training-batch gradient reading.  Hydra still supplies the checkpoint, environment and
two-GPU resource configuration, so these probes exercise the production code path.
"""
import json
import os

import hydra
import numpy as np

from longnav.conf.register_configs import register_configs
from longnav.config_schema import RLConfig

register_configs()


def _cleanup(ctx):
    import ray

    for trainer in ctx.trainers:
        ray.kill(trainer)
    for sim in ctx.sims:
        ray.kill(sim)
    if ctx.wandb_actor is not None:
        try:
            ray.get(ctx.wandb_actor.close.remote(), timeout=30)
        except Exception as exc:  # noqa: BLE001 - teardown must preserve the diagnosis
            print(f"W&B close failed during diagnostic teardown: {exc}", flush=True)
        ray.kill(ctx.wandb_actor)
    ray.shutdown()


def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as file:
        json.dump(payload, file, indent=2, sort_keys=True)


def _trace_eval(cfg):
    from longnav.utils.train_loop import (
        bootstrap_all,
        build_eval_partition,
        build_vector_eval_groups,
        run_eval_cycle,
    )

    ctx = bootstrap_all(cfg, training=False)
    typed = ctx.bootstrapper.typed_cfg
    run_dir = os.path.join(typed.task.output_dir, typed.task.run_name)
    try:
        if not typed.task.eval_uids_file:
            raise ValueError("STOP trace diagnostic requires task.eval_uids_file")
        with open(typed.task.eval_uids_file) as file:
            uids = [line.strip() for line in file if line.strip()]
        eval_pool = bool(typed.sim.eval_episodes)
        if ctx.vector_envs_per_sim > 1:
            _, parts = build_vector_eval_groups(
                ctx.sims,
                len(uids),
                typed.task.eval_seed,
                ctx.vector_envs_per_sim,
                uids=uids,
                eval_pool=eval_pool,
            )
        else:
            _, parts = build_eval_partition(
                ctx.sims,
                len(uids),
                typed.task.eval_seed,
                uids=uids,
                eval_pool=eval_pool,
            )
        row = run_eval_cycle(
            ctx.sims,
            ctx.trainers,
            parts,
            len(uids),
            None,
            0,
            run_dir,
            ode=True,
            vector_envs_per_sim=ctx.vector_envs_per_sim,
            sim_rebuilder=ctx.sim_rebuilder,
            sim_rebuild_validator=ctx.sim_rebuild_validator,
            episode_soft_timeout_seconds=ctx.episode_soft_timeout_seconds,
            episode_hard_timeout_seconds=ctx.episode_hard_timeout_seconds,
            sim_restart_limit=ctx.sim_restart_limit,
            stop_success_radius=float(typed.sim.success_distance),
            save_stop_traces=True,
            record_media=False,
        )
        _write_json(os.path.join(run_dir, "stop_trace_summary.json"), row)
        print(json.dumps(row, indent=2, sort_keys=True), flush=True)
    finally:
        _cleanup(ctx)


def _gradient_batch(cfg):
    import ray

    from longnav.utils.train_loop import bootstrap_all, run_rollout_cycle

    ctx = bootstrap_all(cfg, training=True)
    typed = ctx.bootstrapper.typed_cfg
    run_dir = os.path.join(typed.task.output_dir, typed.task.run_name)
    n_rollout = int(os.environ.get("STOP_DIAG_N_ROLLOUT", "4"))
    if n_rollout < len(ctx.trainers):
        raise ValueError("STOP_DIAG_N_ROLLOUT must cover every training worker")
    try:
        trajectories, model_inputs, _, _, _ = run_rollout_cycle(
            ctx.sims,
            ctx.trainers,
            ctx.shard_iter,
            [],
            n_rollout,
            int(typed.training.rl_config.n_adv),
            ctx.vector_envs_per_sim,
            sim_rebuilder=ctx.sim_rebuilder,
            sim_rebuild_validator=ctx.sim_rebuild_validator,
            episode_soft_timeout_seconds=ctx.episode_soft_timeout_seconds,
            episode_hard_timeout_seconds=ctx.episode_hard_timeout_seconds,
            sim_restart_limit=ctx.sim_restart_limit,
        )
        selected = []
        for index in range(n_rollout):
            targets = trajectories["stop_target"][index].cpu().numpy()
            if np.isfinite(targets).any() and np.any(targets > 0.5):
                selected.append(index)
            if len(selected) == len(ctx.trainers):
                break
        if len(selected) != len(ctx.trainers):
            raise RuntimeError(
                "gradient diagnostic batch contains too few reached-goal histories; "
                "increase STOP_DIAG_N_ROLLOUT"
            )
        futures = []
        for trainer, index in zip(ctx.trainers, selected):
            tensors, metadata = model_inputs[index]
            valid = trajectories["response_mask"][index].bool()
            futures.append(trainer.diagnose_stop_loss_gradients.remote(
                tensors,
                metadata,
                # `model_inputs[index]` contains only real action turns.  The
                # collated rollout columns remain right-padded until this same
                # response-mask selection in `run_training_epochs`.
                trajectories["stop_target"][index:index + 1, valid].cpu().numpy(),
                trajectories["shadow_stop_action"][index:index + 1, valid].cpu().numpy(),
                trajectories["shadow_stop_reward"][index:index + 1, valid].cpu().numpy(),
            ))
        readings = ray.get(futures)
        payload = {
            "n_rollout_collected": n_rollout,
            "selected_rollout_indices": selected,
            "selected_lengths": [
                int(trajectories["response_mask"][index].sum().item()) for index in selected
            ],
            "readings": readings,
        }
        path = os.path.join(run_dir, "stop_gradient_diagnostic.json")
        _write_json(path, payload)
        print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    finally:
        _cleanup(ctx)


def _fit_history(cfg):
    """Fit a disposable probe on saved production histories, without changing the run."""
    import math
    from pathlib import Path

    import torch
    from longnav.utils.state_probe import load_state_probe

    torch.set_num_threads(4)
    directory = Path(cfg.training.diagnostic_history_dir)
    paths = sorted(directory.glob("rank*_ep*.pt"))
    expected = int(cfg.training.diagnostic_history_episodes)
    if len(paths) != expected:
        raise ValueError(f"expected {expected} complete histories, found {len(paths)}")
    histories = [torch.load(path, map_location="cpu", weights_only=True) for path in paths]
    probe = load_state_probe(cfg.vlm.policy_head.checkpoint_dir, input_dim=histories[0]["hidden"].shape[-1])
    probe.train()
    threshold = float(cfg.rollout.stop_prob_threshold)
    boundary = math.log(threshold / (1.0 - threshold))

    def objective(history):
        logits = probe.stop_head(history["hidden"])
        return probe.stop_head.loss(logits - boundary, history["targets"], balanced=True)

    def measure():
        with torch.no_grad():
            losses = [float(objective(history)) for history in histories]
            tp = fp = fn = 0
            first_stop_correct = first_stop_false = no_stop = 0
            for history in histories:
                logits = probe.stop_head(history["hidden"])
                target = history["targets"]
                valid = torch.isfinite(target)
                predicted, positive = logits[valid] >= boundary, target[valid] > 0.5
                tp += int((predicted & positive).sum())
                fp += int((predicted & ~positive).sum())
                fn += int((~predicted & positive).sum())
                indices = torch.nonzero(predicted.reshape(-1)).flatten()
                if not len(indices):
                    no_stop += 1
                elif bool(positive.reshape(-1)[indices[0]]):
                    first_stop_correct += 1
                else:
                    first_stop_false += 1
            return {"balanced_bce": sum(losses) / len(losses), "tp": tp, "fp": fp, "fn": fn,
                    "prefix_first_stop_correct": first_stop_correct,
                    "prefix_first_stop_false": first_stop_false, "prefix_no_stop": no_stop}

    before = measure()
    optimizer = torch.optim.AdamW(probe.stop_head.parameters(), lr=1e-4, weight_decay=0)
    curve = []
    for step in range(100):
        optimizer.zero_grad(set_to_none=True)
        for offset in range(8):
            (objective(histories[(step * 8 + offset) % expected]) / 8).backward()
        norm = torch.nn.utils.clip_grad_norm_(probe.stop_head.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if (step + 1) % 10 == 0:
            curve.append({"step": step + 1, "gradient_norm": float(norm), **measure()})
    after = measure()
    saved = directory / "diagnostic_fitted_probe.pt"
    torch.save(probe.state_dict(), saved)
    reloaded = load_state_probe(cfg.vlm.policy_head.checkpoint_dir, input_dim=histories[0]["hidden"].shape[-1])
    reloaded.load_state_dict(torch.load(saved, weights_only=True))
    with torch.no_grad():
        parity = max(float((probe.stop_head(h["hidden"]) - reloaded.stop_head(h["hidden"])).abs().max())
                     for h in histories)
    payload = {
        "histories": len(paths), "threshold": threshold, "before": before, "after": after,
        "curve": curve, "save_load_max_abs_error": parity,
        "episodes_with_positive": sum(bool((h["targets"] > 0.5).any()) for h in histories),
        "positive_frames": sum(int((h["targets"] > 0.5).sum()) for h in histories),
        "loss_decreased": after["balanced_bce"] < before["balanced_bce"],
        "production_weights_modified": False,
        "counterfactual_caveat": "Recorded prefixes only; no-stop may be censored. Not closed-loop SR.",
    }
    _write_json(str(directory / "fit_report.json"), payload)
    print(json.dumps(payload, indent=2), flush=True)
    if parity != 0 or not np.isfinite(after["balanced_bce"]):
        raise RuntimeError("fixed-history fit failed numerical/save-load correctness")


@hydra.main(version_base=None, config_name="rl_config", config_path="../config")
def main(cfg: RLConfig):
    kind = os.environ.get("STOP_DIAG_KIND", "trace")
    if kind == "trace":
        _trace_eval(cfg)
    elif kind == "gradient":
        _gradient_batch(cfg)
    elif kind == "fit":
        _fit_history(cfg)
    else:
        raise ValueError("STOP_DIAG_KIND must be 'trace', 'gradient', or 'fit'")


if __name__ == "__main__":
    main()
