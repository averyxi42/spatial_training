'''H2 probe: can the credited SDE channel see episode outcomes at all?

The chain-head policy gradient reaches h exclusively through the epsilon-advantage
correlation (grad into h ~ sum_k w_k * A * eps_k/sigma_k * dmu/dh). If SDE-scale
perturbations never change an episode outcome, A is independent of eps, the expected
h-gradient is zero, and no learning rate fixes it. This probe measures exactly that:

  arm ODE      : M repeated eval passes, pure ODE  -> outcome variance from z0 alone
  arm SDE@a    : M repeated passes, normal sampler -> variance from z0 + credited noise

per fixed episode: flip rate under each arm. SDE@0.15 ~ ODE  => credited channel blind
at launch settings (H2 confirmed); rising flip rates along the a-ladder locate the
noise scale at which the channel opens.

Reuses the interleaved-eval machinery wholesale (same actors, fixed set, exhaustion
slices). Results: per-arm per-episode jsonl + a summary table.

Launch (fleet must be free; ~M*(1+len(A_LADDER))*24 episodes total):
  python3 -m longnav.scripts.probe_credited_channel +training=hapo +resources=octo \
      +experiment=flow_sde_objectnav_v3_overfit32 resources.vlm_conda_env=longnav_vlm \
      task.run_name=h2_probe
'''
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import hydra
from longnav.conf.register_configs import register_configs
from longnav.config_schema import RLConfig

register_configs()

SET_SIZE = 32          # pre-screen leaves ~24 servable
SET_SEED = 0           # same fixed set as the overfit diagnostic
# Env-overridable so one script serves both readings without an edit per run:
#   PROBE_M=4 PROBE_A=0.15,0.30,0.50,0.70  -> the `a` CALIBRATION (piRL's init-gap
#     rule: pick the largest `a` whose success gap against the ODE arm stays inside
#     the z0 band). Needs the ladder wide, repeats only deep enough for a mean.
#   PROBE_M=6 PROBE_A=0.15,0.30,0.50       -> the original H2 flip-rate reading, where
#     per-episode outcome VARIANCE is the statistic and repeats matter more than reach.
# Both come out of the same pass; only the emphasis differs.
M_REPEATS = int(os.environ.get("PROBE_M", 6))
A_LADDER = [float(x) for x in os.environ.get("PROBE_A", "0.15,0.30,0.50").split(",")]


@hydra.main(version_base=None, config_name="rl_config", config_path='../config')
def main(cfg: RLConfig):
    cfg.vlm.save_outputs = True
    import json
    import ray
    from collections import defaultdict

    from longnav.utils.train_loop import (bootstrap_all, build_eval_partition,
                                          run_eval_cycle)

    ctx = bootstrap_all(cfg, training=True)   # training=True: same weights path as RL
    trainers, sims = ctx.trainers, ctx.sims
    tcfg = ctx.bootstrapper.typed_cfg
    run_dir = os.path.join(tcfg.task.output_dir, tcfg.task.run_name)
    os.makedirs(run_dir, exist_ok=True)

    eval_uids, eval_parts = build_eval_partition(sims, SET_SIZE, SET_SEED)
    print(f"probe set: {len(eval_uids)} episodes, {M_REPEATS} repeats per arm")

    def set_noise_a(a):
        # dataclasses.replace on the frozen SDEConfig, installed on every head instance.
        ray.get([t.set_sde_noise_a.remote(float(a)) for t in trainers])

    arms = [("ode", None)] + [(f"sde_a{a:g}", a) for a in A_LADDER]
    outcomes = defaultdict(lambda: defaultdict(list))   # arm -> uid -> [0/1,...]
    tick = 0
    try:
        for arm, a in arms:
            if a is not None:
                set_noise_a(a)
            for rep in range(M_REPEATS):
                row = run_eval_cycle(sims, trainers, eval_parts, len(eval_uids),
                                     None, tick, run_dir, ode=(a is None))
                tick += 1
                # per-episode results were appended to eval_episodes.jsonl by the cycle;
                # re-read the tail we just wrote (cycle id == tick-1)
                with open(os.path.join(run_dir, "eval_episodes.jsonl")) as f:
                    for line in f:
                        r = json.loads(line)
                        if r["cycle"] == tick - 1:
                            outcomes[arm][r["uid"]].append(r["success"])
                print(f"[{arm} rep {rep}] success {row['eval/success']:.3f}")

        # ---- summary: per-arm flip statistics over the fixed set ----------------
        print("\n=== H2 PROBE SUMMARY ===")
        print(f"{'arm':12s} {'mean succ':>9s} {'episodes flipping':>18s} {'mean per-ep var':>16s}")
        summary = {}
        for arm, _ in arms:
            eps = outcomes[arm]
            import numpy as np
            per_ep = {u: np.array(v) for u, v in eps.items() if len(v) >= 2}
            flips = sum(1 for v in per_ep.values() if 0 < v.mean() < 1)
            mvar = float(np.mean([v.var() for v in per_ep.values()])) if per_ep else float("nan")
            msucc = float(np.mean([v.mean() for v in per_ep.values()])) if per_ep else float("nan")
            summary[arm] = {"mean_success": msucc, "n_flipping": flips,
                            "mean_outcome_var": mvar, "n_episodes": len(per_ep)}
            print(f"{arm:12s} {msucc:9.3f} {flips:14d}/{len(per_ep):<3d} {mvar:16.4f}")
        with open(os.path.join(run_dir, "h2_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print("\nREAD: sde_a0.15 var ~ ode var  => credited channel BLIND at launch "
              "settings; var rising along the ladder locates where it opens.")
    finally:
        for t in trainers:
            ray.kill(t)
        for s in sims:
            ray.kill(s)
        ray.shutdown()


if __name__ == "__main__":
    main()
