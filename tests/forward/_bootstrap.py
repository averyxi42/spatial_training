"""Real construction of the VLM + sim actors for the forward-pass tier's
component tests. Mirrors train_loop.bootstrap_all(cfg, training=True)
almost exactly (same real Hydra-composed config, same real
ExpBootstrapper.bootstrap_vlms_rl/bootstrap_sims), with one substitution:
the sim backend is swapped to the scripted, deterministic ReplayEnvActor
instead of Habitat/the randomized dummy envs, since a component test needs
a reproducible input, not a different random episode every run.

bootstrap_all itself has no seam for injecting a sim script (Habitat
doesn't need one), so this inlines its ~10 lines rather than adding
test-only behavior to the production function.
"""
import numpy as np
import ray

from longnav.utils.factories import ExpBootstrapper, get_console_logger, get_shard_iterator


def make_replay_script(n_steps: int = 8, obs_text: str = "reach the goal"):
    """`info.oracle_action` is a fixed, deterministic discrete action index
    (cycling forward/left/right, ending in stop) -- consumed by
    EpisodeRolloutMixin.run_episode when rollout_config['use_oracle_action']
    is set, so the *actual* action taken (and therefore every subsequent
    turn's prompt text) is deterministic across runs. Without this, the
    model's own action sampling is genuinely random per-process
    (RolloutWorker.__init__ seeds from os.getpid()), and that sampled
    action's text feeds back into later turns' prompts -- so turn>=2 model
    outputs are not reproducible no matter how deterministic the env's
    observations are."""
    oracle_cycle = [1, 2, 1, 3]  # forward, left, forward, right -- valid indices into action_space
    script = []
    for i in range(n_steps):
        rgb = np.random.default_rng(i).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        is_last = i == n_steps - 1
        script.append(
            {
                "rgb": rgb,
                "obs": {"instr_or_goal": obs_text},
                "reward": 0.1 if not is_last else 1.0,
                "done": is_last,
                "info": {"oracle_action": 0 if is_last else oracle_cycle[i % len(oracle_cycle)]},
            }
        )
    return script


def bootstrap_with_replay_sim(cfg, script):
    logger = get_console_logger()
    # Defensive: setup_cluster()'s ray.init(..., ignore_reinit_error=True)
    # silently keeps whatever cluster is already running (wrong resource
    # tags/sizes) if Ray was already initialized elsewhere in the session.
    ray.shutdown()
    bootstrapper = ExpBootstrapper(cfg)
    bootstrapper.setup_cluster()

    trainers = bootstrapper.bootstrap_vlms_rl(training=True)

    wandb_objs = bootstrapper.bootstrap_logger()
    if wandb_objs is not None:
        wandb_actor, excluded_episodes = wandb_objs
    else:
        wandb_actor, excluded_episodes = None, None

    bootstrapper.resolved_dict["sim"]["script"] = script
    sims = bootstrapper.bootstrap_sims(wandb_actor)

    shard_iter = get_shard_iterator(
        subset_label=bootstrapper.typed_cfg.task.subset_label,
        episode_json=bootstrapper.typed_cfg.task.episode_json,
        shard_size=bootstrapper.typed_cfg.task.shard_size,
        logger=logger,
        excluded_episodes=excluded_episodes,
    )
    return bootstrapper, trainers, sims, wandb_actor, shard_iter, logger


def teardown(trainers, sims, wandb_actor):
    for trainer in trainers:
        ray.kill(trainer)
    for sim in sims:
        ray.kill(sim)
    if wandb_actor is not None:
        ray.kill(wandb_actor)
    ray.shutdown()
