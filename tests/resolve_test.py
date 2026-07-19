"""Non-interactive compose-assertion smoke test for the Hydra ConfigStore groups
(sim / policy_head / logger) and every shipped yaml combo. Exercises the real
hydra.compose path (as an actual `+experiment=...` CLI invocation would), unlike
tests/rl_*.py which construct RLConfig() directly in Python and so never touch
Hydra's defaults-list resolution.

Run with: python3 tests/resolve_test.py
"""
import glob
import os

import hydra
from hydra import compose, initialize
from omegaconf import OmegaConf

from longnav.conf.register_configs import register_configs

register_configs()

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "src", "longnav", "config")

SIM_TARGETS = {
    "habitat": "longnav.env.habitat.HabitatEnvActor",
    "dummy_discrete": "longnav.env.env_base.DummyEnvActor",
    "dummy_continuous": "longnav.env.env_base.DummyContinuousEnvActor",
    "color_bandit": "longnav.env.color_bandit.ColorBanditEnvActor",
    "voxel": "longnav.env.habitat.HabitatEnvActor",
}

POLICY_HEAD_TARGETS = {
    "lm_head": None,  # no _target_ -- config-only metadata
    "gaussian_head": "longnav.utils.vlm_worker.ContinuousActionHead",
}


def check(label, fn):
    try:
        fn()
        print(f"[OK] {label}")
    except Exception as e:
        print(f"[FAIL] {label}: {e}")
        raise


def main():
    with initialize(version_base=None, config_path="../src/longnav/config"):
        # 1. every sim group member resolves to the right actor class
        for name, target in SIM_TARGETS.items():
            def _check(name=name, target=target):
                cfg = compose(config_name="rl_config", overrides=[f"sim={name}"])
                OmegaConf.to_container(cfg, resolve=True)
                assert cfg.sim._target_ == target, f"expected {target}, got {cfg.sim._target_}"
            check(f"sim={name}", _check)

        # 2. every policy_head group member resolves correctly
        for name, target in POLICY_HEAD_TARGETS.items():
            def _check(name=name, target=target):
                cfg = compose(config_name="rl_config", overrides=[f"policy_head@vlm.policy_head={name}"])
                OmegaConf.to_container(cfg, resolve=True)
                got = cfg.vlm.policy_head.get("_target_", None)
                assert got == target, f"expected {target}, got {got}"
            check(f"policy_head@vlm.policy_head={name}", _check)

        # 3. bare defaults (no overrides at all): sim=habitat, policy_head=lm_head, logger=None
        def _check_defaults():
            cfg = compose(config_name="rl_config")
            OmegaConf.to_container(cfg, resolve=True)
            assert cfg.sim._target_ == SIM_TARGETS["habitat"]
            assert cfg.vlm.policy_head.get("_target_", None) is None
            assert cfg.task.logger is None
        check("bare defaults (sim=habitat, policy_head=lm_head, logger=None)", _check_defaults)

        # 4. every shipped experiment/training/checkpoint/dataset/resources yaml composes
        base = os.path.join(os.path.dirname(__file__), "..", "src", "longnav", "config")
        for group in ["experiment", "training", "checkpoint", "dataset", "resources"]:
            for f in sorted(glob.glob(os.path.join(base, group, "*.yaml"))):
                name = os.path.splitext(os.path.basename(f))[0]
                def _check(group=group, name=name):
                    prefix = "+" if group != "experiment" else "+"
                    cfg = compose(config_name="rl_config", overrides=[f"{prefix}{group}={name}"])
                    OmegaConf.to_container(cfg, resolve=True)
                check(f"{group}={name}", _check)

        # 5. the actual known-tricky case: an experiment yaml overriding sim via the
        # correct `defaults: - override /sim: <name>` syntax (not a plain `sim: <name>`
        # field, which fails validation -- see the comment on InferenceConfig).
        def _check_discrete_dummy():
            cfg = compose(config_name="rl_config", overrides=["+experiment=discrete_dummy", "+training=rpp"])
            OmegaConf.to_container(cfg, resolve=True)
            assert cfg.sim._target_ == SIM_TARGETS["dummy_discrete"], cfg.sim._target_
        check("experiment=discrete_dummy selects sim=dummy_discrete via override syntax", _check_discrete_dummy)

    print("\nALL COMPOSE CHECKS PASSED")


if __name__ == "__main__":
    main()
