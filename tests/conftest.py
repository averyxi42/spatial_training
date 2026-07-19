"""Shared pytest fixtures for the LongNav test suite.

Two tiers of fixtures live here:
- CPU-only config-building helpers (`compose_rl_config` and friends) --
  these route through real `hydra.compose()` (register_configs() + the
  `initialize()`/`compose()` pattern already used by
  tests/compose_demos/compose_demo.py) rather than hand-built dataclasses,
  so tests exercise the same config resolution the real CLI does and can't
  silently drift from what Hydra actually produces.
- `checkpoint_model` -- a session-scoped, GPU-only fixture that loads the
  real `+checkpoint=longnav` checkpoint in-process (no Ray: VLMWorker /
  VLMTrainingMixin are plain Python classes, Ray-wrapping only happens at
  the factory call site in utils/factories.py). Skipped (not errored) when
  no GPU is present -- this fixture backs the Phase 5 forward-pass tier,
  which is not yet implemented (see tests/forward/README.md).
"""
import os

import pytest
from hydra import compose, initialize
from omegaconf import OmegaConf

from longnav.conf.register_configs import register_configs

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.relpath(
    os.path.join(HERE, "..", "src", "longnav", "config"), HERE
)


@pytest.fixture(scope="session")
def _hydra_registered():
    register_configs()
    return True


def _compose(overrides, _hydra_registered):
    with initialize(version_base=None, config_path=CONFIG_PATH):
        cfg = compose(config_name="rl_config", overrides=overrides)
    return cfg


@pytest.fixture
def compose_rl_config(_hydra_registered):
    """Factory fixture: compose_rl_config(overrides: list[str]) -> RLConfig.

    Thin wrapper around hydra.compose() using the real config tree under
    src/longnav/config, exactly like tests/compose_demos/compose_demo.py.
    Usage:
        cfg = compose_rl_config(["+experiment=discrete_dummy", "+training=rpp"])
    """

    def _factory(overrides):
        return _compose(overrides, _hydra_registered)

    return _factory


@pytest.fixture
def discrete_rl_config(compose_rl_config):
    """Minimal discrete-head RLConfig, resolved through the real dummy
    discrete-env experiment config."""
    return compose_rl_config(["+experiment=discrete_dummy", "+training=rpp"])


@pytest.fixture
def continuous_rl_config(compose_rl_config):
    """Minimal continuous-head RLConfig, resolved through the real dummy
    continuous-env experiment config."""
    return compose_rl_config(["+experiment=continuous_dummy", "+training=rpp"])


def _has_gpu() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


@pytest.fixture(scope="session")
def checkpoint_model(_hydra_registered):
    """Session-scoped: builds an RLWorker in-process from the real
    +checkpoint=longnav config with load_model=True, and runs setup_training
    so the forward/loss paths are exercisable. Skipped when no GPU is
    available -- the 2B-param checkpoint isn't a CPU-feasible load for a
    test fixture.

    NOT YET CONSUMED: Phase 5 (tests/forward/) is not implemented yet (needs
    a GPU box to calibrate numeric tolerances -- see tests/forward/README.md).
    This fixture is scaffolding for that phase.
    """
    if not _has_gpu():
        pytest.skip("checkpoint_model requires a GPU; none available")

    import socket

    from longnav.config_schema import RolloutConfig, VLMTrainingConfig
    from longnav.utils.factories import get_base_model, resolve_checkpoint_path
    from longnav.utils.rollout_core import RLWorker

    checkpoint_path = resolve_checkpoint_path("Aasdfip/hm3d_rpp_ke_standard-checkpoint_231")
    base_model_path = get_base_model(checkpoint_path)

    from dataclasses import asdict

    rollout_cfg = RolloutConfig()
    training_cfg = VLMTrainingConfig()

    worker = RLWorker(
        asdict(rollout_cfg),
        model_id=base_model_path,
        attn_impl="sdpa",
        load_model=True,
    )

    def _find_free_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    worker.setup_training(
        config=training_cfg,
        rank=0,
        world_size=1,
        master_addr="localhost",
        master_port=_find_free_port(),
    )
    worker.load_checkpoint(checkpoint_path, False, False)

    return worker
