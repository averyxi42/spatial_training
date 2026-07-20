"""Shared pytest fixtures for the LongNav test suite.

Two tiers of fixtures live here:
- CPU-only config-building helpers (`compose_rl_config` and friends) --
  these route through real `hydra.compose()` (register_configs() + the
  `initialize()`/`compose()` pattern already used by
  tests/compose_demos/compose_demo.py) rather than hand-built dataclasses,
  so tests exercise the same config resolution the real CLI does and can't
  silently drift from what Hydra actually produces.
- `build_rl_worker` -- a session-scoped factory fixture that builds a real
  `RLActor` in-process (no Ray: VLMWorker/VLMTrainingMixin/RLActor are plain
  Python classes, Ray-wrapping only happens at the factory call site in
  utils/factories.py) from a real Hydra-composed `RLConfig`, mirroring
  `ExpBootstrapper.bootstrap_vlms_rl(training=True)` exactly (checkpoint
  resolution, base-model swap, `setup_training`) so the forward-pass tier
  (tests/forward/) exercises the same construction path production does,
  not a hand-abbreviated one. Skipped (not errored) when no GPU is present
  -- the 2B-param checkpoint isn't CPU-feasible to load in a test fixture.
  Built workers are cached per config-override tuple within a session,
  since re-loading the checkpoint per test is the dominant cost here.
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


def _find_free_port():
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def build_rl_worker_from_cfg(cfg):
    """Builds a real RLActor in-process from a real (Hydra-composed) RLConfig.

    Mirrors ExpBootstrapper.bootstrap_vlms_rl(training=True) /
    RLWorkerFactory.create + _enable_training exactly (checkpoint
    resolution, base-model swap, setup_training which loads the checkpoint
    internally) -- just without the Ray actor wrapping, so it can run
    directly in a pytest process.
    """
    from longnav.utils.factories import get_base_model, resolve_checkpoint_path
    from longnav.utils.rollout_core import RLActor

    resolved = OmegaConf.to_container(cfg, resolve=True)

    if cfg.training.checkpoint is not None:
        checkpoint_path = resolve_checkpoint_path(cfg.training.checkpoint)
        base_model_path = get_base_model(checkpoint_path)
        if base_model_path is not None:
            resolved["vlm"]["model_id"] = base_model_path
        cfg.training.checkpoint = checkpoint_path

    resolved["vlm"]["save_outputs"] = True
    worker = RLActor(rollout_config=resolved["rollout"], **resolved["vlm"])
    worker.setup_training(
        config=cfg.training,
        rank=0,
        world_size=1,
        master_addr="localhost",
        master_port=_find_free_port(),
    )
    return worker


@pytest.fixture(scope="session")
def build_rl_worker(_hydra_registered):
    """Factory fixture: build_rl_worker(overrides: list[str], cache_key=None,
    mutate_fn=None) -> RLActor.

    `overrides` is passed straight to hydra.compose() (config_name="rl_config"),
    same as compose_rl_config. `mutate_fn`, if given, is called with the
    composed cfg (before the worker is built) -- this is how a test injects
    a `value_head:` block, which per the regression-test plan's decision #2
    has no ConfigStore group and must be supplied as a plain field on the
    resolved config object directly, not via a `+group=name` override.
    Skips the test cleanly if no GPU is available. Only one worker is kept
    resident on the GPU at a time (see below) -- pass a distinct `cache_key`
    when `mutate_fn` makes two calls with identical `overrides` build
    meaningfully different workers.
    """
    if not _has_gpu():
        pytest.skip("build_rl_worker requires a GPU; none available")

    # Single-slot cache, not a real LRU: a 2B-param model + LoRA + optimizer
    # state is ~9GB on this class of GPU, so keeping more than one or two
    # workers resident at once risks CUDA OOM (observed empirically running
    # 3 forward-pass tests back to back). Only the most recently built
    # worker is kept alive; building a different config tears down the
    # previous one first.
    state = {"key": None, "worker": None}

    def _factory(overrides, cache_key=None, mutate_fn=None):
        import gc

        import torch

        key = (tuple(overrides), cache_key)
        if key == state["key"]:
            return state["worker"]

        if state["worker"] is not None:
            del state["worker"]
            state["worker"] = None
            gc.collect()
            torch.cuda.empty_cache()

        cfg = _compose(overrides, True)
        if mutate_fn is not None:
            mutate_fn(cfg)
        state["worker"] = build_rl_worker_from_cfg(cfg)
        state["key"] = key
        return state["worker"]

    return _factory
