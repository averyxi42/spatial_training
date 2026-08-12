import ray
import pytest


@pytest.fixture(scope="module")
def ray_session():
    """Local, CPU-only Ray session for this test module.

    `EpisodeRolloutMixin.run_episode` calls `env_handle.step.remote(...)` /
    `ray.get(...)` directly -- the sim side of the interface is always a real
    Ray actor handle, even in a "no Ray" pure-math sense this isn't. This
    fixture is scoped to tests/rollout/ only; the pure-math tier
    (tests/rl_math/) needs no Ray at all.
    """
    ray.init(
        num_cpus=2,
        include_dashboard=False,
        ignore_reinit_error=True,
        object_store_memory=200 * 1024 * 1024,
    )
    yield
    ray.shutdown()
