"""Blind-episode rejection: the worker's verdict, the collector's decision.

The split under test (settled 2026-08-25): the WORKER decides what counts as
unusable (`EpisodeRolloutMixin._episode_rejected`), the COLLECTOR decides per call
whether to act on it (`collect_rollouts(respect_rejections=...)`) -- training does,
eval does not. What is pinned here:

  * the verdict itself, including the physical-terminal carve-out (99.1% of escapes
    are blind; rejecting them would delete the escape population and with it every
    escape_penalty the shaping applies);
  * PURE EXTENSION -- with the feature off, the verdict is always False and the
    collector's assembly is unchanged;
  * the assembly indexes by surviving dispatch id, so gapped ids cannot misalign
    trajectories against results.
"""
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from longnav.utils.rollout_core import EpisodeRolloutMixin  # noqa: E402


class _Verdict(EpisodeRolloutMixin):
    """Only the rejection surface -- no Ray, no VLM, no env."""

    def __init__(self, **cfg):
        self.rollout_config = cfg


def _info(**kw):
    base = {"episode_label": "s:1", "blind_total": 0, "escaped": False}
    base.update(kw)
    return base


class TestWorkerVerdict:
    def test_off_by_default_is_never_rejected(self):
        """PURE EXTENSION: a run that does not configure rejection sees no verdict,
        whatever the episode looked like."""
        w = _Verdict()
        assert w._episode_rejected(_info(blind_total=42)) is False
        assert w._episode_rejected(_info(blind_total=0)) is False

    def test_rejects_a_blind_episode_when_enabled(self):
        w = _Verdict(reject_blind_episodes=True)
        assert w._episode_rejected(_info(blind_total=1)) is True
        assert w._episode_rejected(_info(blind_total=0)) is False

    def test_physical_terminal_carve_out_is_on_by_default(self):
        """An escaped episode goes blind BECAUSE it is falling out of the world --
        median 2 blind rows against 25 for other blind episodes. It keeps its
        escape_penalty instead of vanishing."""
        w = _Verdict(reject_blind_episodes=True)
        assert w._episode_rejected(_info(blind_total=3, escaped=True)) is False
        assert w._episode_rejected(_info(blind_total=3, tipped_over=True)) is False
        assert w._episode_rejected(_info(blind_total=3)) is True

    def test_carve_out_can_be_disabled(self):
        w = _Verdict(reject_blind_episodes=True,
                     reject_blind_keep_physical_terminal=False)
        assert w._episode_rejected(_info(blind_total=3, escaped=True)) is True

    def test_env_without_the_counter_is_never_rejected(self):
        """The dummy/discrete envs have no notion of a blind step; absence of
        `blind_total` must read as 'nothing to reject', not as a KeyError."""
        w = _Verdict(reject_blind_episodes=True)
        assert w._episode_rejected({"episode_label": "s:1"}) is False


class TestCollectorAssembly:
    """The alignment fix, exercised on the pure-python assembly expression."""

    @staticmethod
    def _assemble(trajectory_ids, result_dict, log_dict):
        kept = sorted(trajectory_ids)
        return [result_dict[i] for i in kept], [log_dict.get(i) for i in kept]

    def test_contiguous_ids_unchanged(self):
        ids = [0, 1, 2]
        res = {0: "a", 1: "b", 2: "c"}
        got, _ = self._assemble(ids, res, {})
        assert got == ["a", "b", "c"]          # identical to range(len)

    def test_gapped_ids_keep_pairing(self):
        """Ids 1 and 3 were rejected; range(len) would return a/b (the FIRST two),
        pairing trajectory 2 with result 1."""
        ids = [0, 2, 4]
        res = {0: "a", 1: "rejected", 2: "c", 3: "rejected", 4: "e"}
        got, _ = self._assemble(ids, res, {})
        assert got == ["a", "c", "e"]
        assert got != [res[i] for i in range(len(ids))]


def test_collect_rollouts_signature_defaults_to_off():
    """The collector's new argument must default to the historical behaviour, so
    every existing caller is unaffected."""
    import inspect

    from longnav.utils.rollout_core import collect_rollouts
    sig = inspect.signature(collect_rollouts)
    assert sig.parameters["respect_rejections"].default is False
    assert sig.parameters["max_dispatch_factor"].default == 3.0
