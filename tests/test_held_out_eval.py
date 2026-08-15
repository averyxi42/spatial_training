"""The train/eval disjointness machinery: `sim.train_uids` + `task.eval_uids_file`.

These two are what make an in-training `eval/*` number a generalisation signal instead of
a quasi-held-out one, and both fail SILENTLY when wrong -- a filter that matches nothing
trains on everything, and an eval set that is not pinned is simply a different set. Hence
tests rather than a look at the config.
"""
import os
import numpy as np
import pytest

ES = "/Projects/spatial_training/dump/eval_system/episode_sets"
TRAIN, EVAL = f"{ES}/trainset128.txt", f"{ES}/fullpool_evalset_servable26.txt"


def _read(path):
    return [u.strip() for u in open(path).read().replace("\n", ",").split(",") if u.strip()]


needs_sets = pytest.mark.skipif(not os.path.exists(TRAIN), reason="uid sets not generated")


@needs_sets
def test_train_and_eval_sets_are_disjoint():
    train, ev = set(_read(TRAIN)), set(_read(EVAL))
    assert len(train) == 128 and len(ev) == 26
    assert not (train & ev), f"train/eval overlap: {sorted(train & ev)}"


@needs_sets
def test_train_set_is_balanced():
    """Not decoration: an unbalanced 128 makes the eval delta a composition artifact."""
    from collections import Counter
    scenes = Counter(u.split(":", 1)[0] for u in _read(TRAIN))
    assert len(scenes) == 80 and max(scenes.values()) <= 2


class _Env:
    """The two `_reshuffle` behaviours, on a stub -- constructing the real actor needs ray,
    habitat and a scene tree, none of which this logic touches."""

    def __init__(self, uids, train_uids, shard):
        from longnav.env.objectnav_continuous import ContinuousObjectNavEnvActor as A
        self._episodes = [type("E", (), {"uid": u})() for u in uids]
        self._shard, self._rng = shard, np.random.default_rng(0)
        self._train_uids = frozenset(train_uids) if train_uids else None
        self._reshuffle = A._reshuffle.__get__(self)

    def served(self):
        self._reshuffle()
        return {self._episodes[i].uid for i in self._order}


POOL = ["s:1", "s:2", "s:3", "s:4"]


def test_filter_restricts_the_unsharded_training_stream():
    assert _Env(POOL, ["s:1", "s:2"], None).served() == {"s:1", "s:2"}


def test_filter_is_bypassed_for_an_explicit_shard():
    """`run_eval_cycle` assigns eval shards; this is what lets eval reach held-out uids."""
    assert _Env(POOL, ["s:1", "s:2"], ["s:3"]).served() == set(POOL)


def test_no_filter_serves_the_whole_pool():
    assert _Env(POOL, None, None).served() == set(POOL)


def test_a_filter_matching_nothing_raises_rather_than_serving_everything():
    with pytest.raises(ValueError, match="train_uids"):
        _Env(POOL, ["other:9"], None).served()


def test_pinned_eval_partition_is_verbatim_and_rejects_a_uid_outside_the_pool():
    from longnav.utils import train_loop

    class _Sims:
        class list_episode_uids:
            @staticmethod
            def remote():
                return POOL
    sims, orig = [_Sims()], train_loop.ray.get
    train_loop.ray.get = lambda x: x() if callable(x) else x
    try:
        uids, parts = train_loop.build_eval_partition(sims, 2, 0, uids=["s:3", "s:1"])
        assert uids == ["s:3", "s:1"]          # verbatim, order preserved
        assert sorted(u for p in parts for u in p) == ["s:1", "s:3"]
        with pytest.raises(KeyError, match="pinned"):
            train_loop.build_eval_partition(sims, 2, 0, uids=["s:1", "nope:1"])
    finally:
        train_loop.ray.get = orig
