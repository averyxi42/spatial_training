import longnav.utils.rollout_core as rollout_core


class RemoteMethod:
    def __init__(self, function):
        self.function = function

    def remote(self, *args, **kwargs):
        return self.function(*args, **kwargs)


class FakeEnv:
    def __init__(self):
        self.labels = []
        self.assignments = []
        self.assign_shard = RemoteMethod(self._assign_shard)
        self.configure_benchmark_video_capture = RemoteMethod(lambda config: None)
        self.reset_batch = RemoteMethod(lambda: list(self.labels))
        self.flush_logs_to_disk = RemoteMethod(lambda: None)

    def _assign_shard(self, labels):
        self.labels = list(labels)
        self.assignments.append(list(labels))


class FakeVlm:
    def __init__(self):
        self.current = []
        self.run_episode_batch = RemoteMethod(self._run_episode_batch)
        self.postprocess_batch = RemoteMethod(self._postprocess_batch)

    def _run_episode_batch(self, env, initial_state, eval_config=None):
        self.current = list(initial_state)
        results = [{"episode_label": label} for label in self.current]
        return None, results, {"wall_seconds": 1.0}

    def _postprocess_batch(self, **kwargs):
        return [(label, None, None) for label in self.current]


def test_collect_vector_rollouts_supports_multiple_waves(monkeypatch):
    monkeypatch.setattr(
        rollout_core.ray,
        "get",
        lambda value, **kwargs: value,
    )
    envs = [FakeEnv(), FakeEnv()]
    vlms = [FakeVlm(), FakeVlm()]
    shards = iter(
        [
            ["scene_a_0", "scene_a_1"],
            ["scene_b_0", "scene_b_1"],
            ["scene_c_0", "scene_c_1"],
            ["scene_d_0", "scene_d_1"],
        ]
    )

    rollouts, results, _, timings = rollout_core.collect_vector_rollouts(
        envs,
        vlms,
        shards,
        target_episodes=8,
        episodes_per_worker=2,
        return_timings=True,
    )

    assert len(rollouts) == len(results) == 8
    assert len(timings["wave_runtimes"]) == 2
    assert envs[0].assignments == [
        ["scene_a_0", "scene_a_1"],
        ["scene_c_0", "scene_c_1"],
    ]
    assert envs[1].assignments == [
        ["scene_b_0", "scene_b_1"],
        ["scene_d_0", "scene_d_1"],
    ]


def test_collect_vector_rollouts_times_out_a_stuck_wave(monkeypatch):
    calls = 0

    def fake_get(value, **kwargs):
        nonlocal calls
        calls += 1
        if kwargs.get("timeout") is not None:
            raise rollout_core.ray.exceptions.GetTimeoutError
        return value

    monkeypatch.setattr(rollout_core.ray, "get", fake_get)
    monkeypatch.setattr(
        rollout_core.ray,
        "wait",
        lambda values, **kwargs: (values[:1], values[1:]),
    )
    envs = [FakeEnv(), FakeEnv()]
    vlms = [FakeVlm(), FakeVlm()]
    shards = iter(
        [
            ["scene_a_0", "scene_a_1"],
            ["scene_b_0", "scene_b_1"],
        ]
    )

    try:
        rollout_core.collect_vector_rollouts(
            envs,
            vlms,
            shards,
            target_episodes=4,
            episodes_per_worker=2,
        )
    except RuntimeError as exc:
        assert "completed_workers=1/2" in str(exc)
        assert "wave=1/1" in str(exc)
    else:
        raise AssertionError("Expected the stuck wave to time out")
    assert calls == 3
