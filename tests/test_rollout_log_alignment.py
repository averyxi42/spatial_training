from types import SimpleNamespace

from longnav.utils import rollout_core


def test_out_of_order_postprocessing_keeps_trajectory_inputs_results_and_logs_aligned(
    monkeypatch,
):
    class Ref:
        def __init__(self, name, value):
            self.name, self.value = name, value

    def remote(name, value):
        return SimpleNamespace(remote=lambda *args, **kwargs: Ref(name, value))

    def get(ref, **kwargs):
        return [get(item) for item in ref] if isinstance(ref, list) else ref.value

    order = ["reset0", "reset1", "episode1", "post1", "episode0", "post0"]
    completed = []

    def wait(refs, **kwargs):
        chosen = min(refs, key=lambda ref: order.index(ref.name)
                     if ref.name in order else len(order))
        completed.append(chosen.name)
        return [chosen], [ref for ref in refs if ref is not chosen]

    monkeypatch.setattr(rollout_core.ray, "get", get)
    monkeypatch.setattr(rollout_core.ray, "wait", wait)
    sims, vlms = [], []
    for i in range(2):
        sims.append(SimpleNamespace(
            is_exhausted=remote(f"exhausted{i}", False),
            reset=remote(f"reset{i}", None),
            flush_logs_to_disk=remote(f"log{i}", f"summary{i}"),
        ))
        vlms.append(SimpleNamespace(
            run_episode=remote(f"episode{i}", (False, {"uid": i})),
            postprocess_episode=remote(f"post{i}", (f"trajectory{i}", f"inputs{i}")),
        ))

    rollouts, results, logs = rollout_core.collect_rollouts(sims, vlms, iter([]), 2)

    assert completed.index("post1") < completed.index("post0")
    for i, (rollout, result, log) in enumerate(zip(rollouts, results, logs)):
        assert rollout == (f"trajectory{i}", f"inputs{i}")
        assert result == {"uid": i}
        assert get(log) == f"summary{i}"
