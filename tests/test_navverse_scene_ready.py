import sys
import types

from longnav.env.navverse import (
    NavVerseHostActor,
    _scene_request_is_ready,
    _set_robot_visuals_hidden,
)


def test_scene_request_requires_requested_episode_to_be_applied():
    sim = types.SimpleNamespace(
        current_episode=types.SimpleNamespace(episode_label="scene_a_0"),
        reset_flag=True,
        sim_state="running",
    )

    assert not _scene_request_is_ready(sim, "scene_b_0")
    sim.reset_flag = False
    assert not _scene_request_is_ready(sim, "scene_b_0")
    sim.current_episode = types.SimpleNamespace(episode_label="scene_b_0")
    assert _scene_request_is_ready(sim, "scene_b_0")


def test_wait_for_scene_ready_steps_pending_scene_switch():
    class Sim:
        current_episode = types.SimpleNamespace(episode_label="scene_a_0")
        reset_flag = True
        sim_state = "running"
        step_count = 0

        def step(self):
            self.step_count += 1
            self.current_episode = types.SimpleNamespace(episode_label="scene_b_0")
            self.reset_flag = False

    host = object.__new__(NavVerseHostActor)
    host.vln_sim = Sim()

    host._wait_for_scene_ready("scene_b_0")

    assert host.vln_sim.step_count == 1


def test_hidden_robot_visuals_only_toggle_render_geometry(monkeypatch):
    visibility_calls = []
    fake_utils = types.SimpleNamespace(
        set_prim_visibility=lambda prim, visible: visibility_calls.append(
            (prim.path, visible)
        )
    )
    fake_sim = types.ModuleType("isaaclab.sim")
    fake_sim.utils = fake_utils
    fake_isaaclab = types.ModuleType("isaaclab")
    fake_isaaclab.sim = fake_sim
    monkeypatch.setitem(sys.modules, "isaaclab", fake_isaaclab)
    monkeypatch.setitem(sys.modules, "isaaclab.sim", fake_sim)

    class Prim:
        def __init__(self, path):
            self.path = path

        def IsValid(self):
            return True

    class Stage:
        @staticmethod
        def GetPrimAtPath(path):
            return Prim(path)

    manager_env = types.SimpleNamespace(
        num_envs=2, scene=types.SimpleNamespace(stage=Stage())
    )
    vln_sim = types.SimpleNamespace(manager_env=manager_env)

    assert _set_robot_visuals_hidden(vln_sim, True) == 2
    assert visibility_calls == [
        ("/World/envs/env_0/Robot", False),
        ("/World/envs/env_1/Robot", False),
    ]
