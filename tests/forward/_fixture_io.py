"""Exact (non-lossy) storage for the forward-pass tier's chained component
fixtures. Per design: each component's stored output is the next
component's stored input -- so fidelity here matters more than
readability. Tensors are saved via torch.save (exact float precision, no
JSON truncation); only the final, already-scalar train_rl_step metrics are
plain JSON (see test_component_c_train_step.py), since those need no
tensor round-tripping at all.
"""
import os

import torch
from tensordict import TensorDict

FIXTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def _fixture_path(scenario_name: str, filename: str) -> str:
    """Fixtures live under one subfolder per scenario:
    tests/forward/fixtures/<scenario_name>/<filename> -- so a scenario's
    whole chain of stored fixtures (A's traj_batch/model_inputs, B's
    traj_batch, C's metrics) is one directory, easy to add/remove/diff as a
    unit when a scenario is added or dropped."""
    return os.path.join(FIXTURES_DIR, scenario_name, filename)


def save_traj_batch(scenario_name: str, filename: str, traj_batch: TensorDict):
    path = _fixture_path(scenario_name, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "batch_size": tuple(traj_batch.batch_size),
        "tensors": {k: v for k, v in traj_batch.items()},
    }
    torch.save(payload, path)


def load_traj_batch(scenario_name: str, filename: str) -> TensorDict:
    payload = torch.load(_fixture_path(scenario_name, filename), weights_only=False)
    return TensorDict(payload["tensors"], batch_size=torch.Size(payload["batch_size"]))


def save_model_inputs(scenario_name: str, filename: str, model_inputs: list):
    """model_inputs is the list of (inputs_tensors_np, inputs_meta) tuples
    `run_rollout_cycle` returns -- exactly what train_rl_step's first two
    positional args come from, one tuple per rollout."""
    path = _fixture_path(scenario_name, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model_inputs, path)


def load_model_inputs(scenario_name: str, filename: str) -> list:
    return torch.load(_fixture_path(scenario_name, filename), weights_only=False)
