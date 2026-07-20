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


def save_traj_batch(name: str, traj_batch: TensorDict):
    os.makedirs(FIXTURES_DIR, exist_ok=True)
    payload = {
        "batch_size": tuple(traj_batch.batch_size),
        "tensors": {k: v for k, v in traj_batch.items()},
    }
    torch.save(payload, os.path.join(FIXTURES_DIR, f"{name}.pt"))


def load_traj_batch(name: str) -> TensorDict:
    payload = torch.load(os.path.join(FIXTURES_DIR, f"{name}.pt"), weights_only=False)
    return TensorDict(payload["tensors"], batch_size=torch.Size(payload["batch_size"]))


def save_model_inputs(name: str, model_inputs: list):
    """model_inputs is the list of (inputs_tensors_np, inputs_meta) tuples
    `run_rollout_cycle` returns -- exactly what train_rl_step's first two
    positional args come from, one tuple per rollout."""
    os.makedirs(FIXTURES_DIR, exist_ok=True)
    torch.save(model_inputs, os.path.join(FIXTURES_DIR, f"{name}.pt"))


def load_model_inputs(name: str) -> list:
    return torch.load(os.path.join(FIXTURES_DIR, f"{name}.pt"), weights_only=False)
