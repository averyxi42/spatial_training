"""Auto-discovers every scenario in tests/forward/scenarios/*.yaml.

A scenario is one production config combination to fingerprint through the
Component A/B/C chain. Adding coverage for a new run setup means dropping a
new yaml file in tests/forward/scenarios/ -- nothing here or in the three
component test files needs to change; they're all parametrized over
`load_scenarios()`'s return value and derive fixture names from each
scenario's filename-derived `name`.
"""
import glob
import os
from dataclasses import dataclass, field
from typing import List, Optional

import yaml

SCENARIOS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenarios")


@dataclass(frozen=True)
class Scenario:
    name: str
    overrides: List[str] = field(default_factory=list)
    n_steps: int = 8
    # None -> discrete oracle action (cycling index into action_space).
    # <int> -> continuous oracle action (fixed per-step float vector of
    # this dimension). Must match the scenario's policy_head action_space_dim.
    oracle_action_dim: Optional[int] = None


def load_scenarios() -> List[Scenario]:
    scenarios = []
    for path in sorted(glob.glob(os.path.join(SCENARIOS_DIR, "*.yaml"))):
        name = os.path.splitext(os.path.basename(path))[0]
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        scenarios.append(
            Scenario(
                name=name,
                overrides=raw.get("overrides", []),
                n_steps=raw.get("n_steps", 8),
                oracle_action_dim=raw.get("oracle_action_dim", None),
            )
        )
    if not scenarios:
        raise RuntimeError(f"No scenario yaml files found in {SCENARIOS_DIR}")
    return scenarios


SCENARIOS = load_scenarios()
SCENARIO_IDS = [s.name for s in SCENARIOS]

# Shared fixture filenames within each scenario's fixtures/<scenario_name>/
# subfolder -- kept here (not re-derived per test file) so all three
# component tests agree on where to read/write without cross-importing
# each other.
COMPONENT_A_TRAJ_BATCH_FILE = "component_a_traj_batch.pt"
COMPONENT_A_MODEL_INPUTS_FILE = "component_a_model_inputs.pt"
COMPONENT_B_TRAJ_BATCH_FILE = "component_b_traj_batch.pt"
COMPONENT_C_METRICS_FILE = "component_c_metrics.json"
