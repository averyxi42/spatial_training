"""Quick sanity check for the container: python version, habitat, torch, CUDA.

    docker compose run --rm longnav python docker/verify_install.py
"""

import sys

print(f"python           {sys.version.split()[0]}")

import numpy as np

print(f"numpy            {np.__version__}")

import habitat_sim

print(f"habitat_sim      {habitat_sim.__version__}")
print(f"  cuda enabled   {habitat_sim.cuda_enabled}")
print(f"  bullet enabled {habitat_sim.built_with_bullet}")

import habitat

print(f"habitat_lab      {habitat.__version__}")

import torch

print(f"torch            {torch.__version__}")
print(f"  cuda available {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  devices        {torch.cuda.device_count()} x {torch.cuda.get_device_name(0)}")

cfg = habitat_sim.SimulatorConfiguration()
cfg.gpu_device_id = 0
print("sim config       ok")
print(
    "\nNote: this does not open an EGL context. To confirm headless GPU rendering "
    "end to end, load a scene (e.g. `bash setup/download_example_data.sh`) and run "
    "tests/eval_smoke.py."
)
