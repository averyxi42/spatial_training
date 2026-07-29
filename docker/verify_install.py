"""Sanity check the container environment.

    docker compose run --rm longnav python docker/verify_install.py
    docker compose run --rm -e LONGNAV_ENV=vln longnav python docker/verify_install.py

Checks what is expected of whichever conda env is active, and reports the rest.
Exits non-zero if a required piece is missing.
"""

import os
import sys

RAY_PIN = "2.53.0"
NUMPY_PIN = "1.26.4"

# Derive the env from the running interpreter rather than CONDA_DEFAULT_ENV,
# which is only set when the env was activated by a shell. `docker compose exec`
# bypasses the entrypoint, so it is often unset even though the right python is
# on PATH.
env_name = os.path.basename(sys.prefix)
if env_name != os.environ.get("CONDA_DEFAULT_ENV", env_name):
    env_name = os.environ["CONDA_DEFAULT_ENV"]

# Check by capability, not by env name: the names are configurable
# (HABITAT_ENV_NAME / VLM_ENV_NAME) and what matters is what is installed.
try:
    import habitat_sim as _habitat_sim  # noqa: F401

    HAS_HABITAT_SIM = True
except Exception:
    HAS_HABITAT_SIM = False

failures = []


def check(label, fn, required=True):
    try:
        print(f"{label:<18} {fn()}")
    except Exception as exc:  # noqa: BLE001 - report, don't mask
        mark = "FAIL" if required else "skip"
        print(f"{label:<18} {mark}: {type(exc).__name__}: {exc}")
        if required:
            failures.append(label)


def ray_version():
    import ray

    if ray.__version__ != RAY_PIN:
        raise AssertionError(f"{ray.__version__} != pinned {RAY_PIN}")
    return ray.__version__


def numpy_version():
    """Only the habitat env is ABI-pinned.

    habitat-sim's magnum bindings are compiled against numpy 1.26.4, so that env
    must not drift. The VLM env is free to run numpy 2.x: it has no compiled
    habitat extensions, and arrays crossing between the envs go through Ray's
    serialisation rather than a shared ABI, so the two versions need not match.
    """
    import numpy

    if HAS_HABITAT_SIM and numpy.__version__ != NUMPY_PIN:
        raise AssertionError(
            f"{numpy.__version__} != {NUMPY_PIN}, required by the habitat_sim build"
        )
    return numpy.__version__


def longnav_importable():
    import longnav.config_schema as cs

    r = cs.ResourceConfig()
    return f"ok (habitat_conda_env={r.habitat_conda_env}, vlm_conda_env={r.vlm_conda_env})"


def habitat_sim_info():
    import habitat_sim

    return (
        f"{habitat_sim.__version__} cuda={habitat_sim.cuda_enabled} "
        f"bullet={habitat_sim.built_with_bullet}"
    )


def torch_info():
    import torch

    out = f"{torch.__version__} cuda_available={torch.cuda.is_available()}"
    if torch.cuda.is_available():
        out += f" ({torch.cuda.device_count()}x {torch.cuda.get_device_name(0)})"
    return out


print(f"conda env          {env_name}")
print(f"python             {sys.version.split()[0]}")

# Required in both envs. Ray re-execs actors into the *other* env by name
# (runtime_env={"conda": ...}), so both need ray at the same version or actor
# startup fails. Both also need the project importable via the baked .pth.
check("ray", ray_version)
check("numpy", numpy_version)
check("longnav", longnav_importable)

if HAS_HABITAT_SIM:
    # The habitat env carries only the project core (ray + hydra-core) on top of
    # habitat. torch deliberately is not installed here — the sim actors do not
    # need it, and duplicating a multi-GB CUDA wheel per env is not free.
    check("habitat_sim", habitat_sim_info)
    check("habitat_lab", lambda: __import__("habitat").__version__)
else:
    check("torch", torch_info)
    check("transformers", lambda: __import__("transformers").__version__)
    # verl commonly pulls vllm / flash-attn at import, which are optional here,
    # so a failure is informational rather than a broken environment.
    check("verl", lambda: __import__("verl").__file__, required=False)

if failures:
    print(f"\nFAILED: {', '.join(failures)}")
    sys.exit(1)

print(
    "\nAll required checks passed. This does not open an EGL context — for "
    "end-to-end headless rendering run tests/eval_smoke.py with a scene present."
)
