"""Compose every demo case under cases/ (each built from named override bundles
under composables/, plus its own raw overrides) against the real hydra.compose
path, and dump the fully-resolved config to output/<case_name>.yaml for review.

This exercises actual `hydra.compose()` / real experiment yamls -- unlike
tests/rl_*.py, which construct RLConfig() directly in Python and never touch
Hydra's defaults-list resolution. That gap is exactly how a config-group
override mistake (a plain `sim: <name>` field instead of a `defaults: -
override /sim: <name>` entry) went unnoticed in an experiment yaml -- see the
built-in regression check at the bottom of this file.

To add a demo case: drop a yaml under cases/ with `composables: [...]` (names
of files under composables/, without extension) and/or `overrides: [...]`
(raw Hydra override strings). Optional `assert_sim_target: <dotted.path>`,
`assert_policy_head_type: <discrete|continuous>`, and `expect_failure: true`
keys add a pass/fail assertion.

Run with: python3 tests/compose_demos/compose_demo.py
"""
import glob
import os
import sys

import yaml as pyyaml
from hydra import compose, initialize
from omegaconf import OmegaConf

from longnav.conf.register_configs import register_configs

HERE = os.path.dirname(os.path.abspath(__file__))
COMPOSABLES_DIR = os.path.join(HERE, "composables")
CASES_DIR = os.path.join(HERE, "cases")
OUTPUT_DIR = os.path.join(HERE, "output")
CONFIG_PATH = os.path.relpath(
    os.path.join(HERE, "..", "..", "src", "longnav", "config"), HERE
)


def load_yaml_dir(path):
    result = {}
    for f in sorted(glob.glob(os.path.join(path, "*.yaml"))):
        name = os.path.splitext(os.path.basename(f))[0]
        with open(f) as fh:
            result[name] = pyyaml.safe_load(fh) or {}
    return result


def resolve_case_overrides(case, composables):
    overrides = []
    for comp_name in case.get("composables", []):
        overrides.extend(composables[comp_name]["overrides"])
    overrides.extend(case.get("overrides", []))
    return overrides


def run_case(name, case, composables):
    overrides = resolve_case_overrides(case, composables)
    expect_failure = case.get("expect_failure", False)
    try:
        cfg = compose(config_name="rl_config", overrides=overrides)
        container = OmegaConf.to_container(cfg, resolve=True)
    except Exception as e:
        if expect_failure:
            with open(os.path.join(OUTPUT_DIR, f"{name}.txt"), "w") as f:
                f.write(f"overrides: {overrides}\n\nEXPECTED failure:\n{e}\n")
            return True, f"correctly failed as expected ({type(e).__name__})"
        with open(os.path.join(OUTPUT_DIR, f"{name}.error.txt"), "w") as f:
            f.write(f"overrides: {overrides}\n\nUNEXPECTED failure:\n{e}\n")
        return False, f"UNEXPECTED failure: {e}"

    if expect_failure:
        return False, "expected a failure but compose succeeded"

    with open(os.path.join(OUTPUT_DIR, f"{name}.yaml"), "w") as f:
        f.write(f"# case: {name}\n# overrides: {overrides}\n\n")
        f.write(OmegaConf.to_yaml(cfg))

    assert_target = case.get("assert_sim_target")
    if assert_target is not None:
        got = container.get("sim", {}).get("_target_")
        if got != assert_target:
            return False, f"assert_sim_target failed: expected {assert_target!r}, got {got!r}"

    # LMHeadConfig (discrete) has no _target_ field at all (nothing extra to
    # construct beyond the backbone), so we assert on `type` rather than
    # `_target_` -- unlike sim backends, every policy_head variant sets `type`.
    assert_policy_head_type = case.get("assert_policy_head_type")
    if assert_policy_head_type is not None:
        got = container.get("vlm", {}).get("policy_head", {}).get("type")
        if got != assert_policy_head_type:
            return False, f"assert_policy_head_type failed: expected {assert_policy_head_type!r}, got {got!r}"

    return True, "ok"


def run_bad_sim_override_regression():
    """Regression check for the exact bug this demo folder exists to catch:
    a plain `sim: <name>` field in an experiment yaml does NOT select the
    ConfigStore group member (it assigns the literal string, which fails
    validation) -- the correct syntax is `defaults: - override /sim: <name>`.
    Writes a throwaway broken yaml into the real experiment/ search path,
    composes it, asserts it raises, then cleans up.
    """
    broken_path = os.path.join(HERE, "..", "..", "src", "longnav", "config", "experiment", "_tmp_bad_sim_override.yaml")
    with open(broken_path, "w") as f:
        f.write("# @package _global_\nsim: dummy_discrete\n")
    try:
        try:
            compose(config_name="rl_config", overrides=["+experiment=_tmp_bad_sim_override"])
        except Exception as e:
            return True, f"correctly failed as expected ({type(e).__name__})"
        return False, "expected a ValidationError but compose succeeded -- has the yaml-vs-CLI override behavior changed?"
    finally:
        os.remove(broken_path)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for f in glob.glob(os.path.join(OUTPUT_DIR, "*")):
        os.remove(f)

    register_configs()
    composables = load_yaml_dir(COMPOSABLES_DIR)
    cases = load_yaml_dir(CASES_DIR)

    results = []
    with initialize(version_base=None, config_path=CONFIG_PATH):
        for name, case in cases.items():
            ok, msg = run_case(name, case, composables)
            results.append((name, ok, msg))

        ok, msg = run_bad_sim_override_regression()
        results.append(("regression: plain `sim: <name>` field must fail", ok, msg))

    width = max(len(name) for name, _, _ in results)
    print()
    for name, ok, msg in results:
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name.ljust(width)}  {msg}")
    print(f"\nResolved configs written to {OUTPUT_DIR}/")

    if not all(ok for _, ok, _ in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
