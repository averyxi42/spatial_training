# Forward-pass tier (GPU, real checkpoint) -- NOT YET IMPLEMENTED

This directory is a placeholder for Phase 5/6 of the regression-test plan
(`/home/avery/.claude/plans/wise-gathering-music.md`). It intentionally
contains no tests yet -- implementing this tier needs a GPU box to do the
one-time numeric-tolerance calibration the plan requires, which isn't
available in this environment/session.

## What's already in place for when this is picked up

- `tests/conftest.py::checkpoint_model` -- session-scoped fixture that
  builds an `RLWorker` in-process (no Ray) from the real
  `+checkpoint=longnav` config (`Aasdfip/hm3d_rpp_ke_standard-checkpoint_231`)
  and runs `setup_training`. Skips cleanly (not an error) when no GPU is
  available. Not yet consumed by any test.
- `pyproject.toml` registers the `gpu` and `slow` pytest markers this tier's
  tests should carry (`@pytest.mark.gpu @pytest.mark.slow`).
- `src/longnav/env/replay.py::ReplayEnvActor` (Phase 3) can supply the
  teacher-forced, deterministic observation sequences this tier needs to
  build frozen batches against a fixed, hand-chosen action sequence.
- `tests/rl_math/_snapshot.py` has the JSON snapshot read/write mechanics
  this tier will reuse (see schema in the plan, Phase 6) -- but its
  `DEFAULT_TOLERANCE` (rtol/atol ~1e-5) is calibrated for the pure-math
  tier's exact-determinism guarantee. It is NOT valid for this tier as-is.

## What's not done, and why it needs a GPU first

Per the plan (Phase 5's "Numeric tolerance procedure"): forward-pass outputs
here are not bit-exact even with a fixed seed/input, because flash-attention
and cuDNN kernels are themselves nondeterministic
(`torch.use_deterministic_algorithms` doesn't work with flash-attn, and a
CPU fallback for a 2B-param VLM is too slow to be a practical test). Before
any snapshot can be committed for this tier, someone with a GPU box needs
to:

1. Run each planned forward-pass test's numeric output ~10 times with a
   fixed seed/input/weights, measuring the empirical spread (std) per
   metric.
2. Set `atol`/`rtol` comfortably above that observed jitter -- not a guess.
3. Round the committed snapshot values to a precision coarser than that
   tolerance, so routine GPU jitter doesn't produce a spurious diff on
   every re-run.

Only after that calibration should the four planned test files
(`test_rl_loss.py`, `test_lora_merge_unmerge.py`, `test_value_head.py`,
`test_bc_loss.py` -- see the plan's Phase 5 section for what each covers)
be written and their snapshots captured/reviewed/committed.

## Running this tier once it exists

```bash
pytest tests/forward -m gpu   # local/pre-PR gate only, never wired into CI
```
