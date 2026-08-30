"""
Load a FLOW-MATCHING-head checkpoint into the existing `VectorRolloutPolicy`.

The exact analogue of `bin_rollout.load_bin_policy`, and for the same reason: the closed-
loop ObjectNav harness (`habitat_physical_nav/src/objectnav_eval/bridge.py`) drives a
`VectorRolloutPolicy`, and a `TurnFlowActionRegressor` fits that contract already --
`FlowActionCodec` occupies the `normalizer` slot, so `step()`'s last line,
`self.model.normalizer.denormalize(vector)`, runs the Euler loop from t=1 to t=0 and
`compose_chunk`s the resulting per-tick differentials into the `(T, 3)` chunk the executor
expects. Nothing in `vector_rollout.py` is touched, and nothing on the simulator side knows
this head exists. `FlowRolloutBackend` in that bridge is the caller; it registers as
`flow_rollout`.

Three things this loader has to get right that the bin loader does not.

**DECODE. `FlowActionCodec.DECODES` is `("sample", "context")` and there is deliberately no
`mean`.** `"sample"` is the only executable policy: `"context"` is the offline passthrough
that lets a script save the cheap-to-decode context once per real VLM forward and try decode
rules later on CPU, exactly the role it plays for the AR head. Averaging several ODE
solutions is NOT a sample from the conditional -- it is the conditional mean, which is the
creeping failure this whole line of work exists to remove. Do not add one here, and do not
average chunks at the call site. The validation below runs BEFORE the checkpoint is touched,
against a class constant, so a bad rule fails identically in an interpreter that cannot load
a 2B-parameter VLM at all.

**INTEGRATION STEPS ARE A CHECKPOINT FIELD, NOT A CONSTANT.** `FlowMatchingConfig.
num_inference_steps` round-trips through `turn_vector_head_config.json`, and
`from_pretrained` restores it onto the codec, so the default here is whatever the run
trained under -- never a literal. `num_inference_steps=` overrides it *for a sweep*, which
is the one knob that changes what the same weights produce without changing the weights.

**SEEDING. `FlowActionCodec.generator` is the settable fallback** `generate()` consults when
no generator is passed explicitly -- which is `denormalize`'s case, and therefore the rollout
path. Reseed once per episode (`seed_sampling(policy, base + episode)`) for a
reproducible-per-episode noise stream rather than one that keeps mutating across a whole
process's worth of episodes. `FlowRolloutBackend.reset()` does exactly this.

The naming trap that makes `seed_sampling` differ from the AR backend's `_reseed`, recorded
in `TurnFlowActionRegressor.codec`'s own docstring: `TurnBinClassifier.codec` is the
normalizer-slot object, but `TurnARActionClassifier.codec` is the VQ codebook and its
generator lives on `.normalizer`. This head has no lookup table, so `.codec` IS the
normalizer-slot object -- the bin head's pattern transfers directly, with
`codec.action_scales.device` standing in for `codec.centroids.device`.

Pose injection needs nothing from this module. `VectorRolloutPolicy.__init__` resolves the
checkpoint's own `modality_specs` (by transform, not by token name) and `pose_values` routes
the raw planar `(x, y, theta)` through `pose_frame.relative_se2_last`, which is the same
function the training collator calls. Reimplementing the frame here -- even correctly --
would create the second implementation that `pose_frame.py`'s docstring exists to prevent.
A checkpoint declaring no specs loads and runs with the mechanism inert.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import torch

from longnav.utils.flow_matching_head import FlowActionCodec, TurnFlowActionRegressor
from longnav.utils.vector_rollout import RolloutConfig, VectorRolloutPolicy


def load_flow_policy(
    checkpoint_dir: Union[str, Path],
    cfg: Optional[RolloutConfig] = None,
    decode: str = "sample",
    num_inference_steps: Optional[int] = None,
    processor=None,
    latent_mode: str = "mean",
    pin_flow_noise: Optional[int] = None,
) -> VectorRolloutPolicy:
    """Build a `VectorRolloutPolicy` over a `TurnFlowActionRegressor` checkpoint.

    Mirrors `VectorRolloutPolicy.from_checkpoint` except for the model class: loading a flow
    checkpoint through the regression path would fail on the state dict (the readout MLP
    emits a context vector, not 30 numbers, and the velocity field has no counterpart at
    all), which is the loud failure to prefer over a silently reshaped context.

    `latent_mode` selects what a CVAE checkpoint does at rollout. `mean` takes the prior's
    mean and is the PARITY path -- bit-identical to the deterministic model at init, and the
    arm to compare against a pre-latent checkpoint. `sample` draws `c ~ p(c|o)`, which is
    what an RL policy does. Requesting `sample` from a checkpoint with no latent raises
    rather than quietly giving the deterministic answer, because those two results are
    indistinguishable once written to a results file.

    `pin_flow_noise` freezes the ODE base noise at that seed, so execution is constant across
    decodes and only the intent varies. Without it, `sample` mode varies `c` AND the flow
    noise together -- reproducible, but not the semantics RL will run under.

    `decode` and `num_inference_steps` are validated first, so the error does not require a
    loadable checkpoint. `num_inference_steps=None` keeps the checkpoint's own value --
    `from_pretrained` restores `FlowMatchingConfig` and hands it to the codec -- so an eval
    integrates the ODE the same way the run that produced the metrics did unless it says
    otherwise.
    """
    if decode not in FlowActionCodec.DECODES:
        raise ValueError(
            f"decode must be one of {FlowActionCodec.DECODES}, got {decode!r}"
            + (" -- this head has no 'mean' decode and must not get one: averaging ODE "
               "samples reproduces the conditional mean, which is the creeping failure "
               "the head exists to avoid" if decode == "mean" else "")
        )
    if num_inference_steps is not None and int(num_inference_steps) < 1:
        raise ValueError(
            f"num_inference_steps must be >= 1, got {num_inference_steps}"
        )
    if latent_mode not in ("mean", "sample"):
        raise ValueError(f"latent_mode must be 'mean' or 'sample', got {latent_mode!r}")

    from transformers import AutoProcessor

    cfg = cfg or RolloutConfig()
    checkpoint_dir = Path(checkpoint_dir)
    if processor is None:
        processor = AutoProcessor.from_pretrained(str(checkpoint_dir))
    model = TurnFlowActionRegressor.from_pretrained(
        checkpoint_dir, processor, dtype=cfg.dtype, device=cfg.device
    )
    # `.codec` is the normalizer-slot object for THIS head; see the module docstring.
    model.codec.decode = decode
    if num_inference_steps is not None:
        model.codec.num_inference_steps = int(num_inference_steps)
    if latent_mode != "mean" and getattr(model.codec, "latent", None) is None:
        raise ValueError(
            f"--latent-mode {latent_mode} was asked for but {checkpoint_dir} has no latent "
            "split; it is a deterministic checkpoint. Refusing rather than silently running "
            "the mean path, whose results are indistinguishable from a sampled run."
        )
    model.codec.latent_mode = latent_mode
    if pin_flow_noise is not None:
        model.codec.pin_flow_noise(int(pin_flow_noise))
    print(f"[flow] {model.codec.describe()}", flush=True)
    policy = VectorRolloutPolicy(model, processor, cfg)
    # A flow checkpoint's stop head, when it has one, lives on the state probe -- the
    # flow head itself has none by design. Silent no-op on a checkpoint without one.
    # getattr because the tests substitute a stub policy rather than load a
    # multi-billion-parameter VLM; a real VectorRolloutPolicy always has the method.
    attach = getattr(policy, "attach_state_probe", None)
    if callable(attach):
        attach(checkpoint_dir)
    return policy


def seed_latent(policy: VectorRolloutPolicy, seed: Optional[int]) -> None:
    """Seed the LATENT stream only, leaving the ODE base-noise stream alone.

    With `seed_sampling` fixed to a common value and this one varied, `z_0` is identical
    across runs while `c` differs -- common random numbers, which removes `z_0` from the
    between-run variance without pinning it to a single atypical draw.
    """
    codec = policy.model.codec
    if seed is None:
        codec.latent_generator = None
        return
    device = codec.action_scales.device
    gen = torch.Generator(device=device if device.type == "cuda" else "cpu")
    gen.manual_seed(int(seed))
    codec.latent_generator = gen


def seed_code(policy: VectorRolloutPolicy, seed: Optional[int]) -> None:
    """Seed the CODE stream -- `c ~ p(c|h)` -- leaving `z_0` and the latent alone.

    Needed for reproducibility, not tidiness: `codec.code_generator` defaults to `None`,
    and `torch.multinomial` with no generator draws from the GLOBAL RNG, which
    `seed_sampling` does not touch (it seeds `codec.generator`). So a `code_mode="sample"`
    rollout is NOT reproducible across invocations unless this is called.

    A third stream rather than a reuse of the other two, for the same common-random-numbers
    reason `latent_generator` is separate: holding `z_0` fixed while varying `c` is how the
    code's own contribution is isolated, and that needs them independently seedable.
    """
    codec = policy.model.codec
    if seed is None:
        codec.code_generator = None
        return
    device = codec.action_scales.device
    gen = torch.Generator(device=device if device.type == "cuda" else "cpu")
    gen.manual_seed(int(seed))
    codec.code_generator = gen


def seed_sampling(policy: VectorRolloutPolicy, seed: Optional[int]) -> None:
    """Reseed the codec's fallback noise generator, or clear it if `seed is None`.

    Call once per episode with a fixed, distinct seed (a base seed plus an episode counter).
    The generator must live on the device the context does, because that is where
    `generate()` allocates `x_1 ~ N(0, I)`; `action_scales` is a buffer on the codec, so it
    moved with the model and is the cheapest correct way to ask where that is.

    Every ODE integration in an episode draws from this one stream, so a rollout is
    reproducible end to end -- which is what makes two closed-loop numbers comparable at all.
    """
    codec = policy.model.codec
    if seed is None:
        codec.generator = None
        return
    device = codec.action_scales.device
    gen = torch.Generator(device=device if device.type == "cuda" else "cpu")
    gen.manual_seed(int(seed))
    codec.generator = gen
