"""
Load a discrete-bin-head checkpoint into the existing `VectorRolloutPolicy`.

Track D of `dump/overnight/PLAN.md` (`dump/bin_head_control/FINDINGS.md`) established
open-loop that sampling from the bin head's learned categorical nearly reproduces the
data's true stop-or-drive mass split, while the posterior mean reproduces the regression
head's exact creeping failure. This module is the thin loader that lets a closed-loop
rollout exercise that same decode choice, for `dump/bin_head_control/closed_loop/`.

`VectorRolloutPolicy` (`vector_rollout.py`) is untouched, and `step()` needs no override:
its last line calls `self.model.normalizer.denormalize(chunk)`, and `BinCodec` (the
`normalizer` slot for a `TurnBinClassifier`, see `bin_head.py`) already implements
`denormalize(logits) -> anchor-relative chunk`, decoding per its own `.decode` attribute
(`"argmax" | "sample" | "mean"`) and composing the per-tick differentials back into a
chunk via `compose_chunk`. So loading a bin checkpoint through `VectorRolloutPolicy` and
setting `model.codec.decode` is the whole adaptation -- exactly the pattern
`dump/bin_head_control/predict_bins.py` already uses for its open-loop predictions
(there with `decode = "logits"`, a passthrough for offline analysis; here with a real
decode rule, so `step()` returns an actual `(T, 3)` chunk ready to execute).

Reproducible sampling: `BinCodec.generator` (see `bin_codec.py`) is a sticky fallback the
codec's `sample` decode consults when no generator is passed explicitly, which is exactly
`denormalize`'s case. Reseed it once per episode (`policy.model.codec.generator =
torch.Generator(device=...).manual_seed(seed)`) for a fixed-per-episode, reproducible
sampling stream -- see `BinRolloutBackend` in
`habitat_physical_nav/src/objectnav_eval/bridge.py`, which does exactly this on
`reset()`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import torch

from longnav.utils.bin_head import TurnBinClassifier
from longnav.utils.vector_rollout import RolloutConfig, VectorRolloutPolicy


def load_bin_policy(
    checkpoint_dir: Union[str, Path],
    cfg: Optional[RolloutConfig] = None,
    decode: str = "argmax",
    processor=None,
) -> VectorRolloutPolicy:
    """Build a `VectorRolloutPolicy` over a `TurnBinClassifier` checkpoint.

    Mirrors `VectorRolloutPolicy.from_checkpoint`, except the model class is the bin
    classifier rather than the hardcoded regression `TurnVectorRegressor` -- loading a bin
    checkpoint through the regression path fails loudly on a state-dict mismatch (verified
    in `dump/bin_head_control/FINDINGS.md`) rather than silently reshaping logits as a pose.
    """
    from transformers import AutoProcessor

    cfg = cfg or RolloutConfig()
    checkpoint_dir = Path(checkpoint_dir)
    if processor is None:
        processor = AutoProcessor.from_pretrained(str(checkpoint_dir))
    model = TurnBinClassifier.from_pretrained(
        checkpoint_dir, processor, dtype=cfg.dtype, device=cfg.device
    )
    if decode not in model.codec.DECODES:
        raise ValueError(f"decode must be one of {model.codec.DECODES}, got {decode!r}")
    model.codec.decode = decode
    return VectorRolloutPolicy(model, processor, cfg)


def seed_sampling(policy: VectorRolloutPolicy, seed: Optional[int]) -> None:
    """Reseed the codec's fallback sampling generator, or clear it if `seed is None`.

    Call once per episode with a fixed, distinct seed (e.g. a base seed plus an episode
    counter) so `decode="sample"` is reproducible per episode rather than continuing to
    mutate across an entire process's worth of episodes.
    """
    codec = policy.model.codec
    if seed is None:
        codec.generator = None
        return
    device = codec.centroids.device
    gen = torch.Generator(device=device if device.type == "cuda" else "cpu")
    gen.manual_seed(int(seed))
    codec.generator = gen
