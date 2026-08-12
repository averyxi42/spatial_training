"""The CVAE prior as an RL policy head, and the flow decoder as its actuator.

Design: `docs/LATENT_RL_ENV.md`. The whole point of this module is how LITTLE it has to be:
`ContinuousActionHead` already returns `{"mu", "log_std"}`, `rollout_core` already samples a
diagonal Gaussian and PPO-ratios it, and `LatentSplit` already emits `(mu, log_sigma)`. So
the RL math, the advantage estimators, the policy loss and `train_loop` are untouched -- they
consume log-probs and are indifferent to what the action means.

WHAT IS ACTUALLY DIFFERENT, and why each difference needs saying:

* `mu` and `log_sigma` come from the TRAINED split, not a fresh `nn.Linear`. A head that
  re-initialises `log_std` from a constant has thrown the CVAE away and is a Gaussian head
  wearing our weights: `sigma` is the quantity the SFT run exists to fit, and its scale is
  the exploration distribution.
* THE ACTION IS NOT WHAT THE ENVIRONMENT RECEIVES. `c` is a 1024-d intent; the robot needs a
  chunk of poses. `decode_action` runs the flow decoder, on the VLM side, because the decoder
  is a torch module holding trained weights and putting it in the simulator process would
  couple the two. `rollout_core` calls it through a `hasattr` hook that no existing head has.
* THE DECODE MUST BE DETERMINISTIC GIVEN `c`. `FlowActionCodec.pin_flow_noise` freezes the
  ODE base noise, so `c -> chunk` is a fixed, differentiable map. Without it the same `c`
  decodes differently on every call and the stored `old_log_prob` stops describing the action
  that actually executed -- which surfaces as a plausible wrong ratio, never as an error.
* ONLY THE FIRST `gap` ROWS ARE EXECUTED. The rest of the chunk is discarded, exactly as in
  training and in the eval executor: the remainder would be superseded by the next
  observation. Inference is synchronous and assumed instantaneous -- no latency masking, no
  prefix conditioning. Both are future work and neither is stubbed here, because a stub that
  silently does nothing is worse than an absence.

The head loads the readout MLP and the codec from a checkpoint WITHOUT instantiating the 2B
backbone: both live in `turn_vector_head.pt` and are described by
`turn_vector_head_config.json`, so they can be rebuilt standalone.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

HEAD_CONFIG_FILE = "turn_vector_head_config.json"
HEAD_WEIGHTS_FILE = "turn_vector_head.pt"


class LatentIntentHead(nn.Module):
    """`hidden_states -> {"mu", "log_std"}` over the latent intent `c`.

    Shape contract is `ContinuousActionHead`'s exactly: `(B, T, hidden)` in, `(B, T, dim)`
    out for both keys, so `rollout_core`'s continuous branch needs no shape special-casing.

    Pooling note: the readout is applied per position. In SFT it saw a single pooled position
    (`train_content_len=1`, `pool_mode=mean`, over which the mean is the identity), and the
    rollout calls the head on `last_hidden_state[:, -1:, :]` -- also one position. So the map
    from hidden state to `h` is the same function in both places, which is what makes the RL
    policy start from the SFT policy rather than near it.
    """

    def __init__(self, readout: nn.Module, codec: nn.Module, gap: int,
                 min_log_std: float = -20.0, max_log_std: float = 2.0):
        super().__init__()
        if getattr(codec, "latent", None) is None:
            raise ValueError(
                "this checkpoint has no latent split, so there is no `c` to be a policy "
                "over -- it is a deterministic checkpoint. Convert it first (see "
                "docs/LATENT_RL.md) or use `gaussian_head`."
            )
        if int(gap) < 1:
            raise ValueError(f"gap must be >= 1, got {gap}")
        self.readout = readout
        self.codec = codec
        self.gap = int(gap)
        # Far wider than the Gaussian head's -5.0 default. `sigma` here is trained by the
        # ELBO and lands around 1% of `h`'s per-dim std (~1e-3), i.e. log_std ~ -6.9; the
        # discrete head's floor would clamp it and silently inflate exploration by ~500x.
        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)

    # -- the policy -----------------------------------------------------------------
    def forward(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        b, t, _ = hidden_states.shape
        flat = hidden_states.reshape(b * t, 1, hidden_states.shape[-1])
        h = self.readout(flat)                                   # (B*T, dim)
        mu, log_sigma = self.codec.latent(h.float())
        log_sigma = log_sigma.clamp(self.min_log_std, self.max_log_std)
        return {"mu": mu.reshape(b, t, -1), "log_std": log_sigma.reshape(b, t, -1)}

    # -- the actuator ---------------------------------------------------------------
    @torch.no_grad()
    def decode_action(self, action: np.ndarray) -> np.ndarray:
        """Sampled `c` -> the `(gap, 3)` chunk the environment executes.

        Returns only the first `gap` rows. The chunk is cumulative anchor-relative poses, so
        truncating it is a prefix of the same trajectory, not a rescaling of it.

        THE DECODER IS FORCED INTO EVAL MODE FOR THE DURATION. Pinning the base noise is not
        by itself enough: the velocity field carries dropout (0.1 by default), and RL calls
        `model.train()` on the whole wrapper, so during training the same `c` would decode
        differently on every call. `old_log_prob` would then describe an action that was
        never executed -- a wrong PPO ratio that stays perfectly finite. Same reasoning, and
        the same remedy, as `_generation_metrics`' `decoder.eval()`.
        """
        dev = self.codec.action_scales.device
        c = torch.as_tensor(np.asarray(action, dtype=np.float32), device=dev).reshape(1, -1)
        was_training = self.codec.decoder.training
        self.codec.decoder.eval()
        try:
            chunk = self.codec.denormalize_from_latent(
                c, noise=self.codec.pinned_flow_noise
            )                                                    # (1, T, 3)
        finally:
            self.codec.decoder.train(was_training)
        return chunk[0, : self.gap].float().cpu().numpy()

    # -- construction ---------------------------------------------------------------
    @classmethod
    def from_policy_head_config(cls, cfg: Dict[str, Any], input_dim: int,
                                dtype: torch.dtype) -> "LatentIntentHead":
        ckpt = cfg.get("checkpoint_dir")
        if not ckpt:
            raise ValueError(
                "latent_head requires `checkpoint_dir`: the split, the readout MLP and the "
                "velocity field are all TRAINED, and a freshly initialised latent head is "
                "not a policy -- it is noise with the right shape."
            )
        seed = cfg.get("pin_flow_noise_seed")
        if seed is None:
            raise ValueError(
                "latent_head requires `pin_flow_noise_seed`. Without a pinned base noise the "
                "same `c` decodes to a different chunk on every call, so `old_log_prob` "
                "describes an action that was never executed and the PPO ratio is wrong "
                "while staying finite. This is not a default worth having."
            )
        readout, codec, ctx_dim = load_latent_stack(ckpt, dtype=dtype)
        if int(cfg.get("action_space_dim", ctx_dim)) != ctx_dim:
            raise ValueError(
                f"action_space_dim={cfg['action_space_dim']} but the checkpoint's latent is "
                f"{ctx_dim}-dimensional. The action IS `c`; these cannot disagree."
            )
        codec.pin_flow_noise(int(seed))
        head = cls(
            readout=readout, codec=codec, gap=int(cfg["gap"]),
            min_log_std=float(cfg.get("gaussian_min_log_std", -20.0)),
            max_log_std=float(cfg.get("gaussian_max_log_std", 2.0)),
        )
        _ = input_dim   # accepted for signature parity with ContinuousActionHead
        return head


def load_latent_stack(checkpoint_dir, dtype: torch.dtype = torch.float32):
    """`(readout, codec, context_dim)` from a checkpoint, WITHOUT the backbone.

    Both modules are described entirely by `turn_vector_head_config.json` and stored in
    `turn_vector_head.pt`, so a 2B-parameter model does not have to be instantiated to get at
    them. Loading is strict in both directions: a checkpoint whose config and weights
    disagree should stop here rather than at the first rollout.
    """
    from longnav.utils.flow_matching_head import FlowActionCodec, FlowActionDecoder
    from longnav.utils.latent_intent import LatentSplit
    from longnav.utils.turn_vectors import TurnVectorHead

    d = Path(checkpoint_dir)
    meta = json.loads((d / HEAD_CONFIG_FILE).read_text())
    latent_meta = meta.get("fm_latent")
    if not latent_meta:
        raise ValueError(
            f"{d} has no `fm_latent` in its head config: it predates the latent split, so it "
            "is a deterministic checkpoint and cannot be a Gaussian policy over `c`."
        )
    model_cfg, ctx_dim = meta["model"], int(meta["fm_context_dim"])

    readout = TurnVectorHead(
        hidden_size=int(meta.get("backbone_hidden_size") or model_cfg.get(
            "backbone_hidden_size") or 2048),
        out_dim=ctx_dim,
        mode=model_cfg.get("pool_mode", "mean"),
        content_len=1 if model_cfg.get("pool_mode") == "flat" else None,
        hidden_dims=tuple(model_cfg.get("head_hidden_dims", ())),
        dropout=float(model_cfg.get("head_dropout", 0.0)),
        layer_norm=bool(model_cfg.get("head_layer_norm", True)),
        standardize=bool(model_cfg.get("standardize_head_inputs", False)),
        dtype=dtype,
    )
    decoder = FlowActionDecoder(
        context_dim=ctx_dim, n_ticks=int(meta["fm_n_ticks"]),
        **(meta.get("fm_decoder_kwargs") or {}),
    )
    dim = int(latent_meta["dim"])
    codec = FlowActionCodec(
        decoder,
        num_inference_steps=int((meta.get("fm_config") or {}).get("num_inference_steps", 10)),
        action_scales=(meta.get("fm_config") or {}).get("action_scales", (0.03, 0.03, 0.05)),
        latent=LatentSplit(
            dim=dim, sigma0=float(latent_meta["sigma0"]),
            rotation=torch.eye(dim) if latent_meta.get("rotated") else None,
        ),
    )
    blob = torch.load(d / HEAD_WEIGHTS_FILE, map_location="cpu", weights_only=False)
    readout.load_state_dict(blob["head"], strict=True)
    codec.load_state_dict(blob["normalizer"], strict=True)
    # `sample` is never used here -- the head reads `mu`/`log_sigma` off the split directly
    # and `decode_action` calls `denormalize_from_latent` with an explicit `c`. Set anyway so
    # that anything reaching for `describe()` reports what this object actually does.
    codec.latent_mode = "sample"
    return readout.eval(), codec.eval(), ctx_dim
