"""Flow-SDE policy head: RL over the denoising chain of the SFT flow-matching head.

Design and every decision's evidence: `docs/FLOW_SDE_RL.md`. The one-paragraph version: the
probability-flow ODE and its SDE share marginals, so sampling stochastically reproduces the
action distribution the SFT model already defines and RL initialises AT the SFT policy. The
sampler is hybrid -- of the `K` denoising steps, `N` (default 1-3) are Euler-Maruyama SDE steps
at positions drawn uniformly per chunk, the rest plain ODE steps. Only the `N` stochastic
transitions carry a Gaussian density; their sum is the policy log-prob; one env-step advantage
multiplies it uniformly (no inner GAE, no inner discounting). Eval and deployment run the pure
ODE (`N = 0`) and never see this module's stochastic path.

The head satisfies the same seams `LatentIntentHead` uses, so the existing continuous RL path
needs only capability dispatch (`getattr(head, "chain_log_prob_batch", None)`), never a new
`policy_head.type`:

    forward(hidden)        -> {"h": readout}          # the dict passthrough at
                                                      # vlm_worker.py:644 / :888 carries any dict
    sample_chain_np(h)     -> chain, positions, logprob, chunk    # rollout sampling
    decode_action(chain)   -> (gap, 3)                # the existing actuator seam
    chain_log_prob_batch   -> (B, S) fp32             # old_log_prob recompute AND rl_loss

THE INVARIANT THIS FILE IS ORGANISED AROUND: `sample_chain_np` and `chain_log_prob_batch` must
use the identical transition density, or the PPO ratio is silently wrong. Both therefore call
ONE function, `_sde_transition`, and neither contains its own copy of the math. A convention
mismatch between sampler and scorer is thereby a compile-time impossibility rather than a test
obligation.

Sign conventions are the codebase's own (`euler_integrate`, flow_matching_head.py:349): t = 1 is
noise, `dt = -1/K`, `x <- x + dt * v`, and the field was trained on `u_t = noise - actions` over
`x_t = t * noise + (1 - t) * actions`. Under that convention (derived, not copied -- the
external spec warns the formulas flip with the convention):

    eps_hat = x_t + (1 - t) * v          # exact for the trained target: substitute and check
    score   = -eps_hat / max(t, t_min)   # grad log p_t for the conditional path N((1-t)a, t^2)
    drift   = v + (sigma_t^2 / 2) * score
    mu      = x_t + dt * drift           # dt NEGATIVE -- same sign as the ODE update
    std     = sigma_t * sqrt(|dt|)

with `sigma_t = a * sqrt(t / (1 - t))` on `t` clamped to `[t_sched_min, t_sched_max]`. The
clamp's upper end matters: the schedule is singular at t = 1, which is exactly position 0.
`n_exclude_last` keeps stochastic positions away from the 1/t singularity at the OTHER end.

Numerics, each a recorded failure mode (docs/FLOW_SDE_RL.md "Failure modes"):
  * the velocity net carries dropout=0.1 and RL runs under `model.train()`; every entry point
    here pins the decoder to eval for its duration, so both recomputes and the rollout sample
    describe the same policy (the `LatentIntentHead.decode_action` precedent);
  * all log-prob math is float32 regardless of model dtype -- a bf16 sum of ~60-180 terms
    cannot resolve the fractions of a nat the ratio needs;
  * `z_0`'s density is parameter-free and cancels in the ratio: stored (the chain layout keeps
    it as block 0, and the first transition needs it) but never summed.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from longnav.utils.bin_codec import compose_chunk

HEAD_CONFIG_FILE = "turn_vector_head_config.json"
HEAD_WEIGHTS_FILE = "turn_vector_head.pt"


# ======================================================================================
# Configuration
# ======================================================================================
@dataclass(frozen=True)
class SDEConfig:
    """The sampler's knobs. `n` and `noise_a` are THE two hyperparameters; the rest are
    guards with defaults chosen in docs/FLOW_SDE_RL.md and not expected to move.

    `noise_a` has no safe default: it is the exploration scale, it couples to the learning
    rate (~1/sigma^2), and its usable range is bounded on BOTH sides (fidelity above,
    behavioural indistinguishability below -- the z_0 scatter on this checkpoint is 0.737 rad,
    so too-small `a` explores less than the environment's own noise). It must come from the
    noise sweep, not from a constructor default.
    """

    n: int                      # stochastic steps per chunk, 1 <= n <= K - n_exclude_last
    noise_a: float              # sigma_t = noise_a * sqrt(t / (1 - t))
    n_exclude_last: int = 1     # keep stochastic positions away from the 1/t singularity
    t_min: float = 1e-3         # floor inside the score, belt on top of n_exclude_last
    t_sched_min: float = 1e-3   # sigma_t schedule clamp, low side
    t_sched_max: float = 0.95   # ...high side: position 0 sits at t = 1.0 where the schedule
    #                             is singular. 0.95 is half a step below at K = 10 --
    #                             deliberate, config-visible, and validated by the marginal-
    #                             preservation test rather than derived.

    def __post_init__(self):
        if self.n < 1:
            raise ValueError(f"n must be >= 1, got {self.n} (use the plain ODE for n = 0)")
        if not (self.noise_a > 0.0):
            raise ValueError(
                f"noise_a must be positive, got {self.noise_a}. There is no default: it is "
                "the exploration scale and must come from the noise sweep "
                "(docs/FLOW_SDE_RL.md, Validation)."
            )
        if self.n_exclude_last < 1:
            raise ValueError(
                "n_exclude_last must be >= 1: the score carries 1/t and the final "
                "transition sits closest to t = 0."
            )


# ======================================================================================
# The one transition density (the sampler/scorer invariant lives HERE)
# ======================================================================================
def _sde_transition(decoder: nn.Module, ctx: torch.Tensor, x_t: torch.Tensor,
                    t: torch.Tensor, dt: float, cfg: SDEConfig
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """One SDE step's Gaussian: `(mu, std)` of `z_next` given `(ctx, x_t, t)`.

    Everything is float32 on entry and exit. `t` is per-row `(B,)` -- the batched scorer calls
    this with different positions per row and the decoder already takes vector time.

    The score is NOT detached: it contains the velocity net, and detaching it silently changes
    the gradient estimator into a different algorithm (external spec, wiring rule 2).
    `sigma_t` scales the drift correction (as sigma^2/2) and the injected noise (as
    sigma*sqrt|dt|) from the SAME tensor -- the two are halves of one identity and any knob
    that scales one without the other is wrong (wiring rule 1).
    """
    x_t = x_t.float()
    t = t.float()
    v = decoder(ctx.float(), x_t, t)                                    # (B, T, 3)
    tb = t.view(-1, 1, 1)
    eps_hat = x_t + (1.0 - tb) * v
    score = -eps_hat / tb.clamp_min(cfg.t_min)
    tc = t.clamp(cfg.t_sched_min, cfg.t_sched_max).view(-1, 1, 1)
    sigma_t = cfg.noise_a * torch.sqrt(tc / (1.0 - tc))
    mu = x_t + dt * (v + 0.5 * sigma_t.pow(2) * score)
    std = sigma_t * math.sqrt(abs(dt))
    return mu, std.expand_as(mu)


def _gaussian_logprob(x: torch.Tensor, mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Sum over the chunk dims, keep the batch dim. Float32 throughout."""
    var = std.pow(2)
    lp = -0.5 * ((x - mu).pow(2) / var + torch.log(2.0 * math.pi * var))
    return lp.flatten(1).sum(dim=1)


class _decoder_eval:
    """Pin the velocity net to eval() for the duration, restoring the caller's mode.

    Inside the head's methods rather than at any call site, so the guarantee is
    caller-independent -- the exact remedy `LatentIntentHead.decode_action` uses, needed
    doubly here because dropout in a RECOMPUTED log-prob corrupts the PPO ratio rather than
    just an action."""

    def __init__(self, decoder: nn.Module):
        self.decoder = decoder

    def __enter__(self):
        self.was_training = self.decoder.training
        self.decoder.eval()

    def __exit__(self, *exc):
        self.decoder.train(self.was_training)
        return False


# ======================================================================================
# The head
# ======================================================================================
class FlowSDEHead(nn.Module):
    """`hidden_states -> {"h"}`, plus the chain sampler/scorer/actuator.

    `forward` deliberately returns only the readout: the policy's distribution is over the
    chain, which no `(mu, log_std)` pair can describe, and anything downstream that needs the
    density calls `chain_log_prob_batch`. The dict passthrough carries `{"h"}` untouched;
    the capability checks (`sample_chain_np`, `chain_log_prob_batch`) are how the three
    branch sites recognise this head -- the actuator-seam idiom, not a type string.
    """

    def __init__(self, readout: nn.Module, codec: nn.Module, gap: int, sde: SDEConfig):
        super().__init__()
        if int(gap) < 1:
            raise ValueError(f"gap must be >= 1, got {gap}")
        self.readout = readout
        self.codec = codec
        self.gap = int(gap)
        self.sde = sde
        dec = codec.decoder
        self.K = int(codec.num_inference_steps)
        self.n_ticks, self.n_dims = int(dec.n_ticks), int(dec.n_dims)
        if sde.n > self.K - sde.n_exclude_last:
            raise ValueError(
                f"n={sde.n} stochastic steps but only {self.K - sde.n_exclude_last} "
                f"admissible positions (K={self.K}, n_exclude_last={sde.n_exclude_last})."
            )
        # The rollout draws through a private generator so a seeded run is reproducible
        # without touching the global stream (the latent head's `latent_generator` pattern).
        self._gen: Optional[torch.Generator] = None

    # -- shapes ---------------------------------------------------------------------
    @property
    def block(self) -> int:
        return self.n_ticks * self.n_dims                      # floats per chain element

    @property
    def chain_len(self) -> int:
        return (self.K + 1) * self.block                       # z_0 .. z_K, flattened

    def seed(self, seed: int) -> None:
        dev = next(self.parameters()).device
        self._gen = torch.Generator(device=dev)
        self._gen.manual_seed(int(seed))

    # -- the policy_stats contract ----------------------------------------------------
    def forward(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        b, t, _ = hidden_states.shape
        flat = hidden_states.reshape(b * t, 1, hidden_states.shape[-1])
        h = self.readout(flat)                                  # (B*T, ctx)
        return {"h": h.reshape(b, t, -1).float()}

    # -- rollout: sample -------------------------------------------------------------
    @torch.no_grad()
    def sample_chain_np(self, h: np.ndarray
                        ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        """One chunk: `(chain, positions, logprob, chunk)`.

        `chain` is `((K+1) * T * 3,)` float32 -- z_0 through z_K flattened, so what is stored
        for the ratio is byte-identical to what produced the executed action. `positions` is
        `(n,)` int64, ascending; transition k maps chain block k to block k+1. `logprob` is
        the summed density of the `n` stochastic transitions only.
        """
        dev = next(self.parameters()).device
        ctx = torch.as_tensor(np.asarray(h, np.float32), device=dev).reshape(1, -1)
        cfg, K, dt = self.sde, self.K, -1.0 / self.K
        admissible = K - cfg.n_exclude_last
        perm = torch.randperm(admissible, generator=self._gen, device=dev)[: cfg.n]
        positions = perm.sort().values
        pos_set = set(positions.tolist())

        with _decoder_eval(self.codec.decoder):
            x = torch.randn(1, self.n_ticks, self.n_dims, device=dev,
                            dtype=torch.float32, generator=self._gen)
            blocks = [x]
            logprob = torch.zeros(1, device=dev)
            for k in range(K):
                t_k = torch.full((1,), 1.0 + k * dt, device=dev)
                if k in pos_set:
                    mu, std = _sde_transition(self.codec.decoder, ctx, x, t_k, dt, cfg)
                    eps = torch.randn(mu.shape, device=dev, dtype=torch.float32,
                                      generator=self._gen)
                    x = mu + std * eps
                    logprob = logprob + _gaussian_logprob(x, mu, std)
                else:
                    v = self.codec.decoder(ctx, x, t_k)
                    x = x + dt * v
                blocks.append(x)

        chain = torch.cat([b.reshape(1, -1) for b in blocks], dim=1)[0]
        chunk = self._compose(blocks[-1])
        return (chain.cpu().numpy().astype(np.float32),
                positions.cpu().numpy().astype(np.int64),
                float(logprob.item()),
                chunk)

    # -- the actuator seam (rollout_core.py:204, unchanged) ----------------------------
    @torch.no_grad()
    def decode_action(self, action: np.ndarray) -> np.ndarray:
        """Stored flat chain -> the `(gap, 3)` chunk the environment executes.

        The action IS the chain; the executed chunk is its last block, unscaled and composed
        exactly as `denormalize_from_latent` composes -- then truncated to `gap` rows, which
        is a prefix of the same trajectory because the chunk is cumulative anchor-relative."""
        dev = next(self.parameters()).device
        flat = torch.as_tensor(np.asarray(action, np.float32), device=dev)
        z_K = flat[-self.block:].reshape(1, self.n_ticks, self.n_dims)
        return self._compose(z_K)

    def _compose(self, z_K: torch.Tensor) -> np.ndarray:
        chunk = compose_chunk(self.codec.unscale(z_K.double()))          # (1, T, 3)
        return chunk[0, : self.gap].float().cpu().numpy()

    # -- training: score stored transitions under current theta ------------------------
    def chain_log_prob_batch(self, h: torch.Tensor, chains: torch.Tensor,
                             positions: torch.Tensor) -> torch.Tensor:
        """`(B, S)` summed log-probs, differentiable, float32.

        `h` is `(B, S, ctx)` (the `policy_stats["h"]` of a training forward), `chains` is
        `(B, S, chain_len)` -- the STORED latents; `positions` is `(B, S, n)`.

        Stored latents are data. The deterministic prefix is NOT re-integrated under the
        current parameters -- re-integrating would score a different chain than the one that
        acted, which changes the estimator (docs/FLOW_SDE_RL.md, storage). One decoder call
        per stochastic slot scores the whole batch: the decoder takes vector time, so rows
        with different positions batch together.
        """
        B, S = chains.shape[0], chains.shape[1]
        n, dt = self.sde.n, -1.0 / self.K
        rows = B * S
        ctx = h.reshape(rows, -1).float()
        ch = chains.reshape(rows, self.K + 1, self.n_ticks, self.n_dims).float()
        pos = positions.reshape(rows, n).long()

        with _decoder_eval(self.codec.decoder):
            total = torch.zeros(rows, device=ctx.device, dtype=torch.float32)
            for j in range(n):
                k = pos[:, j]                                            # (rows,)
                t_k = 1.0 + k.float() * dt
                idx = torch.arange(rows, device=ctx.device)
                x_k = ch[idx, k]                                         # (rows, T, 3)
                x_next = ch[idx, k + 1]
                mu, std = _sde_transition(self.codec.decoder, ctx, x_k, t_k, dt, self.sde)
                total = total + _gaussian_logprob(x_next, mu, std)
        return total.reshape(B, S)

    # -- run-log honesty ---------------------------------------------------------------
    def describe(self) -> str:
        return (f"flow-SDE head: K={self.K} n={self.sde.n} a={self.sde.noise_a} "
                f"exclude_last={self.sde.n_exclude_last} gap={self.gap} "
                f"chain={self.chain_len} floats  (eval/deploy = pure ODE)")

    # -- construction ------------------------------------------------------------------
    @classmethod
    def from_policy_head_config(cls, cfg: Dict[str, Any], input_dim: int,
                                dtype: torch.dtype) -> "FlowSDEHead":
        ckpt = cfg.get("checkpoint_dir")
        if not ckpt:
            raise ValueError(
                "flow_sde head requires `checkpoint_dir`: the readout and the velocity "
                "field are trained modules, and a fresh one is not a policy."
            )
        readout, codec, _ = load_flow_stack(ckpt, dtype=dtype)
        head = cls(
            readout=readout, codec=codec, gap=int(cfg["gap"]),
            sde=SDEConfig(
                n=int(cfg.get("sde_n", 1)),
                noise_a=float(cfg["sde_noise_a"]),
                n_exclude_last=int(cfg.get("sde_exclude_last", 1)),
            ),
        )
        seed = cfg.get("sde_seed")
        if seed is not None:
            head.seed(int(seed))
        _ = input_dim   # signature parity with ContinuousActionHead
        return head


# ======================================================================================
# Loading (no backbone, latent optional)
# ======================================================================================
def load_flow_stack(checkpoint_dir, dtype: torch.dtype = torch.float32):
    """`(readout, codec, context_dim)` without instantiating the backbone.

    Unlike `load_latent_stack` this does NOT require `fm_latent`: the flow-SDE policy's
    natural starting point is the DETERMINISTIC checkpoint (that is the entire point -- RL
    initialises at the strongest SFT policy). A latent checkpoint still loads -- the split is
    reconstructed so the strict state-dict load passes -- and its latent is simply never used:
    this head conditions the decoder on `h` directly.
    """
    from longnav.utils.flow_matching_head import FlowActionCodec, FlowActionDecoder
    from longnav.utils.latent_intent import LatentSplit
    from longnav.utils.turn_vectors import TurnVectorHead

    d = Path(checkpoint_dir)
    meta = json.loads((d / HEAD_CONFIG_FILE).read_text())
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
    latent_meta = meta.get("fm_latent")
    latent = None
    if latent_meta:
        dim = int(latent_meta["dim"])
        latent = LatentSplit(
            dim=dim, sigma0=float(latent_meta["sigma0"]),
            rotation=torch.eye(dim) if latent_meta.get("rotated") else None,
        )
    codec = FlowActionCodec(
        decoder,
        num_inference_steps=int((meta.get("fm_config") or {}).get("num_inference_steps", 10)),
        action_scales=(meta.get("fm_config") or {}).get("action_scales", (0.03, 0.03, 0.05)),
        latent=latent,
    )
    blob = torch.load(d / HEAD_WEIGHTS_FILE, map_location="cpu", weights_only=False)
    readout.load_state_dict(blob["head"], strict=True)
    codec.load_state_dict(blob["normalizer"], strict=True)
    return readout.eval(), codec.eval(), ctx_dim
