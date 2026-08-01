"""The action autoencoder, and the one piece of it that matters: a decoder output
nonlinearity with a genuinely flat region.

    a = sign(x) * relu(|x| - tau)          tau > 0, learned per output dimension

Everything with |x| < tau maps to *exactly* zero, so a positive-volume set of
latents produces a literal stop. That non-injectivity is the whole point -- it is
what lets a continuous latent density push forward onto the data's exact-zero atom,
and it is also what forfeits change-of-variables in action space (accepted: RL
operates on z, the decoder is part of the actuator).

Two implementation details are load-bearing:

1. **Gradients in the dead zone.** `relu(|x| - tau)` has zero gradient wherever it
   is flat, so a unit that lands in the dead zone for a *non*-zero target can never
   climb out -- classic dead-ReLU, except here it is the output layer and it would
   silently cap reconstruction quality. Forward uses the hard threshold; backward
   uses a softplus surrogate of the same shape (a straight-through estimator), which
   is small but non-zero inside the dead zone. Targets that genuinely are zero still
   see zero gradient once they are safely inside, so the atom is a stable fixed
   point rather than something the optimiser fights.

2. **Latent noise.** With a deterministic encoder, every stopped chunk maps to the
   same point and the latent distribution acquires an atom of its own -- which a
   continuous flow cannot fit either, so the problem would simply move. Injecting
   fixed-scale Gaussian noise into z during training does two things at once: it
   forces the decoder's flat region to swallow a *ball* of radius ~sigma rather than
   a point (so the flat region has real volume), and it makes the exported latent
   distribution continuous. The flow trains on `z = mu + sigma * eps`; sigma is
   recorded in the checkpoint and in the export.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

CHUNK_LEN = 10
N_CHANNELS = 3


class SoftThreshold(nn.Module):
    """sign(x) * relu(|x| - tau) with tau > 0 learned per dimension.

    `beta` sets the sharpness of the softplus surrogate used only in the backward
    pass. Larger beta = closer to the true (zero) gradient in the dead zone, i.e.
    less escape pressure. The forward pass is always the exact hard threshold, so
    the outputs are exactly zero regardless of beta.
    """

    def __init__(self, dim: int, tau_init=0.02, beta: float = 40.0,
                 per_dim: bool = True):
        super().__init__()
        n = dim if per_dim else 1
        t = torch.as_tensor(tau_init, dtype=torch.float32).flatten()
        if t.numel() == 1:
            t = t.expand(n).clone()
        assert t.numel() == n, f"tau_init has {t.numel()} entries, expected {n}"
        # softplus^-1(tau_init), so tau starts exactly at tau_init and stays positive
        self.raw_tau = nn.Parameter(torch.log(torch.expm1(t)))
        self.beta = float(beta)

    @property
    def tau(self) -> torch.Tensor:
        return F.softplus(self.raw_tau)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tau = self.tau
        a = x.abs()
        s = torch.sign(x)
        hard = s * (a - tau).clamp(min=0.0)
        soft = s * F.softplus((a - tau) * self.beta) / self.beta
        return soft + (hard - soft).detach()      # forward hard, backward smooth


def mlp(sizes, act=nn.SiLU, norm=True):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            if norm:
                layers.append(nn.LayerNorm(sizes[i + 1]))
            layers.append(act())
    return nn.Sequential(*layers)


class ActionAutoencoder(nn.Module):
    """(B, 10, 3) raw differentials  <->  (B, latent_dim) latent.

    Normalisation is a pure per-channel scale carried as a buffer, so the checkpoint
    is self-contained and 0 always maps to 0 (a mean shift would move the atom off
    the origin and break the flat region's alignment with it).
    """

    def __init__(self, latent_dim: int = 12, hidden: int = 512, depth: int = 3,
                 tau_init=0.02, beta: float = 40.0, noise_std: float = 0.10,
                 per_dim_tau: bool = True, scale=None,
                 chunk_len: int = CHUNK_LEN, n_channels: int = N_CHANNELS):
        super().__init__()
        self.chunk_len, self.n_channels = chunk_len, n_channels
        self.latent_dim, self.noise_std = latent_dim, float(noise_std)
        d_io = chunk_len * n_channels
        t = torch.as_tensor(tau_init, dtype=torch.float32).flatten()
        if t.numel() == n_channels and per_dim_tau:   # per-channel init, tiled over ticks
            t = t.repeat(chunk_len)                   # layout is (tick, channel), C-order
        tau_init = t

        self.encoder = mlp([d_io] + [hidden] * depth + [latent_dim])
        self.decoder = mlp([latent_dim] + [hidden] * depth + [d_io])
        self.threshold = SoftThreshold(d_io, tau_init=tau_init, beta=beta,
                                       per_dim=per_dim_tau)

        s = torch.ones(n_channels) if scale is None else torch.as_tensor(
            scale, dtype=torch.float32)
        self.register_buffer("scale", s)

    # -- normalisation ------------------------------------------------------------
    def normalise(self, x: torch.Tensor) -> torch.Tensor:
        return (x / self.scale).reshape(x.shape[0], -1)

    def denormalise(self, y: torch.Tensor) -> torch.Tensor:
        return y.reshape(y.shape[0], self.chunk_len, self.n_channels) * self.scale

    # -- the two halves -----------------------------------------------------------
    def encode(self, x_raw: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.normalise(x_raw))

    def decode_normalised(self, z: torch.Tensor) -> torch.Tensor:
        return self.threshold(self.decoder(z))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.denormalise(self.decode_normalised(z))

    def forward(self, x_raw: torch.Tensor, noise: bool = True):
        z = self.encode(x_raw)
        z_in = z + self.noise_std * torch.randn_like(z) if noise and self.noise_std > 0 else z
        y = self.decode_normalised(z_in)
        return {"z": z, "z_noised": z_in, "y_norm": y, "recon": self.denormalise(y)}

    # -- checkpointing ------------------------------------------------------------
    def config(self) -> dict:
        return {
            "latent_dim": self.latent_dim, "noise_std": self.noise_std,
            "chunk_len": self.chunk_len, "n_channels": self.n_channels,
            "hidden": self.encoder[0].out_features,
            "depth": sum(isinstance(m, nn.Linear) for m in self.encoder) - 1,
            "beta": self.threshold.beta,
            "per_dim_tau": self.threshold.raw_tau.numel() > 1,
            "scale": self.scale.tolist(),
        }

    @classmethod
    def from_checkpoint(cls, path, map_location="cpu"):
        ck = torch.load(path, map_location=map_location, weights_only=False)
        c = dict(ck["config"])
        c.pop("tau_init", None)
        model = cls(latent_dim=c["latent_dim"], hidden=c["hidden"], depth=c["depth"],
                    beta=c["beta"], noise_std=c["noise_std"],
                    per_dim_tau=c["per_dim_tau"], scale=c["scale"],
                    chunk_len=c["chunk_len"], n_channels=c["n_channels"])
        model.load_state_dict(ck["state_dict"])
        model.eval()
        return model, ck


def moment_penalty(z: torch.Tensor) -> torch.Tensor:
    """Pull the aggregate latent towards per-dimension mean 0, std 1.

    This is the LDM-style latent scale normalisation, learned in rather than applied
    after the fact: it keeps the space the flow has to model well-conditioned, and it
    keeps `noise_std` meaningful (a fixed sigma is only interpretable relative to a
    fixed latent scale). It does *not* force a Gaussian -- the stop mode stays a
    distinct, concentrated cluster, which is what we want the flow to see.
    """
    m = z.mean(0)
    s = z.std(0)
    return (m.pow(2).sum() + (s - 1.0).pow(2).sum()) / z.shape[1]
