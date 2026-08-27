"""The frozen dual-FSQ chunk tokenizer `g: A -> (c_xy, c_theta)`.

`docs/CODE_CONDITIONED_POLICY.md` section 4. One code pair per `(T, 3)` chunk, NEVER
per tick. Two streams with no shared parameters and no shared loss scale, because a
single reconstruction metric over metres and radians has to trade them through some
scalar and a scan -- large heading excursion, near-zero positional signature -- loses
that trade and earns no codes.

The model lives here rather than in `data_scripts/` because three consumers need it: the
tokenizer trainer, the code-conditioned SFT head, and the RL-time obedience gauge. A
second copy would drift and every number downstream would still look plausible.

INPUT CONVENTION, easy to get wrong: these encoders eat CUMULATIVE anchor-relative poses
divided by the corpus scales stored in the checkpoint. The flow head eats PER-TICK
BODY-FRAME differentials at `action_scales`. Both derive from the same chunk and neither
normalisation is valid for the other object. `encode_chunk` owns the scaling so callers
cannot cross them.
"""
import math
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

#: `compose_chunk` folds cumulative heading into [-pi, pi] and the fold is invisible
#: downstream -- a wrapped chunk re-encodes to a WRONG code with no error raised. At the
#: shipped robot this is unreachable (`w_max_radps` 2.0 x `chunk_duration` 0.8 = 1.6 rad,
#: a 1.96x margin) but the margin belongs to THIS robot and THIS chunk length, not to the
#: design: at 4 rad/s, or at H = 40, it is gone.
THETA_WRAP_GUARD = 0.9 * math.pi


class FSQ(nn.Module):
    """Finite Scalar Quantization (Mentzer et al. 2023), `d` dims x `L_i` levels.

    No codebook, no commitment loss, no dead-code revival: every grid point is reachable
    by construction. Chosen over VQ/RVQ because dead codes are not cosmetic downstream --
    a rare code has a poorly-trained rendering, so RL would learn to avoid the CODE for
    reasons unrelated to the behaviour it names. Indices are also ordinal, so neighbouring
    codes are behaviourally adjacent.
    """

    def __init__(self, levels):
        super().__init__()
        self.levels = list(levels)
        self.register_buffer("_levels", torch.tensor(self.levels, dtype=torch.float32))
        basis = np.concatenate([[1], np.cumprod(self.levels[:-1])]).astype(np.int64)
        self.register_buffer("_basis", torch.tensor(basis))

    @property
    def vocab(self) -> int:
        return int(np.prod(self.levels))

    def _bound(self, z):
        """Squash into the quantisation range. The half-level offset for even `L` is what
        keeps the grid symmetric about zero; without it an even codebook is biased one
        step to one side."""
        half_l = (self._levels - 1) * (1 - 1e-3) / 2
        offset = torch.where(self._levels % 2 == 0,
                             torch.tensor(0.5, device=z.device),
                             torch.tensor(0.0, device=z.device))
        shift = torch.tan(offset / half_l)
        return torch.tanh(z + shift) * half_l - offset

    def forward(self, z):
        """`z`: (B, d) -> (quantised (B, d) in [-1, 1], index (B,) int64)."""
        bounded = self._bound(z)
        quant = bounded + (torch.round(bounded) - bounded).detach()   # straight-through
        half_width = self._levels // 2
        codes = quant / half_width
        idx_per_dim = (torch.round(bounded).detach() + half_width).long()
        return codes, (idx_per_dim * self._basis).sum(dim=-1)

    def per_dim_index(self, index: torch.Tensor) -> torch.Tensor:
        """(N,) flat index -> (N, d) per-dimension grid coordinates.

        The basis for ORDINAL metrics: FSQ indices carry a geometry, so the distance
        between a prediction and its target is meaningful and a miss to an adjacent grid
        point is a neighbouring behaviour rather than a different one."""
        levels = self._levels.long().to(index.device)
        basis = self._basis.to(index.device)
        return (index[:, None] // basis[None, :]) % levels[None, :]

    def grid_l1(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Grid-step distance between two flat indices -- 0 is exact, 1 is adjacent."""
        return (self.per_dim_index(a) - self.per_dim_index(b)).abs().sum(dim=-1)

    def codes_from_index(self, index: torch.Tensor) -> torch.Tensor:
        """Inverse of `forward`'s index: (N,) -> (N, d) normalised grid points."""
        half = (self._levels // 2).to(index.device)
        return (self.per_dim_index(index) - half[None, :]).float() / half[None, :].float()


class StreamTokenizer(nn.Module):
    """Encode a `(T, C)` profile to ONE FSQ code and reconstruct it.

    Mean-pooled encoder: the code describes the WHOLE chunk, which is the invariant the
    policy head depends on -- one categorical per decision, not a per-tick vocabulary.
    `decode` consumes only the quantised code, so there are exactly `vocab` possible
    reconstructions and the round-trip test over them is exhaustive rather than sampled.
    """

    def __init__(self, n_channels, levels, n_ticks=20, d_model=128, n_layers=2,
                 n_head=4):
        super().__init__()
        self.n_channels, self.n_ticks = n_channels, n_ticks
        self.fsq = FSQ(levels)
        d_latent = len(levels)

        self.in_proj = nn.Linear(n_channels, d_model)
        self.enc_pos = nn.Parameter(torch.randn(1, n_ticks, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, n_head, dim_feedforward=4 * d_model, batch_first=True,
            norm_first=True, dropout=0.0)
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers)
        self.to_latent = nn.Linear(d_model, d_latent)

        self.from_latent = nn.Linear(d_latent, d_model)
        self.dec_pos = nn.Parameter(torch.randn(1, n_ticks, d_model) * 0.02)
        dec_layer = nn.TransformerEncoderLayer(
            d_model, n_head, dim_feedforward=4 * d_model, batch_first=True,
            norm_first=True, dropout=0.0)
        self.decoder = nn.TransformerEncoder(dec_layer, n_layers)
        self.out_proj = nn.Linear(d_model, n_channels)

    def encode(self, x):
        h = self.encoder(self.in_proj(x) + self.enc_pos).mean(dim=1)
        return self.fsq(self.to_latent(h))

    def decode(self, codes):
        h = self.from_latent(codes)[:, None, :].expand(-1, self.n_ticks, -1)
        return self.out_proj(self.decoder(h + self.dec_pos))

    def forward(self, x):
        codes, index = self.encode(x)
        return self.decode(codes), index


class DualTokenizer(nn.Module):
    """One codebook for `(x, y)`, one for `theta`. See the module docstring for why."""

    def __init__(self, xy_levels, theta_levels, **kw):
        super().__init__()
        self.xy = StreamTokenizer(2, xy_levels, **kw)
        self.theta = StreamTokenizer(1, theta_levels, **kw)

    @property
    def vocab_xy(self) -> int:
        return self.xy.fsq.vocab

    @property
    def vocab_theta(self) -> int:
        return self.theta.fsq.vocab

    def forward(self, chunk):
        """`chunk`: (B, T, 3) NORMALISED cumulative profile."""
        xy_hat, xy_idx = self.xy(chunk[..., :2])
        th_hat, th_idx = self.theta(chunk[..., 2:3])
        return xy_hat, th_hat, xy_idx, th_idx


class FrozenChunkTokenizer(nn.Module):
    """A trained `DualTokenizer` loaded, frozen, and given the scaling contract.

    Frozen is not a convenience: trained jointly with a decoder that also sees `h`, the
    code would be allocated only what `h` cannot already explain -- and `h` linear-probes
    to R^2 0.63 on the whole chunk -- which is the latent programme's collapse reproduced.
    Freezing also gives the code head a stationary target.

    Registered as a submodule so `.to(device)` and `.eval()` reach it, but its parameters
    are `requires_grad_(False)` and it is skipped by optimiser construction that filters
    on `requires_grad`.
    """

    def __init__(self, checkpoint: Union[str, Path], strict_wrap_guard: bool = True):
        super().__init__()
        ck = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        self.model = DualTokenizer(ck["xy_levels"], ck["theta_levels"],
                                   d_model=ck["d_model"], n_layers=ck["n_layers"])
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.register_buffer("xy_scale", torch.tensor(float(ck["xy_scale"])))
        self.register_buffer("theta_scale", torch.tensor(float(ck["theta_scale"])))
        self.strict_wrap_guard = bool(strict_wrap_guard)
        self.checkpoint_path = str(checkpoint)

    @property
    def vocab_xy(self) -> int:
        return self.model.vocab_xy

    @property
    def vocab_theta(self) -> int:
        return self.model.vocab_theta

    def train(self, mode: bool = True):        # noqa: D102 -- frozen, always eval
        return super().train(False)

    @torch.no_grad()
    def encode_chunk(self, chunk: torch.Tensor,
                     valid: Optional[torch.Tensor] = None
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        """`chunk`: (N, T, 3) PHYSICAL anchor-relative poses -> `(c_xy, c_theta)` int64.

        Owns the scaling so a caller cannot feed differentials by accident. `valid`
        (N,) bool marks rows with a real chunk; invalid rows (probe rows carry all-NaN
        targets) are encoded from zeros and must be masked by the caller -- encoding NaN
        would poison the FSQ bound with NaN and silently return grid index 0.
        """
        x = chunk.float()
        if valid is not None:
            x = torch.where(valid[:, None, None], x, torch.zeros_like(x))
        if self.strict_wrap_guard:
            m = float(x[..., 2].abs().max()) if x.numel() else 0.0
            if m > THETA_WRAP_GUARD:
                raise ValueError(
                    f"|theta| reached {m:.3f} rad against the wrap guard "
                    f"{THETA_WRAP_GUARD:.3f} (pi = {math.pi:.3f}): compose_chunk folds "
                    "past pi and the fold is invisible downstream. Check "
                    "w_max_radps * chunk_duration against pi."
                )
        x = torch.stack([x[..., 0] / self.xy_scale,
                         x[..., 1] / self.xy_scale,
                         x[..., 2] / self.theta_scale], dim=-1)
        _, xi = self.model.xy.encode(x[..., :2])
        _, ti = self.model.theta.encode(x[..., 2:3])
        return xi.long(), ti.long()
