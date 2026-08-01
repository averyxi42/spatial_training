"""Gate metrics for the autoencoder.

These are *not* reimplemented: `band_stats` and `divergences` are imported straight
out of `dump/data_diagnostics/sweep_analysis.py`, the same functions that produced
the checkpoint sweep table (data 73.9% exact stop / 4.0% creep band, policy 0.2-1.2%
/ 29-66%). Reusing them is the point -- the autoencoder's reconstructions land in
the same units, the same bands and the same divergences as every number already on
record, so "did the round trip preserve the atom" is directly comparable to "did the
policy ever emit one".
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
DIAGNOSTICS = REPO / "dump" / "data_diagnostics"
if str(DIAGNOSTICS) not in sys.path:
    sys.path.insert(0, str(DIAGNOSTICS))

from sweep_analysis import (  # noqa: E402
    CREEP, DECISIVE, EXACT, TURN_CREEP, TURN_EXACT, band_stats, divergences,
)

# (name, column, exact, creep, decisive, hist_lo, hist_hi) -- forward/rotation match
# sweep_analysis.analyse() exactly; strafe uses the translation bands.
CHANNEL_SPEC = (
    ("forward", 0, EXACT, CREEP, DECISIVE, -0.02, 0.09),
    ("strafe", 1, EXACT, CREEP, DECISIVE, -0.06, 0.06),
    ("rotation", 2, TURN_EXACT, TURN_CREEP, 0.04, -0.12, 0.12),
)


def channel_report(recon: np.ndarray, data: np.ndarray) -> dict:
    """recon/data: (n_ticks, 3) per-tick differentials. Bands + divergences + R^2."""
    out = {}
    for name, col, exact, creep, dec, lo, hi in CHANNEL_SPEC:
        p, g = recon[:, col], data[:, col]
        r = {"recon": band_stats(p, exact, creep, dec),
             "data": band_stats(g, exact, creep, dec),
             **divergences(p, g, lo, hi)}
        r["creep_excess"] = r["recon"]["creep_band"] - r["data"]["creep_band"]
        r["exact_stop_deficit"] = r["data"]["exact_stop"] - r["recon"]["exact_stop"]
        denom = ((g - g.mean()) ** 2).sum()
        r["r2"] = float(1 - ((p - g) ** 2).sum() / max(1e-12, denom))
        r["mse"] = float(((p - g) ** 2).mean())
        r["rmse"] = float(np.sqrt(r["mse"]))
        out[name] = r
    out["mse_all"] = float(((recon - data) ** 2).mean())
    out["ticks"] = int(len(data))
    return out


def gate_verdict(rep: dict, stop_tol: float = 0.03, creep_tol: float = 0.005) -> dict:
    """Pass condition from the brief, made explicit.

    - reconstructions preserve roughly the data's exact-stop mass on forward motion
      (within `stop_tol` absolute), and
    - do not inflate creep-band mass above the data's (within `creep_tol` absolute).
    """
    f = rep["forward"]
    stop_ok = abs(f["exact_stop_deficit"]) <= stop_tol
    creep_ok = f["creep_excess"] <= creep_tol
    return {
        "forward_exact_stop_recon": f["recon"]["exact_stop"],
        "forward_exact_stop_data": f["data"]["exact_stop"],
        "forward_creep_recon": f["recon"]["creep_band"],
        "forward_creep_data": f["data"]["creep_band"],
        "stop_ok": bool(stop_ok), "creep_ok": bool(creep_ok),
        "pass": bool(stop_ok and creep_ok),
    }


def flat_region_volume(model, n: int = 200_000, device="cuda", seed: int = 0,
                       source: str = "normal", latents: np.ndarray | None = None) -> dict:
    """How much of latent space decodes to the exact-zero atom?

    Design parameter, not a curiosity (dump/overnight/RL_INTERFACE.md): the flat
    region's preimage is behaviourally unidentifiable, so an entropy bonus during RL
    can inflate it for free while the robot stands still. Too small and the flow
    cannot hit the atom; too large and the pathology above dominates.

    `source="normal"`: z ~ N(0, I) in the normalised latent space -- the base measure
    the flow will be pushing forward, and therefore the number that matters for RL.
    `source="posterior"`: z drawn from the aggregate posterior (encoded data + noise)
    -- the number that should match the data's own stopped-chunk rate.
    """
    import torch

    g = torch.Generator(device="cpu").manual_seed(seed)
    if source == "normal":
        z = torch.randn(n, model.latent_dim, generator=g)
    else:
        assert latents is not None
        idx = torch.randint(0, len(latents), (n,), generator=g)
        z = torch.as_tensor(latents[idx.numpy()], dtype=torch.float32)
        z = z + model.noise_std * torch.randn(z.shape, generator=g)
    outs = []
    model = model.to(device).eval()
    with torch.no_grad():
        for i in range(0, n, 65536):
            outs.append(model.decode(z[i:i + 65536].to(device)).cpu().numpy())
    a = np.abs(np.concatenate(outs))                    # (n, 10, 3)
    fwd_zero = a[:, :, 0] < EXACT
    all_zero = (a < EXACT).all(axis=2)
    return {
        "source": source, "samples": int(n),
        "tick_forward_atom": float(fwd_zero.mean()),
        "chunk_forward_all_zero": float(fwd_zero.all(axis=1).mean()),
        "tick_all_channels_atom": float(all_zero.mean()),
        "chunk_all_channels_all_zero": float(all_zero.all(axis=1).mean()),
        "mean_abs_forward": float(a[:, :, 0].mean()),
    }


def atom_radius(model, latents: np.ndarray, diffs: np.ndarray, n: int = 4096,
                n_dirs: int = 8, t_max: float = 4.0, steps: int = 80,
                device="cuda", seed: int = 0) -> dict:
    """How far can you walk from a stopped latent before the robot starts moving?

    The base-measure volume answers "how much of latent space is the atom"; this
    answers it in latent *units*, which is what the flow's and the RL policy's step
    sizes are measured in. Walk from encoded fully-stopped chunks along random unit
    directions and record the distance at which forward motion leaves the atom.

    Read against two yardsticks: the injected `noise_std` (the flat region must be
    wider than that or the gate's noisy round trip could not pass) and the latent's
    own per-dimension scale of 1 (a radius approaching that would mean the
    unidentifiable region is a large part of the space).
    """
    import torch

    stopped = np.where((np.abs(diffs[:, :, 0]) < EXACT).all(axis=1))[0]
    if len(stopped) == 0:
        return {"n_stopped": 0}
    rng = np.random.default_rng(seed)
    pick = rng.choice(stopped, min(n, len(stopped)), replace=False)
    z0 = torch.as_tensor(latents[pick], dtype=torch.float32, device=device)
    model = model.to(device).eval()
    ts = torch.linspace(0, t_max, steps, device=device)
    out = []
    with torch.no_grad():
        for _ in range(n_dirs):
            u = torch.randn(z0.shape, device=device)
            u = u / u.norm(dim=1, keepdim=True)
            alive = torch.ones(len(z0), dtype=torch.bool, device=device)
            radius = torch.full((len(z0),), float("nan"), device=device)
            for t in ts[1:]:
                a = model.decode(z0 + t * u)[:, :, 0].abs()
                moving = (a >= EXACT).any(dim=1) & alive
                radius[moving] = t
                alive &= ~moving
                if not alive.any():
                    break
            radius[alive] = t_max
            out.append(radius.cpu().numpy())
    r = np.concatenate(out)
    return {
        "n_stopped_chunks": int(len(stopped)),
        "median": float(np.median(r)), "p10": float(np.percentile(r, 10)),
        "p90": float(np.percentile(r, 90)), "mean": float(r.mean()),
        "noise_std": float(model.noise_std),
        "median_over_noise_std": float(np.median(r) / max(1e-9, model.noise_std)),
    }


def data_atom_rates(diffs: np.ndarray) -> dict:
    """The same quantities measured on the data, so the volume has something to
    be 'too large' or 'too small' relative to."""
    a = np.abs(diffs)                                    # (n, 10, 3)
    fwd_zero = a[:, :, 0] < EXACT
    all_zero = (a < EXACT).all(axis=2)
    return {
        "tick_forward_atom": float(fwd_zero.mean()),
        "chunk_forward_all_zero": float(fwd_zero.all(axis=1).mean()),
        "tick_all_channels_atom": float(all_zero.mean()),
        "chunk_all_channels_all_zero": float(all_zero.all(axis=1).mean()),
    }
