"""The gate that decides whether the flow stage proceeds.

Round-trip the *whole* validation set through the autoencoder and score the
reconstructions with the band statistics from
`dump/data_diagnostics/sweep_analysis.py` -- the same code that produced the
checkpoint sweep, so "did the round trip keep the atom" reads on the same axis as
"did the policy ever emit one".

Pass condition (from the brief):
  * reconstructions preserve roughly the data's exact-stop mass on forward motion,
  * and do not inflate creep-band mass above the data's.

Reconstruction MSE is reported but is explicitly *not* the gate. An autoencoder can
have excellent MSE while smearing the atom into a blob a tenth of a millimetre wide;
that is precisely the failure this whole design exists to escape, and MSE cannot see
it.

Two round trips are scored, and the second one is the real test:
  clean   decode(encode(x))                 -- can the decoder represent the atom
  noisy   decode(encode(x) + sigma * eps)   -- does the flat region have VOLUME

The flow will never be handed the encoder's exact output. It samples a latent, and
the sample lands near the encoded point, not on it. If the atom survives only at
sigma = 0 then the flat region is a measure-zero surface, the flow cannot hit it,
and the gate has not really passed.

Also reports the fraction of latent space that decodes to the atom, under both the
base measure N(0, I) and the aggregate posterior -- a design parameter for the whole
head rather than a curiosity, see dump/overnight/RL_INTERFACE.md.

    <vln python> -m src.longnav.cnf_head.gate --ckpt <run>/ae.pt          # one model
    <vln python> -m src.longnav.cnf_head.gate --sweep                     # all runs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import data as D
from . import metrics as M
from .models import ActionAutoencoder

OUT = D.REPO / "dump" / "cnf_head" / "autoencoder"

# Visual language borrowed wholesale from dump/data_diagnostics/analyze_chunk_motion.py
INK, MUTED, GRID = "#1f2328", "#5b6570", "#e3e6ea"
ACCENT, WARN, SECOND = "#2f6f9f", "#b3542e", "#7a5195"


@torch.no_grad()
def round_trip(model, diffs, device="cuda", batch=65536, seed=0):
    model = model.to(device).eval()
    g = torch.Generator(device=device).manual_seed(seed)
    clean, noisy, zs = [], [], []
    for i in range(0, len(diffs), batch):
        x = torch.from_numpy(diffs[i:i + batch]).to(device)
        z = model.encode(x)
        zs.append(z.cpu().numpy())
        clean.append(model.decode(z).cpu().numpy())
        eps = torch.randn(z.shape, generator=g, device=device)
        noisy.append(model.decode(z + model.noise_std * eps).cpu().numpy())
    return np.concatenate(clean), np.concatenate(noisy), np.concatenate(zs)


def run_gate(ckpt: Path, device="cuda", split="validation") -> dict:
    model, ck = ActionAutoencoder.from_checkpoint(ckpt, map_location=device)
    diffs, _, _ = D.load_cache(split)
    clean, noisy, z = round_trip(model, diffs, device=device)

    rep = {"checkpoint": str(ckpt), "config": ck["config"], "args": ck.get("args", {}),
           "split": split, "chunks": int(len(diffs)),
           "train_chunks": int(ck.get("train_chunks", 0))}
    for tag, rec in (("clean", clean), ("noisy", noisy)):
        r = M.channel_report(D.gate_view(rec, D.GATE_STRIDE),
                             D.gate_view(diffs, D.GATE_STRIDE))
        r["all10"] = M.channel_report(D.flat_view(rec), D.flat_view(diffs))
        r["verdict"] = M.gate_verdict(r)
        rep[tag] = r
    rep["latent"] = {
        "dim": int(z.shape[1]),
        "mean": z.mean(0).tolist(), "std": z.std(0).tolist(),
        "abs_mean_max": float(np.abs(z.mean(0)).max()),
        "std_min": float(z.std(0).min()), "std_max": float(z.std(0).max()),
        "noise_std": model.noise_std,
    }
    tau = model.threshold.tau.detach().cpu().numpy().reshape(model.chunk_len, -1)
    scale = model.scale.cpu().numpy()
    rep["tau_raw_units"] = {n: float(tau[:, c].mean() * scale[c])
                            for c, n in enumerate(D.CHANNELS)}
    rep["tau_normalised"] = {n: float(tau[:, c].mean()) for c, n in enumerate(D.CHANNELS)}
    rep["volume"] = {
        "base_normal": M.flat_region_volume(model, n=200_000, device=device,
                                            source="normal"),
        "aggregate_posterior": M.flat_region_volume(model, n=200_000, device=device,
                                                    source="posterior", latents=z),
        "data": M.data_atom_rates(diffs),
        "atom_radius_latent_units": M.atom_radius(model, z, diffs, device=device),
    }
    return rep, clean, noisy, diffs, z


# ======================================================================================
def _style(ax):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, lw=0.7, alpha=0.9)
    ax.set_axisbelow(True)


def distribution_figure(rep, clean, noisy, data, path):
    """Reconstructed vs data marginals, in the style of sweep_distribution_grid.png:
    ground truth as flat grey, reconstruction as the single accent hue, log-y,
    creep band shaded."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gt = D.gate_view(data, D.GATE_STRIDE)
    recs = {"clean round trip": D.gate_view(clean, D.GATE_STRIDE),
            f"noisy round trip (sigma={rep['config']['noise_std']:g})":
                D.gate_view(noisy, D.GATE_STRIDE)}
    specs = [("forward", 0, 100.0, "forward motion per tick [cm]", (-1.0, 8.5),
              (M.EXACT * 100, M.CREEP * 100)),
             ("strafe", 1, 100.0, "strafe per tick [cm]", (-5.0, 5.0),
              (M.EXACT * 100, M.CREEP * 100)),
             ("rotation", 2, np.degrees(1.0), "rotation per tick [deg]", (-8, 8),
              (np.degrees(M.TURN_EXACT), np.degrees(M.TURN_CREEP)))]

    fig, axes = plt.subplots(2, 3, figsize=(14.2, 7.4), sharex="col")
    for row, (label, rec) in enumerate(recs.items()):
        for col, (name, ci, mul, xlabel, rng, band) in enumerate(specs):
            ax = axes[row, col]
            ax.hist(np.clip(gt[:, ci] * mul, *rng), bins=110, range=rng, color=GRID,
                    lw=0, label="data")
            ax.hist(np.clip(rec[:, ci] * mul, *rng), bins=110, range=rng, color=ACCENT,
                    alpha=0.78, lw=0, label="reconstruction")
            ax.axvspan(band[0], band[1], color=WARN, alpha=0.12, lw=0)
            ax.axvspan(-band[1], -band[0], color=WARN, alpha=0.12, lw=0)
            _style(ax)
            ax.set_yscale("log")
            s = rep["noisy" if row else "clean"][name]
            ax.set_title(f"{name}   stop {100 * s['recon']['exact_stop']:.1f}% "
                         f"(data {100 * s['data']['exact_stop']:.1f}%)   "
                         f"creep {100 * s['recon']['creep_band']:.1f}% "
                         f"(data {100 * s['data']['creep_band']:.1f}%)",
                         fontsize=9.5, color=INK, loc="left", pad=6)
            if row == 1:
                ax.set_xlabel(xlabel, color=INK, fontsize=10)
        axes[row, 0].set_ylabel(f"{label}\nticks (log)", color=MUTED, fontsize=9)
    axes[0, 0].legend(frameon=False, fontsize=9, labelcolor=MUTED)
    v = rep["noisy"]["verdict"]
    fig.suptitle(
        f"Autoencoder round trip preserves the stop-or-drive gap (shaded)   "
        f"z={rep['config']['latent_dim']}   "
        f"gate: {'PASS' if v['pass'] else 'FAIL'}",
        color=INK, fontsize=13, x=0.005, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=140, facecolor="white")
    plt.close(fig)
    return path


def sweep_figure(reports, path):
    """Latent-dimension sweep: does the atom survive, and at what volume."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reports = sorted(reports, key=lambda r: r["config"]["latent_dim"])
    dims = [r["config"]["latent_dim"] for r in reports]
    x = np.arange(len(dims))
    fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.3))

    ax = axes[0]
    ax.bar(x, [100 * r["noisy"]["forward"]["recon"]["exact_stop"] for r in reports],
           color=ACCENT, width=0.6)
    ax.axhline(100 * reports[0]["noisy"]["forward"]["data"]["exact_stop"], ls="--",
               lw=1.6, color=WARN, label="data")
    ax.set_ylabel("% of ticks at exact stop", color=MUTED, fontsize=9)
    ax.set_title("Exact-stop mass, forward", color=INK, fontsize=11, loc="left")

    ax = axes[1]
    ax.bar(x, [100 * r["noisy"]["forward"]["recon"]["creep_band"] for r in reports],
           color=ACCENT, width=0.6)
    ax.axhline(100 * reports[0]["noisy"]["forward"]["data"]["creep_band"], ls="--",
               lw=1.6, color=WARN, label="data")
    ax.set_ylabel("% of ticks in the creep band", color=MUTED, fontsize=9)
    ax.set_title("Creep band, forward (lower is safe)", color=INK, fontsize=11,
                 loc="left")

    ax = axes[2]
    for key, c, mk, lab in (("forward", ACCENT, "-o", "forward"),
                            ("rotation", SECOND, "-s", "rotation"),
                            ("strafe", WARN, "-^", "strafe")):
        ax.plot(x, [r["noisy"][key]["jensen_shannon_bits"] for r in reports], mk,
                color=c, lw=2, ms=6, label=lab)
    ax.set_ylabel("Jensen-Shannon divergence [bits]", color=MUTED, fontsize=9)
    ax.set_title("Distance from the data distribution", color=INK, fontsize=11,
                 loc="left")

    ax = axes[3]
    ax.plot(x, [100 * r["volume"]["base_normal"]["tick_forward_atom"] for r in reports],
            "-o", color=ACCENT, lw=2, ms=6, label="per tick")
    ax.plot(x, [100 * r["volume"]["base_normal"]["chunk_forward_all_zero"]
                for r in reports], "-s", color=SECOND, lw=2, ms=6, label="whole chunk")
    ax.set_ylabel("% of z ~ N(0,I) decoding to the atom", color=MUTED, fontsize=9)
    ax.set_title("Flat-region volume under the base measure", color=INK, fontsize=11,
                 loc="left")

    for ax in axes:
        _style(ax)
        ax.set_xticks(x)
        ax.set_xticklabels([f"z={d}" for d in dims])
        ax.legend(frameon=False, fontsize=9, labelcolor=MUTED)
    fig.suptitle("Latent-dimension sweep, noisy round trip on the full validation set",
                 color=INK, fontsize=13, x=0.005, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=140, facecolor="white")
    plt.close(fig)
    return path


def summarise(rep) -> str:
    n, f = rep["noisy"], rep["noisy"]["forward"]
    v = n["verdict"]
    return (f"z={rep['config']['latent_dim']:>2} sig={rep['config']['noise_std']:g} "
            f"| stop {100 * f['recon']['exact_stop']:5.1f}% "
            f"(data {100 * f['data']['exact_stop']:.1f}%) "
            f"creep {100 * f['recon']['creep_band']:5.2f}% "
            f"(data {100 * f['data']['creep_band']:.2f}%) "
            f"JS {f['jensen_shannon_bits']:.4f} R2 {f['r2']:.4f} "
            f"rmse {1000 * f['rmse']:.2f}mm "
            f"| rot JS {n['rotation']['jensen_shannon_bits']:.4f} "
            f"str JS {n['strafe']['jensen_shannon_bits']:.4f} "
            f"| vol {100 * rep['volume']['base_normal']['tick_forward_atom']:.1f}% "
            f"| {'PASS' if v['pass'] else 'FAIL'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--runs", type=str, default=str(OUT / "runs"))
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--tag", type=str, default="sweep1")
    a = p.parse_args()

    ckpts = ([Path(a.ckpt)] if a.ckpt else
             sorted(Path(a.runs).glob(f"*{a.tag}*/ae.pt")))
    if not ckpts:
        raise SystemExit(f"no checkpoints under {a.runs}")

    reports = []
    for c in ckpts:
        rep, clean, noisy, diffs, z = run_gate(c, device=a.device)
        reports.append(rep)
        (c.parent / "gate.json").write_text(json.dumps(rep, indent=2))
        print(f"{c.parent.name}\n  {summarise(rep)}", flush=True)
        fig = OUT / f"gate_distribution_{c.parent.name}.png"
        print("  ->", distribution_figure(rep, clean, noisy, diffs, fig))

    (OUT / "gate_all.json").write_text(json.dumps(reports, indent=2))
    # Latent-dim sweep figure: hold every other knob fixed at the modal setting.
    keys = [(r["config"]["noise_std"], r["args"].get("l1_weight")) for r in reports
            if not r["args"].get("drop_strafe")]
    modal = max(set(keys), key=keys.count) if keys else None
    same = [r for r in reports if not r["args"].get("drop_strafe")
            and (r["config"]["noise_std"], r["args"].get("l1_weight")) == modal]
    if len(same) > 1:
        print("->", sweep_figure(same, OUT / "gate_latent_sweep.png"))
    print("->", OUT / "gate_all.json")


if __name__ == "__main__":
    main()
