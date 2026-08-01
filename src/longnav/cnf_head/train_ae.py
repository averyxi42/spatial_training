"""Train the action autoencoder.

    <longnav_vlm python> -m src.longnav.cnf_head.train_ae --latent-dim 12

Objective: MSE on *normalised* differentials after a round trip through
`encode -> +sigma*eps -> decode -> soft threshold`, plus a small moment penalty that
holds the aggregate latent at per-dimension mean 0 / std 1.

That is the entire loss. There is deliberately no term rewarding exact zeros: if the
threshold works, zero targets are free (any pre-activation inside the dead zone costs
nothing) and the atom is a stable fixed point of the optimisation rather than
something bribed into existence. If it needed a bribe, the design would be wrong and
the gate should say so.

The gate statistics are evaluated on a validation subsample every `--eval-every`
steps and logged to wandb, so the run is readable live rather than only at the end.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import data as D
from . import metrics as M
from .models import ActionAutoencoder, moment_penalty

OUT = D.REPO / "dump" / "cnf_head" / "autoencoder"
WANDB_PROJECT = "cnf-head-autoencoder"


def run_name(a) -> str:
    ch = "_".join(f"{v:g}" for v in a.l1_channel)
    tau = "_".join(f"{v:g}" for v in a.tau_init)
    return (f"ae-z{a.latent_dim}-sig{a.noise_std:g}-tau{tau}"
            f"-l1{a.l1_weight:g}x{ch}-{a.tag}")


@torch.no_grad()
def evaluate(model, val_raw, val_diffs, stride, device, n_max=200_000):
    """Round-trip the validation set and score it with the existing band stats."""
    model.eval()
    n = min(n_max, len(val_raw))
    recon_clean, recon_noisy, zs = [], [], []
    for i in range(0, n, 65536):
        x = val_raw[i:i + 65536].to(device)
        z = model.encode(x)
        zs.append(z.cpu())
        recon_clean.append(model.decode(z).cpu())
        zn = z + model.noise_std * torch.randn_like(z)
        recon_noisy.append(model.decode(zn).cpu())
    z = torch.cat(zs).numpy()
    out = {}
    gt = val_diffs[:n]
    for tag, rec in (("clean", torch.cat(recon_clean).numpy()),
                     ("noisy", torch.cat(recon_noisy).numpy())):
        rep = M.channel_report(D.gate_view(rec, stride), D.gate_view(gt, stride))
        rep["all10"] = M.channel_report(D.flat_view(rec), D.flat_view(gt))
        rep["verdict"] = M.gate_verdict(rep)
        out[tag] = rep
    out["latent"] = {
        "mean_abs": float(np.abs(z.mean(0)).max()),
        "std_min": float(z.std(0).min()), "std_max": float(z.std(0).max()),
        "std_mean": float(z.std(0).mean()),
        "abs_max": float(np.abs(z).max()),
    }
    out["tau"] = {
        "mean": float(model.threshold.tau.mean()),
        "min": float(model.threshold.tau.min()),
        "max": float(model.threshold.tau.max()),
    }
    tau = model.threshold.tau.detach().cpu().numpy().reshape(model.chunk_len, -1) \
        if model.threshold.raw_tau.numel() > 1 else None
    if tau is not None:
        scale = model.scale.cpu().numpy()
        for c, name in enumerate(D.CHANNELS):
            out["tau"][f"{name}_raw_units"] = float(tau[:, c].mean() * scale[c])
    model.train()
    return out, z


def flatten_log(prefix, rep):
    f, r, s = rep["forward"], rep["rotation"], rep["strafe"]
    return {
        f"{prefix}/fwd_exact_stop": f["recon"]["exact_stop"],
        f"{prefix}/fwd_creep": f["recon"]["creep_band"],
        f"{prefix}/fwd_decisive": f["recon"]["decisive"],
        f"{prefix}/fwd_js": f["jensen_shannon_bits"],
        f"{prefix}/fwd_w1": f["wasserstein1"],
        f"{prefix}/fwd_r2": f["r2"],
        f"{prefix}/fwd_rmse_m": f["rmse"],
        f"{prefix}/rot_exact_stop": r["recon"]["exact_stop"],
        f"{prefix}/rot_js": r["jensen_shannon_bits"],
        f"{prefix}/rot_r2": r["r2"],
        f"{prefix}/strafe_exact_stop": s["recon"]["exact_stop"],
        f"{prefix}/strafe_js": s["jensen_shannon_bits"],
        f"{prefix}/strafe_r2": s["r2"],
        f"{prefix}/mse_all": rep["mse_all"],
        f"{prefix}/pass": float(rep["verdict"]["pass"]),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--latent-dim", type=int, default=12)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--tau-init", type=float, nargs="+", default=[0.10, 0.06, 0.02],
                   metavar="TAU",
                   help="Initial threshold width in NORMALISED units, either one value "
                        "or one per channel (forward, strafe, dtheta). Same reasoning as "
                        "--l1-channel: forward's stop-or-drive gap spans roughly 0.00125 "
                        "to 0.125 normalised, so a threshold anywhere inside it is nearly "
                        "lossless; rotation has no such gap.")
    p.add_argument("--beta", type=float, default=40.0)
    p.add_argument("--noise-std", type=float, default=0.10)
    p.add_argument("--per-dim-tau", type=int, default=1)
    p.add_argument("--moment-weight", type=float, default=0.01)
    p.add_argument("--l1-weight", type=float, default=0.0,
                   help="L1 on the normalised reconstruction. The soft threshold is the "
                        "proximal operator of exactly this penalty, and MSE alone gives "
                        "tau no gradient worth having: a 1 mm residual on a stopped tick "
                        "costs 1e-6 of squared error, so the atom is left to chance. L1 "
                        "supplies a constant pull to zero that vanishing squared error "
                        "does not, and it is per-dimension adaptive -- channels with real "
                        "mass at small magnitudes (rotation) resist it, channels with a "
                        "genuine stop-or-drive gap (forward) do not.")
    p.add_argument("--l1-channel", type=float, nargs=3, default=[1.0, 0.3, 0.1],
                   metavar=("FWD", "STRAFE", "DTHETA"),
                   help="Per-channel multipliers on --l1-weight. Not a free knob -- it "
                        "tracks how much of each channel's mass genuinely lives just "
                        "above zero. On validation ticks the creep band (0.1 mm-1 cm, "
                        "or its angular equivalent) holds 4.5%% of forward mass, 20%% of "
                        "strafe mass and 44%% of rotation mass. Forward has a real "
                        "stop-or-drive gap and a threshold placed in it is nearly free; "
                        "rotation has no gap at all, so the same pressure would snap "
                        "small genuine turns to zero and cost turn fidelity, which is "
                        "the one thing the current policy already does well.")
    p.add_argument("--l1-mode", type=str, default="zero_target",
                   choices=["all", "zero_target"],
                   help="Where the L1 pull applies. 'all' shrinks every output by a "
                        "constant lambda/2, which on this data is ~1.2 mm of forward "
                        "motion -- invisible in MSE but not in Jensen-Shannon, because "
                        "the targets are quantised into hard spikes and a 1.2 mm shift "
                        "walks a spike out of its histogram bin. 'zero_target' applies "
                        "the pull only where the data itself is inside the atom "
                        "(|value| < 0.1 mm), which is exactly where squared error goes "
                        "blind, and leaves decisive motion unbiased.")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=8192)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--scale-percentile", type=float, default=99.0)
    p.add_argument("--drop-strafe", type=int, default=0,
                   help="zero the strafe channel before training (ablation)")
    p.add_argument("--max-train-chunks", type=int, default=0)
    p.add_argument("--train-split", type=str, default="train")
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--eval-chunks", type=int, default=100_000)
    p.add_argument("--tag", type=str, default="v1")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb", type=int, default=1)
    p.add_argument("--out", type=str, default=str(OUT / "runs"))
    a = p.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    name = run_name(a)
    outdir = Path(a.out) / name
    outdir.mkdir(parents=True, exist_ok=True)

    train_diffs, _, _ = D.load_cache(a.train_split)
    val_diffs, _, _ = D.load_cache("validation")
    if a.max_train_chunks:
        idx = np.random.default_rng(a.seed).choice(
            len(train_diffs), min(a.max_train_chunks, len(train_diffs)), replace=False)
        train_diffs = train_diffs[np.sort(idx)]
    scale = np.maximum(D.channel_scale(train_diffs, q=a.scale_percentile), 1e-6)
    if a.drop_strafe:
        # Ablation: the model never sees strafe, on either side. Scored against the
        # REAL validation strafe, so the number is the honest cost of dropping it.
        train_diffs = train_diffs.copy()
        train_diffs[:, :, 1] = 0.0
    print(f"chunks train={len(train_diffs):,} val={len(val_diffs):,}  scale={scale}")

    device = a.device
    model = ActionAutoencoder(
        latent_dim=a.latent_dim, hidden=a.hidden, depth=a.depth, tau_init=a.tau_init,
        beta=a.beta, noise_std=a.noise_std, per_dim_tau=bool(a.per_dim_tau),
        scale=scale).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
    l1_w = torch.tensor(a.l1_channel, dtype=torch.float32, device=device)

    xtr = torch.from_numpy(train_diffs).to(device)
    n = len(xtr)
    steps_per_epoch = n // a.batch
    total_steps = steps_per_epoch * a.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=total_steps, pct_start=0.05)
    val_in = val_diffs[:a.eval_chunks]
    if a.drop_strafe:
        val_in = val_in.copy()
        val_in[:, :, 1] = 0.0
    val_raw = torch.from_numpy(val_in)

    cfg = {**vars(a), **model.config(), "train_chunks": int(n),
           "total_steps": int(total_steps)}
    wb = None
    if a.wandb:
        import wandb
        wb = wandb.init(project=WANDB_PROJECT, name=name, config=cfg, mode="online",
                        group=f"latent-sweep-{a.tag}", dir=str(OUT))
        print("wandb:", wb.url)
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2, default=str))

    step, t0, best = 0, time.time(), None
    for epoch in range(a.epochs):
        perm = torch.randperm(n, device=device)
        for b in range(steps_per_epoch):
            x = xtr[perm[b * a.batch:(b + 1) * a.batch]]
            o = model(x, noise=True)
            target = model.normalise(x)
            recon_loss = torch.nn.functional.mse_loss(o["y_norm"], target)
            mom = moment_penalty(o["z"])
            y = o["y_norm"].reshape(-1, model.chunk_len, model.n_channels).abs()
            l1_mask = (x.abs() < D.EXACT).float() if a.l1_mode == "zero_target" else 1.0
            l1 = (y * l1_w * l1_mask).mean()
            loss = recon_loss + a.moment_weight * mom + a.l1_weight * l1
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            step += 1

            if step % 100 == 0 and wb is not None:
                wb.log({"train/loss": loss.item(), "train/recon_mse_norm": recon_loss.item(),
                        "train/moment": mom.item(), "train/l1": l1.item(),
                        "train/lr": sched.get_last_lr()[0],
                        "train/tau_mean": float(model.threshold.tau.mean()),
                        "epoch": epoch}, step=step)
            if step % a.eval_every == 0 or step == total_steps:
                rep, z = evaluate(model, val_raw, val_diffs[:a.eval_chunks],
                                  D.GATE_STRIDE, device)
                vol = M.flat_region_volume(model, n=50_000, device=device)
                msg = (f"step {step:>6}/{total_steps} ep{epoch:>3} "
                       f"loss {loss.item():.5f}  "
                       f"stop {100 * rep['noisy']['forward']['recon']['exact_stop']:5.1f}% "
                       f"(data {100 * rep['noisy']['forward']['data']['exact_stop']:.1f}%)  "
                       f"creep {100 * rep['noisy']['forward']['recon']['creep_band']:4.2f}% "
                       f"JS {rep['noisy']['forward']['jensen_shannon_bits']:.4f}  "
                       f"R2 {rep['noisy']['forward']['r2']:.3f} | "
                       f"rot JS {rep['noisy']['rotation']['jensen_shannon_bits']:.4f} "
                       f"stop {100 * rep['noisy']['rotation']['recon']['exact_stop']:.1f}% "
                       f"(data {100 * rep['noisy']['rotation']['data']['exact_stop']:.1f}%) | "
                       f"str JS {rep['noisy']['strafe']['jensen_shannon_bits']:.4f} | "
                       f"vol N(0,I) tick {100 * vol['tick_forward_atom']:.1f}% "
                       f"chunk {100 * vol['chunk_forward_all_zero']:.1f}%")
                print(msg, flush=True)
                if wb is not None:
                    wb.log({**flatten_log("val_clean", rep["clean"]),
                            **flatten_log("val_noisy", rep["noisy"]),
                            **{f"latent/{k}": v for k, v in rep["latent"].items()},
                            **{f"tau/{k}": v for k, v in rep["tau"].items()},
                            **{f"volume/{k}": v for k, v in vol.items()
                               if isinstance(v, (int, float))}}, step=step)
                best = rep

    ck = {"state_dict": model.state_dict(), "config": model.config(),
          "args": vars(a), "scale": scale.tolist(), "final_eval": best,
          "train_chunks": int(n)}
    torch.save(ck, outdir / "ae.pt")
    (outdir / "final_eval.json").write_text(json.dumps(best, indent=2))
    print(f"\n-> {outdir / 'ae.pt'}   ({time.time() - t0:.0f}s)")
    if wb is not None:
        wb.summary.update({"final/" + k.split("/", 1)[1]: v
                           for k, v in flatten_log("val_noisy", best["noisy"]).items()})
        wb.finish()


if __name__ == "__main__":
    main()
