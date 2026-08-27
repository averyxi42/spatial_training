#!/usr/bin/env python3
"""Dual FSQ tokenizer over action chunks: one codebook for (x, y), one for theta.

`docs/CODE_CONDITIONED_POLICY.md` section 4. `g: A -> c` maps ONE `(20, 3)` chunk to a
pair of discrete codes -- never per tick. Trained standalone on trajectories with no
access to `h`, then frozen, so the code cannot collapse into the `h` channel the way
the latent programme's `c` did.

WHY TWO STREAMS. A single reconstruction loss over `(x, y, theta)` in metres and
radians has to trade positional error against heading error through some scalar, and
under a position-dominated metric a scan -- a large heading excursion with near-zero
positional signature -- contributes little total error and earns no codes. Separate
streams have no exchange rate: the theta quantizer allocates its codes to variance
WITHIN theta, whatever translation is doing. Each encoder therefore sees only its own
channels; letting the theta encoder read x/y would restore the exchange rate through
a shared representation.

WHY CUMULATIVE. A scan is an excursion-and-return in cumulative heading, a shape a
temporal encoder represents readily. In per-tick differential form it is a sign flip
whose meaning only appears after integrating, which is exactly what a reconstruction
loss under-weights.

WHY FSQ. Dead codes are not cosmetic downstream: a rarely-used code has a
poorly-trained rendering, so when RL explores it the decoder emits something bad and
the policy learns to avoid the CODE for reasons unrelated to the behaviour it names.
FSQ has no codebook to collapse -- the codes are grid points and utilisation is
structural -- and its indices are ordinal, so neighbouring codes are behaviourally
adjacent and an exploration step is a local change rather than an arbitrary one.

Usage:

    python data_scripts/train_chunk_tokenizer.py \\
        --dataset data/v2_25hz_obs2.5hz/formatted_nopose \\
        --xy-levels 8 5 --theta-levels 8 5 \\
        --out dump/tokenizer/dual_fsq_40x40
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# --------------------------------------------------------------------------- FSQ
class FSQ(nn.Module):
    """Finite Scalar Quantization (Mentzer et al. 2023), `d` dims x `L_i` levels.

    No codebook, no commitment loss, no dead-code revival: every grid point is
    reachable by construction. The implicit vocabulary is `prod(levels)`.
    """

    def __init__(self, levels):
        super().__init__()
        self.levels = list(levels)
        self.register_buffer("_levels", torch.tensor(self.levels, dtype=torch.float32))
        basis = np.concatenate([[1], np.cumprod(self.levels[:-1])]).astype(np.int64)
        self.register_buffer("_basis", torch.tensor(basis))

    @property
    def vocab(self):
        return int(np.prod(self.levels))

    def _bound(self, z):
        """Squash `z` into the quantisation range. The half-level offset for even `L`
        is what keeps the grid symmetric about zero -- without it an even codebook is
        biased one step to one side."""
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
        index = (idx_per_dim * self._basis).sum(dim=-1)
        return codes, index


# ----------------------------------------------------------------- one code stream
class StreamTokenizer(nn.Module):
    """Encode a `(T, C)` profile to one FSQ code and reconstruct it.

    Mean-pooled encoder: the code describes the WHOLE chunk, which is the invariant
    the policy head depends on (one categorical per decision, not a per-tick
    vocabulary).
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
    def __init__(self, xy_levels, theta_levels, **kw):
        super().__init__()
        self.xy = StreamTokenizer(2, xy_levels, **kw)
        self.theta = StreamTokenizer(1, theta_levels, **kw)

    def forward(self, chunk):
        """`chunk`: (B, T, 3) normalised CUMULATIVE profile."""
        xy_hat, xy_idx = self.xy(chunk[..., :2])
        th_hat, th_idx = self.theta(chunk[..., 2:3])
        return xy_hat, th_hat, xy_idx, th_idx


# ------------------------------------------------------------------------- data
def load_chunks(dataset, split, row_stride, limit):
    from datasets import load_from_disk

    ds = load_from_disk(dataset)
    ds = ds[split] if hasattr(ds, "keys") and split in ds else ds
    # Selecting the column FIRST matters: touching a row otherwise decodes its images.
    ds = ds.select_columns(["action_chunks"])
    rows = range(0, len(ds), row_stride)
    out = []
    for n, i in enumerate(rows):
        a = np.asarray(ds[int(i)]["action_chunks"], dtype=np.float32)
        if a.ndim == 3 and a.shape[1:] == (20, 3):
            out.append(a)
        if limit and sum(len(x) for x in out) >= limit:
            break
        if n % 500 == 0:
            print(f"  loaded {n}/{len(rows)} rows, "
                  f"{sum(len(x) for x in out)} chunks", flush=True)
    return np.concatenate(out, axis=0)


def to_cumulative(chunks):
    """Return cumulative anchor-relative profiles, and say which form the data was in.

    A cumulative chunk's per-tick magnitude grows with the tick index; a differential
    one's is stationary. Deciding by measurement rather than by assumption, because
    getting it backwards would silently train the tokenizer on the wrong object.
    """
    early = np.abs(chunks[:, :3, :]).mean()
    late = np.abs(chunks[:, -3:, :]).mean()
    cumulative = late > 2.5 * early
    return (chunks if cumulative else np.cumsum(chunks, axis=1)), cumulative, early, late


# ------------------------------------------------------------------------- train
def usage_stats(index, vocab):
    counts = np.bincount(index, minlength=vocab).astype(np.float64)
    p = counts / max(counts.sum(), 1.0)
    nz = p[p > 0]
    entropy = float(-(nz * np.log(nz)).sum())
    return {
        "used": int((counts > 0).sum()),
        "vocab": int(vocab),
        "perplexity": float(np.exp(entropy)),
        "entropy_frac": float(entropy / math.log(vocab)) if vocab > 1 else 0.0,
        "top1_share": float(p.max()),
        "counts": counts.astype(np.int64).tolist(),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--row-stride", type=int, default=1)
    ap.add_argument("--limit-chunks", type=int, default=0)
    ap.add_argument("--xy-levels", type=int, nargs="+", default=[8, 5])
    ap.add_argument("--theta-levels", type=int, nargs="+", default=[8, 5])
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mem-fraction", type=float, default=0.06,
                    help="hard cap on this process's share of the GPU. The held128 RL "
                         "run owns these cards; a cap is enforcement, not etiquette.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda") and args.mem_fraction > 0:
        torch.cuda.set_per_process_memory_fraction(args.mem_fraction)
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        print(f"memory cap: {args.mem_fraction:.3f} x {total:.0f} GiB = "
              f"{args.mem_fraction * total:.1f} GiB")

    print("loading chunks ...", flush=True)
    raw = load_chunks(args.dataset, args.split, args.row_stride, args.limit_chunks)
    chunks, was_cum, early, late = to_cumulative(raw)
    print(f"{len(chunks)} chunks, shape {chunks.shape[1:]}; "
          f"|early|={early:.4f} |late|={late:.4f} -> "
          f"{'CUMULATIVE as stored' if was_cum else 'differential, cumsum applied'}")

    # Per-stream normalisation. x and y SHARE a scale -- they are both metres and the
    # relation between them is physical -- while theta gets its own, which is the
    # whole point of the split.
    xy_scale = float(np.abs(chunks[..., :2]).std())
    th_scale = float(np.abs(chunks[..., 2]).std())
    print(f"scales: xy {xy_scale:.4f} m, theta {th_scale:.4f} rad")
    norm = chunks.copy()
    norm[..., :2] /= xy_scale
    norm[..., 2] /= th_scale

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(norm))
    n_val = max(1024, int(args.val_frac * len(norm)))
    val = torch.from_numpy(norm[perm[:n_val]])
    train = torch.from_numpy(norm[perm[n_val:]])
    print(f"train {len(train)}  val {len(val)}")

    dev = args.device
    model = DualTokenizer(args.xy_levels, args.theta_levels,
                          d_model=args.d_model, n_layers=args.n_layers).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"params {n_par/1e6:.2f}M | xy vocab {model.xy.fsq.vocab} "
          f"| theta vocab {model.theta.fsq.vocab} "
          f"| joint {model.xy.fsq.vocab * model.theta.fsq.vocab}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * (len(train) // args.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps,
                                                pct_start=0.05)

    def evaluate():
        model.eval()
        xs, ts, xi, ti = [], [], [], []
        with torch.no_grad():
            for i in range(0, len(val), args.batch_size):
                b = val[i:i + args.batch_size].to(dev)
                xy_hat, th_hat, x_idx, t_idx = model(b)
                xs.append(((xy_hat - b[..., :2]) ** 2).mean(dim=(1, 2)).cpu())
                ts.append(((th_hat - b[..., 2:3]) ** 2).mean(dim=(1, 2)).cpu())
                xi.append(x_idx.cpu()); ti.append(t_idx.cpu())
        model.train()
        return (float(torch.cat(xs).mean().sqrt()), float(torch.cat(ts).mean().sqrt()),
                torch.cat(xi).numpy(), torch.cat(ti).numpy())

    t0, step = time.time(), 0
    for ep in range(args.epochs):
        order = torch.randperm(len(train))
        for i in range(0, len(train) - args.batch_size + 1, args.batch_size):
            b = train[order[i:i + args.batch_size]].to(dev)
            xy_hat, th_hat, _, _ = model(b)
            # Two losses, never summed through a shared scale: this is the exchange
            # rate the split exists to remove. They are added only because the two
            # streams share no parameters, so the sum's gradient is block-diagonal.
            loss_xy = ((xy_hat - b[..., :2]) ** 2).mean()
            loss_th = ((th_hat - b[..., 2:3]) ** 2).mean()
            (loss_xy + loss_th).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
        rx, rt, xi, ti = evaluate()
        ux = usage_stats(xi, model.xy.fsq.vocab)
        ut = usage_stats(ti, model.theta.fsq.vocab)
        joint = len(np.unique(xi * model.theta.fsq.vocab + ti))
        print(f"ep {ep+1}/{args.epochs} step {step} {time.time()-t0:.0f}s | "
              f"val rmse xy {rx*xy_scale:.4f} m  theta {rt*th_scale:.4f} rad | "
              f"used xy {ux['used']}/{ux['vocab']} ppl {ux['perplexity']:.1f} | "
              f"theta {ut['used']}/{ut['vocab']} ppl {ut['perplexity']:.1f} | "
              f"joint observed {joint}", flush=True)

    rx, rt, xi, ti = evaluate()
    summary = {
        "args": vars(args),
        "n_chunks": int(len(chunks)),
        "stored_cumulative": bool(was_cum),
        "xy_scale_m": xy_scale, "theta_scale_rad": th_scale,
        "val_rmse_xy_m": rx * xy_scale, "val_rmse_theta_rad": rt * th_scale,
        "xy_usage": usage_stats(xi, model.xy.fsq.vocab),
        "theta_usage": usage_stats(ti, model.theta.fsq.vocab),
        "joint_observed_on_val": int(
            len(np.unique(xi * model.theta.fsq.vocab + ti))),
        "params": int(n_par),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    torch.save({"model": model.state_dict(), "xy_levels": args.xy_levels,
                "theta_levels": args.theta_levels, "xy_scale": xy_scale,
                "theta_scale": th_scale, "d_model": args.d_model,
                "n_layers": args.n_layers}, out / "tokenizer.pt")
    print(json.dumps({k: v for k, v in summary.items() if k != "args"}, indent=2)[:1200])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
