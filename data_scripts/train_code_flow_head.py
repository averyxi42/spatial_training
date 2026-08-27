#!/usr/bin/env python3
"""The shipped flow head, conditioned ONLY on the discrete code. No backbone.

`docs/CODE_CONDITIONED_POLICY.md`. This measures `p(A | c)` directly: what a code cell
CONTAINS, how tightly the head obeys the code, and how much of the remaining variation
`z_0` chooses. All three are the section 3.2 questions, answerable with no `h`, no RL
and no 2B backbone in the loop.

READ THE RESULT AS AN UPPER BOUND. A c-only head models `p(A | c)` marginalised over
states; the real design conditions on `(c, r(h))` as well, and adding `r(h)` can only
narrow the conditional. So a tight cell here is strong evidence and a loose cell
quantifies exactly how much work `r(h)` has to do -- which is also the number that
decides `r(h)` against `r(h, c)`.

WHAT IS IDENTICAL to the SFT/RL head: `FlowActionDecoder` and every hyperparameter,
`FlowActionCodec`, `action_scales`, the time law, `k_samples`, `num_inference_steps`,
and the objective itself -- imported, never reimplemented, because a second copy of the
loss would drift from the one that produced the weights and every number here would
still look plausible.

WHAT DIFFERS: the 1024-dim context comes from two frozen code embeddings instead of a
readout MLP over `h`.

CONTEXT TOKEN LAYOUT. The decoder has no context projection -- it reshapes (N, 1024)
to (N, 8, 128) -- so the layout is a deliberate reservation, not an implementation
detail:

    token 0-1  <- c_xy   embedding (2 x 128)
    token 2-3  <- c_theta embedding (2 x 128)
    token 4-7  <- ZERO, reserved for the continuous branch r(h)

Leaving the second half zeroed now means `r(h)` lands later without touching the
architecture, and the code-conditioned weights transfer unchanged.

TWO REPRESENTATIONS, TWO NORMALISATIONS, DO NOT CROSS THEM. The tokenizer eats
CUMULATIVE anchor-relative poses normalised by corpus std (xy 0.346 m shared, theta
0.383 rad). The flow head eats PER-TICK BODY-FRAME differentials via `decompose_chunk`,
normalised by `action_scales` (0.03, 0.03, 0.05). Both derive from the same (20, 3)
chunk; neither normalisation is valid for the other object, and swapping them trains
fine and prints plausible numbers.
"""
import argparse
import json
import sys
import time as _time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "src"), str(_ROOT / "data_scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from longnav.utils.flow_matching_head import (                             # noqa: E402
    ACTION_SCALES, NUM_INFERENCE_STEPS, TIME_ALPHA, TIME_BETA, TIME_OFFSET,
    TIME_SCALE, FlowActionCodec, FlowActionDecoder, flow_interpolate, sample_noise,
    sample_time,
)
from train_chunk_tokenizer import DualTokenizer, load_chunks, to_cumulative  # noqa: E402

N_CONTEXT_TOKENS, D_MODEL = 8, 128
CODE_TOKENS_XY, CODE_TOKENS_TH = 2, 2      # tokens 0-1 and 2-3; 4-7 stay zero


class CodeContext(nn.Module):
    """`(c_xy, c_theta) -> (N, 1024)`, with the back half reserved and zero.

    Two tables rather than one joint 1600-entry table: the factors stay legible to
    attention, and every combination is representable including rare ones, which is
    the compositional generalisation a pruned joint vocabulary would throw away.
    """

    def __init__(self, n_xy, n_theta, d_model=D_MODEL, n_tokens=N_CONTEXT_TOKENS,
                 tok_xy=CODE_TOKENS_XY, tok_th=CODE_TOKENS_TH):
        super().__init__()
        self.d_model, self.n_tokens = d_model, n_tokens
        self.tok_xy, self.tok_th = tok_xy, tok_th
        if tok_xy + tok_th > n_tokens:
            raise ValueError("code tokens exceed the context budget")
        self.emb_xy = nn.Embedding(n_xy, tok_xy * d_model)
        self.emb_th = nn.Embedding(n_theta, tok_th * d_model)
        nn.init.normal_(self.emb_xy.weight, std=0.02)
        nn.init.normal_(self.emb_th.weight, std=0.02)
        self.reserved = n_tokens - tok_xy - tok_th

    @property
    def context_dim(self):
        return self.n_tokens * self.d_model

    def forward(self, c_xy, c_th):
        parts = [self.emb_xy(c_xy), self.emb_th(c_th)]
        if self.reserved:
            # The r(h) slot. Explicit zeros, not a shorter context: the decoder's
            # width contract is fixed and the reservation must be visible in the shape.
            parts.append(torch.zeros(c_xy.shape[0], self.reserved * self.d_model,
                                     device=c_xy.device, dtype=parts[0].dtype))
        return torch.cat(parts, dim=-1)


def encode_corpus(tok, chunks, xy_scale, th_scale, device, batch=8192):
    """Frozen tokenizer over the whole corpus, once. Cached so training never runs it."""
    out_xy, out_th = [], []
    with torch.no_grad():
        for i in range(0, len(chunks), batch):
            b = chunks[i:i + batch].copy()
            b[..., :2] /= xy_scale
            b[..., 2] /= th_scale
            t = torch.from_numpy(b).to(device)
            _, xi = tok.xy.encode(t[..., :2])
            _, ti = tok.theta.encode(t[..., 2:3])
            out_xy.append(xi.cpu().numpy().astype(np.int16))
            out_th.append(ti.cpu().numpy().astype(np.int16))
    return np.concatenate(out_xy), np.concatenate(out_th)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--row-stride", type=int, default=1)
    ap.add_argument("--n-ticks", type=int, default=20)
    ap.add_argument("--k-samples", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-chunks", type=int, default=16384)
    ap.add_argument("--spread-samples", type=int, default=16,
                    help="z_0 draws per validation chunk for the cell-spread measurement")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mem-fraction", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume-from", default=None,
                    help="warm-start decoder+context from a previous code_flow_head.pt. "
                         "The LR schedule restarts, which is the point: a OneCycle anneal "
                         "to ~0 makes a flat tail nearly certain regardless of whether "
                         "more training would help, so a second cycle from these weights "
                         "separates 'converged' from 'out of learning rate'.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    dev = args.device
    if dev.startswith("cuda") and args.mem_fraction > 0:
        torch.cuda.set_per_process_memory_fraction(args.mem_fraction)
        tot = torch.cuda.get_device_properties(0).total_memory / 2**30
        print(f"memory cap {args.mem_fraction*tot:.1f} GiB of {tot:.0f}")

    # ---- frozen tokenizer
    ck = torch.load(args.tokenizer, map_location="cpu", weights_only=False)
    tok = DualTokenizer(ck["xy_levels"], ck["theta_levels"],
                        d_model=ck["d_model"], n_layers=ck["n_layers"])
    tok.load_state_dict(ck["model"]); tok.eval().to(dev)
    for p in tok.parameters():
        p.requires_grad_(False)
    n_xy, n_th = tok.xy.fsq.vocab, tok.theta.fsq.vocab

    # ---- corpus, and the codes, once
    raw = load_chunks(args.dataset, args.split, args.row_stride, 0)
    chunks, was_cum, _, _ = to_cumulative(raw)
    print(f"{len(chunks)} chunks, stored_cumulative={was_cum}", flush=True)
    c_xy, c_th = encode_corpus(tok, chunks, ck["xy_scale"], ck["theta_scale"], dev)
    print(f"codes cached: xy {len(np.unique(c_xy))}/{n_xy} distinct, "
          f"theta {len(np.unique(c_th))}/{n_th}", flush=True)

    # ---- head, built exactly as shipped
    ctx = CodeContext(n_xy, n_th).to(dev)
    decoder = FlowActionDecoder(context_dim=ctx.context_dim, n_ticks=args.n_ticks,
                                d_model=D_MODEL, n_layers=4, n_heads=4, dim_ff=512,
                                dropout=0.1, n_context_tokens=N_CONTEXT_TOKENS,
                                n_dims=3, use_incoming_motion=False).to(dev)
    codec = FlowActionCodec(decoder, num_inference_steps=NUM_INFERENCE_STEPS,
                            action_scales=ACTION_SCALES).to(dev)
    if args.resume_from:
        prev = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        decoder.load_state_dict(prev["decoder"])
        ctx.load_state_dict(prev["ctx"])
        print(f"warm start from {args.resume_from} (epoch tag {prev.get('epoch')})")
    n_par = sum(p.numel() for p in decoder.parameters()) + \
        sum(p.numel() for p in ctx.parameters())
    print(f"head params {n_par/1e6:.3f}M | context tokens: "
          f"{ctx.tok_xy} xy + {ctx.tok_th} theta + {ctx.reserved} RESERVED (zero)")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(chunks))
    v_idx, t_idx = perm[:args.val_chunks], perm[args.val_chunks:]
    chunks_t = torch.from_numpy(chunks)
    cxy_t, cth_t = torch.from_numpy(c_xy.astype(np.int64)), \
        torch.from_numpy(c_th.astype(np.int64))

    params = list(decoder.parameters()) + list(ctx.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * (len(t_idx) // args.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps,
                                                pct_start=0.05)
    K, T, D = args.k_samples, args.n_ticks, 3

    def batch_of(idx):
        ch = chunks_t[idx].to(dev)
        return (codec.normalize(ch).float(), cxy_t[idx].to(dev), cth_t[idx].to(dev), ch)

    def flow_loss(actions, cx, ct):
        """The shipped objective, function for function."""
        N = actions.shape[0]
        c = ctx(cx, ct)
        ctx_k = c.repeat_interleave(K, dim=0)
        act_k = actions.repeat_interleave(K, dim=0)
        tt = sample_time(N, K, alpha=TIME_ALPHA, beta=TIME_BETA, scale=TIME_SCALE,
                         offset=TIME_OFFSET, stratified=True, device=dev)
        noise = sample_noise(N, K, T, D, antithetic=False, device=dev)
        x_t, u_t = flow_interpolate(act_k, noise, tt)
        v_t = decoder(ctx_k, x_t, tt)
        return F.mse_loss(v_t, u_t)

    @torch.no_grad()
    def evaluate(n=4096, spread_m=None, chunk_rows=2048):
        """`chunk_rows` caps the working batch. The K-sample expansion multiplies it by
        `k_samples`, so evaluating 16k rows in one call asks for 131k decoder rows --
        which is how the first run of this script died AFTER finishing training."""
        decoder.eval(); ctx.eval()
        idx = torch.from_numpy(v_idx[:n])
        mses, e_xy, e_th, ox, ot = [], [], [], [], []
        for i in range(0, len(idx), chunk_rows):
            sl = idx[i:i + chunk_rows]
            actions, cx, ct, true_chunk = batch_of(sl)
            mses.append(float(flow_loss(actions, cx, ct)) * len(sl))
            gen = codec.denormalize(ctx(cx, ct)).float()
            err = gen - true_chunk
            e_xy.append(err[..., :2].pow(2).mean(dim=(1, 2)).cpu())
            e_th.append(err[..., 2].pow(2).mean(dim=1).cpu())
            g = gen.clone()
            g[..., :2] /= ck["xy_scale"]; g[..., 2] /= ck["theta_scale"]
            _, gx = tok.xy.encode(g[..., :2])
            _, gt = tok.theta.encode(g[..., 2:3])
            ox.append((gx == cx).cpu()); ot.append((gt == ct).cpu())
        ox, ot = torch.cat(ox), torch.cat(ot)
        res = {
            "flow_mse": float(sum(mses) / len(idx)),
            "gen_rmse_xy_m": float(torch.cat(e_xy).mean().sqrt()),
            "gen_rmse_theta_rad": float(torch.cat(e_th).mean().sqrt()),
            "obey_xy": float(ox.float().mean()),
            "obey_theta": float(ot.float().mean()),
            "obey_both": float((ox & ot).float().mean()),
        }

        # -- z_0-INDUCED SPREAD at fixed c, against the EMPIRICAL within-cell spread.
        # The model's spread is only meaningful next to what the cell actually holds:
        # matching it means the head represents the cell, exceeding it means the head
        # is over-dispersed and z_0 is choosing more than the code left open.
        M = spread_m or args.spread_samples
        sub = idx[:512]
        _, sx, st, _ = batch_of(sub)
        rep_x = sx.repeat_interleave(M); rep_t = st.repeat_interleave(M)
        gen_m = codec.denormalize(ctx(rep_x, rep_t)).float().view(len(sub), M, T, 3)
        res["z0_spread_xy_m"] = float(gen_m[..., :2].std(dim=1).mean())
        res["z0_spread_theta_rad"] = float(gen_m[..., 2].std(dim=1).mean())
        res["z0_spread_terminal_xy_m"] = float(
            gen_m[:, :, -1, :2].std(dim=1).norm(dim=-1).mean())

        joint = c_xy.astype(np.int64) * n_th + c_th.astype(np.int64)
        sub_joint = joint[sub.numpy()]
        sx_, st_ = [], []
        for j in np.unique(sub_joint)[:64]:
            members = chunks[joint == j]
            if len(members) < 8:
                continue
            sx_.append(members[..., :2].std(axis=0).mean())
            st_.append(members[..., 2].std(axis=0).mean())
        res["corpus_within_cell_xy_m"] = float(np.mean(sx_)) if sx_ else float("nan")
        res["corpus_within_cell_theta_rad"] = float(np.mean(st_)) if st_ else float("nan")
        decoder.train(); ctx.train()
        return res

    def save(tag):
        """3.5 MB. Written EVERY epoch, because a mid-run crash that costs 30 minutes of
        GPU is not a trade anyone would take against a few megabytes of disk."""
        torch.save({"decoder": decoder.state_dict(), "ctx": ctx.state_dict(),
                    "decoder_config": decoder.to_config(), "epoch": tag},
                   out / "code_flow_head.pt")

    metrics_path = out / "metrics.jsonl"
    t0, step = _time.time(), 0
    for ep in range(args.epochs):
        order = t_idx[rng.permutation(len(t_idx))]
        for i in range(0, len(order) - args.batch_size + 1, args.batch_size):
            actions, cx, ct, _ = batch_of(torch.from_numpy(order[i:i + args.batch_size]))
            loss = flow_loss(actions, cx, ct)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
        save(ep + 1)
        r = evaluate()
        # Appended as it is produced, not assembled at the end: the first run's per-epoch
        # numbers survived only in a log I had to scrape.
        with metrics_path.open("a") as fh:
            fh.write(json.dumps({"epoch": ep + 1, "step": step, **r}) + "\n")
        print(f"ep {ep+1}/{args.epochs} step {step} {_time.time()-t0:.0f}s | "
              f"mse {r['flow_mse']:.4f} | gen rmse xy {r['gen_rmse_xy_m']:.4f} m "
              f"th {r['gen_rmse_theta_rad']:.4f} rad | obey xy {r['obey_xy']:.3f} "
              f"th {r['obey_theta']:.3f} both {r['obey_both']:.3f} | "
              f"z0 spread xy {r['z0_spread_xy_m']:.4f} (cell "
              f"{r['corpus_within_cell_xy_m']:.4f}) th {r['z0_spread_theta_rad']:.4f} "
              f"(cell {r['corpus_within_cell_theta_rad']:.4f})", flush=True)

    save("final")
    final = evaluate(n=args.val_chunks)
    (out / "summary.json").write_text(json.dumps(
        {"args": vars(args), "params": int(n_par), "final": final}, indent=2))
    print(json.dumps(final, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
