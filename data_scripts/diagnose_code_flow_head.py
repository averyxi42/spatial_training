#!/usr/bin/env python3
"""Why does the code-conditioned flow head disobey, when the codebook is 96.5% obedient?

Capacity is not a candidate: 0.87M parameters against a 1600-way discrete conditioning
could memorise a canonical chunk per code, which is roughly what the tokenizer's own
decoder does. And a perfect model of `p(A | c)` would obey EXACTLY -- every sample from
`p(A|c)` lies in cell `c` by the definition `c = g(A)` -- so there is no geometric floor
and the whole shortfall is modelling error.

That leaves three contributors, which this separates because they call for different
fixes and would otherwise be confused for one another:

1. **Coverage of the velocity field.** The head must specify `v(x, t, c)` over the whole
   path from `N(0,I)` into the cell, not just the cell's centre. Per-code *content*
   needs per-code data, while the shared trunk supplies the *form*. If this dominates,
   obedience should rise with per-code corpus frequency -- rare codes integrate through
   under-sampled regions and drift into a neighbour. Measured as a rank correlation.

2. **Euler discretisation.** Integration error smooths the sampled distribution exactly
   as an under-specified field does. Swept at EVAL ONLY, no retraining: if obedience
   climbs with step count, that part was never a training problem.

3. **Training-time smoothing** (dropout) -- not measured here; it needs a retrain.

Also reports obedience on the scan-enriched theta codes separately. Those are rare, and
if (1) dominates they should be the worst served -- which would be the fourth
independent strike against the scan region of this codebook.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "src"), str(_ROOT / "data_scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from longnav.utils.flow_matching_head import (                             # noqa: E402
    ACTION_SCALES, FlowActionCodec, FlowActionDecoder,
)
from train_chunk_tokenizer import DualTokenizer, load_chunks, to_cumulative  # noqa: E402
from train_code_flow_head import CodeContext, encode_corpus                # noqa: E402


def spearman(a, b):
    """Rank correlation without scipy: Pearson on ranks."""
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--head", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--row-stride", type=int, default=1)
    ap.add_argument("--n-ticks", type=int, default=20)
    ap.add_argument("--n-eval", type=int, default=65536)
    ap.add_argument("--steps", type=int, nargs="+", default=[10, 20, 50])
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mem-fraction", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dev = args.device
    if dev.startswith("cuda") and args.mem_fraction > 0:
        torch.cuda.set_per_process_memory_fraction(args.mem_fraction)

    ck = torch.load(args.tokenizer, map_location="cpu", weights_only=False)
    tok = DualTokenizer(ck["xy_levels"], ck["theta_levels"],
                        d_model=ck["d_model"], n_layers=ck["n_layers"])
    tok.load_state_dict(ck["model"]); tok.eval().to(dev)
    n_xy, n_th = tok.xy.fsq.vocab, tok.theta.fsq.vocab

    hd = torch.load(args.head, map_location="cpu", weights_only=False)
    ctx = CodeContext(n_xy, n_th).to(dev)
    ctx.load_state_dict(hd["ctx"])
    # `to_config()` carries the network shape but NOT n_ticks, which lives on the
    # instance; passing it separately rather than assuming the dict is complete.
    decoder = FlowActionDecoder(context_dim=ctx.context_dim, n_ticks=args.n_ticks,
                                **hd["decoder_config"])
    decoder.load_state_dict(hd["decoder"])
    decoder.eval().to(dev)
    codec = FlowActionCodec(decoder, action_scales=ACTION_SCALES).to(dev)
    ctx.eval()

    raw = load_chunks(args.dataset, args.split, args.row_stride, 0)
    chunks, _, _, _ = to_cumulative(raw)
    c_xy, c_th = encode_corpus(tok, chunks, ck["xy_scale"], ck["theta_scale"], dev)
    xy_freq = np.bincount(c_xy.astype(np.int64), minlength=n_xy)
    th_freq = np.bincount(c_th.astype(np.int64), minlength=n_th)

    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(chunks), size=min(args.n_eval, len(chunks)), replace=False)
    cx_all = torch.from_numpy(c_xy[idx].astype(np.int64))
    ct_all = torch.from_numpy(c_th[idx].astype(np.int64))

    @torch.no_grad()
    def obey_at(steps):
        # `denormalize` takes no step count -- it reads `self.num_inference_steps` -- so
        # the sweep sets the attribute rather than passing an argument the API lacks.
        codec.num_inference_steps = int(steps)
        ok_x, ok_t = [], []
        for i in range(0, len(idx), args.batch_size):
            cx = cx_all[i:i + args.batch_size].to(dev)
            ct = ct_all[i:i + args.batch_size].to(dev)
            gen = codec.denormalize(ctx(cx, ct)).float()
            gen[..., :2] /= ck["xy_scale"]; gen[..., 2] /= ck["theta_scale"]
            _, gx = tok.xy.encode(gen[..., :2])
            _, gt = tok.theta.encode(gen[..., 2:3])
            ok_x.append((gx == cx).cpu().numpy())
            ok_t.append((gt == ct).cpu().numpy())
        return np.concatenate(ok_x), np.concatenate(ok_t)

    res = {"n_eval": int(len(idx)), "euler_sweep": {}}
    per_code = None
    for s in args.steps:
        ox, ot = obey_at(s)
        res["euler_sweep"][str(s)] = {
            "obey_xy": float(ox.mean()), "obey_theta": float(ot.mean()),
            "obey_both": float((ox & ot).mean()),
        }
        print(f"steps {s:3d}: xy {ox.mean():.4f} theta {ot.mean():.4f} "
              f"both {(ox & ot).mean():.4f}", flush=True)
        if per_code is None:
            per_code = (ox, ot)

    # ---- coverage test: does obedience track per-code frequency?
    ox, ot = per_code
    cxn, ctn = cx_all.numpy(), ct_all.numpy()
    out = {}
    for name, codes, ok, freq, n in (("xy", cxn, ox, xy_freq, n_xy),
                                     ("theta", ctn, ot, th_freq, n_th)):
        rate, cnt = np.full(n, np.nan), np.zeros(n)
        for c in range(n):
            m = codes == c
            cnt[c] = m.sum()
            if m.sum() >= 30:
                rate[c] = ok[m].mean()
        keep = ~np.isnan(rate)
        out[name] = {
            "codes_scored": int(keep.sum()),
            "spearman_obey_vs_corpus_freq": spearman(freq[keep], rate[keep]),
            "obey_in_freq_quartiles": [
                float(np.nanmean(rate[keep][q])) for q in
                np.array_split(np.argsort(freq[keep]), 4)
            ],
            "worst_codes": [
                {"code": int(c), "obey": float(rate[c]), "corpus_freq": int(freq[c])}
                for c in np.argsort(np.where(keep, rate, np.inf))[:5]
            ],
        }
    res["coverage"] = out

    # ---- the scan codes, from eval_chunk_tokenizer's enrichment ranking
    scan_codes = [29, 21, 3, 4, 30]
    m = np.isin(ctn, scan_codes)
    res["scan_codes"] = {
        "codes": scan_codes,
        "n": int(m.sum()),
        "obey_theta_on_scan_codes": float(ot[m].mean()) if m.any() else float("nan"),
        "obey_theta_elsewhere": float(ot[~m].mean()),
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "euler_sweep"}, indent=2))


if __name__ == "__main__":
    main()
