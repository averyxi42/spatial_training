#!/usr/bin/env python3
"""Is the tokenizer obedient to ITSELF? The reference the flow head's obedience needs.

`train_code_flow_head.py` reports P(g(A') = c) for chunks the flow head generates from
a code. That number is uninterpretable on its own: some of the miss rate belongs to the
flow head and some belongs to the codebook's own geometry, and they call for opposite
fixes.

The reference is the codebook's self-consistency: decode a code to its canonical
profile, re-encode that profile, and ask whether it comes back. `StreamTokenizer.decode`
consumes ONLY the quantised code -- no skip path, nothing else from the encoder -- so
there are exactly `vocab` possible reconstructions per stream and this is EXHAUSTIVE
rather than sampled. The data-weighted rate is the same test weighted by how often each
code occurs, which is what compares directly against the flow head.

Not a strict upper bound: the tokenizer's decoder is trained for reconstruction, not
for landing in-cell, so a different generator could in principle obey more often. It is
the reference that separates "the flow head misses" from "the cell does not contain its
own centroid".

When a code does NOT come back, the distance it lands at matters. FSQ indices are
ordinal, so a miss to an adjacent grid point is a boundary case; a miss several levels
away means the decoder's output for that code sits somewhere else entirely.
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

from train_chunk_tokenizer import DualTokenizer                            # noqa: E402


def codes_from_index(fsq, index):
    """Inverse of `FSQ.forward`'s index: (N,) int -> (N, d) normalised grid points.

    Mirrors the forward path exactly -- `idx_per_dim = round(bounded) + levels // 2`
    and `codes = quant / (levels // 2)` -- so an error here would show up as a
    round-trip failure and be misread as a codebook defect.
    """
    levels = torch.tensor(fsq.levels)
    basis = fsq._basis.cpu()
    half = levels // 2
    per_dim = (index[:, None] // basis[None, :]) % levels[None, :]
    return ((per_dim - half[None, :]).float() / half[None, :].float())


def per_dim_index(fsq, index):
    levels = torch.tensor(fsq.levels)
    basis = fsq._basis.cpu()
    return (index[:, None] // basis[None, :]) % levels[None, :]


def roundtrip(stream, counts, device):
    fsq = stream.fsq
    vocab = fsq.vocab
    idx = torch.arange(vocab)
    codes = codes_from_index(fsq, idx).to(device)
    with torch.no_grad():
        recon = stream.decode(codes)                    # (vocab, T, C)
        _, back = stream.encode(recon)
    back = back.cpu()
    same = (back == idx)

    a, b = per_dim_index(fsq, idx), per_dim_index(fsq, back)
    l1 = (a - b).abs().sum(dim=1)                       # grid steps between them

    w = np.asarray(counts, dtype=np.float64)
    w = w / max(w.sum(), 1.0)
    return {
        "vocab": int(vocab),
        "levels": list(fsq.levels),
        "codes_self_consistent": int(same.sum()),
        "unweighted_rate": float(same.float().mean()),
        "data_weighted_rate": float((same.numpy() * w).sum()),
        "misses_at_grid_l1_1": int(((~same) & (l1 == 1)).sum()),
        "misses_at_grid_l1_ge2": int(((~same) & (l1 >= 2)).sum()),
        "mean_l1_of_misses": float(l1[~same].float().mean()) if (~same).any() else 0.0,
        "worst_codes": [
            {"code": int(idx[i]), "lands_on": int(back[i]), "grid_l1": int(l1[i]),
             "corpus_share": float(w[i])}
            for i in torch.argsort(-l1)[:5] if not bool(same[i])
        ],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--eval-json", required=True,
                    help="eval_chunk_tokenizer.py output, for the corpus code counts")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ck = torch.load(args.tokenizer, map_location="cpu", weights_only=False)
    tok = DualTokenizer(ck["xy_levels"], ck["theta_levels"],
                        d_model=ck["d_model"], n_layers=ck["n_layers"])
    tok.load_state_dict(ck["model"]); tok.eval().to(args.device)

    ev = json.loads(Path(args.eval_json).read_text())
    res = {
        "xy": roundtrip(tok.xy, ev["xy_usage"]["counts"], args.device),
        "theta": roundtrip(tok.theta, ev["theta_usage"]["counts"], args.device),
    }
    res["joint_data_weighted_rate"] = (
        res["xy"]["data_weighted_rate"] * res["theta"]["data_weighted_rate"])
    print(json.dumps(res, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
