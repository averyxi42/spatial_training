#!/usr/bin/env python3
"""Is the low obedience a real property of the flow head, or a bug in the path around it?

The flow head scores 0.672 obedience while the tokenizer's own decoder scores 0.965, and
the flow head is the more expressive model. Three checks, cheapest and most decisive
first.

1. REPRESENTATION ROUND TRIP -- the control that settles it. The flow head's output path
   is `compose_chunk(unscale(v))`, and it is TRAINED on `scale(decompose_chunk(A))`. Push
   real chunks through that exact round trip with the TRUE differentials substituted for
   generated ones, then encode. Obedience must come back ~1.0. Anything less is a defect
   in the representation path that has nothing to do with the flow head, and it would put
   a ceiling on obedience no amount of training could lift.

2. ANGLE WRAPPING -- `decompose_chunk` wraps dtheta and `compose_chunk` wraps the running
   theta into [-pi, pi]. If the corpus stores cumulative headings beyond pi, the composed
   chunk folds and the encoder sees a different object than the tokenizer was fit on.

3. MISS DISTANCE -- when the head disobeys, how far does it land? FSQ indices are ordinal,
   so a miss to an adjacent grid point means the sample fell just across a decision
   boundary, which for RL is a small behavioural error rather than a wrong mode. A metric
   that counts those as total failures understates how well the code steers.
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

from longnav.utils.bin_codec import compose_chunk, decompose_chunk         # noqa: E402
from longnav.utils.flow_matching_head import (                             # noqa: E402
    ACTION_SCALES, FlowActionCodec, FlowActionDecoder,
)
from train_chunk_tokenizer import DualTokenizer, load_chunks, to_cumulative  # noqa: E402
from train_code_flow_head import CodeContext, encode_corpus                # noqa: E402
from tokenizer_roundtrip import per_dim_index                              # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--head", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--row-stride", type=int, default=8)
    ap.add_argument("--n-eval", type=int, default=32768)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--n-ticks", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mem-fraction", type=float, default=0.10)
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

    raw = load_chunks(args.dataset, args.dataset and "train", args.row_stride, 0)
    chunks, _, _, _ = to_cumulative(raw)
    chunks = chunks[:args.n_eval]
    res = {"n": int(len(chunks))}

    # ---- 2. wrapping
    th = chunks[..., 2]
    res["theta_abs_max_rad"] = float(np.abs(th).max())
    res["frac_chunks_theta_beyond_pi"] = float((np.abs(th).max(axis=1) > np.pi).mean())

    def enc(arr):
        """physical anchor-relative poses -> (c_xy, c_theta)."""
        outx, outt = [], []
        with torch.no_grad():
            for i in range(0, len(arr), args.batch_size):
                b = arr[i:i + args.batch_size].clone()
                b[..., :2] /= ck["xy_scale"]; b[..., 2] /= ck["theta_scale"]
                _, x = tok.xy.encode(b[..., :2])
                _, t = tok.theta.encode(b[..., 2:3])
                outx.append(x.cpu()); outt.append(t.cpu())
        return torch.cat(outx), torch.cat(outt)

    A = torch.from_numpy(chunks).to(dev)
    cx0, ct0 = enc(A)

    # ---- 1. THE CONTROL: the flow head's exact output path, true differentials
    codec_scales = torch.tensor(ACTION_SCALES, dtype=torch.float64, device=dev)
    d_true = decompose_chunk(A.double())                    # what the head is TRAINED on
    scaled = d_true / codec_scales                          # codec.scale
    A_rt = compose_chunk((scaled * codec_scales)).float()   # codec.unscale -> compose
    cx1, ct1 = enc(A_rt)
    res["roundtrip"] = {
        "max_abs_pose_err_m": float((A_rt - A).abs().max()),
        "obey_xy": float((cx1 == cx0).float().mean()),
        "obey_theta": float((ct1 == ct0).float().mean()),
        "obey_both": float(((cx1 == cx0) & (ct1 == ct0)).float().mean()),
    }

    # ---- 3. miss distance for the actual head
    hd = torch.load(args.head, map_location="cpu", weights_only=False)
    ctx = CodeContext(n_xy, n_th).to(dev); ctx.load_state_dict(hd["ctx"]); ctx.eval()
    decoder = FlowActionDecoder(context_dim=ctx.context_dim, n_ticks=args.n_ticks,
                                **hd["decoder_config"])
    decoder.load_state_dict(hd["decoder"]); decoder.eval().to(dev)
    codec = FlowActionCodec(decoder, action_scales=ACTION_SCALES).to(dev)

    gen = []
    with torch.no_grad():
        for i in range(0, len(chunks), args.batch_size):
            gen.append(codec.denormalize(
                ctx(cx0[i:i + args.batch_size].to(dev),
                    ct0[i:i + args.batch_size].to(dev))).float())
    gen = torch.cat(gen)
    gx, gt = enc(gen)

    out = {}
    for name, got, want, fsq in (("xy", gx, cx0, tok.xy.fsq),
                                 ("theta", gt, ct0, tok.theta.fsq)):
        l1 = (per_dim_index(fsq, got) - per_dim_index(fsq, want)).abs().sum(dim=1)
        miss = l1 > 0
        out[name] = {
            "obey": float((~miss).float().mean()),
            "miss_l1_1_share_of_all": float((l1 == 1).float().mean()),
            "miss_l1_ge2_share_of_all": float((l1 >= 2).float().mean()),
            "within_1_grid_step": float((l1 <= 1).float().mean()),
            "mean_l1_given_miss": float(l1[miss].float().mean()) if miss.any() else 0.0,
        }
    res["miss_distance"] = out
    res["within_1_step_both"] = float((
        ((per_dim_index(tok.xy.fsq, gx) - per_dim_index(tok.xy.fsq, cx0)).abs().sum(1) <= 1)
        & ((per_dim_index(tok.theta.fsq, gt)
            - per_dim_index(tok.theta.fsq, ct0)).abs().sum(1) <= 1)).float().mean())

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
