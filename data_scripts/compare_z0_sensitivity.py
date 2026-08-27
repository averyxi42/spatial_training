#!/usr/bin/env python3
"""How much behaviour does `z_0` choose, in the SHIPPED head versus the code-conditioned one?

The code-conditioned prototype reports a `z_0` spread of ~0.029 m. That number is
unanchored on its own -- small in absolute terms, but the question is small RELATIVE TO
WHAT. This measures the identical statistic on the shipped `h`-conditioned SFT head so
the two sit on one ladder, alongside two corpus references:

    unconditional corpus spread   what a chunk varies by with nothing held fixed
    within-cell corpus spread     what real chunks sharing a code vary by
    shipped head, fix h           what z_0 chooses today
    code head, fix (c)            what z_0 chooses under code conditioning

THE STATISTIC, identical in both: hold the conditioning fixed, draw M independent `z_0`,
decode, and take the per-tick standard deviation across draws averaged over ticks, in
physical units on the anchor-relative chunk. Also the terminal-pose spread, since the
per-tick mean is dominated by late ticks where magnitude is largest.

Fair-comparison notes. Both heads decode with the same Euler step count. `pin_flow_noise`
is NOT used -- pinning holds `z_0` at one arbitrary slice of the policy, which is the
opposite of what is being measured here. The shipped head needs the 2B backbone to
produce `h`, which is the only expensive part; the code head needs nothing.
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


def spread_stats(chunks_msc):
    """(N, M, T, 3) decodes -> the two spread numbers, in physical units."""
    xy = chunks_msc[..., :2].std(dim=1).mean().item()          # mean over ticks & dims
    th = chunks_msc[..., 2].std(dim=1).mean().item()
    term = chunks_msc[:, :, -1, :2].std(dim=1).norm(dim=-1).mean().item()
    return {"z0_spread_xy_m": xy, "z0_spread_theta_rad": th,
            "z0_spread_terminal_xy_m": term}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sft-ckpt", required=True, help="the shipped h-conditioned head")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--observations", type=int, default=24)
    ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--target-column", default="action_chunks")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mem-fraction", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.device.startswith("cuda") and args.mem_fraction > 0:
        torch.cuda.set_per_process_memory_fraction(args.mem_fraction)

    from datasets import load_from_disk
    from transformers import AutoProcessor

    from longnav.utils.flow_matching_head import TurnFlowActionRegressor
    from longnav.utils.vector_sft import DataConfig, TurnVectorCollator
    sys.path.insert(0, str(_ROOT / "data_scripts"))
    from latent_spread_probe import collect_h

    head_cfg = json.loads((Path(args.sft_ckpt) / "turn_vector_head_config.json").read_text())
    processor = AutoProcessor.from_pretrained(head_cfg["model"]["model_id"])
    model = TurnFlowActionRegressor.from_pretrained(
        args.sft_ckpt, processor, device=args.device)
    model.eval()
    codec = model.normalizer
    T, D = codec.decoder.n_ticks, codec.decoder.n_dims

    ds = load_from_disk(args.dataset)
    ds = ds[args.split] if hasattr(ds, "keys") and args.split in ds else ds
    rng = np.random.default_rng(args.seed)
    rows = rng.choice(len(ds), size=min(args.observations, len(ds)), replace=False)
    collator = TurnVectorCollator(
        processor=processor, data=DataConfig(target_column=args.target_column),
        train=False, modality_specs=tuple(model.model_cfg.modality_specs))

    decodes = []
    for n, r in enumerate(rows):
        h = collect_h(model, ds[int(r)], collator, args.device)
        if h is None:
            continue
        hk = h.expand(args.samples, h.shape[1]).contiguous()
        with torch.no_grad():
            # Fresh z_0 per draw -- the whole point. No pinning.
            ch = codec.denormalize(hk)
        decodes.append(ch.float().cpu())
        print(f"  obs {n+1}/{len(rows)}", flush=True)
    stacked = torch.stack(decodes)                              # (N, M, T, 3)
    res = {"n_obs": len(decodes), "n_samples": args.samples,
           "shipped_head_fix_h": spread_stats(stacked)}
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
