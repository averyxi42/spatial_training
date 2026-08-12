#!/usr/bin/env python3
"""Does sampling `c` change the BEHAVIOUR? The acceptance gate for the latent conversion.

`docs/LATENT_RL.md` makes the gate two-sided on purpose, because parity against the
deterministic model is passed *best* by the degenerate solution: `beta -> inf` drives
`sigma -> 0` and reproduces v3 exactly. Selecting a checkpoint on reconstruction alone
therefore selects the failure mode. The second, opposing requirement is that sampling `c`
actually moves the robot differently:

    spread(vary c, fixed flow seed)  >>  0
    spread(vary c, fixed flow seed)  >=  spread(fixed c, vary flow seed)

The second inequality is the load-bearing one. Behaviour still living in the flow noise is
behaviour that gets FROZEN OUT at RL time, when `z_0` is pinned to make `c -> chunk`
deterministic -- so a model whose variation is all in the noise has an empty action space
however good its loss looks.

MEASURED ON BEHAVIOUR, NEVER ON ||dc||. The decoder reshapes `c` to 8 tokens of 128 and is
`norm_first`, so per-128-block mean and scale are stripped from what the action tokens read
at layer 1 and re-enter only through a second-order residual path. Roughly 16 of the 1024
directions are strongly attenuated, and a norm-based proxy reads them as real spread.

THE MODE/DITHER INSTRUMENT, WHICH IS NOT A GATE. `h` is already action-shaped -- a linear
probe on it reaches R^2 0.63 on the whole chunk -- so perturbing `c` may be trajectory
dithering rather than a choice between behaviours. That distinction does not decide whether
to ship: dither is local action correction, which is the right tool for the state-distribution
failure RL is being pointed at. It is recorded because it interprets the RL result. The
statistic is cheap and reuses a failure mode this codebase already characterises:
`FlowActionCodec` refuses to average ODE samples because the average of a multimodal
conditional CREEPS. So average the decoded chunks over `c ~ p` and compare the average's
path length against the mean of the individual path lengths:

    creep_ratio = ||mean_k chunk_k|| / mean_k ||chunk_k||    (path length, metres)

    ~1.0  -> unimodal dither around one behaviour
    <<1.0 -> genuinely separated modes cancelling each other out

Usage:

    python data_scripts/latent_spread_probe.py \\
        --ckpt dump/pose_injection/run_latent2_full_beta0.003/checkpoint-2000 \\
        --dataset data/v2_25hz_obs2.5hz/formatted_nopose --split validation \\
        --observations 32 --samples 24
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from datasets import load_from_disk                                        # noqa: E402
from transformers import AutoProcessor                                     # noqa: E402

from longnav.utils.flow_matching_head import TurnFlowActionRegressor       # noqa: E402
from longnav.utils.vector_sft import DataConfig, TurnVectorCollator        # noqa: E402


def path_length(chunk: torch.Tensor) -> torch.Tensor:
    """(..., T, 3) cumulative anchor-relative poses -> planar path length in metres.

    Cumulative, so successive rows differ by one tick's displacement; the first row is the
    displacement from the anchor itself.
    """
    xy = chunk[..., :2]
    steps = torch.cat([xy[..., :1, :], xy[..., 1:, :] - xy[..., :-1, :]], dim=-2)
    return steps.norm(dim=-1).sum(dim=-1)


def terminal(chunk: torch.Tensor):
    """Where the chunk ends: planar displacement (m) and heading (rad)."""
    return chunk[..., -1, :2], chunk[..., -1, 2]


@torch.no_grad()
def probe_one(model, h: torch.Tensor, n_samples: int, seed: int):
    """`h` is ONE observation's readout, (1, dim). Returns the three comparisons.

    Both arms decode the same number of chunks; the only difference is which source of
    randomness is held fixed, which is what makes them comparable.
    """
    codec = model.normalizer
    latent = codec.latent
    dec, T, D = codec.decoder, codec.decoder.n_ticks, codec.decoder.n_dims
    dev = h.device
    hk = h.expand(n_samples, h.shape[1]).contiguous()

    # One base noise, reused for every draw: the RL-time actuator, deterministic given c.
    g = torch.Generator(device=dev).manual_seed(seed)
    fixed_noise = torch.randn(1, T, D, device=dev, dtype=torch.float32, generator=g)

    # ARM A -- vary c, freeze the flow noise. This is the RL policy's action space.
    gc = torch.Generator(device=dev).manual_seed(seed + 1)
    c = latent.draw(hk.float(), mode="sample", generator=gc)["c"]
    chunk_c = codec.denormalize_from_latent(c, noise=fixed_noise.expand(n_samples, T, D))

    # ARM B -- freeze c at the prior mean, vary the flow noise. This is what gets frozen out.
    c_mean = latent.draw(hk.float(), mode="mean")["c"]
    gn = torch.Generator(device=dev).manual_seed(seed + 2)
    noise = torch.randn(n_samples, T, D, device=dev, dtype=torch.float32, generator=gn)
    chunk_n = codec.denormalize_from_latent(c_mean, noise=noise)

    out = {}
    for name, ch in (("vary_c", chunk_c), ("vary_noise", chunk_n)):
        xy, th = terminal(ch)
        out[f"{name}_disp_std_m"] = float(xy.std(dim=0).norm())
        out[f"{name}_head_std_rad"] = float(th.std())
        out[f"{name}_pathlen_mean_m"] = float(path_length(ch).mean())
    # The instrument, on ARM A only: averaging a multimodal set cancels; a unimodal one does
    # not. Compare like with like -- both are path lengths in metres.
    out["creep_ratio"] = float(
        path_length(chunk_c.mean(dim=0)) / path_length(chunk_c).mean().clamp_min(1e-9)
    )
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--observations", type=int, default=32,
                    help="distinct observations to probe; spread is reported as the mean "
                         "over them, since a single observation says nothing")
    ap.add_argument("--samples", type=int, default=24, help="draws per observation, per arm")
    ap.add_argument("--target-column", default="action_chunks")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None, help="write the summary as JSON here too")
    args = ap.parse_args()

    processor = AutoProcessor.from_pretrained(
        json.loads((Path(args.ckpt) / "turn_vector_head_config.json").read_text()
                   )["model"]["model_id"])
    model = TurnFlowActionRegressor.from_pretrained(args.ckpt, processor, device=args.device)
    if getattr(model.normalizer, "latent", None) is None:
        raise SystemExit(
            f"{args.ckpt} has no latent split -- there is no `c` to vary, so this probe has "
            "nothing to measure. It is a deterministic checkpoint."
        )
    model.eval()

    ds = load_from_disk(args.dataset)
    ds = ds[args.split] if hasattr(ds, "keys") and args.split in ds else ds
    rng = np.random.default_rng(args.seed)
    rows = rng.choice(len(ds), size=min(args.observations, len(ds)), replace=False)

    collator = TurnVectorCollator(
        processor=processor, data=DataConfig(target_column=args.target_column),
        train=False, modality_specs=tuple(model.modality_specs),
    )
    per_obs = []
    for i, row in enumerate(rows):
        h = collect_h(model, ds[int(row)], collator, args.device)
        if h is None:
            continue
        per_obs.append(probe_one(model, h, args.samples, args.seed + 100 * i))

    if not per_obs:
        raise SystemExit("no observations yielded a readout; check --dataset/--split")

    keys = sorted(per_obs[0])
    summary = {k: float(np.mean([p[k] for p in per_obs])) for k in keys}
    summary["n_observations"] = len(per_obs)
    summary["n_samples"] = args.samples

    ratio = summary["vary_c_head_std_rad"] / max(summary["vary_noise_head_std_rad"], 1e-9)
    print(f"\n{'':32s}{'vary c':>14s}{'vary flow noise':>18s}")
    for label, a, b in (
        ("terminal displacement std (m)", "vary_c_disp_std_m", "vary_noise_disp_std_m"),
        ("terminal heading std (rad)", "vary_c_head_std_rad", "vary_noise_head_std_rad"),
        ("mean path length (m)", "vary_c_pathlen_mean_m", "vary_noise_pathlen_mean_m"),
    ):
        print(f"{label:32s}{summary[a]:14.5f}{summary[b]:18.5f}")
    print(f"\nheading-spread ratio (c / noise) = {ratio:.3f}"
          f"   -- GATE wants >= 1.0")
    print(f"creep ratio                      = {summary['creep_ratio']:.3f}"
          f"   -- instrument: ~1 dither, <<1 modes")
    print(f"observations={summary['n_observations']} samples/arm={summary['n_samples']}")

    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2))
        print(f"wrote {args.out}")


def collect_h(model, row, collator, device):
    """One dataset row -> its readout `h`, `(1, dim)`.

    Reuses `TurnVectorCollator` and the model's own `extract_turn_vectors` rather than
    reimplementing templating, windowing and span pooling. A second implementation here
    would drift from the one that produced the weights, and the drift would be silent --
    every number below would still be a plausible float.
    """
    from longnav.utils.turn_vectors import extract_turn_vectors

    batch = collator([row])
    skip = {"targets", "labels", "num_turns", "stop_targets"}
    inputs = {k: (v.to(device) if torch.is_tensor(v) else v)
              for k, v in batch.items() if k not in skip}
    modality = {k: inputs.pop(k) for k in list(inputs) if k.startswith("modality_")}
    from longnav.utils.modality_embed import ModalityBatch

    mb = ModalityBatch.pop_from({**modality}, known_keys=model.modality_embedder.keys)
    with torch.no_grad(), model.modality_embedder.pending(mb):
        outputs = model.backbone(use_cache=False, logits_to_keep=1, **inputs)
    h, _ = extract_turn_vectors(
        outputs, inputs["input_ids"], model.head,
        prefix_ids=model.prefix_ids, postfix_ids=model.postfix_ids,
        shift_left=model.model_cfg.shift_left, strict=True,
    )
    return h[:1].float()


if __name__ == "__main__":
    main()
