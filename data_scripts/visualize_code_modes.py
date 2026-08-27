#!/usr/bin/env python3
"""Do the discrete codes correspond to visibly distinct behaviours? Look at them.

Replays an offline episode and, at every decision, decodes K codes x M noise draws and
draws all K*M chunks in a body-frame BEV panel beside the observation, coloured by code,
with a HUD listing each code and its probability. Writes an mp4.

BACKBONE-FREE by default. The c-only prototype head maps `c -> chunk` with no `h` at
all, so the whole rendering path can be exercised before any code-conditioned checkpoint
exists -- and it is exercised on real observations and real recorded chunks, not
synthetic ones. `--head` swaps in a full checkpoint once one is available, at which point
the codes come from `p(c|h)` instead of the prior.

WHY THE NOISE IS SHARED ACROSS CODES. Trajectory (k, m) and (k', m) differ ONLY by the
code, so any visible grouping is a code effect rather than the luck of the draw. Drawing
independent noise per (k, m) would confound exactly the thing the picture is for -- the
same argument `latent_spread_probe` encodes in its two arms, and the SFT diversity term
in decoding both `c` draws under one base noise.

READ IT AS A TWO-WAY DECOMPOSITION, not just a picture: spread ACROSS colours is how far
apart the modes are, spread WITHIN a colour is the `z_0` residual inside one cell, and
their ratio says whether the codes separate modes or slice one blob.
"""
import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "src"), str(_ROOT / "data_scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from longnav.utils.chunk_tokenizer import FrozenChunkTokenizer                # noqa: E402
from longnav.utils.flow_matching_head import (                                # noqa: E402
    ACTION_SCALES, FlowActionCodec, FlowActionDecoder, compose_chunk,
)

def code_color(c_xy, c_theta, fsq_xy, fsq_theta):
    """Colour a code by the CODEBOOK'S GEOMETRY, not by its rank in the panel.

    FSQ indices are ordinal -- that is the property FSQ was chosen for -- so mapping them
    onto a colour wheel makes adjacent codes look adjacent, and a code keeps its colour
    across frames and across episodes. Rank-based colouring has neither property: the
    same colour means a different behaviour in every panel.

      hue        <- theta's dominant FSQ coordinate  (which way it turns)
      value      <- xy's dominant FSQ coordinate     (how far it goes)
      saturation <- theta's second coordinate        (turn shape)
    """
    import colorsys

    tp = fsq_theta.per_dim_index(torch.tensor([c_theta]))[0].tolist()
    xp = fsq_xy.per_dim_index(torch.tensor([c_xy]))[0].tolist()
    n_t0 = fsq_theta.levels[0]
    hue = (tp[0] / max(n_t0 - 1, 1)) * 0.82                 # 0 -> red, .82 -> violet
    sat = 0.55 + 0.45 * (tp[1] / max(fsq_theta.levels[1] - 1, 1)) if len(tp) > 1 else 0.9
    val = 0.55 + 0.45 * (xp[0] / max(fsq_xy.levels[0] - 1, 1))
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))


def load_prototype(path, device):
    """The c-only head: `(c_xy, c_theta) -> chunk`, no backbone anywhere."""
    from train_code_flow_head import CodeContext

    hd = torch.load(path, map_location="cpu", weights_only=False)
    ctx = CodeContext(40, 40)
    ctx.load_state_dict(hd["ctx"])
    dec = FlowActionDecoder(context_dim=ctx.context_dim, n_ticks=20,
                            **hd["decoder_config"])
    dec.load_state_dict(hd["decoder"])
    codec = FlowActionCodec(dec, action_scales=ACTION_SCALES)
    return ctx.to(device).eval(), codec.to(device).eval()


@torch.no_grad()
def decode_grid(ctx_mod, codec, codes, n_noise, device, seed=0):
    """`codes` (K, 2) -> chunks (K, M, T, 3), noise SHARED across the K axis."""
    K, T, D = len(codes), codec.decoder.n_ticks, codec.decoder.n_dims
    g = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(n_noise, T, D, generator=g).to(device)          # (M, T, D)
    cx = torch.as_tensor(codes[:, 0], device=device).repeat_interleave(n_noise)
    ct = torch.as_tensor(codes[:, 1], device=device).repeat_interleave(n_noise)
    # repeat_interleave on codes + tile on noise: code varies slowly, noise fast, so
    # row (k*M + m) is (code k, noise m). `repeat` here would transpose the pairing and
    # silently colour the wrong trajectories.
    z = noise.repeat(K, 1, 1)
    chunks = compose_chunk(codec.generate(ctx_mod(cx, ct), noise=z))
    return chunks.float().cpu().numpy().reshape(K, n_noise, T, 3)


GT_COLOR = "#f8fafc"          # the ground-truth code's samples
GT_PATH_COLOR = "#facc15"      # the recorded chunk itself


def render_frame(img, chunks, codes, probs, true_code, span_m, colors, size=480,
                 gt_chunks=None, gt_path=None, gt_prob=None, gt_in_topk=False):
    """Observation on the left, body-frame BEV of the K x M chunks on the right."""
    from PIL import Image, ImageDraw

    K, M = chunks.shape[:2]
    panel = Image.new("RGB", (size, size), "#0b0f19")
    d = ImageDraw.Draw(panel)
    cx, cy, scale = size * 0.5, size * 0.82, (size * 0.72) / max(span_m, 1e-6)

    for r in (0.5, 1.0, 1.5, 2.0):                      # range rings, 0.5 m apart
        rp = r * scale
        if rp < size:
            d.ellipse([cx - rp, cy - rp, cx + rp, cy + rp], outline="#1f2937")
    d.line([cx, cy - size, cx, cy], fill="#1f2937")

    def to_px(xy):
        # body frame: +x forward (up on screen), +y left (left on screen)
        return cx - xy[1] * scale, cy - xy[0] * scale

    for k in range(K):
        col = colors[k]
        for m in range(M):
            pts = [to_px(p) for p in np.vstack([[0.0, 0.0], chunks[k, m, :, :2]])]
            d.line(pts, fill=col, width=2 if M == 1 else 1)
        end = to_px(chunks[k].mean(axis=0)[-1, :2])
        d.ellipse([end[0] - 3, end[1] - 3, end[0] + 3, end[1] + 3], fill=col)

    # The GROUND-TRUTH code's own samples, then the recorded chunk on top of them. The
    # true code is rarely in the prior's top-K, so without this the one mode we KNOW the
    # episode took would simply be absent from the picture.
    if gt_chunks is not None:
        for m in range(gt_chunks.shape[0]):
            pts = [to_px(q) for q in np.vstack([[0.0, 0.0], gt_chunks[m, :, :2]])]
            d.line(pts, fill=GT_COLOR, width=1)
    if gt_path is not None:
        pts = [to_px(q) for q in np.vstack([[0.0, 0.0], gt_path[:, :2]])]
        d.line(pts, fill=GT_PATH_COLOR, width=3)

    d.text((8, 6), f"top {K} codes x {M} noise draws", fill="#e5e7eb")
    for k, (code, p) in enumerate(zip(codes, probs)):
        mark = "*" if true_code is not None and tuple(code) == tuple(true_code) else " "
        d.text((8, 24 + 15 * k),
               f"{mark} xy{int(code[0]):>3} th{int(code[1]):>3}   p={p:.3f}",
               fill=colors[k])
    y = 24 + 15 * K
    if true_code is not None:
        tag = " (also in top-K)" if gt_in_topk else ""
        pstr = f"p={gt_prob:.4f}" if gt_prob is not None else ""
        d.text((8, y + 4), f"GT xy{int(true_code[0]):>3} th{int(true_code[1]):>3}   "
                           f"{pstr}{tag}", fill=GT_COLOR)
        d.text((8, y + 19), "recorded chunk A*", fill=GT_PATH_COLOR)

    # The dataset stores images as PATHS, not decoded arrays; accept either so the
    # renderer works against a formatted corpus and an in-memory frame alike.
    if isinstance(img, str):
        img = Image.open(img)
    elif isinstance(img, dict):
        img = Image.open(img.get("path") or io.BytesIO(img["bytes"]))
    elif isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    left = img.convert("RGB").resize((size, size))
    out = Image.new("RGB", (size * 2, size))
    out.paste(left, (0, 0)); out.paste(panel, (size, 0))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prototype", required=True, help="c-only head (code_flow_head.pt)")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--prior", required=True, help="code_log_prior.npy")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--episodes", type=int, nargs="+", default=[0])
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--code-source", choices=["sample", "topk"], default="sample",
                    help="'sample' draws K DISTINCT codes from the prior afresh each "
                         "step; 'topk' takes the K most probable, which are fixed for "
                         "the whole episode and -- measured -- all decode to nearly the "
                         "same forward line, because the prior's mass sits on driving "
                         "straight. Sampling is what makes the panel show modes.")
    ap.add_argument("--code-temperature", type=float, default=1.6,
                    help="flattens the prior before sampling (p ** 1/T). The raw prior "
                         "is forward-dominated, so T>1 buys turn modes without inventing "
                         "codes the corpus never used.")
    ap.add_argument("--n-noise", type=int, default=4)
    ap.add_argument("--max-turns", type=int, default=60)
    ap.add_argument("--fps", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    import cv2                      # the env has cv2, not imageio
    from datasets import load_from_disk

    dev = args.device
    ctx_mod, codec = load_prototype(args.prototype, dev)
    tok = FrozenChunkTokenizer(args.tokenizer).to(dev)
    logp = np.load(args.prior)
    prior = np.exp(logp)
    # Tempered sampling weights. Only occupied cells are eligible -- an unused code has
    # no rendering worth looking at and would just add a line nobody trained.
    w = np.where(prior > 1e-6, prior ** (1.0 / max(args.code_temperature, 1e-6)), 0.0)
    w = w / w.sum()
    fixed = np.argsort(-logp)[: args.top_k] if args.code_source == "topk" else None
    # Codes are chosen PER STEP now, so there is no episode-wide list to print here.
    src = ("fixed top-K by prior" if fixed is not None
           else f"sampled per step from the prior, T={args.code_temperature}")
    print(f"code source: {src}; {args.top_k} codes x {args.n_noise} draws per step")

    ds = load_from_disk(args.dataset)
    ds = ds[args.split] if hasattr(ds, "keys") and args.split in ds else ds
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    for ep in args.episodes:
        row = ds[int(ep)]
        imgs, acts = row["images"], np.asarray(row["action_chunks"], dtype=np.float32)
        n = min(len(imgs), len(acts), args.max_turns)
        span = float(np.abs(acts[..., :2]).max()) * 1.15
        true_cx, true_ct = tok.encode_chunk(torch.from_numpy(acts[:n]).to(dev))
        frames = []
        for t in range(n):
            if fixed is not None:
                order = fixed
            else:
                rng = np.random.default_rng(1000 * ep + t)
                order = rng.choice(len(w), size=args.top_k, replace=False, p=w)
                order = order[np.argsort(-logp[order])]      # most probable first
            codes = np.stack([order // tok.vocab_theta,
                              order % tok.vocab_theta], axis=1)
            probs = np.exp(logp[order])
            colors = [code_color(int(a), int(b), tok.model.xy.fsq, tok.model.theta.fsq)
                      for a, b in codes]
            grid = decode_grid(ctx_mod, codec, codes, args.n_noise, dev, seed=t)
            gt = np.array([[int(true_cx[t]), int(true_ct[t])]])
            # SAME seed as the top-K grid, so the GT samples share the noise draws and
            # differ from every coloured mode only by the code.
            gt_grid = decode_grid(ctx_mod, codec, gt, args.n_noise, dev, seed=t)[0]
            gt_flat = int(gt[0, 0]) * tok.vocab_theta + int(gt[0, 1])
            frames.append(np.asarray(render_frame(
                imgs[t], grid, codes, probs, tuple(gt[0]), span, colors,
                gt_chunks=gt_grid, gt_path=acts[t],
                gt_prob=float(np.exp(logp[gt_flat])),
                gt_in_topk=bool((codes == gt[0]).all(axis=1).any()))))
        path = out_dir / f"episode_{ep:04d}_modes.mp4"
        # PNG frames -> ffmpeg libx264 -pix_fmt yuv420p. cv2's mp4v writer produces a
        # file that decodes correctly with cv2 but renders ALL GREEN in ordinary players:
        # it is a pixel-format problem, not a pixel problem, and yuv420p is the format
        # every player handles. This cv2 build cannot open avc1/H264 at all, so the
        # ffmpeg binary is the route.
        import subprocess, tempfile
        with tempfile.TemporaryDirectory() as td:
            for i, f in enumerate(frames):
                cv2.imwrite(f"{td}/f_{i:05d}.png", cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps),
                 "-i", f"{td}/f_%05d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                 "-crf", "18", str(path)], check=True)
        print(f"wrote {path}  ({n} turns, {args.top_k}x{args.n_noise} chunks/turn)")


if __name__ == "__main__":
    main()
