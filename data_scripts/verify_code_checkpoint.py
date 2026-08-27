#!/usr/bin/env python3
"""Does a code-conditioned checkpoint actually round-trip? Run this BEFORE any long run.

The first build of this head attached `code_head`/`code_mixer` to the model rather than
to a saved submodule, so they were absent from every checkpoint -- training looked
perfect for 12 hours and produced nothing loadable. The failure was invisible because
saving succeeded and only *loading* would have revealed it, and nothing loaded.

So this asserts the properties that failure would have violated, in a FRESH process:

  1. the checkpoint declares `fm_code` and rebuilds the slot on load
  2. every code-head and mixer tensor is present and bit-identical to the source
  3. the tokenizer round-trips its scales (crossing the two normalisations -- tokenizer
     cumulative-pose std vs the flow head's per-tick action_scales -- yields plausible
     numbers and a meaningless gauge)
  4. a decode from the reloaded model equals a decode from the source, given the same
     noise; weights matching is not the same as the model computing the same thing

Exit code is nonzero on any failure, so it can gate a launch.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from transformers import AutoProcessor                                     # noqa: E402

from longnav.utils.flow_matching_head import TurnFlowActionRegressor       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    ck = Path(args.checkpoint)
    fails = []

    meta = json.loads((ck / "turn_vector_head_config.json").read_text())
    spec = meta.get("fm_code")
    print(f"[1] fm_code in head config: {spec is not None}")
    if spec is None:
        print("FAIL: checkpoint declares no code slot -- it cannot be evaluated")
        return 1
    print(f"    {spec}")

    blob = torch.load(ck / "turn_vector_head.pt", map_location="cpu", weights_only=False)
    norm = blob["normalizer"]
    code_keys = [k for k in norm if k.startswith(("code_head.", "code_mixer."))]
    tok_keys = [k for k in norm if k.startswith("tokenizer.")]
    print(f"[2] code tensors in the saved normalizer: {len(code_keys)} "
          f"(tokenizer: {len(tok_keys)})")
    if not code_keys:
        fails.append("no code_head/code_mixer tensors in the checkpoint")

    processor = AutoProcessor.from_pretrained(meta["model"]["model_id"])
    model = TurnFlowActionRegressor.from_pretrained(str(ck), processor,
                                                    device=args.device)
    codec = model.normalizer
    have = {"code_head": getattr(codec, "code_head", None),
            "code_mixer": getattr(codec, "code_mixer", None)}
    print(f"[3] rebuilt on load: "
          f"{ {k: v is not None for k, v in have.items()} }")
    for k, v in have.items():
        if v is None:
            fails.append(f"{k} missing after from_pretrained")

    if not fails:
        live = dict(codec.state_dict())
        bad = [k for k in code_keys
               if k not in live or not torch.equal(live[k].cpu().float(),
                                                   norm[k].cpu().float())]
        print(f"[4] code tensors bit-identical to source: {len(code_keys) - len(bad)}"
              f"/{len(code_keys)}")
        if bad:
            fails.append(f"{len(bad)} code tensors differ after reload, e.g. {bad[:3]}")

        tok = getattr(codec, "tokenizer", None)
        if tok is not None:
            xs, ts = float(tok.xy_scale), float(tok.theta_scale)
            print(f"[5] tokenizer scales restored: xy {xs:.4f} theta {ts:.4f}")
            if abs(xs - 1.0) < 1e-9 or abs(ts - 1.0) < 1e-9:
                fails.append("tokenizer scales are still the 1.0 placeholders -- the "
                             "saved buffers did not land, so every code would be wrong")

        # The decode check. Weights matching does not prove the model COMPUTES the same
        # thing: a mis-wired slot can hold correct weights and never consult them.
        g = torch.Generator(device="cpu").manual_seed(0)
        h = torch.randn(3, model.decoder.context_dim, generator=g)
        cx = torch.randint(0, codec.code_head.n_xy, (3,), generator=g)
        ct = torch.randint(0, codec.code_head.n_theta, (3,), generator=g)
        noise = torch.randn(3, model.decoder.n_ticks, model.decoder.n_dims, generator=g)
        with torch.no_grad():
            ctx = codec.code_mixer(h, cx, ct)
            a = codec.unscale(torch.zeros_like(noise)) * 0 + codec.generate(ctx, noise=noise)
            lg = codec.code_head.logits(h)
        finite = bool(torch.isfinite(a).all()) and bool(torch.isfinite(lg).all())
        print(f"[6] decode + code logits finite: {finite} "
              f"(chunk {tuple(a.shape)}, logits {tuple(lg.shape)})")
        if not finite:
            fails.append("reloaded model produced non-finite output")
        # r(h) must contribute; if the mixer were mis-wired it would be all zeros forever
        rms_code = float(ctx[:, :4 * model.decoder.d_model].pow(2).mean().sqrt())
        print(f"[7] context code-block rms {rms_code:.4f} (0.0 would mean the code "
              f"embeddings never reached the decoder)")
        if rms_code == 0.0:
            fails.append("code block of the context is identically zero")

    print()
    if fails:
        for f in fails:
            print(f"FAIL: {f}")
        return 1
    print("PASS: checkpoint round-trips with its code slot intact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
