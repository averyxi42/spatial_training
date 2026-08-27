#!/usr/bin/env python3
"""SFT for the `(c, r(h))`-conditioned flow head. A WRAPPER, not a fork.

`docs/CODE_CONDITIONED_POLICY.md` sections 2 and 5. Everything about the training
recipe -- data, mixture, collator, optimiser, schedule, checkpointing, every existing
flag -- comes from `train_flow_matching_sft.py` unchanged. This module adds four flags,
swaps two constructors, and gets out of the way. There is no copy of the SFT script to
drift from the one that produced the shipped weights.

What is swapped:

    TurnFlowActionRegressor.build(...)  ->  the same call, then `attach_code_heads`,
                                            which promotes the instance and adds the
                                            code embeddings, r(h) and the code head
    FlowMatchingSFTTrainer              ->  CodeSFTTrainer (same trainer, plus the
                                            code head's metrics)

`r(h)`'s final layer is ZERO-INITIALISED, so at step 0 the decoder sees exactly what a
`c`-only head with context tokens 4-7 zeroed sees. The continuous branch grows in from a
no-op instead of arriving as noise into a working flow head.

RTC IS OFF here by construction (`--rtc-delay-max` defaults to 0 in the parent), because
the sequence is: a normal run first, then RTC conditioning introduced as a fine-tune at
~75% of compute -- which is how the shipped RTC checkpoint was made.

Usage mirrors the parent exactly, plus:

    --tokenizer PATH        frozen dual-FSQ tokenizer checkpoint (required)
    --code-loss-weight W    weight on the code head's CE, per turn (default 1.0)
    --code-d-factor D       per-factor embedding width in the code head (default 256)
    --code-r-hidden H       hidden width of r(h) (default: context_dim)
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "src"), str(_ROOT / "data_scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import train_flow_matching_sft as base                                     # noqa: E402

from longnav.utils.chunk_tokenizer import FrozenChunkTokenizer             # noqa: E402
from longnav.utils.code_conditioned_head import (                          # noqa: E402
    CodeSFTTrainer, attach_code_heads,
)

_CODE_FLAGS = {
    "--tokenizer": dict(required=True,
                        help="frozen dual-FSQ tokenizer checkpoint (tokenizer.pt)"),
    "--code-loss-weight": dict(type=float, default=1.0,
                               help="weight on the code head's per-turn CE"),
    "--code-prior": dict(default=None,
                         help="npy of shape (vocab,) holding the corpus log-marginal "
                              "over the joint code; initialises the head's per-code "
                              "bias. Worth ~2.5 nats at step 0 (H(c)=4.838 against "
                              "ln(1600)=7.378), because fitting a skewed 1600-way prior "
                              "by gradient descent is slow at any sane LR."),
    "--code-max-grad-norm": dict(type=float, default=1.0,
                                 help="the code head's OWN clip norm; it is clipped as a "
                                      "separate group so its gradient cannot consume the "
                                      "global clip budget and shrink everyone else's step"),
    "--code-head-kind": dict(default="mlp", choices=["mlp", "compositional"],
                             help="code policy head readout. 'mlp' (DEFAULT) is a plain "
                                  "two-layer MLP on h with a free 1600-way output layer. "
                                  "'compositional' builds the table from per-factor "
                                  "embeddings -- parameter sharing at the cost of a "
                                  "constrained table; see CodePolicyHead's docstring."),
    "--code-head-hidden": dict(type=int, default=None,
                               help="hidden width of the mlp head; default in_dim"),
    "--code-d-factor": dict(type=int, default=256,
                            help="per-factor embedding width in the code head"),
    "--code-r-hidden": dict(type=int, default=None,
                            help="hidden width of r(h); default context_dim"),
    "--code-init-from": dict(default=None,
                             help="WARM START from a c-only prototype head "
                                  "(code_flow_head.pt): loads the flow decoder and BOTH "
                                  "code embedding tables, leaving r(h) at its zero init. "
                                  "Step 0 then reproduces the prototype exactly, because "
                                  "the prototype was trained with context tokens 4-7 held "
                                  "at zero and r(h) outputs zero. Must be a DIFFERENTIAL-"
                                  "space prototype: the shipped codec.normalize is "
                                  "scale(decompose_chunk(.)), so a pose-space checkpoint "
                                  "predicts velocities in another representation."),
}


def _take_code_args():
    """Consume our flags from `sys.argv` and hand the rest to the parent untouched.

    Stripping argv rather than intercepting the parent's parser: it runs a two-stage
    `pre.parse_known_args()` -> base-parser handoff with `allow_abbrev=False` called out
    as load-bearing, and reaching into that is exactly the kind of edit that breaks
    quietly a month later.
    """
    import argparse
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    for flag, kw in _CODE_FLAGS.items():
        ap.add_argument(flag, **kw)
    ns, rest = ap.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    return ns


def _load_prior(path):
    if not path:
        return None
    import numpy as np, torch
    a = torch.from_numpy(np.load(path).astype("float32"))
    print(f"[code] prior-init bias from {path}: H(c) = "
          f"{float(-(a.exp() * a).sum()):.4f} nats (uniform ln(V) = "
          f"{float(torch.log(torch.tensor(float(a.numel())))):.4f})", flush=True)
    return a


def _patch_optimizer_groups():
    """Put `code_head`/`code_mixer` in the FRESH group.

    `build_optimizer_param_groups` builds `fresh` from the modules that exist at
    `build()` time; these two are attached afterwards, so they fell into `other` at
    `--lr` 1e-5 instead of `--head-lr` 1e-4 -- a ~10x deficit on the one module the
    function's own docstring describes ("exactly as randomly-initialized as model.head,
    wants the same larger step").
    """
    original = base.build_optimizer_param_groups

    def patched(model, args):
        groups = original(model, args)
        code = [q for n in ("code_head", "code_mixer")
                for q in (getattr(model, n, None).parameters()
                          if getattr(model, n, None) is not None else [])
                if q.requires_grad]
        if not code:
            return groups
        ids = {id(q) for q in code}
        head_lr = args.head_lr if args.head_lr is not None else args.lr
        for g in groups:
            g["params"] = [q for q in g["params"] if id(q) not in ids]
        groups = [g for g in groups if g["params"]]
        groups.append({"params": code, "lr": head_lr,
                       "weight_decay": args.weight_decay})
        print(f"[code] optimizer: {len(code)} code-head tensors moved to the fresh group "
              f"at lr {head_lr}", flush=True)
        return groups

    base.build_optimizer_param_groups = patched


class _BuildProxy:
    """Forwards every attribute to the real class and overrides only `build`.

    A bare stub would break `--init-from` and anything else reaching for the class; a
    mutation of the real class would leak to every other importer in the process.
    """

    def __init__(self, real, code_args):
        self._real, self._code = real, code_args

    def __getattr__(self, name):
        return getattr(self._real, name)

    def build(self, *a, **kw):
        model = self._real.build(*a, **kw)
        ca = self._code
        tok = FrozenChunkTokenizer(ca.tokenizer)
        print(f"[code] tokenizer {ca.tokenizer}: vocab {tok.vocab_xy} x "
              f"{tok.vocab_theta} = {tok.vocab_xy * tok.vocab_theta}, "
              f"{sum(p.numel() for p in tok.parameters())} frozen params", flush=True)
        model = attach_code_heads(model, tok, code_loss_weight=ca.code_loss_weight,
                                 code_d_factor=ca.code_d_factor,
                                 code_r_hidden=ca.code_r_hidden,
                                 code_head_kind=ca.code_head_kind,
                                 code_head_hidden=ca.code_head_hidden,
                                 code_log_prior=_load_prior(ca.code_prior))
        n_new = (sum(p.numel() for p in model.code_mixer.parameters())
                 + sum(p.numel() for p in model.code_head.parameters()))
        print(f"[code] attached: +{n_new/1e6:.2f}M params, r(h) zero-init "
              f"(scale {model.code_mixer.residual_scale():.1f}), "
              f"code_loss_weight {ca.code_loss_weight}, "
              f"head kind {ca.code_head_kind}", flush=True)
        if ca.code_init_from:
            import torch
            ck = torch.load(ca.code_init_from, map_location="cpu", weights_only=False)
            model.decoder.load_state_dict(ck["decoder"], strict=True)
            ctx = ck["ctx"]
            # The prototype names the second table `emb_th`; the mixer names it
            # `emb_theta`. One rename, asserted by shape rather than assumed.
            model.code_mixer.emb_xy.weight.data.copy_(ctx["emb_xy.weight"])
            model.code_mixer.emb_theta.weight.data.copy_(ctx["emb_th.weight"])
            assert model.code_mixer.residual_scale() == 0.0, "r(h) must stay zero"
            print(f"[code] WARM START from {ca.code_init_from}: decoder "
                  f"({len(ck['decoder'])} tensors) + both code tables loaded; "
                  f"r(h) still zero, so step 0 reproduces the prototype", flush=True)
        return model


def main():
    code_args = _take_code_args()
    if "--rtc-delay-max" in sys.argv:
        i = sys.argv.index("--rtc-delay-max")
        if i + 1 < len(sys.argv) and sys.argv[i + 1] not in ("0", "0.0"):
            raise SystemExit(
                "--rtc-delay-max must be 0 for this run: RTC is introduced as a "
                "fine-tune at ~75% of compute, after a normal run, which is how the "
                "shipped RTC checkpoint was made. Run the normal one first."
            )
    _patch_optimizer_groups()
    CodeSFTTrainer.code_max_grad_norm = float(code_args.code_max_grad_norm)
    base.TurnFlowActionRegressor = _BuildProxy(base.TurnFlowActionRegressor, code_args)
    base.FlowMatchingSFTTrainer = CodeSFTTrainer
    base.main()


if __name__ == "__main__":
    main()
