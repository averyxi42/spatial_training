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
    "--code-d-factor": dict(type=int, default=256,
                            help="per-factor embedding width in the code head"),
    "--code-r-hidden": dict(type=int, default=None,
                            help="hidden width of r(h); default context_dim"),
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
                                 code_r_hidden=ca.code_r_hidden)
        n_new = (sum(p.numel() for p in model.code_mixer.parameters())
                 + sum(p.numel() for p in model.code_head.parameters()))
        print(f"[code] attached: +{n_new/1e6:.2f}M params, r(h) zero-init "
              f"(scale {model.code_mixer.residual_scale():.1f}), "
              f"code_loss_weight {ca.code_loss_weight}", flush=True)
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
    base.TurnFlowActionRegressor = _BuildProxy(base.TurnFlowActionRegressor, code_args)
    base.FlowMatchingSFTTrainer = CodeSFTTrainer
    base.main()


if __name__ == "__main__":
    main()
