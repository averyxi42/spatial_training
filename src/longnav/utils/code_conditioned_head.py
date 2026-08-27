"""SFT for the `(c, r(h))`-conditioned flow head, plus the discrete code policy head.

`docs/CODE_CONDITIONED_POLICY.md` sections 2 and 5. Three pieces, all extensions -- the
shipped `TurnFlowActionRegressor` is untouched apart from two default-inert seams
(`_decoder_context`, `_aux_turn_losses`), so a model that does not use this module runs
bit-identically to before it existed.

    CodeContextMixer   (h, c_xy, c_theta) -> the decoder's 8 context tokens
    CodePolicyHead     h -> a categorical over the joint code product
    TurnFlowActionRegressorWithCode   wires both into the shipped forward

CONTEXT TOKEN LAYOUT. The decoder has NO context projection -- it raises unless
`context_dim == n_context_tokens * d_model` and then only reshapes -- so the split is a
hard layout, not a hyperparameter:

    token 0-1  <- c_xy    embedding
    token 2-3  <- c_theta embedding
    token 4-7  <- r(h),   ZERO AT INITIALISATION

WHY r(h) IS ZERO-INITIALISED. The code-conditioned prototype was trained with tokens 4-7
held at exactly zero. Zeroing the final layer of `r` makes this model's step 0 bit-
identical to that checkpoint, so a warm start is seamless and `r(h)` cannot destroy a
working flow head before it has learned anything -- it grows in from a no-op, the
ControlNet zero-convolution argument. A zero-initialised `nn.Linear` still receives
gradient (`dL/dW = dL/dy . x^T`), so it is a soft start, not a dead branch.

WHY THE CODE HEAD IS ONE FLAT CATEGORICAL. `c_xy` and `c_theta` are strongly correlated,
so a factorised head would place mass on combinations that never co-occur. A flat
softmax over the product models the joint exactly and keeps RL single-step. Its output
table is built COMPOSITIONALLY from per-factor embeddings through an MLP, which keeps
parameter sharing across combinations that share a factor -- and the MLP is essential
rather than cosmetic, because ADDITIVE logits `h.(W_xy e_i + W_th e_j)` would factor as
`log p(i,j) = f(i) + g(j)`, i.e. exactly the independence being avoided.
"""
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from longnav.utils.chunk_tokenizer import FrozenChunkTokenizer
from longnav.utils.flow_matching_head import (
    FlowMatchingSFTTrainer, TurnFlowActionRegressor,
)

#: Accumulated by `CodeSFTTrainer`. `*_sum` keys are divided by `code_n_valid`;
#: `code_pred_hist` is a (V,) histogram that sums elementwise into a batch-level
#: prediction distribution, which is how collapse is detected.
CODE_METRIC_KEYS = (
    "code_ce_sum", "code_correct_sum", "code_correct_xy_sum", "code_correct_theta_sum",
    "code_top5_sum", "code_pred_entropy_sum", "code_conf_sum",
    "code_grid_l1_xy_sum", "code_grid_l1_theta_sum",
    "code_within1_xy_sum", "code_within1_theta_sum", "code_within1_both_sum",
    "code_n_valid", "code_pred_hist",
)


class CodeContextMixer(nn.Module):
    """`(h, c_xy, c_theta)` -> `(N, n_tokens * d_model)`, with `r(h)` starting at zero."""

    def __init__(self, n_xy: int, n_theta: int, context_dim: int = 1024,
                 d_model: int = 128, n_tokens: int = 8, tok_xy: int = 2,
                 tok_theta: int = 2, r_hidden: Optional[int] = None):
        super().__init__()
        if tok_xy + tok_theta >= n_tokens:
            raise ValueError(
                f"code tokens ({tok_xy} + {tok_theta}) leave no room for r(h) in "
                f"{n_tokens} context tokens"
            )
        if context_dim != n_tokens * d_model:
            raise ValueError(
                f"context_dim {context_dim} != n_tokens * d_model "
                f"({n_tokens} * {d_model}); the decoder has no projection to absorb it"
            )
        self.d_model, self.n_tokens = int(d_model), int(n_tokens)
        self.tok_xy, self.tok_theta = int(tok_xy), int(tok_theta)
        self.n_reserved = n_tokens - tok_xy - tok_theta
        self.emb_xy = nn.Embedding(n_xy, tok_xy * d_model)
        self.emb_theta = nn.Embedding(n_theta, tok_theta * d_model)
        nn.init.normal_(self.emb_xy.weight, std=0.02)
        nn.init.normal_(self.emb_theta.weight, std=0.02)

        hid = int(r_hidden or context_dim)
        self.r = nn.Sequential(
            nn.Linear(context_dim, hid), nn.SiLU(),
            nn.Linear(hid, self.n_reserved * d_model),
        )
        # THE ZERO INIT. Only the final layer -- zeroing the whole stack would make the
        # gradient w.r.t. the first layer vanish too, which IS a dead branch.
        nn.init.zeros_(self.r[-1].weight)
        nn.init.zeros_(self.r[-1].bias)

    @property
    def context_dim(self) -> int:
        return self.n_tokens * self.d_model

    def forward(self, h: torch.Tensor, c_xy: torch.Tensor,
                c_theta: torch.Tensor) -> torch.Tensor:
        hf = h.float()
        code = torch.cat([self.emb_xy(c_xy), self.emb_theta(c_theta)], dim=-1)
        res = self.r(hf)
        # THE NUMBER THAT DECIDES THIS DESIGN. If r(h)'s block grows much larger than the
        # code block, the decoder fits through `h` and `c` becomes decorative -- the
        # latent programme's collapse, reproduced. `r`'s weight RMS alone cannot say so,
        # because the output scale also depends on how large `h` is.
        with torch.no_grad():
            self._last_code_rms = float(code.pow(2).mean().sqrt())
            self._last_res_rms = float(res.pow(2).mean().sqrt())
        return torch.cat([code, res], dim=-1)

    @torch.no_grad()
    def residual_scale(self) -> float:
        """RMS of the last layer's weight -- 0.0 while `r(h)` is still a no-op.

        Worth logging: it is the one number that says whether the continuous branch has
        started contributing at all, and it is invisible in the loss."""
        return float(self.r[-1].weight.pow(2).mean().sqrt())


class CodePolicyHead(nn.Module):
    """`p(c | h)` as one categorical over the `n_xy * n_theta` product."""

    def __init__(self, n_xy: int, n_theta: int, in_dim: int, d_factor: int = 256):
        super().__init__()
        self.n_xy, self.n_theta = int(n_xy), int(n_theta)
        self.vocab = self.n_xy * self.n_theta
        self.e_xy = nn.Embedding(n_xy, d_factor)
        self.e_theta = nn.Embedding(n_theta, d_factor)
        nn.init.normal_(self.e_xy.weight, std=0.02)
        nn.init.normal_(self.e_theta.weight, std=0.02)
        self.table_mlp = nn.Sequential(
            nn.Linear(2 * d_factor, d_factor), nn.SiLU(), nn.Linear(d_factor, in_dim))
        self.ln = nn.LayerNorm(in_dim)
        grid_i = torch.arange(self.vocab) // self.n_theta
        grid_j = torch.arange(self.vocab) % self.n_theta
        self.register_buffer("grid_i", grid_i)
        self.register_buffer("grid_j", grid_j)

    def table(self) -> torch.Tensor:
        """(V, in_dim) output embeddings, rebuilt per forward -- V is small and the
        factors must stay tied so a rare combination inherits both factors' training."""
        return self.table_mlp(
            torch.cat([self.e_xy(self.grid_i), self.e_theta(self.grid_j)], dim=-1))

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        return self.ln(h.float()) @ self.table().t()

    def joint_index(self, c_xy: torch.Tensor, c_theta: torch.Tensor) -> torch.Tensor:
        return c_xy * self.n_theta + c_theta


class TurnFlowActionRegressorWithCode(TurnFlowActionRegressor):
    """The shipped flow head, conditioned on `(c_teacher, r(h))`, plus the code head.

    `c_teacher = g(A*)` comes from the FROZEN tokenizer at every step -- no dropout, no
    policy-sampled `c`. With teacher `c` the flow objective already enforces obedience
    implicitly (the model is trained to produce `A*`, whose code IS `c`), so a separate
    obedience term would restate the same constraint; and the only way to use a sampled
    `c` is to regress toward an `A*` whose code differs, which trains the decoder to
    OVERRIDE the code -- the opposite of what the design buys.
    """

    # Attached by `attach_code_heads`, never by `__init__`: `build()` is a classmethod
    # chain that cannot carry a required kwarg, and reaching into it would be an edit to
    # the shipped construction path. With them absent the class behaves EXACTLY like its
    # parent, so promoting a model is reversible and a half-built one is not broken.
    #
    # DELIBERATELY NO `code_mixer = None` CLASS ATTRIBUTE, for the reason
    # `train_flow_matching_sft_value.py` already records: nn.Module keeps an assigned
    # submodule in `_modules`, reachable only through `__getattr__`, and a class
    # attribute satisfies normal lookup FIRST -- permanently shadowing the real module
    # with None. Only the plain float is safe to declare here.
    code_loss_weight: float = 1.0

    # -- the two seams -----------------------------------------------------------------
    def _decoder_context(self, context: torch.Tensor,
                         targets: torch.Tensor) -> torch.Tensor:
        if getattr(self, "code_mixer", None) is None:
            return super()._decoder_context(context, targets)
        # A probe row carries an all-NaN chunk. Encoding NaN poisons the FSQ bound and
        # silently returns grid index 0, so mark those rows and drop them from the code
        # loss -- but still run them through the mixer, because a Python skip would leave
        # the embedding parameters unreached and hang DDP.
        valid = torch.isfinite(targets).flatten(1).all(dim=1)
        c_xy, c_theta = self.tokenizer.encode_chunk(targets, valid=valid)
        self._code_cache = (c_xy, c_theta, valid)
        return self.code_mixer(context, c_xy, c_theta)

    def _aux_turn_losses(self, context: torch.Tensor, targets: torch.Tensor):
        if (getattr(self, "code_head", None) is None
                or getattr(self, "_code_cache", None) is None):
            return super()._aux_turn_losses(context, targets)
        c_xy, c_theta, valid = self._code_cache
        self._code_cache = None
        logits = self.code_head.logits(context)
        tgt = self.code_head.joint_index(c_xy, c_theta)
        ce = F.cross_entropy(logits, tgt, reduction="none")
        # Multiply, never index: an all-invalid batch must still reach every code-head
        # parameter with a (zero) gradient or DDP waits forever for a bucket that never
        # fires -- the same failure class as the cycle-142 NCCL hang.
        w = valid.to(ce.dtype)
        per_turn = self.code_loss_weight * ce * w
        return per_turn, self._code_metrics(logits.detach(), tgt, c_xy, c_theta, valid)

    # -- metrics -----------------------------------------------------------------------
    @torch.no_grad()
    def _code_metrics(self, logits, tgt, c_xy, c_theta, valid) -> Dict[str, torch.Tensor]:
        head, tok = self.code_head, self.tokenizer.model
        w = valid.float()
        n = w.sum()
        ce = F.cross_entropy(logits, tgt, reduction="none")
        pred = logits.argmax(dim=-1)
        p_xy, p_theta = pred // head.n_theta, pred % head.n_theta

        logp = F.log_softmax(logits, dim=-1)
        entropy = -(logp.exp() * logp).sum(dim=-1)
        conf = logp.max(dim=-1).values.exp()
        top5 = (logits.topk(min(5, head.vocab), dim=-1).indices
                == tgt[:, None]).any(dim=-1).float()

        # ORDINAL distance, the metric that matters here: FSQ indices carry a geometry,
        # so a miss to an adjacent grid point is a neighbouring behaviour, not a wrong
        # mode. Exact accuracy alone reads a boundary case as a total failure -- measured
        # on the c-only head, strict obedience 0.672 against 0.940 within one step.
        l1_xy = tok.xy.fsq.grid_l1(p_xy, c_xy).float()
        l1_th = tok.theta.fsq.grid_l1(p_theta, c_theta).float()

        hist = torch.zeros(head.vocab, device=logits.device, dtype=torch.float32)
        hist.scatter_add_(0, pred, w)                # collapse detector, see the trainer
        return {
            "code_ce_sum": (ce * w).sum(),
            "code_correct_sum": ((pred == tgt).float() * w).sum(),
            "code_correct_xy_sum": ((p_xy == c_xy).float() * w).sum(),
            "code_correct_theta_sum": ((p_theta == c_theta).float() * w).sum(),
            "code_top5_sum": (top5 * w).sum(),
            "code_pred_entropy_sum": (entropy * w).sum(),
            "code_conf_sum": (conf * w).sum(),
            "code_grid_l1_xy_sum": (l1_xy * w).sum(),
            "code_grid_l1_theta_sum": (l1_th * w).sum(),
            "code_within1_xy_sum": ((l1_xy <= 1).float() * w).sum(),
            "code_within1_theta_sum": ((l1_th <= 1).float() * w).sum(),
            "code_within1_both_sum": (((l1_xy <= 1) & (l1_th <= 1)).float() * w).sum(),
            "code_n_valid": n,
            "code_pred_hist": hist,
        }


class CodeSFTTrainer(FlowMatchingSFTTrainer):
    """`FlowMatchingSFTTrainer` plus the code head's statistics.

    A subclass rather than an edit: the parent accumulates an explicit key allowlist, so
    extending it here leaves every existing SFT run byte-identical.
    """

    def _accumulate(self, outputs: Dict[str, torch.Tensor]):
        super()._accumulate(outputs)
        for key in CODE_METRIC_KEYS:
            if key not in outputs:
                continue
            v = outputs[key].detach().float()
            self._sums[key] = v.clone() if key not in self._sums else self._sums[key] + v

    def _drain_metrics(self, prefix: str = "") -> Dict[str, float]:
        sums = dict(self._sums)
        out = super()._drain_metrics(prefix)
        n = float(sums.get("code_n_valid", 0.0))
        if n <= 0:
            return out
        def mean(k):
            return float(sums[k]) / n
        out[f"{prefix}code_ce"] = mean("code_ce_sum")
        out[f"{prefix}code_perplexity"] = float(torch.tensor(mean("code_ce_sum")).exp())
        out[f"{prefix}code_acc"] = mean("code_correct_sum")
        out[f"{prefix}code_acc_xy"] = mean("code_correct_xy_sum")
        out[f"{prefix}code_acc_theta"] = mean("code_correct_theta_sum")
        out[f"{prefix}code_top5"] = mean("code_top5_sum")
        out[f"{prefix}code_pred_entropy"] = mean("code_pred_entropy_sum")
        out[f"{prefix}code_confidence"] = mean("code_conf_sum")
        # The ordinal metrics. Strict accuracy reads a boundary case as a total failure;
        # `within1` is the number that tracks whether the code actually steers.
        out[f"{prefix}code_grid_l1_xy"] = mean("code_grid_l1_xy_sum")
        out[f"{prefix}code_grid_l1_theta"] = mean("code_grid_l1_theta_sum")
        out[f"{prefix}code_within1_xy"] = mean("code_within1_xy_sum")
        out[f"{prefix}code_within1_theta"] = mean("code_within1_theta_sum")
        out[f"{prefix}code_within1"] = mean("code_within1_both_sum")
        out[f"{prefix}code_n_valid"] = n
        # COLLAPSE DETECTION. A policy head that will later be SAMPLED from is useless if
        # it only ever emits the modal code, and cross-entropy alone will not say so --
        # predicting the mode is a perfectly good way to lower it. `pred_used` counts
        # distinct codes emitted this window; `pred_perplexity` is the effective number,
        # so a head drifting toward the mode shows up before accuracy moves.
        hist = sums.get("code_pred_hist")
        if hist is not None and float(hist.sum()) > 0:
            p = (hist / hist.sum()).clamp_min(1e-12)
            out[f"{prefix}code_pred_used"] = float((hist > 0).sum())
            out[f"{prefix}code_pred_perplexity"] = float((-(p * p.log()).sum()).exp())
            out[f"{prefix}code_pred_top1_share"] = float(p.max())
        mixer = getattr(getattr(self, "model", None), "code_mixer", None)
        if mixer is not None:
            # 0.0 while r(h) is still the no-op it is initialised to.
            out[f"{prefix}code_r_scale"] = mixer.residual_scale()
            code_rms = getattr(mixer, "_last_code_rms", None)
            res_rms = getattr(mixer, "_last_res_rms", None)
            if code_rms is not None and res_rms is not None:
                out[f"{prefix}code_ctx_code_rms"] = code_rms
                out[f"{prefix}code_ctx_res_rms"] = res_rms
                # >> 1 means r(h) is swamping the code tokens.
                out[f"{prefix}code_ctx_res_over_code"] = res_rms / max(code_rms, 1e-9)
        return out


def attach_code_heads(model: TurnFlowActionRegressor, tokenizer: FrozenChunkTokenizer,
                      code_loss_weight: float = 1.0, code_d_factor: int = 256,
                      code_r_hidden: Optional[int] = None
                      ) -> "TurnFlowActionRegressorWithCode":
    """Promote a normally-built flow head into the `(c, r(h))`-conditioned one.

    Attachment rather than a custom `build()`: `build` is a classmethod chain through
    `TurnVectorRegressor`, and threading a required kwarg through it would mean editing
    the shipped construction path for every head that does not want this. Promoting the
    instance afterwards leaves that path untouched.

    The decoder keeps its exact shape, so the parameter count changes only by the two
    code embeddings, `r`, and the code head -- and `r` is zero-initialised, so the model
    at step 0 produces exactly what a `c`-only head with tokens 4-7 zeroed produces.
    """
    dec = model.decoder
    model.__class__ = TurnFlowActionRegressorWithCode
    model.tokenizer = tokenizer
    model.code_mixer = CodeContextMixer(
        tokenizer.vocab_xy, tokenizer.vocab_theta, context_dim=dec.context_dim,
        d_model=dec.d_model, n_tokens=dec.n_context_tokens, r_hidden=code_r_hidden,
    ).to(next(model.parameters()).device)
    model.code_head = CodePolicyHead(
        tokenizer.vocab_xy, tokenizer.vocab_theta, dec.context_dim,
        d_factor=code_d_factor).to(next(model.parameters()).device)
    model.code_loss_weight = float(code_loss_weight)
    model._code_cache = None
    return model
