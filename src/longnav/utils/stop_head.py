"""
A stop classifier on the turn's context vector.

The motion head says *where to go*; nothing in it says *when to stop*. This adds a second
readout on the same pooled per-turn state, trained as binary classification against the
episode end.

Labels
------
These are human-terminated expert demonstrations: the demonstrator drove until they judged
the episode complete and then stopped, so **the final observation of an episode is the
stop** and every earlier one is not. That is the label, taken as given. Occasional expert
error -- a stop declared a step early or late, or in the wrong place -- is simply what the
data is; there is no cleaner signal to validate it against, and a filter built on a guess
about which stops are "real" would inject exactly the bias it claims to remove.

Under windowing the label follows the window: a window that does not reach the episode's
last turn contains no positive at all. That is correct rather than a gap -- those turns
genuinely are not stops -- but it does mean the eval collator's take-the-first-N policy
yields almost no positives once windowing is live, which would make eval AP meaningless.
`max_turns_per_sample` is above the longest episode in the current configuration, so every
row carries its episode end today.

`stop_grad`, and why the loss weight then does not matter
--------------------------------------------------------
With `stop_grad=True` (the default) the pooled context is detached before the stop head
sees it. The stop loss then reaches the stop head's own parameters and nothing else -- not
the backbone, not the LoRA adapters, not the motion head. It therefore *cannot* degrade the
motion objective, whatever weight it carries and however badly it is behaving, which is the
whole point: an auxiliary head is only free if it is provably free. It also means
`loss_weight` needs no tuning, since scaling a loss that reaches one isolated set of
parameters is equivalent to scaling that head's learning rate.

Setting `stop_grad=False` gives up exactly that guarantee and turns the weight into a real
hyperparameter. It exists so the shared-representation arm can be run, not because it is a
better default.

Metrics: ranking, not calibration
---------------------------------
The primary numbers are **average precision** and **ROC AUC**, both threshold-free. This is
deliberate. A head whose probabilities are badly scaled but whose *ordering* is sound is
recoverable after the fact -- temperature and threshold are one cross-validated scalar each,
fitted on held-out logits -- whereas a head that ranks stops below non-stops is not
recoverable by any post-processing. So the head is judged on ranking, the raw logits are
saved so the post-hoc fit is possible at all, and temperature is a runtime knob rather than
something baked into the weights.

Accuracy is not reported and should not be: at roughly one positive per ninety turns,
predicting "never stop" scores about 99%.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#: Inference policies for turning a logit into a stop decision.
STOP_INFERENCE_MODES = ("sample", "threshold", "argmax")


@dataclass
class StopHeadConfig:
    """Declarative stop-head settings, serialised into the checkpoint config.

    Absent (`None` on `ModelConfig`) means no stop head at all -- no parameters, no loss
    term, no metrics, and a checkpoint byte-identical to one written before this existed.
    """

    hidden_dims: Tuple[int, ...] = (256,)
    dropout: float = 0.0
    layer_norm: bool = True
    zero_init: bool = True
    #: Detach the context before the head. ON by default; see the module docstring.
    stop_grad: bool = True
    #: Scale on the stop loss in the total. Safe to leave at 1.0 while `stop_grad` is on.
    loss_weight: float = 1.0
    #: `BCEWithLogitsLoss(pos_weight=...)`. None -> 1.0. Worth setting to roughly the
    #: negative:positive ratio (~90 on this corpus) so the positive class produces a
    #: gradient of comparable magnitude to the ninety negatives it is drowning in. It
    #: shifts the operating point, which is irrelevant here -- AP and AUC are invariant to
    #: any monotone rescaling of the scores, and the threshold is fitted post hoc anyway.
    pos_weight: Optional[float] = None
    #: Divides the logit at inference. A runtime knob, fitted post hoc on saved logits;
    #: it is not trained and does not change AP or AUC.
    temperature: float = 1.0
    #: "sample" (default) | "threshold" | "argmax". Sampling is the default because a
    #: deterministic threshold on a miscalibrated head either never stops or stops at the
    #: first ambiguous frame, and both failure modes look like "the head does not work".
    inference: str = "sample"
    threshold: float = 0.5

    def __post_init__(self):
        self.hidden_dims = tuple(int(h) for h in self.hidden_dims)
        if self.inference not in STOP_INFERENCE_MODES:
            raise ValueError(
                f"stop inference must be one of {STOP_INFERENCE_MODES}, got "
                f"{self.inference!r}"
            )
        if float(self.temperature) <= 0:
            raise ValueError(f"stop temperature must be > 0, got {self.temperature}")
        if self.pos_weight is not None and float(self.pos_weight) <= 0:
            raise ValueError(f"stop pos_weight must be > 0, got {self.pos_weight}")

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)

    @classmethod
    def from_dict(cls, d: Any) -> "StopHeadConfig":
        if isinstance(d, StopHeadConfig):
            return d
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(d) - known
        if unknown:
            raise ValueError(
                f"unknown StopHeadConfig field(s) {sorted(unknown)}; known: {sorted(known)}"
            )
        return cls(**d)


class StopHead(nn.Module):
    """Pooled turn context -> one stop logit per turn.

    Small on purpose. The question being asked is whether the context vector *carries* the
    end-of-episode signal, so the head should be barely more than a probe; a large head
    would answer a different question (whether the signal is extractable with enough
    capacity) and would answer it slowly.
    """

    def __init__(self, input_dim: int, cfg: Optional[StopHeadConfig] = None,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        self.cfg = cfg or StopHeadConfig()
        self.input_dim = int(input_dim)
        layers: List[nn.Module] = []
        if self.cfg.layer_norm:
            layers.append(nn.LayerNorm(self.input_dim, dtype=dtype))
        curr = self.input_dim
        for h in self.cfg.hidden_dims:
            layers += [nn.Linear(curr, h, dtype=dtype), nn.GELU()]
            if self.cfg.dropout:
                layers.append(nn.Dropout(self.cfg.dropout))
            curr = h
        out = nn.Linear(curr, 1, dtype=dtype)
        if self.cfg.zero_init:
            # Starts every turn at p = 0.5 rather than at an arbitrary random ranking. The
            # gradient to this layer's weights is proportional to its nonzero input, so it
            # leaves zero on the first step.
            nn.init.zeros_(out.weight)
            nn.init.zeros_(out.bias)
        layers.append(out)
        self.net = nn.Sequential(*layers)
        self._dtype = dtype

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """`(N, input_dim) -> (N,)` logits."""
        if context.dim() != 2 or context.shape[1] != self.input_dim:
            raise ValueError(
                f"StopHead expects (N, {self.input_dim}), got {tuple(context.shape)}"
            )
        return self.net(context.to(self._dtype)).squeeze(-1)

    def loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Mean `BCEWithLogits` over the turns, with the configured `pos_weight`."""
        labels = labels.to(logits.dtype)
        pos_weight = (
            None if self.cfg.pos_weight is None
            else torch.tensor(float(self.cfg.pos_weight), dtype=logits.dtype,
                              device=logits.device)
        )
        if logits.numel() == 0:
            return logits.sum() * 0.0
        return F.binary_cross_entropy_with_logits(
            logits, labels, pos_weight=pos_weight, reduction="mean"
        )

    # -- inference ---------------------------------------------------------------------
    def probability(self, logits: torch.Tensor) -> torch.Tensor:
        """Temperature-scaled stop probability. Temperature is applied here and nowhere
        else, so it never touches training or the reported ranking metrics."""
        return torch.sigmoid(logits / float(self.cfg.temperature))

    def decide(self, logits: torch.Tensor,
               generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """`(N,)` logits -> `(N,)` bool stop decisions, per `cfg.inference`."""
        p = self.probability(logits)
        mode = self.cfg.inference
        if mode == "sample":
            u = torch.rand(p.shape, device=p.device, dtype=p.dtype, generator=generator)
            return u < p
        if mode in ("threshold", "argmax"):
            # "argmax" is the same rule at p = 0.5; both are named because a caller asking
            # for argmax means "no threshold to tune", which is a different intent.
            thr = 0.5 if mode == "argmax" else float(self.cfg.threshold)
            return p >= thr
        raise ValueError(f"unknown stop inference mode {mode!r}")


# ======================================================================================
# Threshold-free metrics
# ======================================================================================
def _rank_average(x: np.ndarray) -> np.ndarray:
    """Ranks 1..n with ties averaged -- what a tie-correct AUC needs."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def roc_auc(scores: Sequence[float], labels: Sequence[float]) -> float:
    """Mann-Whitney U / ROC AUC, ties averaged. NaN if either class is absent.

    NaN rather than 0.5 for a single-class input: 0.5 is a real value meaning "no better
    than chance", and an eval split that happened to contain no positives would otherwise
    report a plausible number for a quantity that is undefined.
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=np.float64).ravel() > 0.5
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rank_average(s)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(scores: Sequence[float], labels: Sequence[float]) -> float:
    """Area under the precision-recall curve, `sum (R_n - R_{n-1}) * P_n`.

    The headline number for this head. AP is the one that moves when a rare positive is
    ranked badly; AUC stays high because it is dominated by the ninety easy negatives.
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=np.float64).ravel() > 0.5
    n_pos = int(y.sum())
    if n_pos == 0 or len(s) == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    precision = tp / np.arange(1, len(y_sorted) + 1)
    recall = tp / n_pos
    d_recall = np.diff(np.concatenate([[0.0], recall]))
    return float((precision * d_recall).sum())


def stop_metrics(logits: Sequence[float], labels: Sequence[float]) -> Dict[str, float]:
    """Threshold-free summary plus the base rate, which is the number AP must beat.

    `stop_ap` against `stop_pos_rate` is the whole comparison: a head with no signal scores
    AP == base rate, so reporting the base rate alongside is what stops a "0.09 AP" from
    being read as either good or bad without reference.
    """
    s = np.asarray(logits, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=np.float64).ravel()
    if s.size == 0:
        return {}
    return {
        "stop_ap": average_precision(s, y),
        "stop_auc": roc_auc(s, y),
        "stop_pos_rate": float((y > 0.5).mean()),
        "stop_n": float(s.size),
    }


def episode_stop_labels(n_turns: int, window_start: int, n_total_turns: int,
                        device=None) -> torch.Tensor:
    """`(n_turns,)` float labels for one window of one episode.

    The positive is the episode's final turn, which is in this window exactly when the
    window reaches the end of the episode. Expressed in terms of the window rather than
    read from a column so there is no second definition of "the end" to drift from -- the
    row's own turn count is the definition.
    """
    labels = torch.zeros(int(n_turns), dtype=torch.float32, device=device)
    if int(n_turns) > 0 and int(window_start) + int(n_turns) == int(n_total_turns):
        labels[-1] = 1.0
    return labels
