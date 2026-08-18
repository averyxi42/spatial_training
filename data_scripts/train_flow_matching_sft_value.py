"""DERIVED from train_flow_matching_sft.py (verbatim copy, 2026-08-18) + the STATE
PROBE: distributional distance + value heads co-trained on the backbone hidden at a
configurable readout offset from the policy sandwich (docs/STOP_HEAD_PLAN.md). The
original script is PRESERVED and remains the reference; every addition here is fenced
with `# --- PROBE ---`. With --probe absent this script must behave identically to the
original (pinned by the parity smoke in tests/smoke_probe_trainer.sh).
"""
"""
Train the FLOW-MATCHING action head (`longnav.utils.flow_matching_head`).

The baseline the autoregressive VQ head (`train_ar_action_sft_v2.py`) is compared against.
Same data, same backbone, same LoRA config, same optimizer grouping and the same metric
plumbing; the ONLY difference is the output parameterization -- a jointly generated chunk
from an integrated velocity field instead of T sequentially decoded codebook indices. Read
`flow_matching_head`'s module docstring before touching anything here: several conventions
(the reversed time axis, Beta-distributed times, blockwise-causal attention, the action
scaling) are deliberate and are silently wrong if "corrected".

Exactly `train_ar_action_sft_v2.py`'s pattern: every argument that is not the output
parameterization is borrowed from `train_vector_sft.py`'s parser, dataset loader, optimizer
param groups and preflight, so this run differs from the AR head's only in what the head
predicts and how.

    torchrun --nproc_per_node=4 data_scripts/train_flow_matching_sft.py \
        --train-dataset data/v2_25hz/formatted \
        --max-steps 6000 --grad-accum 1 --max-turns 400 \
        --dataloader-workers 8 --output-dir dump/flow_matching_head/run1

NO CODEBOOK. Unlike the AR and bin heads there is nothing to fit offline and nothing to
gate on: the head is continuous end to end, so `--codebook` does not exist here.

`--modality-specs` comes from `train_vector_sft.py`'s parser like everything else, and now
reaches this head's `ModelConfig`, both collators and the fresh-parameter LR group; see
`longnav/utils/flow_matching_head.py`. Inert unless passed. There is deliberately NO
`--stop-head` here -- this head's answer to creeping is modelling the whole conditional
distribution, and a binary readout bolted on would be the heuristic that makes the result
uninformative.

`--init-modality-from` vs `--resume-from`
-----------------------------------------
`--resume-from` continues a run: optimizer state, LR schedule, global step, RNG and
dataloader position all come back. `--init-modality-from` borrows exactly ONE module's
weights -- the modality encoders -- from another run's checkpoint, typically the regression
head's, and leaves the readout MLP, the velocity field and the LoRA adapter fresh. That
narrowness is the point: across heads the encoders are the only module whose meaning
survives (a pose -> a residual on a token embedding), while the readout and the adapter were
trained against a different objective on a different output shape.

It is not free. The encoder's output layer is zero-initialised so that step 0 is invariant
to the injected pose values, which is what makes a pose run's step 0 identical to a no-pose
baseline's; a trained encoder gives that up in exchange for not spending the first ~1000
steps re-learning the same map.

WHAT THE LOGGED METRICS MEAN (this is the part people get wrong across heads):

  * there is NO teacher-forced / free-running split. This head has one generation mode, so
    every number below comes from actually integrating the ODE and NO `free_*` key is
    emitted. Do NOT compare the AR head's teacher-forced `rmse_*` against this head's
    `rmse_*`; compare against the AR head's `free_*` family.
  * `rmse_*` / `mae_*`          per-tick DIFFERENTIAL error, generated
  * `pose_rmse_*` / `pose_mae_*`  COMPOSED pose error -- the AR head's `free_rmse_*` analogue
  * `near_zero_pred_*`          stop + creep mass, the near-zero statistic that IS comparable
    between a discrete and a continuous head. `stop_pred_*` alone is not: a VQ codebook has a
    centroid sitting exactly on zero and clears the 1e-4 threshold trivially, a continuous
    head may read 1e-5 while being behaviourally stopped.
  * `rotation_flip`             same deadband and definition as the AR head's
    `free_rotation_flip` and the closed-loop ObjectNav probes.
"""

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for p in (str(_ROOT), str(_ROOT / "src"), str(_ROOT / "data_scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
from transformers import AutoProcessor, TrainingArguments  # noqa: E402

import train_vector_sft as base  # noqa: E402
from longnav.utils.latent_intent import LatentConfig  # noqa: E402
from longnav.utils.mixture import MixtureSpec, build_mixture  # noqa: E402
from longnav.utils.model_metrics import attach_model_metrics  # noqa: E402
from longnav.utils.flow_matching_head import (  # noqa: E402
    ACTION_SCALES, DEFAULT_DT_NATIVE, FlowMatchingConfig, FlowMatchingSFTTrainer,
    NUM_INFERENCE_STEPS, TurnFlowActionRegressor,
)
from longnav.utils.vector_sft import (  # noqa: E402
    DataConfig, LossConfig, LoraSpec, ModelConfig, TurnVectorCollator,
)



# --- PROBE ---------------------------------------------------------------------------
import numpy as _np
import torch as _torch

from longnav.utils.state_probe import (
    StateProbe, StateProbeConfig, save_state_probe,
)
from longnav.utils.turn_vectors import TurnSpan, find_turn_spans, to_sparse_indices
from longnav.utils.vector_sft import TurnVectorCollator as _BaseCollator


class _RecordingRng:
    """Wraps the collator rng to recover the window start the base __call__ drew.

    The base collator draws `integers` EXACTLY ONCE per example (the window start,
    only when the episode exceeds the cap); asserting that stays true is what makes
    recovering the draw sound rather than a guess.
    """

    def __init__(self, rng):
        self._rng = rng
        self.draws = []

    def integers(self, *a, **k):
        v = self._rng.integers(*a, **k)
        self.draws.append(int(v))
        return v

    def __getattr__(self, name):
        return getattr(self._rng, name)


class ProbeCollator(_BaseCollator):
    """Base collator + per-turn `distance_targets` / `return_targets`, sliced with the
    SAME window the base applied to the action targets."""

    distance_column: str = "distance_targets"
    return_column: str = "return_targets"

    def __call__(self, examples):
        ex = examples[0]
        rec = _RecordingRng(self._rng)
        self._rng = rec
        try:
            out = super().__call__(examples)
        finally:
            self._rng = rec._rng
        assert len(rec.draws) <= 1, (
            f"base collator drew the rng {len(rec.draws)} times; the window-start "
            "recovery assumes exactly one draw -- vector_sft changed, re-derive")
        start = rec.draws[0] if rec.draws else 0
        n_kept = int(out["targets"].shape[0])
        for col, key in ((self.distance_column, "distance_targets"),
                         (self.return_column, "return_targets")):
            raw = ex.get(col)
            if raw is None:
                continue
            if len(raw) < start + n_kept:
                raise ValueError(
                    f"{col} has {len(raw)} entries but the window needs "
                    f"[{start}, {start + n_kept}); the dataset column is misaligned "
                    "with the observation count")
            vals = [float("nan") if v is None else float(v)
                    for v in raw[start:start + n_kept]]
            out[key] = _torch.tensor(vals, dtype=_torch.float32)
        return out


class TurnFlowActionRegressorWithProbe(TurnFlowActionRegressor):
    """Adds the state probe WITHOUT touching the flow head's forward.

    A forward hook on the backbone captures the sparse outputs; probe readout
    positions are recomputed with the same library machinery the head uses
    (`find_turn_spans` + `to_sparse_indices`, strict) at `readout_offset` DENSE
    tokens from the policy readout. `state_probe=None` (the default, and the
    --probe-off path) leaves forward bit-identical to the parent.
    """

    # NOTE deliberately NO `state_probe = None` class attribute: nn.Module stores an
    # assigned submodule in `_modules`, which is only reachable through __getattr__ --
    # and a class attribute would satisfy normal lookup first, permanently shadowing
    # the real probe (observed: zero probe grads, silent). Use getattr with default.
    probe_token_id = None         # pinned on first batch; constancy asserted after

    def forward(self, input_ids, targets, attention_mask=None, num_turns=None,
                num_items_in_batch=None, distance_targets=None, return_targets=None,
                **backbone_inputs):
        probe = getattr(self, "state_probe", None)
        if probe is None:
            return super().forward(input_ids, targets, attention_mask=attention_mask,
                                   num_turns=num_turns,
                                   num_items_in_batch=num_items_in_batch,
                                   **backbone_inputs)
        captured = {}

        def _cap(_m, _i, o):
            captured["outputs"] = o

        handle = self.backbone.register_forward_hook(_cap)
        try:
            out = super().forward(input_ids, targets, attention_mask=attention_mask,
                                  num_turns=num_turns,
                                  num_items_in_batch=num_items_in_batch,
                                  **backbone_inputs)
        finally:
            handle.remove()
        if distance_targets is None and return_targets is None:
            return out
        o = captured["outputs"]
        hidden = o["last_hidden_state"]
        keep = o["seq_keep_mask"] if "seq_keep_mask" in o else None
        # find_turn_spans returns List[List[TurnSpan]] -- one inner list per batch
        # row (batch is 1 here, but keep the nesting: to_sparse_indices expects it).
        rows = find_turn_spans(input_ids, self.prefix_ids, self.postfix_ids,
                               shift_left=self.model_cfg.shift_left)
        off = int(probe.cfg.readout_offset)
        probe_rows = []
        for row in rows:
            prow = []
            for sp in row:
                pos = int(sp.indices[0]) + off  # DENSE position of the probe token
                assert pos >= 0, \
                    f"probe offset {off} underflows the sequence at turn start"
                prow.append(TurnSpan(batch_idx=sp.batch_idx, start=pos, end=pos + 1,
                                     indices=_torch.tensor([pos]), shifted=False))
            probe_rows.append(prow)
        # Token-identity pin: the probe token must be the SAME id every turn -- the
        # design leans on it being a constant text token (survives sparsification).
        flat = [p for row in probe_rows for p in row]
        ids = input_ids[0, [int(p.indices[0]) for p in flat]]
        if self.probe_token_id is None:
            type(self).probe_token_id = int(ids[0])
        if not bool((ids == self.probe_token_id).all()):
            raise RuntimeError(
                f"probe readout token id varies across turns ({ids.tolist()} vs pinned "
                f"{self.probe_token_id}); readout_offset {off} does not land on a "
                "constant-identity token in this template")
        probe_rows = to_sparse_indices(probe_rows, keep, strict=True)
        flat = [p for row in probe_rows for p in row]
        h = _torch.stack([hidden[p.batch_idx, int(p.indices[0])] for p in flat])
        losses = probe.losses(
            h.unsqueeze(0),
            None if distance_targets is None else distance_targets.reshape(1, -1),
            None if return_targets is None else return_targets.reshape(1, -1),
        )
        n_turns_t = _torch.tensor(float(len(flat)))
        for k, v in losses.items():
            out["loss"] = out["loss"] + v
            # sum-style keys for ProbeTrainer._accumulate (turn-weighted, like
            # motion_loss_sum); the raw key stays for anyone reading outputs directly.
            out[k.replace("probe/", "probe_").replace("_loss", "_loss_sum")] = \
                v.detach() * n_turns_t
            out[k] = v.detach()
        out["probe_turns"] = n_turns_t
        return out


from transformers import TrainerCallback as _TrainerCallback


class _ProbeTrainerMixin:
    """Accumulate/drain the probe losses alongside the whitelisted metrics."""

    _PROBE_KEYS = ("probe_distance_loss_sum", "probe_value_loss_sum", "probe_turns")

    def _accumulate(self, outputs):
        super()._accumulate(outputs)
        for key in self._PROBE_KEYS:
            if key in outputs:
                v = outputs[key].detach().float()
                self._sums[key] = v.clone() if key not in self._sums \
                    else self._sums[key] + v

    def _drain_metrics(self, prefix: str = ""):
        sums = dict(self._sums)   # super clears self._sums
        out = super()._drain_metrics(prefix=prefix)
        n = sums.get("probe_turns")
        if n is not None:
            for key, name in (("probe_distance_loss_sum", "probe_distance_loss"),
                              ("probe_value_loss_sum", "probe_value_loss")):
                if key in sums:
                    out[f"{prefix}{name}"] = float(sums[key] / n.clamp(min=1))
        return out



class _SaveProbeCallback(_TrainerCallback):
    """Writes state_probe.pt(+config) into every checkpoint dir the Trainer saves."""

    def __init__(self, model):
        self._model = model

    def on_save(self, args, state, control, **kwargs):
        if getattr(self._model, "state_probe", None) is not None \
                and state.is_world_process_zero:
            ckpt = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            if os.path.isdir(ckpt):
                save_state_probe(ckpt, self._model.state_probe)
        return control
# --- END PROBE -----------------------------------------------------------------------


def build_optimizer_param_groups(model, args):
    """`base.build_optimizer_param_groups`, but with the fresh velocity field grouped with
    the fresh trunk head under `--head-lr` rather than under the adapters' `--lr`.

    Identical reasoning to `train_ar_action_sft_v2.py`'s copy: the decoder is exactly as
    randomly-initialized as `model.head` and exactly as small, so it wants the same larger
    step a fresh head does; only the LoRA adapters on the pretrained backbone should move at
    the conservative rate. Reimplemented here rather than edited into `train_vector_sft.py`
    because the grouping is specific to this head having a second fresh module.
    """
    fresh = list(model.head.parameters()) + list(model.decoder.parameters())
    # --- PROBE --- fresh heads belong with the other fresh modules at --head-lr
    _probe = getattr(model, "state_probe", None)
    if _probe is not None:
        fresh += list(_probe.parameters())
    # The latent split and the posterior are exactly as randomly-initialised as the head, so
    # they belong on --head-lr with it. Leaving either out would SILENTLY FREEZE it, and a
    # frozen split is indistinguishable from "the CVAE conversion did nothing".
    _latent = getattr(model.normalizer, "latent", None)
    if _latent is not None:
        fresh += list(_latent.parameters())
    _pred = getattr(model, "chunk_predictor", None)
    if _pred is not None:
        fresh += list(_pred.parameters())
    _no_decay = []
    _post = getattr(model, "posterior", None)
    if _post is not None:
        _nd = {id(q) for q in _post.no_decay_parameters()}
        fresh += [q for q in _post.parameters() if id(q) not in _nd]
        # delta_mu's output layer is zero-initialised, so weight decay on it is a standing
        # pull back toward delta_mu = 0 -- i.e. toward sigma_p -> 0, the degenerate solution
        # this design exists to avoid. A regulariser aimed at the failure mode.
        _no_decay = [q for q in _post.no_decay_parameters() if q.requires_grad]
    # Same reason, and the same failure mode the base function documents: a module left out
    # of the groups is SILENTLY FROZEN, and for the modality encoders that is
    # indistinguishable from "the injection did not help" -- the experiment's conclusion.
    fresh += list(model.modality_embedder.parameters())
    fresh = [p for p in fresh if p.requires_grad]
    fresh_ids = {id(p) for p in fresh} | {id(p) for p in _no_decay}
    other = [p for p in model.parameters() if p.requires_grad and id(p) not in fresh_ids]
    # `other` is empty when --latent-freeze-trunk froze the backbone and adapters, which is
    # the whole point of the staged run; an empty group is a footgun, not a configuration.
    groups = ([{"params": other, "lr": args.lr, "weight_decay": args.weight_decay}]
              if other else [])
    if fresh:
        groups.append({
            "params": fresh,
            "lr": args.head_lr if args.head_lr is not None else args.lr,
            "weight_decay": args.weight_decay,
        })
    if _no_decay:
        groups.append({
            "params": _no_decay,
            "lr": args.head_lr if args.head_lr is not None else args.lr,
            "weight_decay": 0.0,
        })
    return groups


def parse_args():
    # allow_abbrev=False is load-bearing, not tidiness. This parser runs BEFORE the base
    # one and passes what it does not recognise through `parse_known_args`. With
    # abbreviation on, any base flag that is a strict prefix of a flag declared here gets
    # captured instead of passed through -- `--train-dataset` was silently swallowed by
    # `--train-datasets`, which would have broken every existing single-dataset command
    # with an error pointing at the wrong flag.
    pre = base.argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument("--decoder-d-model", type=int, default=128)
    # --- PROBE ---
    pre.add_argument("--probe", action="store_true",
                     help="co-train the state probe (distance+value heads). Absent: "
                          "this script is the original, bit for bit")
    pre.add_argument("--probe-offset", type=int, default=-2,
                     help="probe readout position, DENSE tokens from the policy "
                          "sandwich readout. Checkpoint contract with the RL side's "
                          "ValueHeadConfig.readout_offset")
    pre.add_argument("--probe-grad-scale", type=float, default=0.1)
    pre.add_argument("--probe-distance-weight", type=float, default=1.0)
    pre.add_argument("--probe-value-weight", type=float, default=1.0)
    pre.add_argument("--probe-gamma", type=float, default=None,
                     help="REQUIRED with --probe: must equal the dataset's "
                          "return_gamma stamp (checked against row 0)")
    pre.add_argument("--probe-head-lr", type=float, default=None,
                     help="probe params LR; default: --head-lr group")
    pre.add_argument("--probe-datasets", default=None,
                     help="comma-separated mixture component NAMES that train the probe. "
                          "The mechanism is the data itself (components lacking "
                          "distance_targets/return_targets are skipped row-by-row); this "
                          "flag makes the intent EXPLICIT and CHECKED: each named "
                          "component must carry the columns, each unnamed one must not. "
                          "A component with columns you forgot to name is an error, not "
                          "a silent inclusion. Single-dataset runs ignore this.")
    # --- END PROBE ---
    pre.add_argument("--decoder-layers", type=int, default=4)
    pre.add_argument("--decoder-heads", type=int, default=4)
    pre.add_argument("--decoder-ff", type=int, default=512)
    pre.add_argument("--decoder-dropout", type=float, default=0.1)
    pre.add_argument("--context-tokens", type=int, default=8,
                     help="prefix tokens the readout MLP emits. A token is d_model wide, "
                          "so this -- not the MLP width -- sets how many dimensions of "
                          "context the field can receive. context_dim is DERIVED as "
                          "context-tokens * decoder-d-model; there is no projection to "
                          "reconcile a mismatch")
    pre.add_argument("--incoming-motion", action="store_true",
                     help="add the pi0-style state token, carrying the motion the robot is "
                          "already executing plus a validity bit. OFF by default because "
                          "the dataset does not yet supply the value and the AR head does "
                          "not have it either -- enabling it on ONE head breaks the "
                          "comparison this run exists to make")

    f = pre.add_argument_group("flow matching")
    f.add_argument("--k-samples", type=int, default=8,
                   help="(t, noise) draws per turn per step, batched into ONE head forward. "
                        "The head is ~2 orders of magnitude cheaper than the backbone here, "
                        "so this is nearly free variance reduction -- but at --max-turns 400 "
                        "the expanded head batch is 400*K sequences, so watch memory above 8")
    f.add_argument("--eval-per-component", action="store_true",
                   help="evaluate each --mixture-datasets component on its OWN "
                        "validation split and log metrics as eval_<component>_*, "
                        "instead of one blended number. A single eval set cannot say "
                        "WHICH component regressed -- a run that forgets ObjectNav "
                        "while improving PointNav looks flat")
    f.add_argument("--log-motion-bands", action="store_true",
                   help="emit the per-dimension stop_*/creep_*/near_zero_* band "
                        "metrics. OFF by default: ~30 keys per log line, built to "
                        "compare a discrete head's codebook occupancy against a "
                        "continuous head's, which the current mixture does not ask. "
                        "The sums are accumulated either way, so this is logging only")
    f.add_argument("--inference-steps", type=int, default=NUM_INFERENCE_STEPS,
                   help="Euler steps at deploy time (a knob, not a trained quantity)")
    f.add_argument("--metric-steps", type=int, default=None,
                   help="Euler steps for the metrics logged during training; defaults to "
                        "--inference-steps. Pinning it lets --inference-steps be swept at "
                        "eval without moving the training curves")
    f.add_argument("--time-alpha", type=float, default=1.5,
                   help="Beta(alpha, beta) time law. 1.5 with beta=1 concentrates draws near "
                        "t=1, the high-noise end, matching openpi. Uniform is NOT the default "
                        "on purpose")
    f.add_argument("--time-beta", type=float, default=1.0)
    f.add_argument("--time-scale", type=float, default=0.999)
    f.add_argument("--time-offset", type=float, default=0.001)
    f.add_argument("--no-stratified-time", action="store_true",
                   help="draw K i.i.d. Beta times instead of one per stratum. The ablation "
                        "that measures what stratification buys; requires nothing, whereas "
                        "stratification requires --time-beta 1.0")
    f.add_argument("--antithetic-noise", action="store_true",
                   help="pair stratum k with k+K/2 as (eps, -eps); needs an even --k-samples")
    f.add_argument("--action-scales", default=None,
                   help="comma-separated dx,dy,dtheta divisors bringing the differentials to "
                        "~unit variance before flow matching (default "
                        f"{','.join(str(s) for s in ACTION_SCALES)}, the same constants the "
                        "v2 AR decoder calibrated). '1,1,1' disables scaling, which is the "
                        "design document's literal recipe and is expected to train badly -- "
                        "see flow_matching_head's ACTION SCALING note")

    pre.add_argument("--latent-cvae", action="store_true",
                     help="convert the deterministic readout into a stochastic INTENT: "
                          "h -> (mu, log sigma), c ~ N(mu + delta_mu(h, A), sigma^2), with "
                          "a KL to the prior. Off is v3 exactly -- no split, no posterior, "
                          "no KL term. See docs/LATENT_RL.md")
    pre.add_argument("--latent-beta", type=float, default=1.0,
                     help="KL weight. NOT in transferable units: the reconstruction term is "
                          "a velocity regression rather than a log-likelihood, so this is a "
                          "bare exchange rate and must be swept per architecture")
    pre.add_argument("--latent-sigma0", type=float, default=0.02,
                     help="initial sigma, as an ABSOLUTE value in h's units. Measure it: "
                          "1-3%% of h's per-dim std (scripts/measure_h_stats.py). sigma is "
                          "deliberately never floored during training -- it is the quantity "
                          "the acceptance test reads, and a floor corrupts it")
    pre.add_argument("--latent-posterior-width", type=int, default=256,
                     help="posterior hidden width. Capacity here is capacity to "
                          "over-migrate, so it acts as a rate limiter complementing beta")
    pre.add_argument("--latent-aux-weight", type=float, default=0.0,
                     help="weight on an auxiliary `c -> chunk` regression. The flow term "
                          "alone gives the latent no reason to exist: it already reaches low "
                          "loss with the residual in its base noise, which is posterior "
                          "collapse under an expressive decoder. 0.0 = objective unchanged")
    pre.add_argument("--latent-decoder", default="flow", choices=("flow", "deterministic"),
                     help="`deterministic` REPLACES the flow objective with the c->chunk "
                          "regression, so `c` is the only source of variation in the decode. "
                          "Not the same as pinning the flow's base noise, which would make it "
                          "a regressor wearing an ODE rather than a flow with less noise")
    pre.add_argument("--latent-diversity", type=float, default=0.0,
                     help="weight on the mode-seeking RATIO ||dA||/||dc||, clamped. The "
                          "normalisation is the point: unnormalised output spread is solved "
                          "by inflating sigma rather than by making the decoder responsive "
                          "to c, which is the degenerate outcome already measured")
    pre.add_argument("--latent-diversity-clamp", type=float, default=1.0)
    pre.add_argument("--latent-diversity-probe", type=float, default=0.001,
                     help="absolute probe step in h units. Fixed, NOT sigma-scaled: a "
                          "learned prior makes ||c1-c2|| gameable, and shrinking sigma "
                          "raises the ratio toward the local Jacobian norm")
    pre.add_argument("--latent-diversity-target", type=float, default=0.0,
                     help="target dA/dc; the weight is adapted to hit it. A fixed weight "
                          "cannot hold a fixed share of a moving objective -- measured at "
                          "200x and at 0.14% with the same term")
    pre.add_argument("--latent-diversity-lr", type=float, default=0.02)
    pre.add_argument("--latent-sigma-floor", type=float, default=0.0,
                     help="hard lower bound on sigma. Free bits removes the only upward "
                          "pressure on it, so without this reconstruction drives it to zero")
    pre.add_argument("--latent-diversity-steps", type=int, default=3,
                     help="Euler steps for the diversity decode only; it estimates a "
                          "sensitivity, not a deployment-accuracy solve")
    pre.add_argument("--latent-free-bits", type=float, default=0.0,
                     help="nats per dim below which the KL stops penalising. The standard "
                          "remedy for a latent that empties under an expressive decoder; "
                          "0.01 admits ~10 nats over 1024 dims")
    pre.add_argument("--latent-kl-warmup", type=int, default=0,
                     help="linear anneal of beta from 0 over N steps, so the decoder cannot "
                          "learn to route around a latent that is still uninformative")
    pre.add_argument("--latent-reinit-decoder", action="store_true",
                     help="keep the warm-started backbone and readout but re-initialise the "
                          "velocity field, so decoder and posterior learn together")
    pre.add_argument("--latent-freeze-trunk", action="store_true",
                     help="freeze backbone, adapters, readout MLP and modality encoders, "
                          "training only the split and the posterior. The cheap staged run: "
                          "hours rather than days, and it answers whether sigma is usable "
                          "at all before anything expensive is spent")
    pre.add_argument("--latent-freeze-decoder", action="store_true",
                     help="also freeze the velocity field, so sigma is fitted against the "
                          "decoder's EXISTING robustness radius. Gives up the decoder "
                          "adapting to noised c -- which is what widens the usable "
                          "exploration band -- so it is a run-1 measurement, not the end state")
    pre.add_argument("--mixture-datasets", action="append", default=None,
                     metavar="NAME=PATH:RATIO",
                     help="MIX several corpora at a stated ratio, repeatable, e.g. "
                          "--mixture-datasets objectnav=data/v2_25hz_obs2.5hz/formatted_pose:1 "
                          "--mixture-datasets pointnav=/Projects/data/pointnav_hm3d_2p5hz_clamp/formatted_pose:1 "
                          ". Ratios are UNNORMALISED weights and are independent of corpus "
                          "size -- which is the point: concatenating a 39k-row corpus with a "
                          "5k-row one trains 89%% the first whatever was intended, and the "
                          "proportion drifts as a corpus grows. Sources are loaded "
                          "separately and never concatenated, so they need not share an "
                          "arrow schema; they must agree on the chunking parameters, which "
                          "is checked up front. Named --mixture-* rather than --train-datasets so it "
                          "cannot be confused with, or abbreviate to, --train-dataset")
    pre.add_argument("--mixture-length", type=int, default=None,
                     help="nominal examples per epoch when mixing (default: the sum of the "
                          "source sizes). Sampling is with replacement, so this is a stream "
                          "length, not a pass over the data -- use --max-steps as the budget")
    pre.add_argument("--mixture-seed", type=int, default=0,
                     help="keys the per-index draw, so every dataloader worker and every "
                          "resume produce the same mixture")
    pre.add_argument("--init-from", default=None,
                     help="WARM START from a checkpoint: load ALL of its trained weights "
                          "(LoRA adapter, readout head, velocity field, modality encoders) "
                          "into this model, then train from step 0 with a fresh optimizer, "
                          "scheduler, RNG and dataloader order. Distinct from --resume-from, "
                          "which additionally restores that run's optimizer state, LR "
                          "schedule and step counter -- i.e. continues it. Distinct from "
                          "--init-modality-from, which borrows the encoders ALONE. Modules "
                          "this run declares that the checkpoint has no entry for at all "
                          "start fresh and are printed; everything shared loads strictly, "
                          "so a shape drift (a different action_chunk_len gives the "
                          "velocity field a different n_ticks) raises rather than silently "
                          "reshaping. --resume-from wins if both are given")
    pre.add_argument("--init-modality-from", default=None,
                     help="warm start the MODALITY ENCODERS ONLY from another run's "
                          "checkpoint (e.g. a regression run's "
                          "dump/pose_injection/run_v7_planar_pose/checkpoint-1400). Loads "
                          "blob['modality']['encoders'] and NOTHING else -- not the head, "
                          "not the normalizer (this head's velocity field lives there), not "
                          "the LoRA adapter; those stay at fresh init. The specs must match "
                          "the source's, and a missing, extra or shape-drifted encoder "
                          "raises rather than being skipped. COST: the encoder's output "
                          "layer is zero-initialised so that step 0 is invariant to the "
                          "injected pose values, which is what makes a pose run's step 0 "
                          "comparable with a no-pose baseline's. A trained encoder gives "
                          "that up -- from step 0 the pose moves the loss. A deliberate "
                          "trade of comparability for convergence speed, not a free win. "
                          "(A spec with a nonzero `gain_init`, e.g. pose_spec_planar.json's "
                          "1.0, has already traded it away at fresh init, so on THAT spec "
                          "the warm start costs nothing further on this axis -- what it "
                          "costs there is agreement with a fresh-encoder run.) "
                          "Distinct from --resume-from (which restores a whole training "
                          "state) and from the AR script's --init-from (which borrows every "
                          "shared module, meaningless across heads)")
    mine, rest = pre.parse_known_args()

    # `--train-dataset` is required by the base parser, so a pure --train-datasets run
    # would fail to parse. Supply the FIRST source as its value rather than a placeholder:
    # everything that reads `args.train_dataset` -- notably the `--eval-dataset` fallback --
    # then gets a real, loadable path instead of something that only fails later.
    if mine.mixture_datasets and not any(a == "--train-dataset" for a in rest):
        first = MixtureSpec.parse(mine.mixture_datasets[0]).path
        rest = [*rest, "--train-dataset", first]

    old_argv, sys.argv = sys.argv, [sys.argv[0], *rest]
    try:
        args = base.parse_args()
    finally:
        sys.argv = old_argv

    # Namespaces merge attribute by attribute, so a flag declared on the pre-parser and not
    # copied across is parsed and then silently dropped -- it reads as a no-op run.
    # --- PROBE --- (the comment above is load-bearing: copy or it silently no-ops)
    args.probe = mine.probe
    args.probe_offset = mine.probe_offset
    args.probe_grad_scale = mine.probe_grad_scale
    args.probe_distance_weight = mine.probe_distance_weight
    args.probe_value_weight = mine.probe_value_weight
    args.probe_gamma = mine.probe_gamma
    args.probe_head_lr = mine.probe_head_lr
    args.probe_datasets = mine.probe_datasets
    # --- END PROBE ---
    args.latent_cvae = mine.latent_cvae
    args.latent_beta = mine.latent_beta
    args.latent_sigma0 = mine.latent_sigma0
    args.latent_posterior_width = mine.latent_posterior_width
    args.latent_diversity = mine.latent_diversity
    args.latent_diversity_clamp = mine.latent_diversity_clamp
    args.latent_diversity_probe = mine.latent_diversity_probe
    args.latent_sigma_floor = mine.latent_sigma_floor
    args.latent_diversity_target = mine.latent_diversity_target
    args.latent_diversity_lr = mine.latent_diversity_lr
    args.latent_diversity_steps = mine.latent_diversity_steps
    args.latent_free_bits = mine.latent_free_bits
    args.latent_kl_warmup = mine.latent_kl_warmup
    args.latent_reinit_decoder = mine.latent_reinit_decoder
    args.latent_aux_weight = mine.latent_aux_weight
    args.latent_decoder = mine.latent_decoder
    args.latent_freeze_trunk = mine.latent_freeze_trunk
    args.latent_freeze_decoder = mine.latent_freeze_decoder
    args.init_modality_from = mine.init_modality_from
    args.init_from = mine.init_from
    args.mixture_datasets = mine.mixture_datasets
    args.mixture_length = mine.mixture_length
    args.mixture_seed = mine.mixture_seed
    # Every pre-parser flag must be copied across BY NAME. A flag added above and not
    # listed here parses without error and is then silently discarded -- the run proceeds
    # with the default and nothing says so.
    args.eval_per_component = mine.eval_per_component
    args.log_motion_bands = mine.log_motion_bands
    # Derived, never accepted: there is no context projection, so the readout MLP must emit
    # exactly one d_model-wide vector per prefix token.
    args.context_dim = mine.context_tokens * mine.decoder_d_model
    args.decoder_kwargs = dict(
        d_model=mine.decoder_d_model, n_layers=mine.decoder_layers,
        n_heads=mine.decoder_heads, dim_ff=mine.decoder_ff, dropout=mine.decoder_dropout,
        n_context_tokens=mine.context_tokens, use_incoming_motion=mine.incoming_motion,
    )
    scales = (base._csv(mine.action_scales, float) if mine.action_scales else ACTION_SCALES)
    if len(scales) != 3:
        raise SystemExit(f"--action-scales needs 3 comma-separated floats, got {scales}")
    args.fm_cfg = FlowMatchingConfig(
        k_samples=mine.k_samples,
        num_inference_steps=mine.inference_steps,
        metric_inference_steps=mine.metric_steps or mine.inference_steps,
        time_alpha=mine.time_alpha, time_beta=mine.time_beta,
        time_scale=mine.time_scale, time_offset=mine.time_offset,
        stratified_time=not mine.no_stratified_time,
        antithetic_noise=mine.antithetic_noise,
        action_scales=tuple(scales),
    )
    if args.output_dir == "dump/vector_sft":
        raise SystemExit(
            "--output-dir defaults to the regression baseline's directory; pass an "
            "explicit one under dump/flow_matching_head/"
        )
    if args.wandb_project == "longnav-vector-sft":
        args.wandb_project = "longnav-flow-matching-head"
    if args.loss != "huber":
        raise SystemExit(
            "--loss does not apply: this head's objective is the flow-matching MSE between "
            "the predicted velocity and u_t = noise - actions"
        )
    return args


def main():
    args = parse_args()
    is_main = int(os.environ.get("RANK", "0")) == 0

    report_to = [] if args.no_wandb else ["wandb"]
    if report_to:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        os.environ.setdefault("WANDB_MODE", "online")

    model_cfg = ModelConfig(
        model_id=args.model_id,
        attn_impl=args.attn_impl,
        prefix=base._unescape(args.prefix),
        postfix=base._unescape(args.postfix),
        shift_left=not args.no_shift_left,
        pool_mode=args.pool_mode,
        head_hidden_dims=base._csv(args.head_hidden_dims, int),
        head_dropout=args.head_dropout,
        freeze_vision_tower=not args.train_vision_tower,
        # `--modality-specs` comes from `base.parse_args()` already (this script only
        # pre-parses the head-shape flags); what was missing was putting it on the
        # ModelConfig. Inert unless passed. There is no `--stop-head` counterpart here on
        # purpose -- see flow_matching_head's NO STOP / DEADBAND section.
        modality_specs=base._modality_specs(args.modality_specs),
    )
    data_cfg = DataConfig(
        target_column=args.target_column,
        messages_column=args.messages_column,
        images_column=args.images_column,
        max_turns_per_sample=args.max_turns or None,
        target_dim_names=("dx", "dy", "dtheta"),  # what the metric table reports on
    )
    loss_cfg = LossConfig(kind="flow_matching", normalize_targets=False)
    lora = None if args.no_lora else LoraSpec(
        r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout,
        target_modules=base._csv(args.lora_target_modules),
    )

    # ---- data ----------------------------------------------------------------------
    # Single dataset unless --train-datasets was given, so the default path is exactly
    # the one it has always been -- same call, same object, no wrapper.
    specs = []
    if args.mixture_datasets:
        specs = [MixtureSpec.parse(t, args.train_split) for t in args.mixture_datasets]
        train_ds = build_mixture(specs, base.load_split, args.train_split,
                                 length=args.mixture_length, seed=args.mixture_seed)
        if is_main:
            print(train_ds.describe())
        # --- PROBE --- explicit per-component contract (see --probe-datasets help)
        if args.probe and args.probe_datasets is not None:
            wanted = {n.strip() for n in args.probe_datasets.split(",") if n.strip()}
            unknown = wanted - {sp.name for sp in specs}
            if unknown:
                raise SystemExit(f"--probe-datasets names unknown components: {sorted(unknown)}")
            for sp in specs:
                cols = set(base.load_split(sp.path, sp.split or args.train_split).column_names)
                has = {"distance_targets", "return_targets"} <= cols
                if sp.name in wanted and not has:
                    raise SystemExit(
                        f"--probe-datasets includes {sp.name!r} but its dataset carries no "
                        "distance/return target columns; join them first "
                        "(add_distance_targets.py)")
                if sp.name not in wanted and has:
                    raise SystemExit(
                        f"component {sp.name!r} carries probe target columns but is not in "
                        "--probe-datasets; name it or point at the un-joined dataset -- "
                        "silent inclusion is the failure mode this flag exists to prevent")
            if is_main:
                print(f"[probe] training value/distance on components: {sorted(wanted)}")
    else:
        train_ds = base.load_split(args.train_dataset, args.train_split)
    eval_ds = None
    if args.eval_split:
        if args.eval_per_component and specs:
            # One eval set per mixture component, keyed by the component's name. HF's
            # Trainer evaluates a dict of datasets separately and prefixes each with its
            # key, and `TurnVectorSFTTrainer.evaluate` already threads `metric_key_prefix`
            # into `_drain_metrics`, so this needs nothing from the trainer -- the metrics
            # come out as `eval_<component>_turn_loss`, `eval_<component>_rmse_dx`, ...
            #
            # Worth having because a single blended eval number cannot say WHICH component
            # regressed: a mixture that quietly forgets ObjectNav while improving PointNav
            # looks flat. That is exactly the failure a co-training run needs to see.
            eval_ds = {}
            for spec in specs:
                try:
                    eval_ds[spec.name] = base.load_split(
                        spec.path, args.eval_split, args.eval_max_samples
                    )
                except (KeyError, ValueError, FileNotFoundError) as exc:
                    # A component without the eval split is skipped by name rather than
                    # failing the run -- but it is announced, because a silently missing
                    # component is a metric that looks fine by being absent.
                    if is_main:
                        print(f"[eval] component {spec.name!r} has no {args.eval_split!r} "
                              f"split ({type(exc).__name__}); it will not be evaluated")
            if not eval_ds:
                raise SystemExit(
                    "--eval-per-component was given but no mixture component has a "
                    f"{args.eval_split!r} split"
                )
            if is_main:
                print("[eval] per component: "
                      + ", ".join(f"{k} n={len(v)}" for k, v in eval_ds.items()))
        else:
            eval_ds = base.load_split(
                args.eval_dataset or args.train_dataset, args.eval_split,
                args.eval_max_samples,
            )
    chunk_shape = base.infer_target_shape(train_ds, args.target_column)   # (T, 3)
    if chunk_shape[-1] != 3:
        raise ValueError(f"expected a 3-dim (dx,dy,dtheta) chunk, got shape {chunk_shape}")
    n_ticks = chunk_shape[0]
    if is_main:
        print(f"Train rows: {len(train_ds)}  chunk {chunk_shape}  n_ticks={n_ticks}"
              + (f"  eval rows: {len(eval_ds)}" if eval_ds is not None else ""))

    # ---- model ---------------------------------------------------------------------
    processor = AutoProcessor.from_pretrained(args.model_id)
    # --- PROBE --- build through the subclass; identical when the probe stays None
    model = TurnFlowActionRegressorWithProbe.build(
        model_cfg, loss_cfg, lora, n_ticks, processor,
        context_dim=args.context_dim, decoder_kwargs=args.decoder_kwargs,
        fm_cfg=args.fm_cfg, dtype=torch.bfloat16,
        latent_cfg=LatentConfig(
            enabled=bool(args.latent_cvae), sigma0=args.latent_sigma0,
            beta=args.latent_beta, posterior_width=args.latent_posterior_width,
            aux_weight=args.latent_aux_weight, decoder_kind=args.latent_decoder,
            diversity_weight=args.latent_diversity,
            diversity_clamp=args.latent_diversity_clamp,
            diversity_steps=args.latent_diversity_steps,
            diversity_probe=args.latent_diversity_probe,
            sigma_floor=args.latent_sigma_floor,
            diversity_target=args.latent_diversity_target,
            diversity_lr=args.latent_diversity_lr,
            free_bits=args.latent_free_bits, kl_warmup_steps=args.latent_kl_warmup,
            reinit_decoder=args.latent_reinit_decoder,
        ),
    )
    # The rotation-flip deadband is a SPEED (rad/s), so the per-tick threshold needs this
    # corpus's tick duration -- v1 is 20 Hz (dt=0.05), v2 25 Hz (dt=0.04). Read it off the
    # data rather than assuming, so the logged flip rate is the same statistic the AR head
    # and objectnav_eval's CoherenceProbe report for the same policy.
    row0 = train_ds[0]
    dt_native = row0.get("dt_native")
    if not dt_native and row0.get("native_fps"):
        dt_native = 1.0 / float(row0["native_fps"])
    model.dt_native = float(dt_native or DEFAULT_DT_NATIVE)

    # --- PROBE ---
    if args.probe:
        if args.probe_gamma is None:
            raise SystemExit("--probe requires --probe-gamma (no default: it must match "
                             "the dataset's return_gamma stamp)")
        stamped_gamma = row0.get("return_gamma")
        if stamped_gamma is None:
            raise SystemExit("--probe: the dataset has no return_gamma column; format "
                             "it with --distance-column/--return-gamma first")
        if abs(float(stamped_gamma) - float(args.probe_gamma)) > 1e-9:
            raise SystemExit(f"--probe-gamma {args.probe_gamma} != dataset return_gamma "
                             f"{stamped_gamma}; the value head's targets would not mean "
                             "what its config claims")
        probe_cfg = StateProbeConfig(
            readout_offset=args.probe_offset, grad_scale=args.probe_grad_scale)
        probe_cfg.distance["loss_weight"] = args.probe_distance_weight
        probe_cfg.value["loss_weight"] = args.probe_value_weight
        probe_cfg.value["gamma"] = float(args.probe_gamma)
        _hs = int(model.backbone.config.text_config.hidden_size
                  if hasattr(model.backbone.config, "text_config")
                  else model.backbone.config.hidden_size)
        model.state_probe = StateProbe(_hs, probe_cfg, dtype=_torch.float32)
        if is_main:
            print(f"[probe] offset {args.probe_offset}, grad_scale "
                  f"{args.probe_grad_scale}, gamma {args.probe_gamma}, hidden {_hs}")
    # --- END PROBE ---

    # The encoders only, on every rank and before the optimizer exists. Everything else --
    # the readout MLP, the velocity field, the adapter -- stays fresh, which is what makes
    # this a head comparison rather than a continuation of the regression run. `--resume-from`
    # restores a whole training state and wins if both are given.
    # The whole model, on every rank and before the optimizer exists. This is the plain
    # "load a checkpoint" case -- the same weights `load_trainable` restores on a resume and
    # the same ones eval loads via `VectorRolloutPolicy.from_checkpoint` -- with the
    # optimizer, schedule, step counter and dataloader order left fresh. That is what makes
    # it a new run on new data rather than a continuation of the old one.
    if args.init_from and not args.resume_from:
        fresh = model.warm_start(args.init_from)
        if is_main:
            print(f"[init-from] loaded trainable weights from {args.init_from}")
            print(f"[init-from] left at fresh init: {fresh or 'nothing'}")
    elif args.init_from and args.resume_from and is_main:
        print(f"[init-from] ignored: --resume-from {args.resume_from} restores the full "
              "training state, including these weights")

    if args.init_modality_from and not args.resume_from:
        loaded = model.init_modality_from(args.init_modality_from)
        if is_main:
            print(f"[init-modality-from] loaded encoder(s) {loaded} from "
                  f"{args.init_modality_from}; head, velocity field and adapter are fresh")
            print("[init-modality-from] NOTE: a trained encoder gives up the zero-init "
                  "property -- step 0 is no longer invariant to the injected pose values, "
                  "so this run is not step-0-comparable with a no-pose baseline")
    elif args.init_modality_from and args.resume_from and is_main:
        print(f"[init-modality-from] ignored: --resume-from {args.resume_from} restores the "
              "full training state, including the encoders")

    # Freezing happens AFTER --init-from: `load_state_dict` writes into frozen tensors fine,
    # but the optimizer groups are built by filtering on requires_grad, so lowering it before
    # the warm start would be fine and lowering it after the optimizer exists would not.
    if args.latent_cvae and (args.latent_freeze_trunk or args.latent_freeze_decoder):
        frozen = []
        if args.latent_freeze_trunk:
            for name in ("backbone", "head", "modality_embedder"):
                for q in getattr(model, name).parameters():
                    q.requires_grad_(False)
                frozen.append(name)
        if args.latent_freeze_decoder:
            for q in model.decoder.parameters():
                q.requires_grad_(False)
            frozen.append("decoder")
        if is_main:
            live = sum(q.numel() for q in model.parameters() if q.requires_grad)
            print(f"[latent] froze {', '.join(frozen)}; {live/1e6:.2f}M parameters live")

    if is_main:
        print("[latent]", LatentConfig(
            enabled=bool(args.latent_cvae), sigma0=args.latent_sigma0,
            beta=args.latent_beta, posterior_width=args.latent_posterior_width).describe())
        fm, dk = args.fm_cfg, args.decoder_kwargs
        decoder_params = sum(p.numel() for p in model.decoder.parameters())
        print("Model:", model.trainable_parameter_report())
        print(f"Velocity field: {decoder_params:,} params  dt_native={model.dt_native}")
        print(f"Context: dim {args.context_dim} -reshape-> {dk['n_context_tokens']} "
              f"token(s) x {dk['d_model']}   blockwise-causal   "
              f"state_token={dk['use_incoming_motion']}")
        print(f"Flow: K={fm.k_samples} draws/turn/step  "
              f"time~Beta({fm.time_alpha},{fm.time_beta})*{fm.time_scale}+{fm.time_offset}  "
              f"stratified={fm.stratified_time}  antithetic={fm.antithetic_noise}")
        print(f"Sampler: {model.normalizer.describe()}  "
              f"metric_steps={fm.metric_inference_steps}")
        print("NOTE: no free_* metrics -- this head has ONE generation mode. Compare "
              "rmse_*/pose_rmse_* against the AR head's free_* family, never its "
              "teacher-forced table.")
        if model.modality_embedder:
            print("Modality embeddings:\n" + model.modality_embedder.describe())

    # The model's own spec list, not a second copy: which column feeds which marker has
    # exactly one definition, and `build()` has already registered the marker tokens on this
    # processor's tokenizer.
    specs = model_cfg.modality_specs
    # --- PROBE --- ProbeCollator only when co-training; otherwise the original
    _Coll = ProbeCollator if args.probe else TurnVectorCollator
    train_collator = _Coll(processor, data_cfg, train=True, seed=args.seed,
                                        modality_specs=specs)
    eval_collator = _Coll(processor, data_cfg, train=False, seed=args.seed,
                                       modality_specs=specs)

    if not args.no_preflight and is_main:
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        if args.resume_from:
            model.load_trainable(args.resume_from)
            print(f"[preflight] loaded trainable weights from {args.resume_from}")
        base.preflight(model, train_collator, train_ds, processor, model_cfg)

    # ---- trainer -------------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        run_name=args.run_name,
        report_to=report_to,
        seed=args.seed,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler,
        max_grad_norm=args.max_grad_norm,
        bf16=True,
        gradient_checkpointing=not args.no_grad_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=args.eval_steps,
        dataloader_num_workers=args.dataloader_workers,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        label_names=[],
        ddp_find_unused_parameters=False,
        average_tokens_across_devices=True,
        optim="adamw_torch",
    )

    # --- PROBE --- mixin only changes metric visibility; probe-off runs identically
    class _ProbeFlowTrainer(_ProbeTrainerMixin, FlowMatchingSFTTrainer):
        pass

    _TrainerCls = _ProbeFlowTrainer if args.probe else FlowMatchingSFTTrainer
    trainer = _TrainerCls(
        model=model,
        args=training_args,
        data_collator=train_collator,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=processor,
        data_config=data_cfg,
        eval_data_collator=eval_collator,
    )
    # Off unless asked for; the band sums are accumulated regardless, so this is logging.
    trainer.emit_motion_bands = bool(args.log_motion_bands)
    # --- PROBE ---
    if args.probe:
        trainer.add_callback(_SaveProbeCallback(model))
    optim_cls, optim_kwargs = trainer.get_optimizer_cls_and_kwargs(training_args)
    optim_kwargs.pop("lr", None)
    trainer.optimizer = optim_cls(
        build_optimizer_param_groups(model, args), lr=args.lr, **optim_kwargs
    )

    # Per-module grad/weight/activation metrics, on the run's existing logging path.
    # The parameter-group check above proves each module is in the optimizer; this
    # proves the optimizer is actually moving it.
    attach_model_metrics(trainer, args=args, verbose=is_main)

    if is_main:
        print(f"Effective batch: 1 x {args.grad_accum} accum x "
              f"{training_args.world_size} rank(s) = "
              f"{args.grad_accum * training_args.world_size} conversations/step")

    trainer.train(resume_from_checkpoint=args.resume_from)
    trainer.save_model(os.path.join(args.output_dir, "final"))
    # --- PROBE ---
    if args.probe and is_main:
        save_state_probe(os.path.join(args.output_dir, "final"),
                         getattr(model, "state_probe"))
    if torch.cuda.is_available() and is_main:
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
