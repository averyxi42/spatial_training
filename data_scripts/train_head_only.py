"""Train the action head, and only the head, on a frozen context. Minutes, not hours.

    python data_scripts/train_head_only.py \
        --store dump/head_only/ctx_v9_ck11400 \
        --output-dir dump/head_only/run_cnf_a --run-name cnf-headonly-a \
        --max-steps 4000

Why this exists is in `docs/HEAD_ONLY_TRAINING.md` and in
`dump/cnf_sigma_ablation/FINDINGS.md`. In one line: the CNF head has never trained
healthily end to end, the diagnosis is that the trunk's conditioning vector collapses,
and the way to test a diagnosis about the trunk is to hold the trunk fixed.

Comparability is the whole design constraint
--------------------------------------------
Every number this emits must be readable against the same-named number from a real
`train_flow_sft.py` run. So none of it is written here:

  * the objective is `flow_head.FlowObjectiveMixin.flow_objective`, the same method
    `TurnFlowPolicy.forward` calls;
  * the metrics are `FlowSFTTrainer._drain_metrics`, inherited unchanged, which is what
    produces `eval_nll_per_dim`, `eval_nll_headroom`, `eval_logdet_absmax`,
    `eval_scale_saturation`, `eval_creep_*`, `eval_rmse_*` and `eval_mae_*`;
  * `loss` and `grad_norm` are HF `Trainer`'s;
  * `attach_model_metrics` adds the same per-module observability every other entry
    point has.

What is added is the collapse diagnostics (`eval_ctx_*`, `eval_w_ctx_over_w_x`), because
they are the reason the pipeline exists, and the `eval_in_ctx_*` versions of the same
statistics computed on the *harvested* vector -- which is constant by construction and is
therefore the control that says what a good context looks like on this corpus.

The defaults mirror v12 (`dump/pose_injection/v12_launch.sh`) so the comparison is a
comparison: context-dim 1024, head-hidden-dims 1024, flow 12x256 depth 2, noise-std 3e-5,
sigma-start 100 over 1000 steps, head-lr 1e-4, flow-lr 3e-4.

One difference is deliberate and is not free to change silently: v12 ran under bf16
autocast because a 2B backbone had to. There is no backbone here, so the head runs in
fp32 by default and `--bf16` restores the exact numerical path. The flow itself is fp32
in both (`flow_objective` disables autocast around it), so the objective is unaffected
either way; only the head's projection differs.

Other heads
-----------
The context is trunk output and therefore head-independent, so the same store trains any
head. `HEAD_BUILDERS` is the seam: an entry supplies a module exposing `encode(context)`,
`forward(context, targets, num_turns, num_items_in_batch) -> {"loss", ...}` and
`save_pretrained`, plus its own metric drain if it has one. Only the CNF is implemented
-- it is the head under investigation -- and a flow-matching or AR entry would reuse the
store, the trainer and the diagnostics unchanged.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "src"), str(_ROOT / "data_scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import TrainingArguments  # noqa: E402

from longnav.utils.head_only import (  # noqa: E402
    MOTION_FILTERS, ContextStore, FrozenContextCollator, FrozenContextDataset,
    FrozenContextFlowPolicy, HeadOnlyFlowTrainer, context_collapse_stats,
    filter_rows_by_motion,
)
from longnav.utils.model_metrics import (  # noqa: E402
    add_model_metrics_args, attach_model_metrics,
)
from longnav.utils.vector_sft import DataConfig  # noqa: E402


def build_cnf(store, args):
    """The CNF head, sized off the store's provenance rather than off flags.

    `pooled_dim` and `chunk_shape` come from the artifact, never from the command line:
    they are facts about the trunk that produced it, and a flag would only be a way to
    disagree with them.
    """
    prov = store.provenance
    identity = args.context_key == "context"
    return FrozenContextFlowPolicy(
        pooled_dim=prov.context_dim if identity else prov.pooled_dim,
        context_dim=prov.context_dim if identity else args.context_dim,
        chunk_shape=prov.chunk_shape,
        head_hidden_dims=_csv(args.head_hidden_dims, int),
        head_dropout=args.head_dropout,
        flow_kwargs={
            "n_layers": args.flow_layers, "hidden": args.flow_hidden,
            "depth": args.flow_depth, "s_max": args.flow_s_max,
            "noise_std": (args.noise_std,) * int(prov.chunk_shape[1]),
        },
        decode_k=args.decode_k,
        identity_head=identity,
    )


HEAD_BUILDERS = {"cnf": build_cnf}


def _csv(s, cast=str):
    return tuple(cast(x) for x in str(s).split(",") if x.strip()) if s else ()


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--store", required=True, nargs="+",
                    help="one or more harvest_context.py output directories. Several "
                         "parts of one parallel harvest are joined; they must agree on "
                         "the source trunk and corpus, which is checked")
    ap.add_argument("--store-shards", default=None,
                    help="pin each --store part to its first N shards, e.g. '11,9,11,9'. "
                         "A harvest is appendable, so without a pin the same store grows "
                         "between runs and split_by_episode reshuffles -- a later run then "
                         "trains on rows an earlier one held out, silently")
    ap.add_argument("--head", default="cnf", choices=sorted(HEAD_BUILDERS))
    ap.add_argument("--context-key", default="pooled", choices=("pooled", "context"),
                    help="'pooled' trains the projection AND the flow (the real trainer's "
                         "trainable set minus LoRA); 'context' freezes the source head "
                         "too and trains only the flow")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--run-name", default=None)

    g = ap.add_argument_group("head shape (v12's, so the comparison is a comparison)")
    g.add_argument("--context-dim", type=int, default=1024)
    g.add_argument("--head-hidden-dims", default="1024")
    g.add_argument("--head-dropout", type=float, default=0.0)
    g.add_argument("--flow-layers", type=int, default=12)
    g.add_argument("--flow-hidden", type=int, default=256)
    g.add_argument("--flow-depth", type=int, default=2)
    g.add_argument("--flow-s-max", type=float, default=5.0)
    g.add_argument("--noise-std", type=float, default=3e-5)
    g.add_argument("--decode-k", type=int, default=16)
    g.add_argument("--init-flow", default=None,
                   help="warm-start the density from a standalone marginal fit, exactly "
                        "as train_flow_sft.py's --init-flow does")

    o = ap.add_argument_group("optimisation")
    o.add_argument("--max-steps", type=int, default=4000)
    o.add_argument("--batch-size", type=int, default=256,
                   help="turns per step. v12 saw ~250 chunks/step from one conversation, "
                        "so 256 keeps the gradient's sample size comparable")
    o.add_argument("--head-lr", type=float, default=1e-4)
    o.add_argument("--flow-lr", type=float, default=3e-4)
    o.add_argument("--weight-decay", type=float, default=1e-4)
    o.add_argument("--warmup-ratio", type=float, default=0.01)
    o.add_argument("--lr-scheduler", default="cosine")
    o.add_argument("--max-grad-norm", type=float, default=1.0)
    o.add_argument("--sigma-start", type=float, default=100.0)
    o.add_argument("--sigma-anneal-steps", type=int, default=None,
                   help="default 1000 cold, 0 with --init-flow, matching train_flow_sft.py")
    o.add_argument("--gate-saturation", type=float, default=0.05)
    o.add_argument("--gate-logdet-slack", type=float, default=1.5)
    o.add_argument("--gate-patience", type=int, default=20)
    o.add_argument("--bf16", action="store_true",
                   help="run the head under bf16 autocast, as a real run does. Off by "
                        "default: there is no backbone forcing it here and fp32 is both "
                        "cheaper to reason about and free")
    o.add_argument("--seed", type=int, default=42)

    f = ap.add_argument_group("support filtering (see head_only.chunk_motion_mask)")
    f.add_argument("--target-filter", default="none", choices=MOTION_FILTERS,
                   help="drop chunks so the density is fitted on a support without the "
                        "zero atom. Applied AFTER the episode split, so every arm holds "
                        "out the same episodes and only the rows differ")
    f.add_argument("--filter-tol", type=float, default=1e-4,
                   help="'in the atom' threshold, in metres/radians per tick. The default "
                        "is the band statistics' EXACT, so the filter and the metric use "
                        "one definition of a stopped tick")
    f.add_argument("--filter-eval", dest="filter_eval", action="store_true", default=True,
                   help="filter the val split the same way (default): a run's own eval is "
                        "then on the support it was fitted on")
    f.add_argument("--no-filter-eval", dest="filter_eval", action="store_false",
                   help="keep the full val split, so eval_* is on the unfiltered "
                        "distribution and is directly comparable to a baseline's -- at the "
                        "cost of scoring the model off its own support")

    d = ap.add_argument_group("data / logging")
    d.add_argument("--val-frac", type=float, default=0.1)
    d.add_argument("--test-frac", type=float, default=0.0)
    d.add_argument("--eval-steps", type=int, default=250)
    d.add_argument("--save-steps", type=int, default=1000)
    d.add_argument("--save-total-limit", type=int, default=3)
    d.add_argument("--logging-steps", type=int, default=25)
    d.add_argument("--probe-rows", type=int, default=4096,
                   help="held-out rows the collapse diagnostics are computed on")
    d.add_argument("--dataloader-workers", type=int, default=2)
    d.add_argument("--wandb-project", default="longnav-head-only")
    d.add_argument("--no-wandb", action="store_true")
    d.add_argument("--resume-from", default=None)
    add_model_metrics_args(ap)

    args = ap.parse_args()
    if args.sigma_anneal_steps is None:
        args.sigma_anneal_steps = 0 if args.init_flow else 1000
    args.run_name = args.run_name or Path(args.output_dir).name
    return args


def build_param_groups(model, args):
    """Head and flow as two groups -- `train_flow_sft.build_param_groups` minus adapters.

    The flow keeps its own larger rate and zero weight decay for the reasons stated
    there: it is fresher and smaller than the projection, and decaying a density's
    parameters biases it toward the base measure.
    """
    head = [p for p in model.head.parameters() if p.requires_grad]
    flow = [p for p in model.normalizer.parameters() if p.requires_grad]
    groups = []
    if head:
        groups.append({"params": head, "lr": args.head_lr,
                       "weight_decay": args.weight_decay})
    if flow:
        groups.append({"params": flow, "lr": args.flow_lr, "weight_decay": 0.0})
    return groups


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    pins = _csv(args.store_shards, int) or None
    store = ContextStore.open_many(args.store, pins)
    prov = store.provenance
    if pins:
        print(f"store pinned to the first {list(pins)} shard(s) of each part")
    print(f"store {' + '.join(args.store)}")
    print(f"  {len(store)} rows over {store.n_episodes} episode(s), "
          f"pooled {prov.pooled_dim} context {prov.context_dim} chunk {prov.chunk_shape}")
    print(f"  from {prov.source_checkpoint} (step {prov.source_step}) "
          f"sha256 {prov.source_hash[:16]}...")
    print(f"  corpus {prov.corpus}[{prov.split}]  modality "
          f"{[s.get('token') for s in prov.modality_specs] or 'none'}")

    tr, va, te = store.split_by_episode(args.val_frac, args.test_frac, seed=args.seed)
    if len(va) == 0:
        raise SystemExit("the val split is empty -- raise --val-frac or harvest more "
                         "episodes; a run with no held-out number is not a measurement")
    print(f"  split by episode: train {len(tr)} / val {len(va)} / test {len(te)} rows")

    # After the split, never before: the episodes held out must not depend on the filter,
    # or two arms would differ in *which* episodes they were scored on as well as in which
    # rows they were fitted on, and the difference between them would be uninterpretable.
    n_tr_all, n_va_all = len(tr), len(va)
    if args.target_filter != "none":
        tr = filter_rows_by_motion(store, tr, args.target_filter, args.filter_tol)
        if args.filter_eval:
            va = filter_rows_by_motion(store, va, args.target_filter, args.filter_tol)
        if len(tr) == 0 or len(va) == 0:
            raise SystemExit(
                f"filter {args.target_filter!r} at tol {args.filter_tol} leaves "
                f"train {len(tr)} / val {len(va)} rows -- nothing to fit"
            )
        print(f"  filter {args.target_filter!r} (tol {args.filter_tol:g}): "
              f"train {len(tr)}/{n_tr_all} ({len(tr) / n_tr_all:.1%}) / "
              f"val {len(va)}/{n_va_all} ({len(va) / n_va_all:.1%}) rows kept")
        print("  NOTE: nll_* are a density on this support and are NOT comparable to a "
              "run fitted on another one. The band and error statistics are.")

    # The control the whole experiment leans on, printed before anything trains: what the
    # frozen vector's own statistics are. If these already look collapsed, nothing the
    # head does downstream can be read as evidence about the head.
    frozen = store.get(args.context_key, va[: min(len(va), 8192)])
    print("  frozen context (val rows): " + "  ".join(
        f"{k}={v:.4g}" for k, v in context_collapse_stats(frozen).items()))

    model = HEAD_BUILDERS[args.head](store, args)
    if args.init_flow:
        from longnav.utils.flow_head import warm_start_from_marginal

        rep = warm_start_from_marginal(model.flow, args.init_flow)
        print(f"  warm start from {rep['source']}: {rep['copied']} copied, "
              f"{rep['context_padded']} context-padded, {len(rep['skipped'])} skipped")
    n_head = sum(p.numel() for p in model.head.parameters())
    n_flow = sum(p.numel() for p in model.flow.parameters())
    print(f"head: {n_head:,} params ({'identity' if model.identity_head else 'trained'}), "
          f"flow: {n_flow:,} params, {args.flow_layers} couplings")
    print(f"log-determinant budget from the noise floor: "
          f"{float(torch.log(model.flow.unit_scale / model.flow.noise_std_raw).sum()):.1f} "
          f"nats/chunk;  min NLL/dim at sigma {model.flow.min_nll_per_dim(1.0):.3f} "
          f"(step 0: {model.flow.min_nll_per_dim(args.sigma_start):.3f})")

    train_ds = FrozenContextDataset(store, tr, args.context_key)
    eval_ds = FrozenContextDataset(store, va, args.context_key)
    collator = FrozenContextCollator(store, args.context_key)

    report_to = [] if args.no_wandb else ["wandb"]
    if report_to:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        os.environ.setdefault("WANDB_MODE", "online")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        run_name=args.run_name,
        report_to=report_to,
        seed=args.seed,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        max_steps=args.max_steps,
        learning_rate=args.head_lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler,
        max_grad_norm=args.max_grad_norm,
        bf16=args.bf16,
        gradient_checkpointing=False,    # nothing here is deep enough to want it
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        dataloader_num_workers=args.dataloader_workers,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        label_names=[],
        ddp_find_unused_parameters=False,
        average_tokens_across_devices=True,
        optim="adamw_torch",
    )

    trainer = HeadOnlyFlowTrainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        # `target_dim_names` is what turns `rmse_0/1/2` into `rmse_dx/dy/dtheta`, i.e.
        # what makes the keys match a real run's rather than merely correspond to them.
        data_config=DataConfig(target_dim_names=("dx", "dy", "dtheta")),
        sigma_start=args.sigma_start,
        sigma_anneal_steps=args.sigma_anneal_steps,
        gate_saturation=args.gate_saturation,
        gate_logdet_slack=args.gate_logdet_slack,
        gate_patience=args.gate_patience,
        probe_store=store,
        probe_rows=va,
        probe_context_key=args.context_key,
        probe_max_rows=args.probe_rows,
    )
    optim_cls, optim_kwargs = trainer.get_optimizer_cls_and_kwargs(training_args)
    optim_kwargs.pop("lr", None)
    trainer.optimizer = optim_cls(
        build_param_groups(model, args), lr=args.head_lr, **optim_kwargs
    )
    attach_model_metrics(trainer, args=args, verbose=True)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # The provenance travels with the run, not just with the store: a head checkpoint
    # whose store has been deleted or moved must still be able to name its trunk, since
    # that is what `reattach_head.py` checks against.
    (Path(args.output_dir) / "head_only_run.json").write_text(json.dumps({
        "args": vars(args), "provenance": prov.to_json(),
        "n_train_rows": int(len(tr)), "n_val_rows": int(len(va)),
        "n_train_rows_unfiltered": int(n_tr_all), "n_val_rows_unfiltered": int(n_va_all),
        "n_test_rows": int(len(te)), "n_episodes": int(store.n_episodes),
    }, indent=2))

    t0 = time.time()
    trainer.train(resume_from_checkpoint=args.resume_from)
    el = time.time() - t0
    trainer.save_model(os.path.join(args.output_dir, "final"))
    print(f"\n{args.max_steps} steps in {el:.0f}s = {args.max_steps / max(el, 1e-9):.2f} "
          f"steps/s ({el / max(1, args.max_steps) * 1000:.1f} ms/step). "
          f"A full train_flow_sft.py step is ~4 s.")
    if torch.cuda.is_available():
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
