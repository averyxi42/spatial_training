"""
Train the conditional normalizing flow head -- the candidate replacement for regression.

Everything that is not the output parameterization is imported from
`train_vector_sft.py`: the same argument parser defaults, the same dataset loader, the
same preflight. Run it with the regression baseline's flags and the only difference
between the two runs is a conditional density fitted by maximum likelihood instead of
Huber against the conditional mean. See `longnav/utils/flow_head.py` for the wiring and
`longnav/cnf_head/flow.py` for the flow itself.

The regression baseline (`dump/vector_sft`, the sweep in `dump/data_diagnostics/`) was:

    python data_scripts/train_vector_sft.py \
        --train-dataset data/continuous_sft_formatted \
        --max-steps 20000 --grad-accum 1 --max-turns 400 \
        --dataloader-workers 8 --no-target-norm

and the bin-head control (`dump/bin_head_control/run_trunk64`) matched it, so the matched
flow run is:

    python data_scripts/train_flow_sft.py \
        --train-dataset data/continuous_sft_formatted \
        --max-steps 20000 --grad-accum 1 --max-turns 400 \
        --dataloader-workers 8 --head-hidden-dims 64 \
        --output-dir dump/cnf_head/flow/run_ctx128 --run-name flow-ctx128-matched

Same backbone, same LoRA, same optimizer, same 64-wide conditioning trunk, same 250-step
checkpoints. What the trunk emits changes -- a context vector rather than an action -- and
the loss becomes NLL per dimension in nats.

Two flags carry the design decisions from `flow.py`'s docstring and should not be moved
casually. `--noise-std 1e-5` is the measured width of the data's own float-residue cloud
and is band-statistic invisible (forward exact-stop 63.1% -> 63.1%); 1e-4 would push a
fifth of the stopped ticks into the creep band and corrupt the metric this whole project
is about. `--sigma-start` anneals that floor down from a wider one over the first steps,
because early optimization is where a flow on atomic data blows up.
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
from longnav.utils.model_metrics import attach_model_metrics  # noqa: E402
from longnav.utils.flow_head import FlowSFTTrainer, TurnFlowPolicy  # noqa: E402
from longnav.utils.vector_sft import (  # noqa: E402
    DataConfig, LossConfig, LoraSpec, ModelConfig, TurnVectorCollator,
)


def parse_args():
    """The baseline's own parser, plus the flow's flags.

    `train_vector_sft.parse_args` builds and parses in one call, so the flow-only flags are
    stripped from `sys.argv` first and the remainder is handed to it verbatim -- the same
    trick `train_bin_sft.py` uses, and for the same reason: every optimizer, data and LoRA
    default is literally the baseline's, so a change there lands in all three heads.
    """
    pre = base.argparse.ArgumentParser(add_help=False)
    f = pre.add_argument_group("flow")
    f.add_argument("--context-dim", type=int, default=128,
                   help="width of the conditioning vector the trunk emits")
    f.add_argument("--flow-layers", type=int, default=12)
    f.add_argument("--flow-hidden", type=int, default=256)
    f.add_argument("--flow-depth", type=int, default=2)
    f.add_argument("--flow-s-max", type=float, default=5.0,
                   help="bound on a coupling's log-scale; the runaway guard")
    f.add_argument("--flow-lr", type=float, default=3e-4,
                   help="the flow is fresh and small, so it wants a larger step than the "
                        "adapters; it shares the head's group otherwise")
    f.add_argument("--noise-std", type=float, default=3e-5,
                   help="raw-unit noise floor per channel. See the module docstring "
                        "before changing this")
    f.add_argument("--init-flow", default=None,
                   help="warm-start the density from a standalone marginal fit "
                        "(dump/cnf_head/flow/pretrain_*.pt), the way the bin control "
                        "fits its edges offline")
    f.add_argument("--gate-saturation", type=float, default=0.05)
    f.add_argument("--gate-logdet-slack", type=float, default=1.5)
    f.add_argument("--gate-patience", type=int, default=20)
    f.add_argument("--sigma-start", type=float, default=100.0,
                   help="noise-floor multiplier at step 0, annealed to 1")
    f.add_argument("--sigma-anneal-steps", type=int, default=None,
                   help="default 1000 from a cold start, and 0 with --init-flow: the "
                        "anneal exists to stabilise early optimisation, which a warm "
                        "start has already done, and starting wide would only blur a "
                        "density that is already at the right sharpness")
    f.add_argument("--decode-k", type=int, default=16,
                   help="samples per context for the best-of-k committed readout")
    mine, rest = pre.parse_known_args()

    old_argv, sys.argv = sys.argv, [sys.argv[0], *rest]
    try:
        args = base.parse_args()      # prints the baseline's --help if asked
    finally:
        sys.argv = old_argv

    for k, v in vars(mine).items():
        setattr(args, k, v)
    if args.sigma_anneal_steps is None:
        args.sigma_anneal_steps = 0 if args.init_flow else 1000
    # The baseline's defaults point at the baseline's artifacts; inheriting them would
    # write flow checkpoints on top of the run this is measured against.
    if args.output_dir == "dump/vector_sft":
        raise SystemExit("--output-dir defaults to the regression baseline's directory; "
                         "pass an explicit one under dump/cnf_head/flow/")
    if args.wandb_project == "longnav-vector-sft":
        args.wandb_project = "longnav-flow-head"
    if args.loss != "huber":
        raise SystemExit("--loss does not apply: this head's objective is the flow's NLL")
    return args


def build_param_groups(model, args):
    """Adapters, trunk and flow as three groups.

    The baseline's `build_optimizer_param_groups` splits head from adapters because a
    fresh head wants a larger step than pretrained weights; the flow is fresher and
    smaller still, and its gradients are an order of magnitude larger near the atom, so it
    gets its own group rather than inheriting either rate by accident.
    """
    head = [p for p in model.head.parameters() if p.requires_grad]
    flow = [p for p in model.normalizer.parameters() if p.requires_grad]
    seen = {id(p) for p in head} | {id(p) for p in flow}
    other = [p for p in model.parameters() if p.requires_grad and id(p) not in seen]
    groups = [{"params": other, "lr": args.lr, "weight_decay": args.weight_decay}]
    if head:
        groups.append({"params": head, "lr": args.head_lr or args.lr,
                       "weight_decay": args.weight_decay})
    if flow:
        groups.append({"params": flow, "lr": args.flow_lr,
                       "weight_decay": 0.0})   # weight decay on a density's parameters
    return groups                              # would bias it toward the base measure


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
    )
    data_cfg = DataConfig(
        target_column=args.target_column,
        messages_column=args.messages_column,
        images_column=args.images_column,
        max_turns_per_sample=args.max_turns or None,
        target_dim_names=base._csv(args.target_dim_names) or ("dx", "dy", "dtheta"),
    )
    loss_cfg = LossConfig(kind="flow_nll", normalize_targets=False)
    lora = None if args.no_lora else LoraSpec(
        r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout,
        target_modules=base._csv(args.lora_target_modules),
    )

    # ---- data ----------------------------------------------------------------------
    train_ds = base.load_split(args.train_dataset, args.train_split)
    eval_ds = None
    if args.eval_split:
        eval_ds = base.load_split(
            args.eval_dataset or args.train_dataset, args.eval_split, args.eval_max_samples
        )
    chunk_shape = base.infer_target_shape(train_ds, args.target_column)   # (T, 3)
    if is_main:
        print(f"Train rows: {len(train_ds)}  chunk {chunk_shape} -> flow over "
              f"{chunk_shape[0] * chunk_shape[1]} dims, context {args.context_dim}"
              + (f"  eval rows: {len(eval_ds)}" if eval_ds is not None else ""))

    # ---- model ---------------------------------------------------------------------
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = TurnFlowPolicy.build(
        model_cfg, loss_cfg, lora, (args.context_dim,), processor, dtype=torch.bfloat16,
        chunk_shape=chunk_shape,
        flow_kwargs={
            "n_layers": args.flow_layers, "hidden": args.flow_hidden,
            "depth": args.flow_depth, "s_max": args.flow_s_max,
            "noise_std": (args.noise_std,) * chunk_shape[1],
        },
        decode_k=args.decode_k,
    )
    if args.init_flow:
        from longnav.utils.flow_head import warm_start_from_marginal

        rep = warm_start_from_marginal(model.flow, args.init_flow)
        if is_main:
            print(f"Warm start from {rep['source']}: {rep['copied']} tensors copied, "
                  f"{rep['context_padded']} context-padded, "
                  f"{len(rep['skipped'])} skipped {rep['skipped'][:3]}")
            if rep["source_config"].get("noise_std", [None])[0] != args.noise_std:
                print(f"  !! marginal was fitted at noise_std="
                      f"{rep['source_config'].get('noise_std')} but this run uses "
                      f"{args.noise_std}; the density starts at the wrong sharpness")
    if is_main:
        print("Model:", model.trainable_parameter_report())
        print(f"Flow: {sum(p.numel() for p in model.flow.parameters()):,} params, "
              f"{args.flow_layers} coupling layers, s_max {args.flow_s_max}")
        print(f"Log-determinant budget from the noise floor: "
              f"{float(torch.log(model.flow.unit_scale / model.flow.noise_std_raw).sum()):.1f} "
              f"nats/chunk (gate fires above x{args.gate_logdet_slack})")
        print(f"Noise floor: sigma={args.noise_std:g} raw units, annealed from "
              f"{args.sigma_start}x over {args.sigma_anneal_steps} steps. "
              f"min NLL/dim at sigma: {model.flow.min_nll_per_dim(1.0):.3f} nats "
              f"(at step 0: {model.flow.min_nll_per_dim(args.sigma_start):.3f})")

    train_collator = TurnVectorCollator(processor, data_cfg, train=True, seed=args.seed)
    eval_collator = TurnVectorCollator(processor, data_cfg, train=False, seed=args.seed)

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

    trainer = FlowSFTTrainer(
        model=model,
        args=training_args,
        data_collator=train_collator,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=processor,
        data_config=data_cfg,
        eval_data_collator=eval_collator,
        sigma_start=args.sigma_start,
        sigma_anneal_steps=args.sigma_anneal_steps,
        gate_saturation=args.gate_saturation,
        gate_logdet_slack=args.gate_logdet_slack,
        gate_patience=args.gate_patience,
    )
    optim_cls, optim_kwargs = trainer.get_optimizer_cls_and_kwargs(training_args)
    optim_kwargs.pop("lr", None)
    trainer.optimizer = optim_cls(
        build_param_groups(model, args), lr=args.lr, **optim_kwargs
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
    if torch.cuda.is_available() and is_main:
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
