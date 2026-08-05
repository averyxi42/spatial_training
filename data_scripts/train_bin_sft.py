"""
Train the discrete-bin cross-entropy head -- the CONTROL for the flow work.

This is not a candidate head. It exists to answer one question: does *any* distributional
head stop the policy creeping, or is the conditional normalizing flow specifically needed?
See `longnav/utils/bin_head.py` for the design and `dump/bin_head_control/FINDINGS.md` for
the answer.

Being a control is the whole point, so everything that is not the output parameterization
is imported from `train_vector_sft.py` rather than re-implemented: the same argument
parser defaults, the same dataset loader, the same optimizer param groups, the same
preflight. Run it with the regression baseline's flags and the only difference between the
two runs is cross-entropy over bins instead of Huber over reals.

The baseline (`dump/vector_sft`, the sweep in `dump/data_diagnostics/`) was:

    python data_scripts/train_vector_sft.py \
        --train-dataset data/continuous_sft_formatted \
        --max-steps 20000 --grad-accum 1 --max-turns 400 \
        --dataloader-workers 8 --no-target-norm

so the matched control is:

    python data_scripts/train_bin_sft.py \
        --train-dataset data/continuous_sft_formatted \
        --bin-edges dump/bin_head_control/bin_edges_65.json \
        --max-steps 20000 --grad-accum 1 --max-turns 400 \
        --dataloader-workers 8 --output-dir dump/bin_head_control/run_bins65

Bin edges are fitted offline by `dump/bin_head_control/fit_bins.py` and passed in, not
fitted here: they are gated on their own (a round trip through them must preserve the
data's exact-stop mass) before any GPU time is spent, and a run whose edges moved would
not be comparable to another run's.
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
from longnav.utils.bin_codec import BinCodec  # noqa: E402
from longnav.utils.bin_head import BinSFTTrainer, TurnBinClassifier  # noqa: E402
from longnav.utils.vector_sft import (  # noqa: E402
    DataConfig, LossConfig, LoraSpec, ModelConfig, TurnVectorCollator,
)


def parse_args():
    """The baseline's own parser, plus `--bin-edges`.

    `train_vector_sft.parse_args` builds and parses in one call, so the bin-only flag is
    stripped from `sys.argv` first and the remainder is handed to it verbatim. Borrowing
    the parser rather than restating it is what keeps this a control: every optimizer,
    data and LoRA default is literally the baseline's, and a change there lands in both.
    """
    pre = base.argparse.ArgumentParser(add_help=False)
    pre.add_argument("--bin-edges", default=None,
                     help="JSON written by dump/bin_head_control/fit_bins.py")
    mine, rest = pre.parse_known_args()

    old_argv, sys.argv = sys.argv, [sys.argv[0], *rest]
    try:
        args = base.parse_args()      # prints the baseline's --help if asked
    finally:
        sys.argv = old_argv

    if not mine.bin_edges:
        raise SystemExit("--bin-edges is required (see dump/bin_head_control/fit_bins.py)")
    args.bin_edges = mine.bin_edges
    # The baseline's defaults point at the baseline's artifacts. Inheriting them would
    # write bin checkpoints into `dump/vector_sft/`, on top of the run this is a control
    # for, so they are refused rather than silently redirected.
    if args.output_dir == "dump/vector_sft":
        raise SystemExit(
            "--output-dir defaults to the regression baseline's directory; pass an "
            "explicit one under dump/bin_head_control/"
        )
    if args.wandb_project == "longnav-vector-sft":
        args.wandb_project = "longnav-bin-head-control"
    if args.loss != "huber":
        raise SystemExit("--loss does not apply: this head's objective is cross-entropy")
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
    )
    data_cfg = DataConfig(
        target_column=args.target_column,
        messages_column=args.messages_column,
        images_column=args.images_column,
        max_turns_per_sample=args.max_turns or None,
        target_dim_names=base._csv(args.target_dim_names) or ("dx", "dy", "dtheta"),
    )
    # `kind` is recorded in the checkpoint for provenance; the objective is fixed.
    loss_cfg = LossConfig(kind="cross_entropy", normalize_targets=False)
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
    edges = BinCodec.from_json(args.bin_edges)
    target_shape = (*chunk_shape, edges.n_bins)
    if chunk_shape[-1] != edges.n_dims:
        raise ValueError(
            f"chunk has {chunk_shape[-1]} dimension(s) but the bin edges were fitted for "
            f"{edges.n_dims}"
        )
    if is_main:
        print(f"Train rows: {len(train_ds)}  chunk {chunk_shape} x {edges.n_bins} bins "
              f"-> head out_dim {target_shape[0] * target_shape[1] * target_shape[2]}"
              + (f"  eval rows: {len(eval_ds)}" if eval_ds is not None else ""))

    # ---- model ---------------------------------------------------------------------
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = TurnBinClassifier.build(
        model_cfg, loss_cfg, lora, target_shape, processor, dtype=torch.bfloat16
    )
    model.codec.load_state_dict(edges.state_dict())
    if is_main:
        print("Model:", model.trainable_parameter_report())
        print(f"Bin edges: {args.bin_edges}  zero_tol={float(model.codec.zero_tol):g}")

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

    trainer = BinSFTTrainer(
        model=model,
        args=training_args,
        data_collator=train_collator,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=processor,
        data_config=data_cfg,
        eval_data_collator=eval_collator,
    )
    optim_cls, optim_kwargs = trainer.get_optimizer_cls_and_kwargs(training_args)
    optim_kwargs.pop("lr", None)
    trainer.optimizer = optim_cls(
        base.build_optimizer_param_groups(model, args), lr=args.lr, **optim_kwargs
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
