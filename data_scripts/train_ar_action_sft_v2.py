"""
Train the AR action head, VERSION 2 (`longnav.utils.ar_action_head_v2`).

Same objective, data, optimizer grouping and metrics as `train_ar_action_sft.py`; the only
difference is which decoder is built. See `ar_action_head_v2`'s docstring for what v2
changes and why each change is traceable to a measurement. v1 and its script stay frozen
so `run_v2_lloyd_ddp4` / `run_v3_ctx8_pose` remain reproducible.

v2 removes knobs rather than adding them: there is no --context-dim (derived), no
--direct-context (always), no --prefix-pose (the pose state is intrinsic) and no
--pose-scales legacy path.

Exactly the discrete-bin control's own pattern (`data_scripts/train_bin_sft.py`): every
argument that is not the output parameterization is borrowed from `train_vector_sft.py`'s
parser, dataset loader, optimizer param groups and preflight -- so this run differs from
the regression baseline (and from the bin-head control) only in what the head predicts and
how, not in data, backbone, LoRA config or optimizer.

    python data_scripts/train_ar_action_sft.py \
        --train-dataset data/continuous_sft_formatted \
        --codebook dump/autoregressive_head/codebook_256.json \
        --max-steps 6000 --grad-accum 1 --max-turns 400 \
        --dataloader-workers 8 --output-dir dump/autoregressive_head/run1

Codebook is fitted offline by `dump/autoregressive_head/fit_codebook.py` and passed in,
not fitted here -- gated on its own (round trip must preserve the data's exact-stop mass)
before any GPU time is spent, mirroring the bin head's `fit_bins.py` gate.
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
from longnav.utils.ar_action_head_v2 import (  # noqa: E402
    DEFAULT_DT_NATIVE, ARActionSFTTrainer, ChunkVQCodec, TurnARActionClassifierV2,
)
from longnav.utils.vector_sft import (  # noqa: E402
    DataConfig, LossConfig, LoraSpec, ModelConfig, TurnVectorCollator,
)


def build_optimizer_param_groups(model, args):
    """`base.build_optimizer_param_groups`, but with the fresh causal decoder grouped
    with the fresh trunk head under `--head-lr` rather than under the adapters' `--lr`.
    The decoder is exactly as randomly-initialized as `model.head` and exactly as small,
    so it wants the same larger step a fresh head usually does -- only the LoRA adapters
    on the pretrained backbone should move at the conservative rate. Reimplemented here
    (not editing `train_vector_sft.py`) precisely because that grouping choice is specific
    to this head having a second fresh module the base function does not know about.
    """
    fresh = list(model.head.parameters()) + list(model.decoder.parameters())
    fresh = [p for p in fresh if p.requires_grad]
    fresh_ids = {id(p) for p in fresh}
    other = [p for p in model.parameters() if p.requires_grad and id(p) not in fresh_ids]
    groups = [{"params": other, "lr": args.lr, "weight_decay": args.weight_decay}]
    if fresh:
        groups.append({
            "params": fresh,
            "lr": args.head_lr if args.head_lr is not None else args.lr,
            "weight_decay": args.weight_decay,
        })
    return groups


def parse_args():
    pre = base.argparse.ArgumentParser(add_help=False)
    pre.add_argument("--codebook", default=None,
                     help="JSON written by dump/autoregressive_head/fit_codebook.py")
    pre.add_argument("--decoder-d-model", type=int, default=128)
    pre.add_argument("--decoder-layers", type=int, default=4)
    pre.add_argument("--decoder-heads", type=int, default=4)
    pre.add_argument("--decoder-ff", type=int, default=512)
    pre.add_argument("--decoder-dropout", type=float, default=0.1)
    pre.add_argument("--context-tokens", type=int, default=8,
                     help="prefix tokens the readout MLP emits. A token is d_model wide, "
                          "so this -- not the MLP width -- sets how many dimensions of "
                          "context the decoder can receive. context_dim is DERIVED as "
                          "context-tokens * decoder-d-model; v2 has no projection to "
                          "reconcile a mismatch")
    pre.add_argument("--attn-mode", default="causal", choices=("causal", "markov"),
                     help="'markov': each tick attends to the context slots and itself "
                          "only. Legitimate in v2 because the pose state is always "
                          "present, so a tick's own input is a sufficient conditioning set")
    pre.add_argument("--incoming-motion", action="store_true",
                     help="reserve the tick-0 incoming-velocity slot + validity bit. The "
                          "dataset does not yet supply the value, so this currently only "
                          "changes the architecture; see ar_action_head_v2's docstring")
    mine, rest = pre.parse_known_args()

    old_argv, sys.argv = sys.argv, [sys.argv[0], *rest]
    try:
        args = base.parse_args()
    finally:
        sys.argv = old_argv

    if not mine.codebook:
        raise SystemExit(
            "--codebook is required (see dump/autoregressive_head/fit_codebook.py)"
        )
    args.codebook = mine.codebook
    # Derived, never accepted: v2 has no context_proj, so the readout MLP must emit
    # exactly one d_model-wide vector per prefix token.
    args.context_dim = mine.context_tokens * mine.decoder_d_model
    args.decoder_kwargs = dict(
        d_model=mine.decoder_d_model, n_layers=mine.decoder_layers,
        n_heads=mine.decoder_heads, dim_ff=mine.decoder_ff, dropout=mine.decoder_dropout,
        n_context_tokens=mine.context_tokens, attn_mode=mine.attn_mode,
        use_incoming_motion=mine.incoming_motion,
    )
    if args.output_dir == "dump/vector_sft":
        raise SystemExit(
            "--output-dir defaults to the regression baseline's directory; pass an "
            "explicit one under dump/autoregressive_head/"
        )
    if args.wandb_project == "longnav-vector-sft":
        args.wandb_project = "longnav-autoregressive-head-v2"
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
        target_dim_names=("dx", "dy", "dtheta"),  # what _band_metrics/trainer report on
    )
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
    if chunk_shape[-1] != 3:
        raise ValueError(f"expected a 3-dim (dx,dy,dtheta) chunk, got shape {chunk_shape}")
    n_ticks = chunk_shape[0]
    codebook = ChunkVQCodec.from_json(args.codebook)
    if is_main:
        print(f"Train rows: {len(train_ds)}  chunk {chunk_shape}  "
              f"n_ticks={n_ticks}  n_codes={codebook.n_codes}"
              + (f"  eval rows: {len(eval_ds)}" if eval_ds is not None else ""))

    # ---- model ---------------------------------------------------------------------
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = TurnARActionClassifierV2.build(
        model_cfg, loss_cfg, lora, n_ticks, processor,
        n_codes=codebook.n_codes, context_dim=args.context_dim,
        decoder_kwargs=args.decoder_kwargs, dtype=torch.bfloat16,
    )
    model.codec.load_state_dict(codebook.state_dict())
    # The rotation-flip deadband is a SPEED (rad/s), so the per-tick threshold needs this
    # corpus's tick duration -- v1 is 20 Hz (dt=0.05), v2 25 Hz (dt=0.04). Read it off the
    # data rather than assuming, so the logged flip rate is the same statistic
    # analyze_ar.py and objectnav_eval's CoherenceProbe report for the same policy.
    row0 = train_ds[0]
    dt_native = row0.get("dt_native")
    if not dt_native and row0.get("native_fps"):
        dt_native = 1.0 / float(row0["native_fps"])
    model.dt_native = float(dt_native or DEFAULT_DT_NATIVE)
    if is_main:
        print("Model:", model.trainable_parameter_report())
        decoder_params = sum(p.numel() for p in model.decoder.parameters())
        dk = args.decoder_kwargs
        print(f"Codebook: {args.codebook}  n_codes={model.n_codes}  "
              f"decoder params: {decoder_params:,}  dt_native={model.dt_native}")
        print(f"Context: dim {args.context_dim} -reshape-> {dk['n_context_tokens']} "
              f"token(s) x {dk['d_model']}   attn={dk['attn_mode']}  "
              f"incoming_motion={dk['use_incoming_motion']}  "
              f"scales={tuple(dk['scales'])}")

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

    trainer = ARActionSFTTrainer(
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
    if torch.cuda.is_available() and is_main:
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
