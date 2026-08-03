"""
Train the autoregressive-over-ticks action head -- the candidate hedged against the
conditional normalizing flow, per `dump/overnight/PLAN.md` and
`dump/autoregressive_head/FINDINGS.md`.

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

`--modality-specs` and the `--stop-head` group come from `train_vector_sft.py`'s parser
like everything else, and now reach this head's `ModelConfig` and collators too; see
`longnav/utils/ar_action_head.py`. Both are inert unless passed.

`--init-from` vs `--resume-from`
--------------------------------
`--resume-from` continues a run: optimizer state, LR schedule, global step, RNG and
dataloader position all come back, and the config must describe the same model.
`--init-from` only borrows the *weights* -- train from step 0 with a fresh optimizer, and
let modules this run declares that the source checkpoint predates start at their own init.
That is the right tool for taking a trained no-pose run as the starting point for a
pose-injection one: the objective composition changes (a second loss term, a new encoder),
so continuing the old schedule and step counter would be describing a run that never
happened.
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
from longnav.utils.ar_action_head import (  # noqa: E402
    DEFAULT_DT_NATIVE, ARActionSFTTrainer, CausalActionDecoder, ChunkVQCodec,
    TurnARActionClassifier,
)
from longnav.utils.stop_head import StopHeadConfig  # noqa: E402
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
    # Same reason, and the same failure mode the base function documents: a module left
    # out of the groups is silently frozen, and for the modality encoders that is
    # indistinguishable from "the injection did not help" -- the experiment's conclusion.
    fresh += list(model.modality_embedder.parameters())
    if model.stop_head is not None:
        fresh += list(model.stop_head.parameters())
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
    pre.add_argument("--context-dim", type=int, default=None,
                     help="OUTPUT width of the readout MLP -- the MLP/decoder interface "
                          "(default 256). Distinct from --head-hidden-dims, which sizes "
                          "that MLP's INTERNAL layers and is where the only nonlinearity "
                          "lives. Ignored (and derived) under --direct-context. See the "
                          "'context pathway' section of ar_action_head.py")
    pre.add_argument("--direct-context", action="store_true",
                     help="drop the decoder's context_proj: the readout MLP emits "
                          "n_context_tokens * d_model directly and the decoder only "
                          "reshapes. Removes a redundant linear factor (~1M params at "
                          "8x128) and makes the width agreement structural instead of a "
                          "warning. --context-dim is derived, not accepted")
    pre.add_argument("--decoder-d-model", type=int, default=128)
    pre.add_argument("--decoder-layers", type=int, default=4)
    pre.add_argument("--decoder-heads", type=int, default=4)
    pre.add_argument("--decoder-ff", type=int, default=512)
    pre.add_argument("--decoder-dropout", type=float, default=0.1)
    pre.add_argument("--context-tokens", type=int, default=1,
                     help="split the trunk's context vector into this many decoder prefix "
                          "tokens. A single token caps the context at d_model dimensions "
                          "no matter how wide the trunk head is, so widening the head only "
                          "helps in combination with this. Set --context-dim to "
                          "context-tokens * decoder-d-model to make the split exactly "
                          "rank-preserving (e.g. 8 tokens x 128 -> --context-dim 1024)")
    pre.add_argument("--prefix-pose", action="store_true",
                     help="feed each tick the pose accumulated so far plus the previous "
                          "tick's differential. Adds no information (both are functions of "
                          "the prefix) but removes the SE(2) composition the decoder would "
                          "otherwise have to learn to do internally")
    pre.add_argument("--pose-scales", default="calibrated",
                     choices=("calibrated", "legacy"),
                     help="feature normalisation for --prefix-pose. 'calibrated' gives "
                          "each state feature ~unit variance on v2_25hz; 'legacy' is the "
                          "original single 0.1 constant, which left the accumulated-pose "
                          "features ~20x wider than the differential ones. Only 'legacy' "
                          "reproduces checkpoints trained before the split")
    pre.add_argument("--init-from", default=None,
                     help="WARM START from a checkpoint: load its trained weights into "
                          "this model and then train from step 0 with a fresh optimizer, "
                          "scheduler and dataloader order. Distinct from --resume-from, "
                          "which continues that run's schedule and step counter. Modules "
                          "this run declares that the checkpoint predates (a modality "
                          "encoder, a stop head) start at their own init; every shared "
                          "module loads strictly. --resume-from wins if both are given")
    pre.add_argument("--attn-mode", default="causal", choices=("causal", "markov"),
                     help="'markov' lets each tick attend only to the context slots and "
                          "itself, making the per-tick input a sufficient state. Requires "
                          "--prefix-pose, which is what makes that state sufficient")
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
    args.init_from = mine.init_from
    # Under --direct-context the interface width is not a free choice: the readout MLP
    # must emit exactly one d_model-wide vector per prefix token. Derive it rather than
    # letting a stale --context-dim silently disagree.
    derived = mine.context_tokens * mine.decoder_d_model
    if mine.direct_context:
        if mine.context_dim is not None and mine.context_dim != derived:
            raise SystemExit(
                f"--direct-context derives --context-dim as --context-tokens x "
                f"--decoder-d-model = {mine.context_tokens} x {mine.decoder_d_model} = "
                f"{derived}; drop the explicit --context-dim {mine.context_dim}"
            )
        args.context_dim = derived
    else:
        args.context_dim = mine.context_dim if mine.context_dim is not None else 256
    args.decoder_kwargs = dict(
        d_model=mine.decoder_d_model, n_layers=mine.decoder_layers,
        n_heads=mine.decoder_heads, dim_ff=mine.decoder_ff, dropout=mine.decoder_dropout,
        n_context_tokens=mine.context_tokens, use_prefix_pose=mine.prefix_pose,
        attn_mode=mine.attn_mode, direct_context=mine.direct_context,
        pose_scales=(CausalActionDecoder.POSE_SCALES_CALIBRATED
                     if mine.pose_scales == "calibrated"
                     else CausalActionDecoder.POSE_SCALES_LEGACY),
    )
    if mine.attn_mode == "markov" and not mine.prefix_pose:
        raise SystemExit(
            "--attn-mode markov without --prefix-pose: masking the token history leaves "
            "each tick with only its own code embedding and tick index, which is NOT a "
            "sufficient state (no accumulated pose). Pass --prefix-pose too."
        )
    if (not mine.direct_context and mine.context_tokens > 1
            and args.context_dim != derived):
        print(f"[warn] --context-dim {args.context_dim} != --context-tokens "
              f"{mine.context_tokens} x --decoder-d-model {mine.decoder_d_model}; the "
              f"context projection will not be rank-preserving across the split. "
              f"--direct-context makes this structural instead of advisory")
    if args.output_dir == "dump/vector_sft":
        raise SystemExit(
            "--output-dir defaults to the regression baseline's directory; pass an "
            "explicit one under dump/autoregressive_head/"
        )
    if args.wandb_project == "longnav-vector-sft":
        args.wandb_project = "longnav-autoregressive-head-study"
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
        # `--modality-specs` and the `--stop-head` group come from `base.parse_args()`
        # already (this script only pre-parses the head-shape flags); what was missing was
        # putting them on the ModelConfig. Both default to inert.
        modality_specs=base._modality_specs(args.modality_specs),
        stop_head=(
            StopHeadConfig(
                hidden_dims=base._csv(args.stop_hidden_dims, int),
                stop_grad=not args.no_stop_grad,
                loss_weight=args.stop_loss_weight,
                pos_weight=args.stop_pos_weight,
                temperature=args.stop_temperature,
                inference=args.stop_inference,
                threshold=args.stop_threshold,
            )
            if args.stop_head else None
        ),
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
    model = TurnARActionClassifier.build(
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

    # Warm start, on every rank and before the optimizer exists -- the whole point is that
    # only the *weights* come across. `trainer.train()` below is NOT told about this
    # directory, so the optimizer state, the LR schedule, the global step, the RNG and the
    # dataloader position are all fresh, which is what makes it a new run rather than a
    # continuation of one with a different objective composition.
    if args.init_from and not args.resume_from:
        fresh = model.warm_start(args.init_from)
        if is_main:
            print(f"[init-from] loaded trainable weights from {args.init_from}")
            print(f"[init-from] left at fresh init: {fresh or 'nothing'}")
    elif args.init_from and args.resume_from and is_main:
        print(f"[init-from] ignored: --resume-from {args.resume_from} restores the full "
              "training state, including these weights")

    if is_main:
        print("Model:", model.trainable_parameter_report())
        decoder_params = sum(p.numel() for p in model.decoder.parameters())
        dk = args.decoder_kwargs
        print(f"Codebook: {args.codebook}  n_codes={model.n_codes}  "
              f"decoder params: {decoder_params:,}  dt_native={model.dt_native}")
        route = "reshape (direct)" if dk["direct_context"] else "context_proj"
        print(f"Context: dim {args.context_dim} -{route}-> {dk['n_context_tokens']} "
              f"token(s) x {dk['d_model']}   prefix_pose={dk['use_prefix_pose']}  "
              f"attn={dk['attn_mode']}  pose_scales={tuple(dk['pose_scales'])}")
        if model.modality_embedder:
            print("Modality embeddings:\n" + model.modality_embedder.describe())
        if model.stop_head is not None:
            print(f"Stop head: {model_cfg.stop_head}")

    # The model's own spec list, not a second copy -- `build()` has already registered the
    # marker tokens on this processor's tokenizer. Stop labels are driven off the model
    # config rather than a second flag: the collator emits them exactly when the model has
    # somewhere to put them.
    specs = model_cfg.modality_specs
    stop_labels = model_cfg.stop_head is not None
    train_collator = TurnVectorCollator(processor, data_cfg, train=True, seed=args.seed,
                                        modality_specs=specs, stop_labels=stop_labels)
    eval_collator = TurnVectorCollator(processor, data_cfg, train=False, seed=args.seed,
                                       modality_specs=specs, stop_labels=stop_labels)

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
