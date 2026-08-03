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
from longnav.utils.flow_matching_head import (  # noqa: E402
    ACTION_SCALES, DEFAULT_DT_NATIVE, FlowMatchingConfig, FlowMatchingSFTTrainer,
    NUM_INFERENCE_STEPS, TurnFlowActionRegressor,
)
from longnav.utils.vector_sft import (  # noqa: E402
    DataConfig, LossConfig, LoraSpec, ModelConfig, TurnVectorCollator,
)


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
    pre.add_argument("--decoder-d-model", type=int, default=128)
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
    mine, rest = pre.parse_known_args()

    old_argv, sys.argv = sys.argv, [sys.argv[0], *rest]
    try:
        args = base.parse_args()
    finally:
        sys.argv = old_argv

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
    if is_main:
        print(f"Train rows: {len(train_ds)}  chunk {chunk_shape}  n_ticks={n_ticks}"
              + (f"  eval rows: {len(eval_ds)}" if eval_ds is not None else ""))

    # ---- model ---------------------------------------------------------------------
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = TurnFlowActionRegressor.build(
        model_cfg, loss_cfg, lora, n_ticks, processor,
        context_dim=args.context_dim, decoder_kwargs=args.decoder_kwargs,
        fm_cfg=args.fm_cfg, dtype=torch.bfloat16,
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
    if is_main:
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

    trainer = FlowMatchingSFTTrainer(
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
