"""
Train a per-turn continuous-vector head + LoRA adapters on a conversational VLM.

This is the entrypoint; the machinery lives in `longnav.utils.vector_sft` (objective,
collation, checkpointing, metrics) and `longnav.utils.turn_vectors` (turn location and
the dense->sparse index remap). Read `vector_sft.py`'s module docstring first -- it
explains the span convention, the B == 1 constraint and the loss scaling.

The dataset is a standard HF conversational multimodal table plus one extra column of
per-assistant-turn regression targets:

    messages          [{"role": ..., "content": [{"type": "image"}, {"type": "text", ...}]}, ...]
    images            [PIL | path | HF Image, ...]  in placeholder order
    <target-column>   (n_assistant_turns, *target_shape) -- e.g. (n_turns, 10, 3) for a
                      10-step action chunk of (dx, dy, dtheta)

Single GPU:

    python data_scripts/train_vector_sft.py \
        --train-dataset ~/codes/habitat/continuous_demos/data/arrowtable \
        --train-split train --target-column action_chunks \
        --target-dim-names dx,dy,dtheta --max-turns 16 \
        --output-dir dump/vector_sft_run1

Multi-GPU (DDP; per-device batch is forced to 1, so scale with accumulation x ranks):

    torchrun --nproc_per_node=4 data_scripts/train_vector_sft.py \
        --train-dataset ... --grad-accum 8 --output-dir dump/vector_sft_run1

Resume:

    ... --resume-from dump/vector_sft_run1/checkpoint-500
"""

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for p in (str(_ROOT), str(_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

# Long, variable-length multimodal sequences fragment the allocator badly; this is the
# one env var worth setting before torch initializes its caching allocator.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from transformers import AutoProcessor, TrainingArguments

from longnav.utils.vector_sft import (
    DataConfig,
    LoraSpec,
    LossConfig,
    ModelConfig,
    TurnVectorCollator,
    TurnVectorRegressor,
    TurnVectorSFTTrainer,
    fit_target_normalizer,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Per-turn vector-regression SFT for sparse Qwen3-VL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    d = p.add_argument_group("data")
    d.add_argument("--train-dataset", required=True,
                   help="load_from_disk() directory or a hub dataset id")
    d.add_argument("--train-split", default="train")
    d.add_argument("--eval-dataset", default=None,
                   help="defaults to --train-dataset if --eval-split is given")
    d.add_argument("--eval-split", default=None)
    d.add_argument("--eval-max-samples", type=int, default=64)
    d.add_argument("--target-column", default="action_chunks")
    d.add_argument("--messages-column", default="messages")
    d.add_argument("--images-column", default="images")
    d.add_argument("--target-dim-names", default=None,
                   help="comma-separated names for the target's last dim, e.g. dx,dy,dtheta")
    d.add_argument("--max-turns", type=int, default=16,
                   help="turns per sample; random contiguous window at train, first-N at "
                        "eval. 0 disables the cap (watch the token budget)")

    m = p.add_argument_group("model / head")
    m.add_argument("--model-id", default="Qwen/Qwen3-VL-2B-Instruct")
    m.add_argument("--attn-impl", default="sdpa", choices=["sdpa", "flash_attention_2"])
    m.add_argument("--affixes", default="action", choices=["action", "template"])
    m.add_argument("--no-shift-left", action="store_true",
                   help="pool the assistant content instead of the position that precedes "
                        "it (only sensible when the content is real text, not a placeholder)")
    m.add_argument("--pool-mode", default="mean", choices=["mean", "last", "attn", "flat"])
    m.add_argument("--head-hidden-dims", default="1024,1024",
                   help="comma-separated MLP trunk widths; empty for a linear head")
    m.add_argument("--head-dropout", type=float, default=0.0)
    m.add_argument("--train-vision-tower", action="store_true",
                   help="leave the ViT trainable (frozen by default)")

    l = p.add_argument_group("lora")
    l.add_argument("--lora-r", type=int, default=64)
    l.add_argument("--lora-alpha", type=int, default=128)
    l.add_argument("--lora-dropout", type=float, default=0.05)
    l.add_argument("--lora-target-modules",
                   default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    l.add_argument("--no-lora", action="store_true",
                   help="head only: no adapters and the backbone is frozen (leaving it "
                        "trainable would put 2B params in the optimizer)")

    o = p.add_argument_group("objective")
    o.add_argument("--loss", default="huber", choices=["huber", "mse", "l1"])
    o.add_argument("--huber-beta", type=float, default=1.0)
    o.add_argument("--no-target-norm", action="store_true",
                   help="skip target standardization (see TargetNormalizer's docstring)")
    o.add_argument("--normalizer-rows", type=int, default=512,
                   help="train rows used to fit target statistics; 0 for all")

    t = p.add_argument_group("optimization")
    t.add_argument("--output-dir", default="dump/vector_sft")
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--head-lr", type=float, default=None,
                   help="separate LR for the head (defaults to --lr). A fresh head "
                        "usually wants a larger step than the adapters")
    t.add_argument("--weight-decay", type=float, default=0.01)
    t.add_argument("--grad-accum", type=int, default=8)
    t.add_argument("--max-steps", type=int, default=5000)
    t.add_argument("--warmup-ratio", type=float, default=0.03)
    t.add_argument("--lr-scheduler", default="cosine")
    t.add_argument("--max-grad-norm", type=float, default=1.0)
    t.add_argument("--no-grad-checkpointing", action="store_true")
    t.add_argument("--logging-steps", type=int, default=1)
    t.add_argument("--save-steps", type=int, default=250)
    t.add_argument("--eval-steps", type=int, default=250)
    t.add_argument("--save-total-limit", type=int, default=3)
    t.add_argument("--dataloader-workers", type=int, default=4)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--resume-from", default=None)

    w = p.add_argument_group("logging")
    w.add_argument("--wandb-project", default="longnav-vector-sft")
    w.add_argument("--run-name", default=None)
    w.add_argument("--no-wandb", action="store_true")
    w.add_argument("--no-preflight", action="store_true",
                   help="skip the one-batch alignment check before training")

    return p.parse_args()


def _csv(text, cast=str):
    if text is None or text == "":
        return ()
    return tuple(cast(x) for x in text.split(",") if x != "")


def load_split(path, split, max_samples=None):
    from datasets import load_dataset, load_from_disk

    if os.path.isdir(os.path.expanduser(path)):
        ds = load_from_disk(os.path.expanduser(path))
        if hasattr(ds, "keys"):
            ds = ds[split]
    else:
        ds = load_dataset(path, split=split)
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds


def infer_target_shape(dataset, column):
    """Per-turn target shape, from row 0. Every row must agree (the head is fixed-size)."""
    row = dataset[0] if hasattr(dataset, "__getitem__") else next(iter(dataset))
    arr = np.asarray(row[column], dtype=np.float32)
    if arr.ndim < 1:
        raise ValueError(f"column {column!r} must be indexed by turn")
    return tuple(arr.shape[1:]) if arr.ndim > 1 else (1,)


def build_optimizer_param_groups(model, args):
    """Head and adapters as separate groups so the fresh head can take a larger step."""
    head_params = [p for p in model.head.parameters() if p.requires_grad]
    head_ids = {id(p) for p in head_params}
    other = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    groups = [{"params": other, "lr": args.lr, "weight_decay": args.weight_decay}]
    if head_params:
        groups.append(
            {
                "params": head_params,
                "lr": args.head_lr if args.head_lr is not None else args.lr,
                "weight_decay": args.weight_decay,
            }
        )
    return groups


def preflight(model, collator, dataset, processor, cfg_model):
    """One forward pass, with the located spans decoded, before committing to a run.

    The failure this catches is the expensive one: a span/target count mismatch or a
    tokenizer convention change is invisible until the first forward, and printing what
    was actually pooled is how you confirm the head is reading the intended positions.
    """
    from longnav.utils.turn_vectors import find_turn_spans

    batch = collator([dataset[0]])
    device = model.device
    inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    spans = find_turn_spans(
        batch["input_ids"], model.prefix_ids, model.postfix_ids,
        shift_left=cfg_model.shift_left,
    )[0]
    tok = processor.tokenizer
    print(f"[preflight] dense tokens: {batch['input_ids'].shape[1]}, "
          f"turns found: {len(spans)}, targets: {tuple(batch['targets'].shape)}")
    for s in spans[:3]:
        print(f"[preflight]   turn span [{s.start}:{s.end}] -> "
              f"{tok.decode(batch['input_ids'][0, s.start:s.end])!r}")
    with torch.no_grad():
        out = model(**inputs)
    print(f"[preflight] loss {float(out['loss']):.4f}  "
          f"sparse tokens {int(out['n_tokens'])}/{int(out['n_dense_tokens'])}")
    del out, inputs
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


def main():
    args = parse_args()
    is_main = int(os.environ.get("RANK", "0")) == 0

    if args.no_wandb:
        report_to = []
    else:
        report_to = ["wandb"]
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    model_cfg = ModelConfig(
        model_id=args.model_id,
        attn_impl=args.attn_impl,
        affixes=args.affixes,
        shift_left=not args.no_shift_left,
        pool_mode=args.pool_mode,
        head_hidden_dims=_csv(args.head_hidden_dims, int),
        head_dropout=args.head_dropout,
        freeze_vision_tower=not args.train_vision_tower,
    )
    data_cfg = DataConfig(
        target_column=args.target_column,
        messages_column=args.messages_column,
        images_column=args.images_column,
        max_turns_per_sample=args.max_turns or None,
        target_dim_names=_csv(args.target_dim_names) or None,
    )
    loss_cfg = LossConfig(
        kind=args.loss,
        huber_beta=args.huber_beta,
        normalize_targets=not args.no_target_norm,
        normalizer_fit_rows=args.normalizer_rows or None,
    )
    lora = None if args.no_lora else LoraSpec(
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_modules=_csv(args.lora_target_modules),
    )

    # ---- data ----------------------------------------------------------------------
    train_ds = load_split(args.train_dataset, args.train_split)
    eval_ds = None
    if args.eval_split:
        eval_ds = load_split(
            args.eval_dataset or args.train_dataset, args.eval_split, args.eval_max_samples
        )
    target_shape = infer_target_shape(train_ds, args.target_column)
    if is_main:
        print(f"Train rows: {len(train_ds)}  target shape per turn: {target_shape}"
              + (f"  eval rows: {len(eval_ds)}" if eval_ds is not None else ""))

    # ---- model ---------------------------------------------------------------------
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = TurnVectorRegressor.build(
        model_cfg, loss_cfg, lora, target_shape, processor, dtype=torch.bfloat16
    )
    # Fitted identically on every rank (same rows, same order), so no broadcast needed.
    if loss_cfg.normalize_targets:
        n = fit_target_normalizer(
            model.normalizer, train_ds, args.target_column, loss_cfg.normalizer_fit_rows
        )
        if is_main:
            print(f"Target stats over {n} turns: mean="
                  f"{model.normalizer.mean.tolist()} std={model.normalizer.std.tolist()}")
    if is_main:
        print("Model:", model.trainable_parameter_report())

    train_collator = TurnVectorCollator(processor, data_cfg, train=True, seed=args.seed)
    eval_collator = TurnVectorCollator(processor, data_cfg, train=False, seed=args.seed)

    if not args.no_preflight and is_main:
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        if args.resume_from:
            # Preflight runs before Trainer restores the checkpoint, so without this the
            # reported loss would be the freshly initialized head's -- alarming and
            # meaningless on a resumed run. Loading here is idempotent with the restore.
            model.load_trainable(args.resume_from)
            print(f"[preflight] loaded trainable weights from {args.resume_from}")
        preflight(model, train_collator, train_ds, processor, model_cfg)

    # ---- trainer -------------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        run_name=args.run_name,
        report_to=report_to,
        seed=args.seed,
        # B == 1 is a hard requirement of the sparse backbone; scale elsewhere.
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
        dataloader_pin_memory=False,  # batches carry PIL-derived tensors of varying size
        # Every column is consumed by the collator, so Trainer must not prune them.
        remove_unused_columns=False,
        label_names=[],
        # All trainable params (adapters + head) get gradients every step.
        ddp_find_unused_parameters=False,
        average_tokens_across_devices=True,
        optim="adamw_torch",
    )

    trainer = TurnVectorSFTTrainer(
        model=model,
        args=training_args,
        data_collator=train_collator,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=processor,
        data_config=data_cfg,
        eval_data_collator=eval_collator,
    )
    # Head and adapters on separate LRs; Trainer builds the scheduler around this.
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
