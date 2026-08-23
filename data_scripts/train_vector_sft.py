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
        --train-dataset data/continuous_sft_formatted --grad-accum 8 --output-dir dump/vector_sft_run1

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

from longnav.utils.model_metrics import add_model_metrics_args, attach_model_metrics
from longnav.utils.turn_vectors import ACTION_POSTFIX, ACTION_PREFIX
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
from longnav.utils.stop_head import STOP_INFERENCE_MODES, StopHeadConfig


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
    d.add_argument("--eval-split", default="validation", help="if given, triggers evaluation")
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
    m.add_argument("--attn-impl", default="flash_attention_2", choices=["sdpa", "flash_attention_2"])
    # A turn is prefix + content + postfix; these ARE the affixes, not a named preset.
    # Escapes are decoded, so '\n' can be written literally on the command line.
    m.add_argument("--prefix", default=ACTION_PREFIX,
                   help="turn prefix. Default wraps the content in '**'; pass "
                        "'<|im_start|>assistant\\n' to pool the assistant content itself")
    m.add_argument("--postfix", default=ACTION_POSTFIX, help="turn postfix")
    m.add_argument("--no-shift-left", action="store_true",
                   help="pool the assistant content instead of the position that precedes "
                        "it (only sensible when the content is real text, not a placeholder)")
    m.add_argument("--pool-mode", default="mean", choices=["mean", "last", "attn", "flat"])
    m.add_argument("--head-hidden-dims", default="1024,1024",
                   help="comma-separated MLP trunk widths; empty for a linear head. NOT a "
                        "free knob: the CNF head ran at 64 and its conditioning vector "
                        "collapsed to a participation ratio of 2.2 out of 128 dimensions "
                        "(dump/cnf_sigma_ablation/FINDINGS.md). Every current run passes "
                        "1024 explicitly; the default is wide so that forgetting to pass "
                        "it cannot silently reintroduce that bottleneck.")
    m.add_argument("--head-dropout", type=float, default=0.0)
    m.add_argument("--train-vision-tower", action="store_true",
                   help="leave the ViT trainable (frozen by default)")
    m.add_argument("--modality-specs", default=None,
                   help="JSON list of modality embedding specs, or a path to a .json file "
                        "holding one. Each entry: {\"token\": \"<x>\", \"n_features\": 3, "
                        "\"encoder\": \"mlp\", \"encoder_kwargs\": {}, \"column\": \"x\"}. "
                        "Omitted -> the mechanism is inert. See "
                        "longnav.utils.modality_embed")

    s = p.add_argument_group("stop head")
    s.add_argument("--stop-head", action="store_true",
                   help="train a binary 'is this the episode end' readout on the pooled "
                        "turn context. Off by default; without it nothing about the run "
                        "changes")
    s.add_argument("--stop-hidden-dims", default="256",
                   help="comma-separated hidden widths for the stop head")
    s.add_argument("--stop-loss-weight", type=float, default=1.0,
                   help="scale on the stop loss. Needs no tuning while --stop-grad is on, "
                        "since the loss then reaches only the stop head's parameters")
    s.add_argument("--stop-pos-weight", type=float, default=None,
                   help="BCE pos_weight. Roughly the negative:positive ratio (~90 here) "
                        "so the one positive turn per episode is not drowned by the "
                        "gradient of its ninety negatives")
    s.add_argument("--no-stop-grad", action="store_true",
                   help="let the stop loss reach the backbone and the motion head. This "
                        "gives up the guarantee that the auxiliary cannot hurt the motion "
                        "objective, and makes --stop-loss-weight a real hyperparameter")
    s.add_argument("--stop-temperature", type=float, default=1.0,
                   help="inference-time logit scale; tune post hoc on the saved logits")
    s.add_argument("--stop-inference", default="sample",
                   choices=list(STOP_INFERENCE_MODES))
    s.add_argument("--stop-threshold", type=float, default=0.5)

    l = p.add_argument_group("lora")
    # r=128 / alpha=256 is the standard for this project. The earlier r=64 / alpha=128 was
    # a value inherited from agent-written launch scripts, never chosen deliberately, and
    # it under-provisioned the adapter -- VRAM comfortably allows double. Every run before
    # `reg-v6-pose-stop-ddp4` (2026-08-04) used 64, so any comparison spanning that
    # boundary has adapter capacity as a confound alongside whatever it meant to vary.
    l.add_argument("--lora-r", type=int, default=128)
    l.add_argument("--lora-alpha", type=int, default=256)
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
    t.add_argument("--lr", type=float, default=1e-5)
    t.add_argument("--head-lr", type=float, default=1e-4,
                   help="separate LR for the head (defaults to --lr). A fresh head "
                        "usually wants a larger step than the adapters")
    t.add_argument("--weight-decay", type=float, default=0.0001)
    t.add_argument("--grad-accum", type=int, default=8)
    t.add_argument("--max-steps", type=int, default=5000)
    t.add_argument("--warmup-ratio", type=float, default=0.01)
    t.add_argument("--lr-scheduler", default="cosine")
    t.add_argument("--max-grad-norm", type=float, default=1.0)
    t.add_argument("--no-grad-checkpointing", action="store_true")
    t.add_argument("--logging-steps", type=int, default=1)
    t.add_argument("--save-steps", type=int, default=250)
    t.add_argument("--eval-steps", type=int, default=250)
    t.add_argument("--save-total-limit", type=int, default=None)
    t.add_argument("--dataloader-workers", type=int, default=4)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--resume-from", default=None)

    w = p.add_argument_group("logging")
    w.add_argument("--wandb-project", default="longnav-vector-sft")
    w.add_argument("--run-name", default=None)
    w.add_argument("--no-wandb", action="store_true")
    w.add_argument("--no-preflight", action="store_true",
                   help="skip the one-batch alignment check before training")

    # Per-module grad/weight/activation metrics. On by default and purely observational;
    # every head script below re-parses through this function, so they all inherit it.
    add_model_metrics_args(p)

    return p.parse_args()


def _unescape(text):
    """Decode backslash escapes so `--prefix '<|im_start|>assistant\n**'` works in a shell.

    Latin-1 round-trip first so non-ASCII placeholders survive `unicode_escape`.
    """
    return text.encode("latin-1", "backslashreplace").decode("unicode_escape")


def _modality_specs(text):
    """A JSON list of specs, or a path to a file holding one. None -> no modalities."""
    if not text:
        return ()
    import json

    raw = Path(os.path.expanduser(text))
    payload = json.loads(raw.read_text()) if raw.is_file() else json.loads(text)
    if not isinstance(payload, list):
        raise ValueError("--modality-specs must be a JSON *list* of spec objects")
    return tuple(payload)


def _csv(text, cast=str):
    if text is None or text == "":
        return ()
    return tuple(cast(x) for x in text.split(",") if x != "")


def load_split(path, split, max_samples=None, sample_seed=None):
    from datasets import load_dataset, load_from_disk

    if os.path.isdir(os.path.expanduser(path)):
        ds = load_from_disk(os.path.expanduser(path))
        if hasattr(ds, "keys"):
            ds = ds[split]
    else:
        ds = load_dataset(path, split=split)
    if max_samples:
        # A seeded random subset, not the first N. The head of a corpus is not a sample
        # of it: on the on-policy stop corpus the first 4 validation rows are all
        # ORDERED though the split is 47.5% shuffled, and the first 64 objectnav rows
        # come from a SINGLE scene -- so the eval measured one scene and hid the
        # clock-free half entirely. `sample_seed=None` keeps the historical first-N
        # behaviour for reproducing older runs.
        n = min(max_samples, len(ds))
        if sample_seed is None:
            ds = ds.select(range(n))
        else:
            ds = ds.shuffle(seed=int(sample_seed)).select(range(n))
    return ds


def infer_target_shape(dataset, column):
    """Per-turn target shape, from row 0. Every row must agree (the head is fixed-size)."""
    row = dataset[0] if hasattr(dataset, "__getitem__") else next(iter(dataset))
    arr = np.asarray(row[column], dtype=np.float32)
    if arr.ndim < 1:
        raise ValueError(f"column {column!r} must be indexed by turn")
    return tuple(arr.shape[1:]) if arr.ndim > 1 else (1,)


def build_optimizer_param_groups(model, args):
    """Head (and modality encoders) apart from the adapters, so the fresh modules can take
    a larger step.

    Then assert every trainable parameter landed in some group. A module left out of the
    groups is **silently frozen** -- no error, no warning, just a parameter that never
    moves. For the modality encoders that failure is indistinguishable from "the injection
    did not help", which is the conclusion the experiment is trying to draw.
    """
    fresh = [p for p in model.head.parameters() if p.requires_grad]
    fresh += [p for p in model.modality_embedder.parameters() if p.requires_grad]
    if model.stop_head is not None:
        fresh += [p for p in model.stop_head.parameters() if p.requires_grad]
    fresh_ids = {id(p) for p in fresh}
    other = [p for p in model.parameters() if p.requires_grad and id(p) not in fresh_ids]
    groups = [{"params": other, "lr": args.lr, "weight_decay": args.weight_decay}]
    if fresh:
        groups.append(
            {
                "params": fresh,
                "lr": args.head_lr if args.head_lr is not None else args.lr,
                "weight_decay": args.weight_decay,
            }
        )

    grouped = {id(p) for g in groups for p in g["params"]}
    missing = [
        n for n, p in model.named_parameters()
        if p.requires_grad and id(p) not in grouped
    ]
    if missing:
        raise RuntimeError(
            f"{len(missing)} trainable parameter(s) are in no optimizer group and would "
            f"be silently frozen: {missing[:8]}"
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
        prefix=_unescape(args.prefix),
        postfix=_unescape(args.postfix),
        shift_left=not args.no_shift_left,
        pool_mode=args.pool_mode,
        head_hidden_dims=_csv(args.head_hidden_dims, int),
        head_dropout=args.head_dropout,
        freeze_vision_tower=not args.train_vision_tower,
        modality_specs=_modality_specs(args.modality_specs),
        stop_head=(
            StopHeadConfig(
                hidden_dims=_csv(args.stop_hidden_dims, int),
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
        if model.modality_embedder:
            print("Modality embeddings:\n" + model.modality_embedder.describe())

    # The model's own spec list, not a second copy: which column feeds which marker has
    # exactly one definition, and `build()` has already registered the tokens on this
    # processor's tokenizer.
    specs = model_cfg.modality_specs
    # Driven off the model config, not a second flag: the collator emits stop labels
    # exactly when the model has somewhere to put them.
    stop_labels = model_cfg.stop_head is not None
    train_collator = TurnVectorCollator(processor, data_cfg, train=True, seed=args.seed,
                                        modality_specs=specs, stop_labels=stop_labels)
    eval_collator = TurnVectorCollator(processor, data_cfg, train=False, seed=args.seed,
                                       modality_specs=specs, stop_labels=stop_labels)

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
    # `model/<module>/{grad_norm,weight_delta,act_norm_mean,...}` alongside the run's own
    # metrics. The group check above proves a module is in the optimizer; this proves the
    # optimizer is actually moving it -- which is the failure that cost run_v6 3600 steps.
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
