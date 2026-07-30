"""
Smoke tests for `longnav.utils.vector_sft`: the pure-python pieces, then a real
end-to-end overfit run through `Trainer` + LoRA + gradient checkpointing.

The synthetic task mirrors the intended application without needing the navigation data:
each turn shows an image whose dominant color encodes a target action chunk of shape
`(4, 3)`, the assistant message is the fixed `**____**` placeholder, and the head has
to regress the chunk from that turn's vector. Because the mapping color -> chunk is
deterministic, a working pipeline must drive the loss down hard; a broken one (no grad
path through gradient checkpointing, misaligned turns, a detached head) plateaus.

    # unit tests only, no GPU needed
    pytest tests/test_vector_sft_smoke.py -k "not train"

    # full run, and the same thing under torchrun so the DDP wrapping path is exercised
    python tests/test_vector_sft_smoke.py
    torchrun --nproc_per_node=1 tests/test_vector_sft_smoke.py
"""

import os
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for p in (str(_ROOT), str(_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("WANDB_MODE", "disabled")

import numpy as np
import pytest
import torch
from PIL import Image

from longnav.utils.vector_sft import (
    DataConfig,
    LoraSpec,
    LossConfig,
    ModelConfig,
    TargetNormalizer,
    TurnVectorCollator,
    TurnVectorRegressor,
    TurnVectorSFTTrainer,
    assistant_turn_indices,
    fit_target_normalizer,
    n_images_in_message,
    slice_conversation,
)

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
COLOR_RGB = np.array([[200, 40, 40], [40, 200, 40], [40, 40, 200]], dtype=np.float32)
# One distinct (4, 3) chunk per color: what the head must learn to read out.
COLOR_CHUNK = np.array(
    [
        [[0.25, 0.00, 0.00], [0.50, 0.00, 0.00], [0.75, 0.00, 0.00], [1.00, 0.00, 0.00]],
        [[0.20, 0.10, 0.30], [0.40, 0.20, 0.60], [0.60, 0.30, 0.90], [0.80, 0.40, 1.20]],
        [[0.10, -0.15, -0.40], [0.20, -0.30, -0.80], [0.30, -0.45, -1.20], [0.40, -0.60, -1.60]],
    ],
    dtype=np.float32,
)
SYSTEM_PROMPT = "You are a navigation policy. Output the next action."


# ======================================================================================
# Synthetic dataset in the format the trainer expects
# ======================================================================================
def make_episode(rng, image_dir: Path, ep: str, n_turns: int, size: int = 128):
    """One conversation row: prologue + n_turns of (image -> `**____**`)."""
    messages = [{"role": "user", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
    image_paths, chunks = [], []
    for t in range(n_turns):
        color = int(rng.integers(0, 3))
        frame = COLOR_RGB[color] + rng.normal(0, 10.0, (size, size, 3))
        path = image_dir / f"{ep}_t{t}.png"
        Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8)).save(path)
        image_paths.append(str(path))
        chunks.append(COLOR_CHUNK[color] + rng.normal(0, 0.01, COLOR_CHUNK.shape[1:]))
        messages += [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Observation {t}:"},
                    {"type": "image"},
                    {"type": "text", "text": "Action:"},
                ],
            },
            # Fixed placeholder: the real action comes from the head, not these tokens.
            {"role": "assistant", "content": [{"type": "text", "text": "**____**"}]},
        ]
    return {
        "messages": messages,
        "images": image_paths,
        "action_chunks": np.stack(chunks).tolist(),
    }


def make_dataset(image_dir: Path, n_episodes: int = 8, n_turns: int = 4, seed: int = 0,
                 tag: str = "train"):
    """`tag` namespaces the image filenames: two splits sharing a directory would
    otherwise overwrite each other's frames and leave rows pointing at images whose
    color no longer matches their target."""
    from datasets import Dataset

    rng = np.random.default_rng(seed)
    rows = [make_episode(rng, image_dir, f"{tag}_ep{e}", n_turns) for e in range(n_episodes)]
    return Dataset.from_list(rows)


# ======================================================================================
# Unit tests (no model)
# ======================================================================================
def _toy_conversation(n_turns=5):
    messages = [{"role": "user", "content": [{"type": "text", "text": "sys"}]}]
    for t in range(n_turns):
        messages += [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": f"o{t}"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "**____**"}]},
        ]
    images = [f"img{t}" for t in range(n_turns)]
    targets = torch.arange(n_turns, dtype=torch.float32)[:, None].repeat(1, 3)
    return messages, images, targets


def test_turn_and_image_counting():
    messages, images, _ = _toy_conversation(5)
    assert len(assistant_turn_indices(messages)) == 5
    assert sum(n_images_in_message(m) for m in messages) == len(images)


@pytest.mark.parametrize("start,n", [(0, 2), (1, 3), (3, 2)])
def test_slice_conversation_keeps_everything_in_lockstep(start, n):
    messages, images, targets = _toy_conversation(5)
    msgs, imgs, tgts = slice_conversation(messages, images, targets, start, n)
    # The window has exactly n turns, n images, n targets -- and they are the SAME n.
    assert len(assistant_turn_indices(msgs)) == n
    assert len(imgs) == n
    assert imgs == [f"img{start + i}" for i in range(n)]
    assert tgts.shape[0] == n
    assert torch.equal(tgts[:, 0], torch.arange(start, start + n, dtype=torch.float32))
    # The instruction prologue survives windowing away turn 0.
    assert msgs[0]["content"][0]["text"] == "sys"


def test_slice_conversation_rejects_image_count_mismatch():
    messages, images, targets = _toy_conversation(4)
    with pytest.raises(ValueError, match="placeholder"):
        slice_conversation(messages, images[:-1], targets, 0, 2)


def test_target_normalizer_roundtrip_and_state_dict():
    x = torch.randn(64, 4, 3) * torch.tensor([10.0, 1.0, 0.01]) + torch.tensor([5.0, -2.0, 0.0])
    norm = TargetNormalizer(3)
    norm.fit(x)
    z = norm.normalize(x)
    assert torch.allclose(z.reshape(-1, 3).mean(0), torch.zeros(3), atol=1e-5)
    assert torch.allclose(z.reshape(-1, 3).std(0), torch.ones(3), atol=1e-2)
    assert torch.allclose(norm.denormalize(z), x, atol=1e-4)
    # Buffers must travel with the checkpoint, or inference silently mis-scales.
    fresh = TargetNormalizer(3)
    fresh.load_state_dict(norm.state_dict())
    assert bool(fresh.fitted) and torch.allclose(fresh.mean, norm.mean)


def test_normalizer_refuses_unfitted_use():
    with pytest.raises(RuntimeError, match="before fit"):
        TargetNormalizer(3).normalize(torch.zeros(2, 3))


def test_collator_rejects_batch_gt_1():
    messages, images, targets = _toy_conversation(2)
    row = {"messages": messages, "images": images, "action_chunks": targets.tolist()}
    collator = TurnVectorCollator(processor=None, data=DataConfig(), train=True)
    with pytest.raises(ValueError, match="batch size must be 1"):
        collator([row, row])


def test_collator_rejects_target_count_mismatch():
    messages, images, targets = _toy_conversation(4)
    row = {"messages": messages, "images": images, "action_chunks": targets[:2].tolist()}
    collator = TurnVectorCollator(processor=None, data=DataConfig(), train=True)
    with pytest.raises(ValueError, match="assistant turn"):
        collator([row])


# ======================================================================================
# End-to-end training smoke
# ======================================================================================
def run_train_smoke(steps: int = 60, n_episodes: int = 8, n_turns: int = 4,
                    workdir: str = None, drop_factor: float = 3.0):
    """Overfit the synthetic set; return (eval_before, eval_after, train_losses).

    The pass/fail criterion is the *eval* loss before vs after training, not the training
    loss: per-step training loss is one episode's mean over a handful of turns and swings
    by ~10x between episodes, while the eval loop uses deterministic first-N windows over
    a fixed set. A `drop_factor` fall in eval loss is the check that every link in the
    chain carries gradient -- LoRA adapters under gradient checkpointing, the
    dense->sparse span remap, the head, and target normalization.
    """
    from transformers import AutoProcessor, TrainerCallback, TrainingArguments

    tmp = Path(workdir or tempfile.mkdtemp(prefix="vector_sft_smoke_"))
    (tmp / "images").mkdir(parents=True, exist_ok=True)
    train_ds = make_dataset(tmp / "images", n_episodes, n_turns, seed=0, tag="train")
    eval_ds = make_dataset(tmp / "images", 4, n_turns, seed=1, tag="eval")

    model_cfg = ModelConfig(model_id=MODEL_ID, attn_impl="sdpa", head_hidden_dims=(512,))
    data_cfg = DataConfig(max_turns_per_sample=n_turns, target_dim_names=("dx", "dy", "dtheta"))
    loss_cfg = LossConfig(kind="huber", normalizer_fit_rows=None)

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = TurnVectorRegressor.build(
        model_cfg, loss_cfg, LoraSpec(r=8, alpha=16), (4, 3), processor,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    fit_target_normalizer(model.normalizer, train_ds, "action_chunks", None)
    print("Model:", model.trainable_parameter_report())

    losses = []

    class Recorder(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                losses.append(logs["loss"])

    args = TrainingArguments(
        output_dir=str(tmp / "out"),
        report_to=[],
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=2,
        max_steps=steps,
        learning_rate=2e-4,
        lr_scheduler_type="constant",
        warmup_steps=0,
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,
        save_strategy="steps",
        save_steps=steps,  # exercise the custom _save once, at the end
        save_total_limit=1,
        eval_strategy="steps",
        eval_steps=steps,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        label_names=[],
        ddp_find_unused_parameters=False,
        disable_tqdm=True,
        seed=0,
    )
    trainer = TurnVectorSFTTrainer(
        model=model,
        args=args,
        data_collator=TurnVectorCollator(processor, data_cfg, train=True, seed=0),
        eval_data_collator=TurnVectorCollator(processor, data_cfg, train=False, seed=0),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=processor,
        data_config=data_cfg,
        callbacks=[Recorder()],
    )
    before = trainer.evaluate(metric_key_prefix="eval")
    trainer.train()
    after = trainer.evaluate(metric_key_prefix="eval")

    print(f"train loss (mean of first/last 10 logged steps): "
          f"{np.mean(losses[:10]):.4f} -> {np.mean(losses[-10:]):.4f} "
          f"over {len(losses)} steps")
    print("eval before:", {k: round(v, 4) for k, v in before.items()})
    print("eval after :", {k: round(v, 4) for k, v in after.items()})
    assert after["eval_turn_loss"] < before["eval_turn_loss"] / drop_factor, (
        f"eval loss only went {before['eval_turn_loss']:.4f} -> "
        f"{after['eval_turn_loss']:.4f}; expected a {drop_factor}x drop. "
        "Something in the chain is not learning."
    )
    # Denormalized RMSE must improve too -- the loss is computed on normalized targets,
    # so a drop there alone would not prove the predictions are physically closer.
    assert after["eval_rmse_mean"] < before["eval_rmse_mean"] / 2, (
        f"eval RMSE {before['eval_rmse_mean']:.4f} -> {after['eval_rmse_mean']:.4f}"
    )

    # Checkpoint round trip: the saved dir must hold adapter + head + normalizer, and
    # reloading must restore the fitted statistics bit-for-bit.
    ckpts = sorted((tmp / "out").glob("checkpoint-*"))
    assert ckpts, "no checkpoint was written"
    ckpt = ckpts[-1]
    assert (ckpt / "turn_vector_head.pt").exists()
    assert (ckpt / "adapter").exists(), "LoRA adapter missing from the checkpoint"
    saved_mean = model.normalizer.mean.clone()
    model.normalizer.mean.zero_()

    # The adapter half matters more than the head half and is easy to get silently wrong:
    # `build()` already created an adapter named "default", so a loader that refuses to
    # overwrite an existing name would leave the LoRA weights at their init values while
    # reporting success. Zero a trained B matrix and require the reload to bring it back.
    b_name, b_param = next(
        (n, q) for n, q in model.backbone.named_parameters() if "lora_B" in n and q.numel() > 1
    )
    trained_b = b_param.detach().clone()
    assert trained_b.abs().max() > 0, (
        "LoRA B is still all zeros after training, so this check cannot detect a no-op "
        "reload -- train longer before asserting on it"
    )
    b_param.data.zero_()

    model.load_trainable(ckpt)
    assert torch.allclose(model.normalizer.mean, saved_mean), "normalizer did not restore"
    restored = dict(model.backbone.named_parameters())[b_name]
    assert torch.allclose(restored, trained_b), (
        f"LoRA weights were NOT restored from {ckpt}: {b_name} stayed at "
        f"{float(restored.abs().max()):.3g} instead of {float(trained_b.abs().max()):.3g}"
    )
    print(f"checkpoint round trip OK ({ckpt.name}): head, normalizer and {b_name}")

    # `from_pretrained` is the entry point for turn-by-turn inference and rebuilds the
    # backbone from scratch, so it exercises a different loading path than resume.
    del trainer
    reloaded = TurnVectorRegressor.from_pretrained(
        ckpt, processor, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    assert torch.allclose(reloaded.normalizer.mean.cpu(), saved_mean.cpu()), \
        "from_pretrained lost the target statistics"
    reloaded_b = dict(reloaded.backbone.named_parameters())[b_name]
    assert torch.allclose(reloaded_b.cpu().float(), trained_b.cpu().float(), atol=1e-3), \
        "from_pretrained did not restore the adapter weights"
    print("from_pretrained round trip OK")
    return before, after, losses


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_train_smoke():
    run_train_smoke()


if __name__ == "__main__":
    run_train_smoke()
