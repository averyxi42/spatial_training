"""
Toy training + inference on per-turn vectors from the sparse Qwen3-VL model.

Multi-turn, multimodal conversations: every user turn carries one image that is
predominantly red, green or blue; the assistant replies with a randomly chosen action
("**forward**", "**left**", ...). We extract one vector per assistant turn from a
single sparse forward pass and train a head on two labels:

  task=action  the action word spoken in THAT turn. The action tokens are inside the
               pooled span, so this must reach ~100%. It is an alignment test: if
               turn k's vector were pooled at turn j's position, or if the sparse
               index remap were off, this collapses toward chance.

  task=color   the color class of THAT turn's image -- never stated in text, so it can
               only be read out of visual context the turn vector absorbed. Above
               chance means the vector carries its own turn's multimodal context.

`--label-shift K` is the control that makes the `color` result mean something: it
relabels each turn with another turn's color from the SAME conversation. Since all the
conversation's images are in context, a head could otherwise be reading a
conversation-level color signature. Measured on 24 train / 8 eval conversations:

    task=color, correct labels     eval acc 0.906   held-out 8/8
    task=color, --label-shift 1    eval acc 0.391   held-out 2/8
    task=color, --label-shift 2    eval acc 0.391   held-out 1/8
    task=action, correct labels    eval acc 1.000   held-out 8/8
    task=action, --label-shift 1   eval acc 0.438   held-out 2/8

i.e. the shifted controls collapse to roughly chance (0.333), so the information is
turn-local. Train accuracy hits 1.000 in every case including the controls -- the head
memorizes 192 samples either way, so only eval numbers are informative here.

The backbone is frozen; hidden states are extracted once and cached to
`dump/turn_vectors_toy_*.pt`, then the head trains on the cached states. Fits on one
24GB 3090 (~4GB peak).

    python tests/forward/demo_turn_vectors_train.py --task action
    python tests/forward/demo_turn_vectors_train.py --task color --mode attn
    python tests/forward/demo_turn_vectors_train.py --task color --label-shift 1
"""

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for p in (str(_ROOT), str(_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoConfig, AutoProcessor

from longnav.utils.modeling import Qwen3VLSparseForConditionalGeneration
from longnav.utils.turn_vectors import (
    ACTION_POSTFIX,
    ACTION_PREFIX,
    DEFAULT_POSTFIX,
    DEFAULT_PREFIX,
    TurnVectorHead,
    extract_turn_vectors,
    resolve_affix_ids,
)

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
COLORS = ["red", "green", "blue"]
COLOR_RGB = np.array([[200, 40, 40], [40, 200, 40], [40, 40, 200]], dtype=np.float32)
ACTIONS = ["forward", "left", "right"]


def make_conversation(rng, n_turns, size=256, ablate_images=False):
    """One conversation: n_turns (colored image -> action) exchanges.

    The color label is carried ONLY by the pixels: the text of a turn is
    "Step {i}: next move?" plus "**{action}**", and `action` is drawn independently of
    `color`, so no text token correlates with the label. `ablate_images=True` replaces
    every image with the same neutral gray frame, which destroys the only channel the
    label could travel through -- the control that proves the visual dependence.
    """
    colors = rng.integers(0, len(COLORS), n_turns)
    actions = rng.integers(0, len(ACTIONS), n_turns)
    messages, images = [], []
    for i, (color, action) in enumerate(zip(colors, actions)):
        # Base color + blocky texture + pixel noise: not literally constant, and
        # consecutive frames stay partly redundant, which is what the sparsifier keys on.
        blocks = rng.normal(0.0, 26.0, (size // 32, size // 32, 3))
        base = np.full(3, 128.0, dtype=np.float32) if ablate_images else COLOR_RGB[color]
        frame = base + np.kron(blocks, np.ones((32, 32, 1)))
        frame = frame + rng.normal(0.0, 8.0, frame.shape)
        images.append(Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8)))
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": images[-1]},
                    {"type": "text", "text": f"Step {i}: next move?"},
                ],
            }
        )
        messages.append({"role": "assistant", "content": f"**{ACTIONS[action]}**"})
    return messages, images, colors.tolist(), actions.tolist()


@torch.no_grad()
def extract_features(model, processor, messages, images, prefix_ids, postfix_ids,
                     shift_left=False):
    """One sparse forward pass -> (states, mask, spans, stats, inputs).

    `torch.no_grad` rather than `inference_mode`: inference-mode tensors cannot enter
    an autograd graph later, even after cloning.
    """
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    inputs = processor(text=text, images=images, videos=None, padding=False, return_tensors="pt")

    # Whole conversation in one pass, no KV cache: the model computes its own rope
    # indices, and modeling.py sparsifies position_ids/attention_mask internally.
    outputs = model(**inputs.to(model.device), use_cache=False, logits_to_keep=1)

    states, spans = extract_turn_vectors(
        outputs,
        inputs["input_ids"],
        head=None,  # raw content states; the head trains separately on the cache
        prefix_ids=prefix_ids,
        postfix_ids=postfix_ids,
        shift_left=shift_left,
    )
    vis_keep = outputs["vis_keep_mask"]
    stats = {
        "dense": inputs["input_ids"].shape[1],
        "sparse": outputs["last_hidden_state"].shape[1],
        "vis_kept": int(vis_keep.sum()),
        "vis_total": int(vis_keep.numel()),
    }
    lengths = torch.tensor([len(s) for s in spans])
    mask = torch.arange(states.shape[1])[None, :] < lengths[:, None]
    return states.float().cpu().clone(), mask, spans, stats, inputs


def build_split(model, processor, prefix_ids, postfix_ids, n_convs, n_turns, seed, tag,
                ablate_images=False, shift_left=False):
    rng = np.random.default_rng(seed)
    states, masks, colors, actions = [], [], [], []
    for c in range(n_convs):
        messages, images, conv_colors, conv_actions = make_conversation(
            rng, n_turns, ablate_images=ablate_images
        )
        s, m, spans, stats, _ = extract_features(
            model, processor, messages, images, prefix_ids, postfix_ids, shift_left
        )
        assert len(spans) == n_turns, f"found {len(spans)} turns, expected {n_turns}"
        states.append(s)
        masks.append(m)
        colors.append(torch.tensor(conv_colors))
        actions.append(torch.tensor(conv_actions))
        if c == 0:
            print(
                f"  [{tag}] per conv: {stats['dense']} -> {stats['sparse']} tokens, "
                f"visual kept {stats['vis_kept']}/{stats['vis_total']}, "
                f"{len(spans)} turns x {s.shape[1]} content tokens"
            )
    return {
        "states": torch.cat(states),
        "mask": torch.cat(masks),
        "color": torch.cat(colors),
        "action": torch.cat(actions),
    }


def load_or_build_cache(args, cache_path):
    if cache_path.exists() and not args.rebuild_cache:
        print(f"Loading cached features from {cache_path}")
        return torch.load(cache_path, weights_only=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"Loading {args.model_id} (sparse, frozen) on {device}...")
    processor = AutoProcessor.from_pretrained(args.model_id)
    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    model = Qwen3VLSparseForConditionalGeneration.from_pretrained(
        args.model_id,
        config=config,
        dtype=dtype,
        device_map=device,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation=args.attn_impl,
    ).eval()
    model.config.use_cache = False
    model.requires_grad_(False)

    # template: content is the whole assistant message ('**left**', 3 tokens).
    # action:   content is the bare action word ('left', 1 token), VLMWorker's convention.
    prefix, postfix = (
        (DEFAULT_PREFIX, DEFAULT_POSTFIX)
        if args.affixes == "template"
        else (ACTION_PREFIX, ACTION_POSTFIX)
    )
    prefix_ids, postfix_ids = resolve_affix_ids(processor.tokenizer, prefix, postfix)
    print(f"Affixes: prefix={prefix_ids} postfix={postfix_ids}")

    print(f"\nExtracting features ({args.train_convs + args.eval_convs} convs "
          f"x {args.turns} turns)...")
    train = build_split(model, processor, prefix_ids, postfix_ids,
                        args.train_convs, args.turns, seed=0, tag="train",
                        ablate_images=args.ablate_images, shift_left=args.shift_left)
    evald = build_split(model, processor, prefix_ids, postfix_ids,
                        args.eval_convs, args.turns, seed=1234, tag="eval",
                        ablate_images=args.ablate_images, shift_left=args.shift_left)

    # One extra held-out conversation, kept whole, for the inference demo.
    rng = np.random.default_rng(9999)
    messages, images, colors, actions = make_conversation(
        rng, args.turns, ablate_images=args.ablate_images
    )
    d_states, d_mask, d_spans, d_stats, d_inputs = extract_features(
        model, processor, messages, images, prefix_ids, postfix_ids, args.shift_left
    )
    demo = {
        "states": d_states,
        "mask": d_mask,
        "color": torch.tensor(colors),
        "action": torch.tensor(actions),
        "stats": d_stats,
        "input_ids": d_inputs["input_ids"].cpu(),
        "spans": [(s.batch_idx, s.start, s.end) for s in d_spans],
    }

    if device == "cuda":
        print(f"Peak GPU memory during extraction: "
              f"{torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
    cache = {"train": train, "eval": evald, "demo": demo}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_path)
    print(f"Cached features -> {cache_path}")
    return cache


def build_head(args, states, n_classes, device):
    """The head's output vector IS the per-turn vector; for this toy its `out_dim` is
    the number of classes, so the vector doubles directly as class logits."""
    return TurnVectorHead(
        hidden_size=states.shape[-1],
        out_dim=n_classes,
        mode=args.mode,
        content_len=states.shape[1] if args.mode == "flat" else None,
        hidden_dims=(256,),
        dropout=0.1,
        standardize=args.standardize,
    ).to(device)


def evaluate(head, states, mask, labels, device):
    head.eval()
    with torch.no_grad():
        logits = head(states.to(device), mask.to(device))
    return float((logits.argmax(-1).cpu() == labels).float().mean()), logits.cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--task", default="action", choices=["action", "color"])
    ap.add_argument("--turns", type=int, default=8)
    ap.add_argument("--train-convs", type=int, default=24)
    ap.add_argument("--eval-convs", type=int, default=8)
    ap.add_argument("--mode", default="mean", choices=["mean", "last", "attn", "flat"])
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--attn-impl", default="sdpa", choices=["sdpa", "flash_attention_2"])
    ap.add_argument(
        "--label-shift",
        type=int,
        default=0,
        help="CONTROL: label each turn with the label of the turn this many steps "
        "later in the SAME conversation. Nonzero should collapse toward chance -- if "
        "it does not, the vectors encode conversation-level rather than turn-local "
        "information.",
    )
    ap.add_argument(
        "--ablate-images",
        action="store_true",
        help="CONTROL: replace every image with the same neutral gray frame. The color "
        "label then has no channel to travel through, so accuracy must fall to chance. "
        "Uses its own feature cache.",
    )
    ap.add_argument(
        "--standardize",
        action="store_true",
        help="fit per-dimension input statistics into the head's buffers. Measured to "
        "be unnecessary here (see the module docstring); kept as a flag because it is "
        "the conditioning aid to reach for if a harder task trains badly.",
    )
    ap.add_argument(
        "--train-subsample",
        type=int,
        default=0,
        help="use only the first N training turns (for data-scaling ablations)",
    )
    ap.add_argument(
        "--affixes",
        default="template",
        choices=["template", "action"],
        help="template: 3-token content span ('**left**'). action: 1-token span ('left'), "
        "which is the narrowest realistic fixed-template case.",
    )
    ap.add_argument(
        "--shift-left",
        action="store_true",
        help="pool the states that PREDICT the content instead of the content itself. "
        "With --affixes action the span becomes the single '**' token that opens every "
        "turn -- the same token everywhere, so the vector is a pure function of context "
        "with no content leakage. Uses its own feature cache.",
    )
    ap.add_argument("--cache", default=None, help="path for the feature cache (.pt)")
    ap.add_argument("--rebuild-cache", action="store_true")
    args = ap.parse_args()

    if args.attn_impl == "flash_attention_2":
        from longnav.utils.turn_vectors import patch_flash_attention_packing

        patch_flash_attention_packing()

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache_path = Path(
        args.cache
        or _ROOT / "dump" / f"turn_vectors_toy_{args.turns}x"
        f"{args.train_convs}+{args.eval_convs}"
        f"{'_grayablation' if args.ablate_images else ''}"
        f"{'_actionaffix' if args.affixes == 'action' else ''}"
        f"{'_shiftleft' if args.shift_left else ''}.pt"
    )
    cache = load_or_build_cache(args, cache_path)

    tr, ev, demo = cache["train"], cache["eval"], cache["demo"]
    classes = ACTIONS if args.task == "action" else COLORS

    def relabel(y):
        """Roll labels within each conversation (identity when --label-shift is 0)."""
        if args.label_shift == 0:
            return y
        return y.view(-1, args.turns).roll(-args.label_shift, dims=1).reshape(-1)

    tr_y, ev_y = relabel(tr[args.task]), relabel(ev[args.task])
    if args.train_subsample:
        n = args.train_subsample
        tr = {k: v[:n] for k, v in tr.items()}
        tr_y = tr_y[:n]
        print(f"\nSubsampled train split to {n} turns")
    if args.label_shift:
        print(
            f"\n*** CONTROL RUN: labels shifted by {args.label_shift} turn(s) within "
            "each conversation. Expect ~chance. ***"
        )
    print(
        f"\nTask: {args.task} ({len(classes)} classes, chance = {1/len(classes):.3f})\n"
        f"Train: {tuple(tr['states'].shape)}  Eval: {tuple(ev['states'].shape)}  "
        f"train class balance: {torch.bincount(tr_y).tolist()}"
    )

    tr_s, ev_s = tr["states"], ev["states"]
    head = build_head(args, tr_s, len(classes), device)
    if args.standardize:
        # Fitted once on the train split and stored in the head's buffers, so it is
        # applied automatically at eval and inference and saved with the state dict.
        head.fit_input_stats(tr_s, tr["mask"])
        spread = (head.input_mean.abs() / head.input_std).cpu()
        print(f"Input stats fitted: |mean|/std across dims -- median "
              f"{spread.median():.2f}, max {spread.max():.1f} "
              f"(largest |mean| {head.input_mean.abs().max():.1f} vs median "
              f"{head.input_mean.abs().median():.3f})")
    print(f"Head: mode={args.mode}, out_dim={len(classes)}, "
          f"standardize={args.standardize}, "
          f"{sum(p.numel() for p in head.parameters() if p.requires_grad):,} "
          f"trainable params (backbone frozen)")

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-2)
    tr_sd, tr_md, tr_yd = tr_s.to(device), tr["mask"].to(device), tr_y.to(device)

    acc0, _ = evaluate(head, ev_s, ev["mask"], ev_y, device)
    print(f"\nepoch   0: eval acc {acc0:.3f} (untrained)")
    for epoch in range(1, args.epochs + 1):
        head.train()
        opt.zero_grad()
        loss = F.cross_entropy(head(tr_sd, tr_md), tr_yd)
        loss.backward()
        opt.step()
        if epoch % max(1, args.epochs // 6) == 0:
            tr_acc, _ = evaluate(head, tr_s, tr["mask"], tr_y, device)
            ev_acc, _ = evaluate(head, ev_s, ev["mask"], ev_y, device)
            print(f"epoch {epoch:>3}: loss {loss.item():.4f}  "
                  f"train acc {tr_acc:.3f}  eval acc {ev_acc:.3f}")

    # ---- Inference on the held-out conversation, per turn -------------------------
    print("\n--- Inference on a fresh held-out conversation ---")
    head.eval()
    with torch.no_grad():
        # Raw states: the head standardizes internally using its stored buffers.
        probs = head(demo["states"].to(device), demo["mask"].to(device))
        probs = probs.softmax(-1).cpu()
    preds = probs.argmax(-1).tolist()
    truth = relabel(demo[args.task]).tolist()

    print(f"{demo['stats']['dense']} dense -> {demo['stats']['sparse']} sparse tokens, "
          f"visual kept {demo['stats']['vis_kept']}/{demo['stats']['vis_total']}")
    tok = AutoProcessor.from_pretrained(args.model_id).tokenizer
    print(f"{'turn':>4}  {'dense span':>12}  {'content':>12}  "
          f"{'true':>8}  {'pred':>8}  p(pred)")
    for i, (b, start, end) in enumerate(demo["spans"]):
        decoded = tok.decode(demo["input_ids"][b, start:end])
        hit = "OK " if preds[i] == truth[i] else "MISS"
        print(f"{i:>4}  {f'[{start}:{end}]':>12}  {decoded!r:>12}  "
              f"{classes[truth[i]]:>8}  {classes[preds[i]]:>8}  "
              f"{probs[i, preds[i]]:.3f}  {hit}")
    acc = float(np.mean([p == t for p, t in zip(preds, truth)]))
    print(f"\nHeld-out conversation accuracy: {acc:.3f} "
          f"({int(round(acc*len(truth)))}/{len(truth)})")


if __name__ == "__main__":
    main()
