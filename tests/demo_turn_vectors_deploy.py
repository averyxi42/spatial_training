"""
Deployed inference on per-turn vectors: live model -> head -> metrics -> visualization.

`demo_turn_vectors_train.py` proves the per-turn vectors carry turn-local information,
but it trains AND evaluates on hidden states cached to `dump/turn_vectors_toy_*.pt`.
This script closes the loop: the head is trained from that cache (cheap, unchanged),
then every evaluated conversation goes through the real thing at inference time --

    multimodal conversation -> processor -> sparse Qwen3-VL forward (live, no cache)
        -> extract_turn_vectors(head=head) -> per-turn class vector -> metrics

Phase A  train the head on the cached `train` split, save `dump/turn_vector_head_*.pt`.
Phase B  rebuild the head from that checkpoint, then for each fresh conversation run one
         live sparse forward pass and pool one vector per assistant turn out of it.
         Nothing from the cache touches phase B.

Two checks keep the live path honest:

  parity     `--parity` replays the exact seed-1234 conversations that built the eval
             cache and compares live predictions against cache predictions. Agreement
             is direct proof that "live instead of cache" changed nothing.
  alignment  `--task action` must land ~1.000. The action tokens sit inside the pooled
             span, so anything less means the dense->sparse index remap or the affix
             matching is off, and any `color` number would be meaningless.

The label-shift control from the training demo is reported alongside the headline
number rather than as a separate run: the same live probabilities are re-scored against
labels rolled within each conversation, which collapses to chance if the vectors were
carrying a conversation-level signature instead of turn-local context.

    python tests/forward/demo_turn_vectors_deploy.py --task color --parity
    python tests/forward/demo_turn_vectors_deploy.py --task action

Writes `dump/turn_vectors_deploy_<task>.html` (self-contained; every turn's image is
inlined) plus a `.json` of the same records.
"""

import argparse
import base64
import html
import io
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for p in (str(_ROOT), str(_ROOT / "src"), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoProcessor

from longnav.utils.modeling import Qwen3VLSparseForConditionalGeneration
from longnav.utils.turn_vectors import (
    ACTION_POSTFIX,
    ACTION_PREFIX,
    DEFAULT_POSTFIX,
    DEFAULT_PREFIX,
    extract_turn_vectors,
    resolve_affix_ids,
)

# The data generator, label spaces and head builder are imported, not copied: the live
# conversations must come from exactly the generator the cached head was trained on.
from demo_turn_vectors_train import (  # noqa: E402  (sibling file, path added above)
    ACTIONS,
    COLORS,
    MODEL_ID,
    build_head,
    make_conversation,
)

EVAL_CACHE_SEED = 1234  # seed `build_split` used for the cached eval split


# --------------------------------------------------------------------------------------
# Phase A: train the head from the cached features
# --------------------------------------------------------------------------------------
def train_head_from_cache(args, cache, device):
    """Train the head on cached train states; return (head, history, ckpt_path)."""
    tr = cache["train"]
    classes = ACTIONS if args.task == "action" else COLORS
    tr_y = tr[args.task]

    head = build_head(args, tr["states"], len(classes), device)
    if args.standardize:
        head.fit_input_stats(tr["states"], tr["mask"])

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-2)
    states, mask, labels = (
        tr["states"].to(device),
        tr["mask"].to(device),
        tr_y.to(device),
    )
    history = []
    for epoch in range(1, args.epochs + 1):
        head.train()
        opt.zero_grad()
        loss = F.cross_entropy(head(states, mask), labels)
        loss.backward()
        opt.step()
        if epoch % max(1, args.epochs // 6) == 0:
            head.eval()
            with torch.no_grad():
                acc = float(
                    (head(states, mask).argmax(-1) == labels).float().mean()
                )
            history.append({"epoch": epoch, "loss": float(loss), "train_acc": acc})
            print(f"  epoch {epoch:>3}: loss {loss.item():.4f}  train acc {acc:.3f}")

    ckpt = _ROOT / "dump" / (
        f"turn_vector_head_{args.task}_{args.mode}_{args.affixes}"
        f"{'_shiftleft' if args.shift_left else ''}.pt"
    )
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": head.state_dict(),
            "task": args.task,
            "mode": args.mode,
            "classes": classes,
            "hidden_size": tr["states"].shape[-1],
            "content_len": tr["states"].shape[1],
            "standardize": args.standardize,
            "shift_left": args.shift_left,
            "affixes": args.affixes,
        },
        ckpt,
    )
    print(f"  head -> {ckpt}")
    return head, history, ckpt


def load_head(args, ckpt_path, device):
    """Rebuild the head from disk -- phase B never sees the training-time object.

    The checkpoint, not `args`, is the authority on `task`/`mode`/`standardize`: the
    pooling buffers exist in every mode, so a mismatched `--head-ckpt` run would load
    cleanly and then score against the wrong label space.
    """
    blob = torch.load(ckpt_path, weights_only=False)
    # `shift_left`/`affixes` matter as much as the rest: under mode='mean' a head
    # trained on 1-token shifted spans accepts unshifted 3-token states without a shape
    # error and returns a garbage number.
    for key in ("task", "mode", "standardize", "shift_left", "affixes"):
        if blob[key] != getattr(args, key):
            raise SystemExit(
                f"checkpoint {ckpt_path.name} was trained with {key}={blob[key]!r} "
                f"but --{key.replace('_', '-')} is {getattr(args, key)!r}"
            )
    dummy = torch.zeros(1, blob["content_len"], blob["hidden_size"])
    head = build_head(args, dummy, len(blob["classes"]), device)
    head.load_state_dict(blob["state_dict"])
    head.eval()
    return head, blob["classes"]


# --------------------------------------------------------------------------------------
# Phase B: live inference
# --------------------------------------------------------------------------------------
def load_model(args):
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
    return model, processor, device


@torch.inference_mode()
def run_conversation(model, processor, head, messages, images, prefix_ids, postfix_ids,
                     shift_left=False):
    """One live forward pass -> per-turn class vectors + sparsification stats.

    This is the deployed path: `head` is passed straight into `extract_turn_vectors`,
    so the returned tensor is the per-turn vector itself (out_dim == n_classes here, so
    it doubles as logits) and no hidden states are ever cached.
    """
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    inputs = processor(
        text=text, images=images, videos=None, padding=False, return_tensors="pt"
    )
    t0 = time.perf_counter()
    outputs = model(**inputs.to(model.device), use_cache=False, logits_to_keep=1)
    vectors, spans = extract_turn_vectors(
        outputs,
        inputs["input_ids"],
        head=head,
        prefix_ids=prefix_ids,
        postfix_ids=postfix_ids,
        shift_left=shift_left,
        strict=True,  # a dropped content token is an error, never a silent neighbour
    )
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    latency = time.perf_counter() - t0

    vis_keep = outputs["vis_keep_mask"]
    stats = {
        "dense": int(inputs["input_ids"].shape[1]),
        "sparse": int(outputs["last_hidden_state"].shape[1]),
        "vis_kept": int(vis_keep.sum()),
        "vis_total": int(vis_keep.numel()),
        "latency_s": latency,
    }
    return vectors.float().cpu(), spans, stats, inputs["input_ids"].cpu()


def deploy(args, model, processor, head, classes, prefix_ids, postfix_ids, seed, n_convs):
    """Live inference over `n_convs` fresh conversations. Returns per-conv records."""
    tokenizer = processor.tokenizer
    rng = np.random.default_rng(seed)
    records = []

    # Warmup on a throwaway conversation: the first forward pays cuda/kernel autotune
    # costs (~3-4x), which would otherwise land in the reported per-conversation latency.
    warm_msgs, warm_imgs, _, _ = make_conversation(np.random.default_rng(0), args.turns)
    run_conversation(model, processor, head, warm_msgs, warm_imgs, prefix_ids,
                     postfix_ids, args.shift_left)

    for c in range(n_convs):
        messages, images, colors, actions = make_conversation(rng, args.turns)
        vectors, spans, stats, input_ids = run_conversation(
            model, processor, head, messages, images, prefix_ids, postfix_ids,
            args.shift_left,
        )
        assert len(spans) == args.turns, f"found {len(spans)} turns, expected {args.turns}"
        probs = vectors.softmax(-1)
        truth = colors if args.task == "color" else actions

        turns = []
        for i, span in enumerate(spans):
            turns.append(
                {
                    "turn": i,
                    "dense_span": [int(span.start), int(span.end)],
                    "sparse_span": [
                        int(span.indices.min()),
                        int(span.indices.max()) + 1,
                    ],
                    "content": tokenizer.decode(
                        input_ids[span.batch_idx, span.start : span.end]
                    ),
                    # Read back out of the messages actually sent, never re-typed.
                    "user_text": messages[2 * i]["content"][1]["text"],
                    "color": COLORS[colors[i]],
                    "action": ACTIONS[actions[i]],
                    "true": int(truth[i]),
                    "pred": int(probs[i].argmax()),
                    "probs": [round(float(p), 4) for p in probs[i]],
                    "image": _png_data_uri(images[i]),
                }
            )
        records.append(
            {
                "conv": c,
                "stats": stats,
                "turns": turns,
                "truth": [int(t) for t in truth],
                "pred": [int(p) for p in probs.argmax(-1)],
                "probs": probs.tolist(),
            }
        )
        acc = float(np.mean([t["pred"] == t["true"] for t in records[-1]["turns"]]))
        print(
            f"  conv {c}: {stats['dense']:>5} -> {stats['sparse']:>5} tokens, "
            f"visual kept {stats['vis_kept']:>4}/{stats['vis_total']:<4} "
            f"{stats['latency_s']*1e3:>6.0f} ms  acc {acc:.3f}"
        )
    return records


def _png_data_uri(image, size=112):
    buf = io.BytesIO()
    image.resize((size, size)).save(buf, format="JPEG", quality=72)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------
def shifted_labels(truth, shift):
    """Roll labels within the conversation -- the control from the training demo."""
    return list(np.roll(np.asarray(truth), -shift)) if shift else list(truth)


def summarize(records, classes, shift):
    n = len(classes)
    confusion = np.zeros((n, n), dtype=int)
    hits, total, ctrl_hits = 0, 0, 0
    per_conv = []
    for rec in records:
        preds, truth = rec["pred"], rec["truth"]
        ctrl = shifted_labels(truth, shift)
        conv_hits = 0
        for p, t, ct in zip(preds, truth, ctrl):
            confusion[t, p] += 1
            conv_hits += int(p == t)
            ctrl_hits += int(p == ct)
        hits += conv_hits
        total += len(preds)
        per_conv.append(conv_hits / len(preds))
    mean_conf = float(np.mean([max(p) for r in records for p in r["probs"]]))
    return {
        "n_turns": total,
        "accuracy": hits / total,
        "control_accuracy": ctrl_hits / total,
        "control_shift": shift,
        "chance": 1.0 / n,
        "per_conv_accuracy": per_conv,
        "confusion": confusion.tolist(),
        "mean_pred_confidence": mean_conf,
        "mean_latency_ms": float(np.mean([r["stats"]["latency_s"] for r in records]) * 1e3),
        "mean_dense": float(np.mean([r["stats"]["dense"] for r in records])),
        "mean_sparse": float(np.mean([r["stats"]["sparse"] for r in records])),
        "vis_kept": int(sum(r["stats"]["vis_kept"] for r in records)),
        "vis_total": int(sum(r["stats"]["vis_total"] for r in records)),
    }


def parity_check(args, model, processor, head, classes, prefix_ids, postfix_ids, cache, device):
    """Live vs cache on the identical conversations that built the eval cache.

    `build_split` draws all its conversations from one `default_rng(seed)`, so replaying
    the same seed for the same number of conversations reproduces them token for token.
    """
    ev = cache["eval"]
    n_convs = ev["states"].shape[0] // args.turns
    records = deploy(
        args, model, processor, head, classes, prefix_ids, postfix_ids,
        seed=EVAL_CACHE_SEED, n_convs=n_convs,
    )
    live_pred = np.array([p for r in records for p in r["pred"]])
    with torch.no_grad():
        cache_pred = (
            head(ev["states"].to(device), ev["mask"].to(device)).argmax(-1).cpu().numpy()
        )
    truth = ev[args.task].numpy()
    agree = float((live_pred == cache_pred).mean())
    print(
        f"  parity over {len(live_pred)} turns: live vs cache predictions agree "
        f"{agree:.3f}  (live acc {float((live_pred==truth).mean()):.3f}, "
        f"cache acc {float((cache_pred==truth).mean()):.3f})"
    )
    return {
        "n_turns": int(len(live_pred)),
        "agreement": agree,
        "live_accuracy": float((live_pred == truth).mean()),
        "cache_accuracy": float((cache_pred == truth).mean()),
    }


# --------------------------------------------------------------------------------------
# Visualization
# --------------------------------------------------------------------------------------
CSS = """
:root{--bg:#fbfaf8;--panel:#fff;--ink:#1a1a1c;--muted:#6c6a70;--line:#e4e1dc;
--ok:#1f7a4d;--miss:#b3261e;--accent:#3b5bdb;--chip:#f1efec;}
@media(prefers-color-scheme:dark){:root{--bg:#141416;--panel:#1c1c20;--ink:#eceaf0;
--muted:#9b98a2;--line:#2e2e34;--ok:#4ec98a;--miss:#ff8b7e;--accent:#8fa4ff;--chip:#26262c;}}
:root[data-theme=dark]{--bg:#141416;--panel:#1c1c20;--ink:#eceaf0;--muted:#9b98a2;
--line:#2e2e34;--ok:#4ec98a;--miss:#ff8b7e;--accent:#8fa4ff;--chip:#26262c;}
:root[data-theme=light]{--bg:#fbfaf8;--panel:#fff;--ink:#1a1a1c;--muted:#6c6a70;
--line:#e4e1dc;--ok:#1f7a4d;--miss:#b3261e;--accent:#3b5bdb;--chip:#f1efec;}
body{background:var(--bg);color:var(--ink);font:15px/1.55 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
margin:0;padding:2.2rem 1.2rem 4rem;overflow-x:hidden;}
.wrap{max-width:1080px;margin:0 auto;}
h1{font-size:1.7rem;margin:0 0 .3rem;letter-spacing:-.01em;}
h2{font-size:1.05rem;margin:2.4rem 0 .8rem;text-transform:uppercase;letter-spacing:.08em;
color:var(--muted);font-weight:600;}
.sub{color:var(--muted);margin:0 0 1.6rem;max-width:70ch;}
code,.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.86em;}
.pipe{display:flex;flex-wrap:wrap;gap:.4rem;align-items:center;background:var(--panel);
border:1px solid var(--line);border-radius:10px;padding:.7rem .9rem;margin-bottom:1.6rem;}
.pipe span{background:var(--chip);border-radius:6px;padding:.25rem .55rem;}
.pipe i{color:var(--muted);font-style:normal;}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:.8rem;}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:.9rem 1rem;}
.card .k{color:var(--muted);font-size:.78rem;text-transform:uppercase;letter-spacing:.06em;}
.card .v{font-size:1.55rem;font-weight:650;letter-spacing:-.02em;margin-top:.15rem;}
.card .n{color:var(--muted);font-size:.8rem;}
.bars{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:1rem;}
.bar{display:grid;grid-template-columns:11rem 1fr 3.4rem;gap:.7rem;align-items:center;margin:.45rem 0;}
.bar .t{background:var(--chip);border-radius:5px;height:12px;overflow:hidden;}
.bar .f{height:100%;border-radius:5px;background:var(--accent);}
.bar .f.dim{background:var(--muted);opacity:.55;}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch;}
table{border-collapse:collapse;width:100%;min-width:640px;background:var(--panel);}
th,td{padding:.45rem .6rem;border-bottom:1px solid var(--line);text-align:left;font-size:.87rem;
white-space:nowrap;}
th{color:var(--muted);font-weight:600;font-size:.76rem;text-transform:uppercase;letter-spacing:.05em;}
.ok{color:var(--ok);font-weight:600;}.miss{color:var(--miss);font-weight:600;}
.conv{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:1rem;margin-bottom:1.1rem;}
.conv h3{margin:0 0 .2rem;font-size:.98rem;}
.conv .meta{color:var(--muted);font-size:.82rem;margin-bottom:.8rem;}
.turns{display:flex;gap:.6rem;overflow-x:auto;padding-bottom:.4rem;}
.turn{flex:0 0 118px;}
.turn img{width:112px;height:112px;border-radius:8px;display:block;border:2px solid transparent;max-width:100%;}
.turn.hit img{border-color:var(--ok);}.turn.miss img{border-color:var(--miss);}
.turn .lab{font-size:.75rem;color:var(--muted);margin-top:.3rem;line-height:1.35;}
.turn .lab b{color:var(--ink);}
.cm{border-collapse:collapse;min-width:0;width:auto;}
.cm td,.cm th{text-align:center;padding:.4rem .7rem;font-variant-numeric:tabular-nums;}
.cm td.d{font-weight:700;color:var(--ok);}
.note{color:var(--muted);font-size:.85rem;max-width:78ch;}
"""


def render_html(args, summary, records, parity, history, ckpt_path, classes):
    e = html.escape
    task = args.task
    chance = summary["chance"]

    def pct(x):
        return f"{x*100:.1f}%"

    def bar(label, value, dim=False):
        return (
            f'<div class="bar"><div class="mono">{e(label)}</div>'
            f'<div class="t"><div class="f{" dim" if dim else ""}" '
            f'style="width:{min(100, value*100):.1f}%"></div></div>'
            f'<div class="mono">{value:.3f}</div></div>'
        )

    cards = [
        ("Live accuracy", pct(summary["accuracy"]),
         f"{summary['n_turns']} turns, {len(records)} conversations"),
        (f"Control (shift {summary['control_shift']})", pct(summary["control_accuracy"]),
         f"chance {pct(chance)}"),
        ("Mean p(pred)", f"{summary['mean_pred_confidence']:.3f}", "softmax of turn vector"),
        ("Forward latency", f"{summary['mean_latency_ms']:.0f} ms",
         "per conversation, live"),
        ("Token compression",
         f"{summary['mean_dense']:.0f}→{summary['mean_sparse']:.0f}",
         f"visual kept {summary['vis_kept']}/{summary['vis_total']}"),
    ]
    if parity:
        cards.insert(2, ("Live vs cache", pct(parity["agreement"]),
                         f"{parity['n_turns']} replayed turns"))

    card_html = "".join(
        f'<div class="card"><div class="k">{e(k)}</div><div class="v">{e(v)}</div>'
        f'<div class="n">{e(n)}</div></div>'
        for k, v, n in cards
    )

    conv_html = []
    for rec in records:
        acc = float(np.mean([t["pred"] == t["true"] for t in rec["turns"]]))
        s = rec["stats"]
        tiles = []
        for t in rec["turns"]:
            hit = t["pred"] == t["true"]
            tiles.append(
                f'<div class="turn {"hit" if hit else "miss"}">'
                f'<img src="{t["image"]}" alt="turn {t["turn"]} frame">'
                f'<div class="lab">t{t["turn"]} <span class="mono">{e(t["content"])}</span><br>'
                f'true <b>{e(classes[t["true"]])}</b><br>'
                f'pred <b>{e(classes[t["pred"]])}</b> {t["probs"][t["pred"]]:.2f}</div></div>'
            )
        rows = "".join(
            f'<tr><td>{t["turn"]}</td><td class="mono">{e(t["user_text"])}</td>'
            f'<td class="mono">{e(t["content"])}</td>'
            f'<td class="mono">[{t["dense_span"][0]}:{t["dense_span"][1]}]</td>'
            f'<td class="mono">[{t["sparse_span"][0]}:{t["sparse_span"][1]}]</td>'
            f'<td>{e(classes[t["true"]])}</td><td>{e(classes[t["pred"]])}</td>'
            f'<td class="mono">{t["probs"][t["pred"]]:.3f}</td>'
            f'<td class="{"ok" if t["pred"]==t["true"] else "miss"}">'
            f'{"OK" if t["pred"]==t["true"] else "MISS"}</td></tr>'
            for t in rec["turns"]
        )
        conv_html.append(
            f'<div class="conv"><h3>Conversation {rec["conv"]} &mdash; accuracy '
            f'{acc:.3f}</h3><div class="meta mono">{s["dense"]} dense &rarr; '
            f'{s["sparse"]} sparse tokens &middot; visual kept {s["vis_kept"]}/'
            f'{s["vis_total"]} &middot; {s["latency_s"]*1e3:.0f} ms</div>'
            f'<div class="turns">{"".join(tiles)}</div>'
            f'<div class="scroll"><table><thead><tr><th>turn</th><th>user text</th>'
            f'<th>content</th><th>dense span</th><th>sparse span</th><th>true</th>'
            f'<th>pred</th><th>p(pred)</th><th></th></tr></thead>'
            f'<tbody>{rows}</tbody></table></div></div>'
        )

    cm = summary["confusion"]
    cm_rows = "".join(
        f'<tr><th>{e(c)}</th>'
        + "".join(
            f'<td class="{"d" if i==j else ""}">{cm[i][j]}</td>'
            for j in range(len(classes))
        )
        + "</tr>"
        for i, c in enumerate(classes)
    )
    cm_html = (
        '<div class="scroll"><table class="cm"><thead><tr><th></th>'
        + "".join(f"<th>pred {e(c)}</th>" for c in classes)
        + f"</tr></thead><tbody>{cm_rows}</tbody></table></div>"
    )

    bars = "".join(
        [bar(f"conv {i}", a) for i, a in enumerate(summary["per_conv_accuracy"])]
        + [
            bar("live (all turns)", summary["accuracy"]),
            bar(f"control shift {summary['control_shift']}",
                summary["control_accuracy"], dim=True),
            bar("chance", chance, dim=True),
        ]
    )

    parity_note = (
        f'<p class="note">Replaying the seed-{EVAL_CACHE_SEED} conversations that built '
        f'the eval feature cache: live predictions agree with cache-based predictions on '
        f'<b>{pct(parity["agreement"])}</b> of {parity["n_turns"]} turns '
        f'(live accuracy {parity["live_accuracy"]:.3f}, cache accuracy '
        f'{parity["cache_accuracy"]:.3f}). The live path is the cached path.</p>'
        if parity
        else '<p class="note">Parity against the feature cache was not run '
        "(<code>--parity</code>).</p>"
    )

    trained = (
        f"{history[-1]['train_acc']:.3f} train accuracy after {history[-1]['epoch']} "
        f"epochs (loss {history[-1]['loss']:.4f})"
        if history
        else "loaded from checkpoint"
    )

    return f"""<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Deployed turn vectors &mdash; {e(task)}</title>
<style>{CSS}</style>
<div class="wrap">
<h1>Per-turn vectors, deployed</h1>
<p class="sub">Every number below comes from a live sparse Qwen3-VL forward pass over a
full multimodal conversation &mdash; no cached hidden states. One vector is pooled per
assistant turn and read out as a <b>{e(task)}</b> class over
{e(", ".join(classes))}.</p>

<div class="pipe mono">
<span>{len(records)} conversations &times; {args.turns} turns</span><i>&rarr;</i>
<span>processor</span><i>&rarr;</i>
<span>{e(args.model_id)} (sparse, frozen)</span><i>&rarr;</i>
<span>extract_turn_vectors(affixes={e(args.affixes)}, shift_left={e(str(args.shift_left))})</span><i>&rarr;</i>
<span>TurnVectorHead &rarr; {len(classes)}-d vector</span><i>&rarr;</i>
<span>metrics</span>
</div>

<div class="cards">{card_html}</div>

<h2>Is the signal turn-local?</h2>
<div class="bars">{bars}</div>
<p class="note">The control re-scores the <em>same</em> live predictions against labels
rolled by {summary['control_shift']} turn(s) inside each conversation. All of the
conversation's images are in context, so a head reading a conversation-level signature
would score just as well under the roll. Collapsing to chance
({chance:.3f}) is what makes the headline number mean &ldquo;this turn&rdquo;.</p>

<h2>Confusion</h2>
{cm_html}

<h2>Live vs cache</h2>
{parity_note}
<p class="note">Head: <code>mode={e(args.mode)}</code>,
out_dim={len(classes)}, standardize={e(str(args.standardize))} &mdash; {e(trained)} on the
cached train split, then reloaded from <code>{e(ckpt_path.name)}</code> before any live
inference. Backbone frozen throughout.</p>

<h2>Conversations</h2>
{"".join(conv_html)}
</div>
"""


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--task", default="color", choices=["action", "color"])
    ap.add_argument("--turns", type=int, default=8)
    ap.add_argument("--live-convs", type=int, default=6,
                    help="fresh conversations to run through the live model")
    ap.add_argument("--seed", type=int, default=4242,
                    help="rng seed for the live conversations (unseen during training)")
    ap.add_argument("--mode", default="mean", choices=["mean", "last", "attn", "flat"])
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--standardize", action="store_true")
    ap.add_argument("--attn-impl", default="sdpa", choices=["sdpa", "flash_attention_2"])
    ap.add_argument("--affixes", default="template", choices=["template", "action"],
                    help="must match the feature cache the head is trained from")
    ap.add_argument("--shift-left", action="store_true",
                    help="pool the states that PREDICT the content rather than the "
                    "content tokens. Must match the feature cache the head was trained "
                    "on; with --affixes action this is the content-free config.")
    ap.add_argument("--control-shift", type=int, default=1,
                    help="turns to roll labels by for the turn-locality control")
    ap.add_argument("--parity", action="store_true",
                    help="also replay the cached eval conversations live and compare")
    ap.add_argument("--cache", default=None, help="feature cache (.pt) to train the head on")
    ap.add_argument("--head-ckpt", default=None,
                    help="skip phase A and deploy this head checkpoint")
    ap.add_argument("--out", default=None, help="output .html path")
    args = ap.parse_args()

    if args.attn_impl == "flash_attention_2":
        from longnav.utils.turn_vectors import patch_flash_attention_packing

        patch_flash_attention_packing()

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache_path = Path(
        args.cache
        or _ROOT / "dump" / f"turn_vectors_toy_{args.turns}x24+8"
        f"{'_actionaffix' if args.affixes == 'action' else ''}"
        f"{'_shiftleft' if args.shift_left else ''}.pt"
    )

    cache = None
    history = []
    if args.head_ckpt:
        ckpt_path = Path(args.head_ckpt)
        print(f"Phase A skipped; deploying head from {ckpt_path}")
    else:
        if not cache_path.exists():
            raise SystemExit(
                f"feature cache {cache_path} not found -- build it first with "
                "`python tests/forward/demo_turn_vectors_train.py`"
            )
        print(f"Phase A: training head on cached features from {cache_path}")
        cache = torch.load(cache_path, weights_only=False)
        _, history, ckpt_path = train_head_from_cache(args, cache, device)

    head, classes = load_head(args, ckpt_path, device)
    model, processor, _ = load_model(args)
    prefix, postfix = (
        (DEFAULT_PREFIX, DEFAULT_POSTFIX)
        if args.affixes == "template"
        else (ACTION_PREFIX, ACTION_POSTFIX)
    )
    prefix_ids, postfix_ids = resolve_affix_ids(processor.tokenizer, prefix, postfix)
    print(f"Affixes: prefix={prefix_ids} postfix={postfix_ids}")

    print(f"\nPhase B: live inference on {args.live_convs} fresh conversations "
          f"(seed {args.seed})")
    records = deploy(args, model, processor, head, classes, prefix_ids, postfix_ids,
                     args.seed, args.live_convs)
    summary = summarize(records, classes, args.control_shift)

    parity = None
    if args.parity:
        if cache is None:
            cache = torch.load(cache_path, weights_only=False)
        print(f"\nParity: replaying the cached eval conversations through the live model")
        parity = parity_check(args, model, processor, head, classes, prefix_ids,
                              postfix_ids, cache, device)

    print(
        f"\nLive {args.task} accuracy {summary['accuracy']:.3f} over "
        f"{summary['n_turns']} turns  |  control (shift {args.control_shift}) "
        f"{summary['control_accuracy']:.3f}  |  chance {summary['chance']:.3f}"
    )
    if torch.cuda.is_available():
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")

    out = Path(
        args.out
        or _ROOT / "dump" / (
            f"turn_vectors_deploy_{args.task}_{args.affixes}"
            f"{'_shiftleft' if args.shift_left else ''}.html"
        )
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        render_html(args, summary, records, parity, history, ckpt_path, classes),
        encoding="utf-8",
    )
    json_out = out.with_suffix(".json")
    json_out.write_text(
        json.dumps(
            {
                "args": vars(args),
                "summary": summary,
                "parity": parity,
                "train_history": history,
                "records": [
                    {**r, "turns": [{k: v for k, v in t.items() if k != "image"}
                                    for t in r["turns"]]}
                    for r in records
                ],
            },
            indent=2,
        )
    )
    print(f"Visualization -> {out}\nRecords -> {json_out}")


if __name__ == "__main__":
    main()
