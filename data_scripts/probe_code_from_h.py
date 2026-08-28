#!/usr/bin/env python3
"""Is the code head near H(c|h), or is it failing to extract what `h` already contains?

The in-run head reaches eval CE 4.578 on objectnav against a corpus marginal H(c) =
4.838 -- only 0.26 nats of mutual information -- while on pointnav it reaches 2.734.
That split is expected (pointnav is point-to-point, objectnav is search, so `H(c|h)` is
genuinely higher there), but it does not say WHICH limit objectnav is against.

This settles it. Cache `h` for held-out turns once, then fit an unconstrained head on
those FROZEN features to convergence, with no backbone in the loop and no competing
flow-matching objective:

  * plateaus at the in-run CE  -> `h` does not carry more about the code. The head is
    fine and the lever is elsewhere: shorter context, different conditioning, or
    accepting that the code is only weakly predictable in a search task.
  * reaches materially lower   -> the in-run head or its optimisation is the limit, and
    is worth fixing.

Reported against three references, because a CE alone means nothing:
  ln(V)      a uniform head
  H(c)       the corpus marginal -- a constant bias-only predictor
  in-run CE  what the live head achieves

Fits both a LINEAR head and an MLP: if the MLP does not beat the linear one, the
relationship `h -> c` is not the thing being lost.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "src"), str(_ROOT / "data_scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


@torch.no_grad()
def collect(model, ds, rows, collator, device, tok, vocab_theta):
    """(h, joint code) for every turn of every sampled row."""
    from latent_spread_probe import collect_h  # the same readout the head itself uses

    H, C, R, P = [], [], [], []
    for n, r in enumerate(rows):
        row = ds[int(r)]
        chunks = np.asarray(row["action_chunks"], dtype=np.float32)
        if chunks.ndim != 3:
            continue
        batch = collator([row])
        skip = {"targets", "labels", "num_turns", "stop_targets"}
        inputs = {k: (v.to(device) if torch.is_tensor(v) else v)
                  for k, v in batch.items() if k not in skip}
        modality = {k: inputs.pop(k) for k in list(inputs) if k.startswith("modality_")}
        from longnav.utils.modality_embed import ModalityBatch
        from longnav.utils.turn_vectors import extract_turn_vectors

        mb = ModalityBatch.pop_from({**modality}, known_keys=model.modality_embedder.keys)
        with model.modality_embedder.pending(mb):
            # `logits_to_keep=1` is not optional: without it the LM head materialises
            # full-vocab logits for EVERY position, which is gigabytes for a long turn
            # sequence and has nothing to do with the readout we want. Same call
            # `latent_spread_probe.collect_h` makes.
            out = model.backbone(use_cache=False, logits_to_keep=1, **inputs)
        # TWO readouts from the same forward pass, because the code head does not see
        # the backbone's hidden state -- it sees the readout head's OUTPUT.
        #   states -> pooled_context() -> (N, 2048)   the mean-pooled residual stream
        #          -> project()        -> (N, 1024)   `fm_context_dim`, what the flow
        #                                             decoder, r(h) and CodePolicyHead
        #                                             all actually consume.
        # `fm_context_dim = 1024` is a DEFAULT (flow_matching_head.py:1218), never
        # chosen for this model, so the 2048 -> 1024 projection is an unexamined
        # bottleneck sitting directly upstream of the code head. Probing both sides of
        # it costs one extra tensor and answers whether it is throwing code
        # information away.
        states, _, smask = extract_turn_vectors(
            out, inputs["input_ids"], None, prefix_ids=model.prefix_ids,
            postfix_ids=model.postfix_ids, shift_left=model.model_cfg.shift_left,
            strict=True, return_mask=True)
        pooled = model.head.pooled_context(states, smask)
        h = model.head.project(pooled).float().cpu()
        pooled = pooled.float().cpu()
        t = min(len(h), len(chunks))
        cx, ct = tok.encode_chunk(torch.from_numpy(chunks[:t]).to(device))
        P.append(pooled[:t])
        H.append(h[:t])
        C.append((cx * vocab_theta + ct).cpu())
        # WHICH ROW each turn came from. The train/val split must be by ROW, not by
        # turn: turns inside one episode are heavily correlated (same scene, same
        # geometry, overlapping chunks), so a turn-level split leaves nearly every
        # val turn with siblings in train and the probe reads the episode instead of
        # generalising. The first run of this made that mistake.
        R.append(torch.full((t,), int(r), dtype=torch.long))
        if n % 10 == 0:
            print(f"  {n}/{len(rows)} rows, {sum(len(x) for x in H)} turns", flush=True)
    return torch.cat(H), torch.cat(C), torch.cat(R), torch.cat(P)


def fit(name, head, Xtr, Ytr, Xva, Yva, steps, lr, bs, device, wd=0.01, patience=6):
    """EARLY STOPPING ON VAL, and the BEST val CE is what is returned.

    The first version of this ran a fixed step count and reported the last value. Both
    heads were overfitting -- the linear one's val CE ROSE across the second half and the
    MLP diverged to CE 16.8, well above uniform -- so the number it printed measured
    overfitting, not what `h` carries. A probe that is asking "is this the information
    ceiling" has to be at ITS OWN best, or the answer is meaningless.
    """
    head = head.to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)
    best, best_acc, bad, every = float("inf"), 0.0, 0, max(steps // 20, 1)
    for s in range(steps):
        i = torch.randint(0, len(Xtr), (bs,))
        loss = F.cross_entropy(head(Xtr[i].to(device)), Ytr[i].to(device))
        loss.backward(); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        if (s + 1) % every == 0:
            with torch.no_grad():
                ce_sum = n = corr = 0.0
                for j in range(0, len(Xva), 4096):
                    lg = head(Xva[j:j + 4096].to(device))
                    y = Yva[j:j + 4096].to(device)
                    ce_sum += float(F.cross_entropy(lg, y, reduction="sum"))
                    corr += float((lg.argmax(-1) == y).sum()); n += len(y)
                ce, acc = ce_sum / n, corr / n
            if ce < best - 1e-4:
                best, best_acc, bad = ce, acc, 0
            else:
                bad += 1
            print(f"    {name} step {s+1:6d}  val CE {ce:.4f}  acc {acc:.3f}"
                  f"{'  *' if bad == 0 else ''}", flush=True)
            if bad >= patience:
                print(f"    {name}: early stop, best val CE {best:.4f}", flush=True)
                break
    return best, best_acc


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--rows", type=int, default=120)
    ap.add_argument("--max-turns", type=int, default=200,
                    help="turns per sample. MUST match the training run (v3/v4 use 200); "
                         "DataConfig defaults to 16, which computes h from a 16-turn "
                         "window while the in-run head sees up to 200 turns of context. "
                         "That is a different h, and not the one the number compares to")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--in-run-ce", type=float, default=None,
                    help="the live head's eval CE on this component, for comparison")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mem-fraction", type=float, default=0.10)
    ap.add_argument("--cache", default=None,
                    help="path to save/reuse the cached (h, code, row) tensors, so "
                         "re-fitting does not repeat the backbone pass")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from datasets import load_from_disk
    from transformers import AutoProcessor

    from longnav.utils.chunk_tokenizer import FrozenChunkTokenizer
    from longnav.utils.flow_matching_head import TurnFlowActionRegressor
    from longnav.utils.vector_sft import DataConfig, TurnVectorCollator

    dev = args.device
    if dev.startswith("cuda") and args.mem_fraction > 0:
        torch.cuda.set_per_process_memory_fraction(args.mem_fraction)

    meta = json.loads((Path(args.checkpoint) / "turn_vector_head_config.json").read_text())
    processor = AutoProcessor.from_pretrained(meta["model"]["model_id"])
    model = TurnFlowActionRegressor.from_pretrained(args.checkpoint, processor, device=dev)
    model.eval()
    tok = FrozenChunkTokenizer(args.tokenizer).to(dev)
    V = tok.vocab_xy * tok.vocab_theta

    ds = load_from_disk(args.dataset)
    ds = ds[args.split] if hasattr(ds, "keys") and args.split in ds else ds
    rng = np.random.default_rng(0)
    rows = rng.choice(len(ds), size=min(args.rows, len(ds)), replace=False)
    collator = TurnVectorCollator(
        processor=processor,
        data=DataConfig(target_column="action_chunks",
                        max_turns_per_sample=args.max_turns),
        train=False, modality_specs=tuple(model.model_cfg.modality_specs))

    cache = Path(args.cache) if args.cache else None
    if cache is not None and cache.exists():
        blob = torch.load(cache)
        X, Y, Rw, Pl = blob["X"], blob["Y"], blob["R"], blob["P"]
        print(f"loaded cached h from {cache}: {len(X)} turns", flush=True)
        del model
    else:
        print("caching h ...", flush=True)
        X, Y, Rw, Pl = collect(model, ds, rows, collator, dev, tok, tok.vocab_theta)
        print(f"{len(X)} turns, context dim {X.shape[1]}, pooled dim {Pl.shape[1]}")
        del model
        if cache is not None:
            # The backbone pass is the expensive part (~20 min); the fits are seconds.
            # Cache so a change to the probe head does not re-pay for the features.
            cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"X": X, "Y": Y, "R": Rw, "P": Pl}, cache)
    torch.cuda.empty_cache()

    # SPLIT BY ROW. See the note in collect(): a turn-level split leaks episode
    # identity across the boundary and inflates the probe against an in-run eval
    # number that is measured on held-out EPISODES.
    uniq = torch.unique(Rw)
    g = torch.Generator().manual_seed(0)
    order = uniq[torch.randperm(len(uniq), generator=g)]
    n_va_rows = max(1, int(round(len(order) * 0.2)))
    va_rows = set(int(v) for v in order[:n_va_rows])
    is_va = torch.tensor([int(r) in va_rows for r in Rw])
    va, tr = torch.nonzero(is_va).squeeze(1), torch.nonzero(~is_va).squeeze(1)
    Xtr, Ytr, Xva, Yva = X[tr], Y[tr], X[va], Y[va]
    Ptr, Pva = Pl[tr], Pl[va]
    print(f"split by row: {len(uniq) - n_va_rows} train rows / {n_va_rows} val rows "
          f"-> {len(Xtr)} / {len(Xva)} turns", flush=True)

    # The two references any CE has to be read against.
    counts = torch.bincount(Ytr, minlength=V).float() + 1e-6
    p = counts / counts.sum()
    H_marg = float(-(p * p.log()).sum())
    ce_marg = float(F.cross_entropy(p.log().expand(len(Yva), V), Yva))
    print(f"\nreferences: ln(V) = {np.log(V):.4f} | H(c) train marginal = {H_marg:.4f} | "
          f"marginal-predictor CE on val = {ce_marg:.4f}"
          + (f" | in-run head = {args.in_run_ce:.4f}" if args.in_run_ce else ""))

    d = X.shape[1]
    res = {"n_turns": int(len(X)), "n_train_turns": int(len(Xtr)),
           "n_val_turns": int(len(Xva)), "n_val_rows": int(n_va_rows),
           "split": "by_row", "context_dim": int(d), "pooled_dim": int(Pl.shape[1]),
           "vocab": V, "H_marginal": H_marg,
           "marginal_ce_on_val": ce_marg, "in_run_ce": args.in_run_ce}
    print("\nfitting on FROZEN h = the 1024-d CONTEXT the code head reads "
          "(joint, 1600-way):")
    res["linear"], _ = fit("linear", nn.Linear(d, V), Xtr, Ytr, Xva, Yva,
                           args.steps, args.lr, args.batch_size, dev)
    res["mlp"], _ = fit("mlp", nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2048),
                                             nn.SiLU(), nn.Linear(2048, V)),
                        Xtr, Ytr, Xva, Yva, args.steps, args.lr, args.batch_size, dev)

    # UPSTREAM OF THE PROJECTION. If this beats the 1024-d fit by a real margin, the
    # unchosen `fm_context_dim` default is destroying code information before the head
    # ever sees it, and widening it is a cheaper fix than anything downstream.
    dp = Pl.shape[1]
    print(f"\nfitting on the {dp}-d POOLED hidden, upstream of the readout projection:")
    res["pooled_linear"], _ = fit("pooled-linear", nn.Linear(dp, V), Ptr, Ytr, Pva, Yva,
                                  args.steps, args.lr, args.batch_size, dev)

    # THE PER-FACTOR PROBES, which are the better-conditioned question. 1600 classes
    # against the available turns is a hopeless ratio; 40 classes each is 40x better
    # posed and answers the same thing -- how much does `h` carry about the code.
    n_th = tok.vocab_theta
    for tag, y_tr, y_va, nc in (("xy", Ytr // n_th, Yva // n_th, tok.vocab_xy),
                                ("theta", Ytr % n_th, Yva % n_th, n_th)):
        cnt = torch.bincount(y_tr, minlength=nc).float() + 1e-6
        q = cnt / cnt.sum()
        marg = float(F.cross_entropy(q.log().expand(len(y_va), nc), y_va))
        print(f"\nfitting on FROZEN h ({tag}, {nc}-way; marginal CE {marg:.4f}):")
        ce, acc = fit(f"{tag}-mlp",
                      nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1024), nn.SiLU(),
                                    nn.Linear(1024, nc)),
                      Xtr, y_tr, Xva, y_va, args.steps, args.lr, args.batch_size, dev)
        res[f"{tag}_marginal_ce"] = marg
        res[f"{tag}_probe_ce"] = ce
        res[f"{tag}_probe_acc"] = acc

    print("\n" + "=" * 64)
    print(f"turns collected      {len(X)}  (train {len(Xtr)} / val {len(Xva)}, "
          f"SPLIT BY ROW over {len(uniq)} episodes)")
    print(f"marginal (bias only) {ce_marg:.4f}   [joint, {V}-way]")
    print(f"linear on ctx ({d})  {res['linear']:.4f}")
    print(f"MLP on ctx           {res['mlp']:.4f}")
    print(f"linear on pooled ({dp}) {res['pooled_linear']:.4f}   "
          f"[projection cost {res['linear'] - res['pooled_linear']:+.4f} nats]")
    for tag in ("xy", "theta"):
        print(f"{tag:>5}: marginal {res[f'{tag}_marginal_ce']:.4f} -> probe "
              f"{res[f'{tag}_probe_ce']:.4f}  (acc {res[f'{tag}_probe_acc']:.3f})")
    best = min(res["linear"], res["mlp"])
    if args.in_run_ce:
        print(f"in-run head          {args.in_run_ce:.4f}   "
              f"(best probe {best:.4f}, gap {args.in_run_ce - best:+.4f})")
    # No verdict printed -- the first version of this printed a confident conclusion off
    # a diverged fit. But the two directions are NOT symmetric, and that is the thing to
    # keep straight when reading the table:
    #   probe LOSES to the in-run head -> says nothing about `h`. The probe saw a
    #       fraction of the turns and had no backbone; being starved is the obvious
    #       explanation and it cannot be separated from a real ceiling.
    #   probe BEATS the in-run head -> a genuine LOWER BOUND on what `h` carries. A
    #       weaker model class on less data, reading frozen features, extracted more.
    #       Nothing about being starved can manufacture that.
    print("\nRead with care, and note the asymmetry: a probe LOSING to the in-run head "
          "is evidence about the probe (it saw fewer turns and had no backbone). A "
          "probe BEATING it is a lower bound on what h carries -- being starved cannot "
          "manufacture that.")
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
