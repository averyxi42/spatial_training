"""Score head-only checkpoints on the *same* rows, whatever support each was fitted on.

    python data_scripts/eval_head_only.py \
        --store dump/head_only/ctx_v9_ck11400/part{0,1,2,3} --store-shards 11,9,11,9 \
        --checkpoint baseline=dump/head_only/run_A_pooled/final \
        --checkpoint all_move=dump/head_only/run_F2_all_move/final \
        --support none --support all_move \
        --out dump/head_only/atom_filter/eval.json

Why a second entry point rather than reading the runs' own `eval_*`
------------------------------------------------------------------
Filtering changes the support, so each run's own eval is computed on a different set of
rows: the filtered run holds out filtered episodes' filtered chunks, the baseline holds out
all of them. Their `eval_creep_sample_dx` values are then two numbers about two
distributions, and subtracting them measures the filter as much as the model. Worse, the
comparison flatters whichever arm was scored on the easier support, and nothing raises.

So every checkpoint is evaluated here on every named support, off one episode split, with
one metric implementation -- `HeadOnlyFlowTrainer.evaluate`, i.e. the identical code that
produced the numbers inside each run. What this makes readable:

  * a model on **its own** support: is the fitted density leaking into the gap?
  * a model on **another** support: what did the change of support cost elsewhere? A
    filtered model on the full support cannot produce stop mass it was never shown, and
    that is the price of the fix rather than a bug in it.

`sigma_mult` is forced to 1.0 (the end of the anneal) on every checkpoint, because the
noise floor -- and therefore `min_nll_per_dim` and `nll_headroom` -- moves with it, and a
run evaluated mid-anneal is not comparable to one evaluated after. `nll_*` still must not
be read across supports: a density on a subset is a different normalisation constant. The
band statistics (`creep_*`, `stop_*`) and the errors (`rmse_*`, `mae_*`) are the readout,
and each is reported next to the ground truth's own value on that same support.
"""

import argparse
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "src"), str(_ROOT / "data_scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import TrainingArguments  # noqa: E402

from longnav.utils.head_only import (  # noqa: E402
    MOTION_FILTERS, ContextStore, FrozenContextCollator, FrozenContextDataset,
    FrozenContextFlowPolicy, HeadOnlyFlowTrainer, filter_rows_by_motion,
)
from longnav.utils.vector_sft import DataConfig  # noqa: E402


def _csv(s, cast=str):
    return tuple(cast(x) for x in str(s).split(",") if x.strip()) if s else ()


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--store", required=True, nargs="+")
    ap.add_argument("--store-shards", default=None,
                    help="pin each part to its first N shards, e.g. '11,9,11,9'. Must "
                         "match the runs under test or the split is not theirs")
    ap.add_argument("--checkpoint", action="append", required=True, metavar="NAME=PATH",
                    help="a head-only checkpoint directory (the one holding "
                         "head_only_config.json), optionally NAME=PATH")
    ap.add_argument("--support", action="append", choices=MOTION_FILTERS, default=None,
                    help="a filter naming a set of val rows to score on; repeatable. "
                         "Default: every mode")
    ap.add_argument("--filter-tol", type=float, default=1e-4)
    ap.add_argument("--context-key", default="pooled", choices=("pooled", "context"))
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42,
                    help="must be the runs' --seed: it is what fixes the episode split")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--out", default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    store = ContextStore.open_many(args.store, _csv(args.store_shards, int) or None)
    _, va, _ = store.split_by_episode(args.val_frac, args.test_frac, seed=args.seed)
    print(f"store: {len(store)} rows / {store.n_episodes} episodes; val {len(va)} rows")

    supports = args.support or list(MOTION_FILTERS)
    rows = {s: filter_rows_by_motion(store, va, s, args.filter_tol) for s in supports}
    for s, r in rows.items():
        print(f"  support {s:<12} {len(r):>7} val rows ({len(r) / max(1, len(va)):.1%})")

    ckpts = []
    for spec in args.checkpoint:
        name, _, path = spec.partition("=")
        ckpts.append((name if path else Path(name).parent.name, path or name))

    results = {}
    for name, path in ckpts:
        model = FrozenContextFlowPolicy.from_pretrained(path)
        # The end of the anneal, on every arm: the noise floor is a function of sigma, so
        # a checkpoint evaluated at another multiplier reports a different floor and an
        # incomparable headroom.
        model.set_sigma_mult(1.0)
        if torch.cuda.is_available():
            model.cuda()
        model.eval()
        targs = TrainingArguments(
            output_dir="/tmp/eval_head_only", report_to=[], seed=args.seed,
            per_device_eval_batch_size=args.batch_size, max_steps=1,
            dataloader_num_workers=0, dataloader_pin_memory=False,
            remove_unused_columns=False, label_names=[], bf16=False,
        )
        trainer = HeadOnlyFlowTrainer(
            model=model, args=targs,
            data_collator=FrozenContextCollator(store, args.context_key),
            train_dataset=FrozenContextDataset(store, va[:2], args.context_key),
            data_config=DataConfig(target_dim_names=("dx", "dy", "dtheta")),
            sigma_start=1.0, sigma_anneal_steps=0,
        )
        results[name] = {}
        for s in supports:
            if len(rows[s]) == 0:
                continue
            print(f"\n=== {name} on support {s} ({len(rows[s])} rows) ===", flush=True)
            m = trainer.evaluate(
                FrozenContextDataset(store, rows[s], args.context_key),
                metric_key_prefix="eval",
            )
            results[name][s] = {k: float(v) for k, v in m.items()
                                if isinstance(v, (int, float))}
            results[name][s]["n_rows"] = int(len(rows[s]))
        del trainer, model
        torch.cuda.empty_cache()

    print("\n" + "=" * 100)
    print("creep_sample / creep_gt and stop_sample / stop_gt, per support "
          "(the comparable readout)")
    print("=" * 100)
    for s in supports:
        print(f"\n-- support {s} ({len(rows[s])} rows)")
        hdr = f"{'checkpoint':<16}" + "".join(
            f"{c:>13}" for c in ("creep_s_dx", "creep_s_dth", "stop_s_dx",
                                 "stop_ratio_dx", "rmse_mean", "nll/dim"))
        print(hdr)
        for name, _ in ckpts:
            r = results.get(name, {}).get(s)
            if not r:
                continue
            print(f"{name:<16}" + "".join(f"{r.get(k, float('nan')):>13.4f}" for k in (
                "eval_creep_sample_dx", "eval_creep_sample_dtheta", "eval_stop_sample_dx",
                "eval_stop_ratio_dx", "eval_rmse_mean", "eval_nll_per_dim")))
        r0 = next((results[n][s] for n, _ in ckpts if s in results.get(n, {})), None)
        if r0:
            print(f"{'GROUND TRUTH':<16}" + "".join(f"{r0.get(k, float('nan')):>13.4f}" for k in (
                "eval_creep_gt_dx", "eval_creep_gt_dtheta", "eval_stop_gt_dx")))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({
            "args": vars(args), "n_val_rows": int(len(va)),
            "support_rows": {s: int(len(r)) for s, r in rows.items()},
            "results": results,
        }, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
