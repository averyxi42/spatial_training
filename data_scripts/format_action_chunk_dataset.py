"""
Add a `messages` column to an action-chunk episode table, for `train_vector_sft.py`.

This is the one domain-specific piece of the pipeline, kept deliberately outside the
trainer: `continuous_demos`' `build_action_chunk_episode_dataset.py` emits one row per
episode with `images` (paths, one per observation) and `action_chunks`
`(n_obs, action_chunk_len, 3)`, but no conversation formatting. The trainer wants the
standard HF conversational layout, so this script writes it -- and nothing else.

The assistant turns carry a fixed placeholder (`**____**` by default). That is not a
label: the continuous action comes from the regression head reading that turn's vector,
which is why `train_vector_sft.py` defaults to affixes that wrap the content in
`**` plus shift-left, pooling
the single `**` token that opens each turn. Change the placeholder here and nothing
downstream cares, as long as it stays constant across turns and its middle stays a single
token that does not merge into the `**` affixes -- `docs/placeholder_tokens.md` explains
why `**forward**` was a poor choice (it asserts a false action history in every later
turn's context) and why `**[…]**` is broken outright (BPE merges `]**`).

    python data_scripts/format_action_chunk_dataset.py \
        --in-dir  ~/codes/habitat/continuous_demos/data/arrowtable \
        --out-dir dump/datasets/action_chunks_conversational \
        --val-fraction 0.1

Rows whose observation count and chunk count disagree are dropped with a warning rather
than carried into training, where they would trip the trainer's alignment assert.
"""

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for p in (str(_ROOT), str(_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from datasets import DatasetDict, load_from_disk

# Imported, not duplicated: `vector_rollout.py` builds the rollout context from the same
# string, and a silent divergence between the two would change the head's input
# distribution at deployment without any error.
from longnav.utils.vector_rollout import DEFAULT_SYSTEM_PROMPT as SYSTEM_PROMPT


def build_messages(example, placeholder: str, goal_column: str, images_column: str):
    goal = example.get(goal_column) or "the goal object"
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": SYSTEM_PROMPT.format(goal=goal)}],
        }
    ]
    for i in range(len(example[images_column])):
        messages += [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Observation {i}:"},
                    {"type": "image"},
                    {"type": "text", "text": "Action:"},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": placeholder}]},
        ]
    example["messages"] = messages
    return example


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--in-dir", required=True, help="load_from_disk() dir of the episode table")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--split", default="train", help="split to read when the input is a DatasetDict")
    p.add_argument("--target-column", default="action_chunks")
    p.add_argument("--images-column", default="images")
    p.add_argument("--goal-column", default="goal_text")
    p.add_argument("--placeholder", default="**____**",
                   help="constant assistant text; the real action comes from the head. "
                        "Keep the middle a single token that does not merge into '**' "
                        "(see docs/placeholder_tokens.md)")
    p.add_argument("--image-root", default=None,
                   help="prepend to relative image paths (the builder may emit relative ones)")
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--max-turns", type=int, default=0,
                   help="drop episodes with more observations than this (0 = keep all). "
                        "The trainer windows long episodes anyway; this is for pruning "
                        "outliers up front")
    p.add_argument("--num-proc", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    ds = load_from_disk(os.path.expanduser(args.in_dir))
    if hasattr(ds, "keys"):
        ds = ds[args.split]
    print(f"Loaded {len(ds)} episode(s) from {args.in_dir}")

    def consistent(ex):
        return len(ex[args.images_column]) == len(ex[args.target_column])

    before = len(ds)
    ds = ds.filter(consistent, num_proc=args.num_proc, desc="align check")
    if len(ds) != before:
        print(f"  dropped {before - len(ds)} row(s) whose image and chunk counts disagreed")

    if args.max_turns:
        before = len(ds)
        ds = ds.filter(lambda ex: len(ex[args.images_column]) <= args.max_turns,
                       num_proc=args.num_proc, desc="length filter")
        print(f"  dropped {before - len(ds)} row(s) over {args.max_turns} observations")

    if args.image_root:
        root = os.path.expanduser(args.image_root)
        ds = ds.map(
            lambda ex: {
                args.images_column: [
                    p if os.path.isabs(p) else os.path.join(root, p)
                    for p in ex[args.images_column]
                ]
            },
            num_proc=args.num_proc,
            desc="absolute paths",
        )

    ds = ds.map(
        build_messages,
        fn_kwargs={
            "placeholder": args.placeholder,
            "goal_column": args.goal_column,
            "images_column": args.images_column,
        },
        num_proc=args.num_proc,
        desc="messages",
    )

    if args.val_fraction > 0:
        split = ds.train_test_split(test_size=args.val_fraction, seed=args.seed)
        out = DatasetDict({"train": split["train"], "validation": split["test"]})
    else:
        out = DatasetDict({"train": ds})

    Path(args.out_dir).parent.mkdir(parents=True, exist_ok=True)
    out.save_to_disk(args.out_dir)
    n_turns = [len(r) for r in out["train"][args.images_column][: min(200, len(out["train"]))]]
    print(f"Wrote {args.out_dir}: " + ", ".join(f"{k}={len(v)}" for k, v in out.items()))
    print(f"  observations per episode (first {len(n_turns)} train rows): "
          f"min {min(n_turns)}, median {sorted(n_turns)[len(n_turns)//2]}, max {max(n_turns)}")


if __name__ == "__main__":
    main()
