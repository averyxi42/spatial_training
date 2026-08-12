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
# `segment_starts` / `modality_values` / `expected_modality_len` used to live here, next
# to `build_messages`. They now live in `vector_rollout.py` and are imported, for the same
# reason `SYSTEM_PROMPT` is: the ROLLOUT has to produce the same row order for the same
# conversation, and it cannot import a data script. One definition, two callers -- rather
# than two definitions that agree today. They are re-exported by this import, so
# `from format_action_chunk_dataset import modality_values` still resolves.
from longnav.utils.vector_rollout import (
    GOAL_PLACEMENTS,
    SYSTEM_PROMPTS,
    RolloutConfig,
    expected_modality_len,
    anchor_suffix,
    goal_block_content,
    modality_values,
    segment_starts,
    user_block_content,
)

__all__ = [
    "build_messages",
    "expected_modality_len",
    "modality_values",
    "segment_starts",
]


def build_messages(example, placeholder: str, goal_column: str, images_column: str,
                   modality_marker=None, goal_marker=None, goal_placement="segment",
                   segment_column="segment_indices", task_column="task"):
    """Write the conversation for one episode.

    The per-turn user block comes from `vector_rollout.user_block_content`, not from a copy
    of it here: the rollout renders its context from the same function, so the marker's
    presence and its position relative to the image are decided once. A marker that trained
    after the image and deployed before it would be a different context with no error
    anywhere -- the count check would still pass, because the count is right.

    The marker goes in the **user** block. It is therefore not inside the assistant turn,
    which is what keeps the readout spans, `train_content_len` and the affix arithmetic
    exactly as they were; `tests/test_pose_injection.py` asserts that rather than assuming
    it.

    PointNav adds a goal, carried by a *second occurrence of the same `<pose>` marker* --
    one modality type, many occurrences, exactly as `<image>` works for vision. Two
    placements:

      * `segment` (default) -- its own user message wherever a new goal is introduced, so
        the conversation reads the way the task does. Note the first announcement lands in
        the prologue (everything before the first image-bearing message), so it survives
        windowing; later ones do not, which is fine because windowing is an eval ablation
        rather than the training regime.
      * `observation` -- repeated in every turn, like a PointGoal sensor reading.

    With `goal_marker=None` this writes exactly the conversation it always did.
    """
    goal = example.get(goal_column) or "the goal object"
    task = example.get(task_column) or "objectnav"
    prompt = SYSTEM_PROMPTS.get(task, SYSTEM_PROMPT)
    # `goal_placement` rides on the config so `render_user_block` derives `inline_goal`
    # from the same field the rollout does; `inline` below stays explicit because
    # `user_block_content` is the shared primitive and takes it directly.
    cfg = RolloutConfig(modality_marker=modality_marker, goal_marker=goal_marker,
                        goal_placement=goal_placement if goal_marker else None)
    inline = bool(goal_marker) and goal_placement == "observation"
    # Both `segment` and `anchor` emit an announcement message wherever the goal changes;
    # they differ only in what that message says and in whether the prologue carries the
    # frame anchor. `observation` emits none, carrying the goal inline instead.
    starts = (segment_starts(example.get(segment_column) or [])
              if goal_marker and goal_placement in ("segment", "anchor") else set())

    # Under the `anchor` placement the frame is set once, here, by a single agent pose in
    # the prologue; each later announcement then carries the goal alone. Empty string for
    # every other placement, so the prompt is byte-identical to what it always was.
    messages = [
        {"role": "user",
         "content": [{"type": "text",
                      "text": prompt.format(goal=goal) + anchor_suffix(cfg)}]}
    ]
    for i in range(len(example[images_column])):
        if i in starts:
            messages.append({"role": "user", "content": goal_block_content(cfg)})
        messages += [
            {"role": "user", "content": user_block_content(cfg, i, inline_goal=inline)},
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
    p.add_argument("--val-fraction", type=float, default=0.1, help = "only used if --val-split is not set")
    p.add_argument("--val-split", default=None, help="split to read validation rows from when the input is a DatasetDict")
    p.add_argument("--max-turns", type=int, default=0,
                   help="drop episodes with more observations than this (0 = keep all). "
                        "The trainer windows long episodes anyway; this is for pruning "
                        "outliers up front")
    p.add_argument("--modality-marker", default=None,
                   help="write this marker into each turn's user block, after the image "
                        "(e.g. '<pose>'). The values come from a dataset column named by "
                        "the model's modality spec -- for pose that is `obs_poses`, which "
                        "is already carried through untouched. Omit for the classic "
                        "no-marker conversations")
    # NOTE: the conventional value, `obs_poses`, is a misnomer -- the column holds a
    # mixed occurrence-ordered stream of agent AND goal poses, exactly as `images` would
    # hold observation and goal images. A rename to `poses` is queued; see
    # docs/column_naming.md before changing it, because the spec file and every corpus on
    # disk have to move in the same commit.
    p.add_argument("--modality-column", default=None,
                   help="column that must be 1:1 with the observations when "
                        "--modality-marker is set (e.g. 'obs_poses'). Rows that disagree "
                        "are dropped, exactly as misaligned action chunks are")
    p.add_argument("--goal-marker", default=None,
                   help="PointNav's goal marker. Normally the SAME literal as "
                        "--modality-marker: `<pose>` is one modality type and a goal pose "
                        "is another occurrence of it, bound by occurrence order like every "
                        "other. Omit for ObjectNav")
    p.add_argument("--goal-values-column", default="obs_goals",
                   help="column holding the per-observation goal poses. Distinct from "
                        "--goal-column, which is the goal *text*")
    p.add_argument("--goal-placement", choices=GOAL_PLACEMENTS,
                   default="segment",
                   help="segment: its own user message wherever a new goal is introduced. "
                        "observation: repeated in every turn, like a PointGoal sensor")
    p.add_argument("--segment-column", default="segment_indices",
                   help="per-observation segment index; goal announcements are derived "
                        "from where it changes")
    p.add_argument("--task-column", default="task",
                   help="picks the system prompt per row, so a table cannot describe one "
                        "task with the other's prompt")
    p.add_argument("--map-cache-dir", default=None,
                   help="write datasets' map/filter cache here instead of alongside the "
                        "input. Set it when --in-dir is a dataset something else is "
                        "reading: the default puts `cache-*.arrow` files in the input's "
                        "own directory")
    p.add_argument("--num-proc", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.modality_column and not args.modality_marker:
        p.error("--modality-column without --modality-marker: the column would be carried "
                "but never referenced, since nothing in the text would occur to bind it to")
    if args.goal_marker and not args.modality_column:
        p.error("--goal-marker needs --modality-column: the goal rows are written into "
                "that same column, in text order, because both markers are the same token")

    ds = load_from_disk(os.path.expanduser(args.in_dir))

    cache_root = None
    if args.map_cache_dir:
        cache_root = Path(os.path.expanduser(args.map_cache_dir))
        cache_root.mkdir(parents=True, exist_ok=True)

    def cache_kw(step):
        """`cache_file_name(s)` for one map/filter step, or nothing when unset.

        `datasets` derives a map's cache path from the *input's* directory, so without
        this a retabulation drops `cache-*.arrow` files into the dataset it is reading --
        which is exactly the dataset another run is likely to be reading too.
        """
        if cache_root is None:
            return {}
        if hasattr(ds, "keys"):  # DatasetDict
            return {"cache_file_names": {k: str(cache_root / f"{step}-{k}.arrow")
                                         for k in ds.keys()}}
        return {"cache_file_name": str(cache_root / f"{step}.arrow")}

    def size(d):
        """Row count, for a Dataset or a DatasetDict alike -- `len` on the latter is
        the number of splits, which makes every "dropped N rows" line a lie."""
        return sum(len(v) for v in d.values()) if hasattr(d, "keys") else len(d)

    print(f"Loaded {size(ds)} episode(s) from {args.in_dir}")

    def consistent(ex):
        return len(ex[args.images_column]) == len(ex[args.target_column])

    before = size(ds)
    ds = ds.filter(consistent, num_proc=args.num_proc, desc="align check",
                   **cache_kw("align"))
    if size(ds) != before:
        print(f"  dropped {before - size(ds)} row(s) whose image and chunk counts disagreed")

    if args.modality_column:
        # One marker per observation (plus, for PointNav, the goal ones), and the binding
        # is occurrence order, so a row with the wrong number of modality values would bind
        # every later turn's value to the wrong turn. Checked here rather than at training
        # time because here it is one dropped row instead of a crashed run.
        before = size(ds)
        ds = ds.filter(
            lambda ex: len(ex[args.modality_column]) == len(ex[args.images_column]),
            num_proc=args.num_proc, desc=f"{args.modality_column} align check",
            **cache_kw("modality_align"),
        )
        if size(ds) != before:
            print(f"  dropped {before - size(ds)} row(s) whose {args.modality_column} count "
                  f"disagreed with the observation count")

    if args.goal_marker:
        before = size(ds)
        ds = ds.filter(
            lambda ex: len(ex[args.goal_values_column]) == len(ex[args.images_column]),
            num_proc=args.num_proc, desc=f"{args.goal_values_column} align check",
            **cache_kw("goal_align"),
        )
        if size(ds) != before:
            print(f"  dropped {before - size(ds)} row(s) whose {args.goal_values_column} "
                  f"count disagreed with the observation count")
        # Rewrite the modality column in TEXT ORDER, interleaving the goal rows where
        # their markers occur. Done here rather than in the arrow builder because it
        # exists only as a consequence of how the conversation is written, and leaving
        # the raw columns in the table keeps every placement ablatable without a rebuild.
        ds = ds.map(
            lambda ex: {args.modality_column: modality_values(
                ex, args.modality_column, args.goal_values_column,
                args.segment_column, args.goal_placement)},
            num_proc=args.num_proc, desc="goal value rows",
            **cache_kw("goal_values"),
        )

    if args.max_turns:
        before = size(ds)
        ds = ds.filter(lambda ex: len(ex[args.images_column]) <= args.max_turns,
                       num_proc=args.num_proc, desc="length filter",
                       **cache_kw("length"))
        print(f"  dropped {before - size(ds)} row(s) over {args.max_turns} observations")

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
            **cache_kw("abspath"),
        )

    ds = ds.map(
        build_messages,
        fn_kwargs={
            "placeholder": args.placeholder,
            "goal_column": args.goal_column,
            "images_column": args.images_column,
            "modality_marker": args.modality_marker,
            "goal_marker": args.goal_marker,
            "goal_placement": args.goal_placement,
            "segment_column": args.segment_column,
            "task_column": args.task_column,
        },
        num_proc=args.num_proc,
        desc="messages",
        **cache_kw("messages"),
    )

    if args.val_fraction > 0 and args.val_split is None:
        split = ds.train_test_split(test_size=args.val_fraction, seed=args.seed)
        out = DatasetDict({"train": split["train"], "validation": split["test"]})
    else:
        if args.val_split is not None:
            out = DatasetDict({"train": ds[args.split], "validation": ds[args.val_split]})
        else:
            # `ds` is a DatasetDict whenever the input was one, which it is for every
            # corpus the builder writes. Wrapping it again nests DatasetDicts and the
            # summary below then fails on a missing column -- after the save, so the
            # output looked fine and the run looked broken. Unwrap the split instead.
            out = DatasetDict({"train": ds[args.split] if hasattr(ds, "keys") else ds})

    Path(args.out_dir).parent.mkdir(parents=True, exist_ok=True)
    out.save_to_disk(args.out_dir)
    n_turns = [len(r) for r in out["train"][args.images_column][: min(200, len(out["train"]))]]
    print(f"Wrote {args.out_dir}: " + ", ".join(f"{k}={len(v)}" for k, v in out.items()))
    print(f"  observations per episode (first {len(n_turns)} train rows): "
          f"min {min(n_turns)}, median {sorted(n_turns)[len(n_turns)//2]}, max {max(n_turns)}")


if __name__ == "__main__":
    main()
