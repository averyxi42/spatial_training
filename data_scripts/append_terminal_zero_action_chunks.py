#!/usr/bin/env python
"""Append stationary action chunks only after verified successful ObjectNav demos."""

import argparse
import copy
import re
from pathlib import Path

import numpy as np
from datasets import DatasetDict, load_from_disk


_OBSERVATION = re.compile(r"^(Observation )\d+(:)$")


def _terminal_messages(messages, steps):
    if len(messages) < 2:
        raise ValueError("terminal demo needs a user/assistant action pair")
    user, assistant = messages[-2:]
    if user.get("role") != "user" or assistant.get("role") != "assistant":
        raise ValueError("terminal messages must end in a user/assistant action pair")
    content = user.get("content", [])
    if not content or content[0].get("type") != "text":
        raise ValueError("terminal user turn must start with an observation label")
    match = _OBSERVATION.match(str(content[0].get("text", "")))
    start = int(match.group(0).split()[1].rstrip(":")) + 1 if match else None
    appended = []
    for offset in range(steps):
        next_user = copy.deepcopy(user)
        if start is not None:
            index = start + offset
            next_user["content"][0]["text"] = (
                f"{match.group(1)}{index}{match.group(2)}"
            )
        appended.extend((next_user, copy.deepcopy(assistant)))
    return list(messages) + appended


def _append_terminal_zeros(example, terminal_steps, radius):
    chunks = example["action_chunks"]
    images = example["images"]
    if not chunks or len(chunks) != len(images):
        raise ValueError("action_chunks and images must be aligned and nonempty")
    distances = example.get("distance_targets")
    if distances is None or len(distances) != len(chunks):
        raise ValueError("terminal zero actions require aligned distance_targets")
    last_distance = distances[-1]
    if last_distance is None or not np.isfinite(last_distance) or last_distance > radius:
        return example
    for name in ("obs_poses", "stop_targets", "return_targets", "obs_indices", "frame_indices"):
        values = example.get(name)
        if values is not None and len(values) != len(chunks):
            raise ValueError(f"{name} is not aligned with action_chunks")
    zero_chunk = np.zeros_like(np.asarray(chunks[-1], dtype=np.float32)).tolist()
    result = dict(example)
    result["action_chunks"] = list(chunks) + [zero_chunk] * terminal_steps
    result["images"] = list(images) + [images[-1]] * terminal_steps
    result["messages"] = _terminal_messages(example["messages"], terminal_steps)
    terminal_values = {
        "obs_poses": example["obs_poses"][-1] if example.get("obs_poses") is not None else None,
        "distance_targets": float(last_distance),
        "stop_targets": 1.0,
        "return_targets": 0.0,
        "obs_indices": example["obs_indices"][-1] if example.get("obs_indices") is not None else None,
        "frame_indices": example["frame_indices"][-1] if example.get("frame_indices") is not None else None,
    }
    for name, terminal_value in terminal_values.items():
        values = example.get(name)
        if values is not None:
            result[name] = list(values) + [terminal_value] * terminal_steps
    if "num_observations" in example:
        result["num_observations"] = int(example["num_observations"]) + terminal_steps
    return result


def _validate(dataset, radius, terminal_steps):
    total = appended = 0
    for example in dataset:
        total += 1
        chunks = example["action_chunks"]
        distances = example["distance_targets"]
        if len(chunks) != len(example["images"]) or len(chunks) != len(distances):
            raise ValueError("output has misaligned actions, images, or distances")
        final_distance = distances[-1]
        if (final_distance is not None and np.isfinite(final_distance)
                and final_distance <= radius):
            zeros = np.asarray(chunks[-terminal_steps:], dtype=np.float32)
            if zeros.shape[0] == terminal_steps and np.count_nonzero(zeros) == 0:
                appended += 1
    return total, appended


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--success-radius-m", type=float, default=1.0)
    parser.add_argument("--terminal-steps", type=int, default=4)
    parser.add_argument("--num-proc", type=int, default=8)
    args = parser.parse_args()
    if args.terminal_steps < 1:
        parser.error("--terminal-steps must be positive")
    if args.success_radius_m <= 0:
        parser.error("--success-radius-m must be positive")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    source = load_from_disk(str(args.input))
    splits = source if isinstance(source, DatasetDict) else DatasetDict({"train": source})
    transformed = DatasetDict({
        name: split.map(
            _append_terminal_zeros,
            fn_kwargs={"terminal_steps": args.terminal_steps, "radius": args.success_radius_m},
            num_proc=args.num_proc,
            desc=f"append terminal zero actions [{name}]",
        )
        for name, split in splits.items()
    })
    for name, split in transformed.items():
        total, appended = _validate(split, args.success_radius_m, args.terminal_steps)
        print(f"{name}: appended zero actions to {appended}/{total} verified terminal demos")
    transformed.save_to_disk(str(args.output))


if __name__ == "__main__":
    main()
