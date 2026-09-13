"""Compare serial 8-slot rollout inference modes in one loaded VLM process."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from longnav.utils.vlm_worker import VLMWorker


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--images-npy", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--slots", type=int, default=8)
    parser.add_argument("--turns", type=int, default=20)
    parser.add_argument(
        "--modes",
        default="baseline,optimized,baseline",
    )
    parser.add_argument("--vision-only", action="store_true")
    parser.add_argument("--vision-repeats", type=int, default=20)
    return parser.parse_args()


def start_messages(goal):
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are a robot navigating an indoor environment toward a goal "
                        f"object.\nGoal: {goal}\nAt each step you receive the current RGB "
                        "observation. Produce the next short trajectory of poses to follow, "
                        "relative to your current pose."
                    ),
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Observation 0:"},
                {"type": "image"},
                {"type": "text", "text": "Action:"},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "**____**"}]},
    ]


def turn_messages(step):
    return [
        {"role": "assistant", "content": [{"type": "text", "text": "**____**"}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Observation {step}:"},
                {"type": "image"},
                {"type": "text", "text": "Action:"},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "**____**"}]},
    ]


def run_mode(worker, frames, turns, mode):
    if mode not in {
        "baseline",
        "optimized",
        "precomputed_serial",
        "batched_vision",
    }:
        raise ValueError(f"Unknown mode: {mode}")
    worker.rollout_inference_optimizations = mode != "baseline"
    worker._inference_mode_prepared = False
    worker.reset(clear_cuda_cache=False)
    states = [worker.new_sequence_state() for _ in frames]
    head = worker.model.action_head
    head.seed(12345)
    hidden = []
    actions = []
    per_turn = []
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for step in range(turns):
        turn_started = time.perf_counter()
        policies = []
        if mode in {"precomputed_serial", "batched_vision"}:
            worker._prepare_model_for_inference()
            prepared = []
            prepared_states = []
            for index, frame in enumerate(frames):
                worker.restore_sequence_state(states[index])
                messages = (
                    start_messages(f"target-{index}")
                    if step == 0
                    else turn_messages(step)
                )
                prepared.append(
                    worker._prepare_infer_inputs(
                        messages=messages,
                        images=[Image.fromarray(frame)],
                        pos_id_kwargs={"mode": "standard"},
                    )
                )
                prepared_states.append(worker.capture_sequence_state())
            features = (
                worker.batch_image_features(prepared)
                if mode == "batched_vision"
                else [worker.batch_image_features([inputs])[0] for inputs in prepared]
            )
            for index, (inputs, sequence_state, image_features) in enumerate(
                zip(prepared, prepared_states, features)
            ):
                worker.restore_sequence_state(sequence_state)
                policy, _ = worker._forward_infer_inputs(
                    inputs,
                    temperature=1.0,
                    precomputed_image_features=image_features,
                )
                policies.append(policy)
                states[index] = worker.capture_sequence_state()
        else:
            for index, frame in enumerate(frames):
                worker.restore_sequence_state(states[index])
                messages = (
                    start_messages(f"target-{index}")
                    if step == 0
                    else turn_messages(step)
                )
                policy, _, _ = worker.infer_probs(
                    messages=messages,
                    images=[Image.fromarray(frame)],
                    temperature=1.0,
                    pos_id_kwargs={"mode": "standard"},
                )
                policies.append(policy)
                states[index] = worker.capture_sequence_state()
        for policy in policies:
            chain, positions, logprob, chunk = head.sample_chain_np(
                np.asarray(policy["h"], dtype=np.float32).reshape(-1)
            )
            hidden.append(np.asarray(policy["h"], dtype=np.float32))
            actions.append(
                np.concatenate(
                    [chain, positions.astype(np.float32), [logprob], chunk.reshape(-1)]
                )
            )
        torch.cuda.synchronize()
        per_turn.append(time.perf_counter() - turn_started)
    torch.cuda.synchronize()
    wall = time.perf_counter() - started
    packed_shapes = []
    for state in states:
        worker.restore_sequence_state(state)
        packed = worker._pack_embeds()
        packed_shapes.append(
            {
                key: (
                    [list(tensor.shape) for tensor in value]
                    if isinstance(value, list)
                    else list(value.shape)
                )
                for key, value in packed.items()
            }
        )
    hidden = np.stack(hidden)
    actions = np.stack(actions)
    return {
        "mode": mode,
        "wall_seconds": wall,
        "seconds_per_slot_step": wall / (len(frames) * turns),
        "turn_seconds": per_turn,
        "peak_memory_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "packed_shapes": packed_shapes,
        "hidden_sha256": hashlib.sha256(hidden.tobytes()).hexdigest(),
        "actions_sha256": hashlib.sha256(actions.tobytes()).hexdigest(),
        "hidden": hidden,
        "actions": actions,
    }


def run_vision_batch_probe(worker, frames, repeats):
    prepared = [
        worker.tokenize_inputs(start_messages(f"target-{index}"), [Image.fromarray(frame)])
        for index, frame in enumerate(frames)
    ]
    pixels = [item["pixel_values"].to(worker.device) for item in prepared]
    grids = [item["image_grid_thw"].to(worker.device) for item in prepared]
    model = worker.vl_model

    def serial():
        return [model.get_image_features(pixel, grid) for pixel, grid in zip(pixels, grids)]

    def batched():
        return model.get_image_features(torch.cat(pixels), torch.cat(grids))

    serial()
    batched()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(repeats):
        serial_outputs = serial()
    torch.cuda.synchronize()
    serial_seconds = time.perf_counter() - started
    started = time.perf_counter()
    for _ in range(repeats):
        batch_outputs = batched()
    torch.cuda.synchronize()
    batch_seconds = time.perf_counter() - started

    batch_primary, batch_deepstack = batch_outputs
    split_sizes = [int(grid.prod().item() // model.visual.spatial_merge_size**2) for grid in grids]
    split_deepstack = [torch.split(layer, split_sizes) for layer in batch_deepstack]
    primary_max_abs = 0.0
    deepstack_max_abs = 0.0
    for index, (serial_primary, serial_deepstack) in enumerate(serial_outputs):
        primary_max_abs = max(
            primary_max_abs,
            float((serial_primary[0] - batch_primary[index]).abs().max().item()),
        )
        for layer_index, serial_layer in enumerate(serial_deepstack):
            deepstack_max_abs = max(
                deepstack_max_abs,
                float(
                    (serial_layer - split_deepstack[layer_index][index])
                    .abs()
                    .max()
                    .item()
                ),
            )
    return {
        "slots": len(frames),
        "repeats": repeats,
        "serial_seconds": serial_seconds,
        "batch_seconds": batch_seconds,
        "speedup": serial_seconds / batch_seconds,
        "primary_max_abs": primary_max_abs,
        "deepstack_max_abs": deepstack_max_abs,
    }


def main():
    args = parse_args()
    images = np.load(args.images_npy)[: args.slots]
    if len(images) != args.slots:
        raise ValueError(f"Expected {args.slots} images, got {len(images)}")
    worker = VLMWorker(
        model_id=args.model_id,
        attn_impl="flash_attention_2",
        dtype="bfloat16",
        save_outputs=True,
        use_sparse=True,
        policy_head={
            "type": "continuous",
            "_target_": "longnav.utils.flow_sde_policy.FlowSDEHead",
            "checkpoint_dir": args.checkpoint_dir,
            "gap": 10,
            "sde_n": 3,
            "sde_noise_a": 0.9,
            "sde_position_weight": "sigma",
        },
    )

    if args.vision_only:
        output = run_vision_batch_probe(worker, images, args.vision_repeats)
        Path(args.output).write_text(json.dumps(output, indent=2))
        print(json.dumps(output, indent=2), flush=True)
        return

    # Warm CUDA kernels and allocator on a short sequence before timing either mode.
    run_mode(worker, images[:1], turns=2, mode="baseline")
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    results = [run_mode(worker, images, args.turns, mode=mode) for mode in modes]
    baseline_hidden = results[0].pop("hidden")
    baseline_actions = results[0].pop("actions")
    comparisons = []
    for result in results[1:]:
        candidate_hidden = result.pop("hidden")
        candidate_actions = result.pop("actions")
        hidden_delta = candidate_hidden - baseline_hidden
        chain_len = worker.model.action_head.chain_len
        chain_delta = candidate_actions[:, :chain_len] - baseline_actions[:, :chain_len]
        logprob_delta = candidate_actions[:, chain_len + 3] - baseline_actions[:, chain_len + 3]
        chunk_delta = candidate_actions[:, -30:] - baseline_actions[:, -30:]
        chunk_delta_poses = chunk_delta.reshape(-1, 10, 3)
        endpoint_xy = np.linalg.norm(chunk_delta_poses[:, -1, :2], axis=-1)
        endpoint_yaw = np.abs(chunk_delta_poses[:, -1, 2])
        comparisons.append(
            {
                "mode": result["mode"],
                "hidden_max_abs": float(np.max(np.abs(hidden_delta))),
                "hidden_rms": float(np.sqrt(np.mean(hidden_delta**2))),
                "hidden_value_rms": float(np.sqrt(np.mean(baseline_hidden**2))),
                "chain_max_abs": float(np.max(np.abs(chain_delta))),
                "chain_rms": float(np.sqrt(np.mean(chain_delta**2))),
                "logprob_max_abs": float(np.max(np.abs(logprob_delta))),
                "chunk_max_abs": float(np.max(np.abs(chunk_delta))),
                "chunk_mae": float(np.mean(np.abs(chunk_delta))),
                "chunk_rms": float(np.sqrt(np.mean(chunk_delta**2))),
                "chunk_component_mae": np.mean(
                    np.abs(chunk_delta_poses), axis=(0, 1)
                ).tolist(),
                "endpoint_xy_mean_m": float(np.mean(endpoint_xy)),
                "endpoint_xy_p95_m": float(np.quantile(endpoint_xy, 0.95)),
                "endpoint_xy_max_m": float(np.max(endpoint_xy)),
                "endpoint_yaw_mean_deg": float(np.degrees(np.mean(endpoint_yaw))),
                "endpoint_yaw_p95_deg": float(np.degrees(np.quantile(endpoint_yaw, 0.95))),
                "endpoint_yaw_max_deg": float(np.degrees(np.max(endpoint_yaw))),
                "hidden_exact": bool(np.array_equal(candidate_hidden, baseline_hidden)),
                "actions_exact": bool(np.array_equal(candidate_actions, baseline_actions)),
            }
        )
    output = {"runs": results, "comparisons_to_first_baseline": comparisons}
    Path(args.output).write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
