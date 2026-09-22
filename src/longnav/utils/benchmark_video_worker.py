import importlib.util
import io
import math
import os
import time
from pathlib import Path

import imageio
import numpy as np
from PIL import Image


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _benchmark_helpers():
    repo_root = Path(__file__).resolve().parents[6]
    vis = _load_module(
        "navverse_async_benchmark_vis",
        repo_root / "baselines/utils/vis.py",
    )
    bev = _load_module(
        "navverse_async_benchmark_bev",
        repo_root / "baselines/utils/bev_video.py",
    )
    return vis, bev


def _decode_image(data):
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


def render_benchmark_video_batch(capture):
    started = time.perf_counter()
    vis, bev = _benchmark_helpers()
    video_paths = {}
    for job in capture["jobs"]:
        label = job["episode_label"]
        instruction_lines = ["Instruction: "]
        for word in job["instruction"].split():
            if len(instruction_lines[-1]) + len(word) > 30:
                instruction_lines.append(word)
            else:
                instruction_lines[-1] += f" {word}"
        frames = []
        for frame in job["frames"]:
            pov = _decode_image(frame["pov"])
            third_person = _decode_image(frame["third_person"])
            grid = [
                {
                    "type": "text",
                    "args": vis.text_args,
                    "text": f"Episode: {label}",
                },
                *[
                    {"type": "text", "args": vis.text_args, "text": line}
                    for line in instruction_lines
                ],
                {
                    "type": "text",
                    "args": vis.text_args,
                    "text": f"Step: {frame['step']}, action: {frame['action']}",
                },
                {
                    "type": "image_row",
                    "height": int(pov.shape[0]),
                    "image": [
                        ["First person view", pov],
                        ["Third person view", third_person],
                    ],
                },
            ]
            frames.append(vis.render_grid(grid))
        if not frames:
            continue
        panel_width = 760
        margin = 8
        renderer = bev.BEVVideoRenderer(
            episode_folder=job["episode_folder"],
            output_size=(
                panel_width - 2 * margin,
                int(frames[0].shape[0]) - 2 * margin,
            ),
        )
        scene_id, episode_id = label.rsplit("_", 1)
        frame_steps = [frame["step"] for frame in job["frames"]]
        bev_frames = renderer.render_sequence(
            scene_id,
            episode_id,
            job["trajectory"],
            frame_steps,
        )
        if bev_frames:
            frames = bev.append_bev_panels_to_frames(
                frames,
                bev_frames,
                panel_width=panel_width,
                margin=margin,
            )
        frames = vis.pad_frames_to_same_size(frames)
        fps = 2
        if len(frames) / fps > 60:
            fps = max(fps, math.ceil(len(frames) / 60))
        os.makedirs(os.path.dirname(job["output_path"]), exist_ok=True)
        imageio.mimsave(job["output_path"], frames, fps=fps)
        video_paths[label] = job["output_path"]
    return {
        "videos": video_paths,
        "render_seconds": time.perf_counter() - started,
        "export_seconds": float(capture.get("export_seconds", 0.0)),
    }
