"""Mine on-policy (frame, distance, return) training rows from RL rollout dumps.

The state probe trains first on demo corpora, but demos never visit the states the
POLICY visits (settle, creep, wrong-room commitment) -- the standard failure of offline
stop heads. Every RL rollout episode already stores per-policy-step `distance_to_goal`
(sequence.json) and a real-time `video.mp4` whose frames at policy-step boundaries are
the exact observations the policy saw. This walks rollout dirs, decodes ONLY those
frames, and writes an images/ + labels.jsonl store shaped for the probe:

    <out>/images/<episode_tag>/step_00042.jpg
    <out>/labels.jsonl   {image, episode_tag, run, step, distance, return_g<gamma>, success}

Returns are computed per episode from the distance series under the RL reward
(state_probe.distance_return_targets), gamma a flag. Frame->step mapping: videos are
written at `fps` real-time and one policy step spans `gap` ticks of `dt` seconds
(gap*dt*fps frames per step; 25 fps, gap 10, dt 0.04 -> exactly 10 frames/step).
Resumable by episode tag; failures are logged and skipped. CPU only.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np


def episode_dirs(run_dir: Path):
    for scene in sorted((run_dir / "rollout").iterdir()):
        if scene.is_dir():
            for ep in sorted(scene.iterdir()):
                if (ep / "sequence.json").exists() and (ep / "video.mp4").exists():
                    yield ep


def mine_episode(ep_dir: Path, out_images: Path, gamma: float, frames_per_step: float,
                 stride: int, quality: int):
    import imageio
    import imageio.v3 as iio
    from longnav.utils.state_probe import distance_return_targets

    seq = json.loads((ep_dir / "sequence.json").read_text())
    d = seq["distance_to_goal"]
    n_steps = len(seq.get("reward", d))
    returns = distance_return_targets(d[:n_steps + 1] if len(d) > n_steps else d,
                                      gamma=gamma)
    tag = ep_dir.name.replace("@", "_").replace(":", "_").replace("#", "_")
    ep_out = out_images / tag
    ep_out.mkdir(parents=True, exist_ok=True)
    wanted = {}
    for k in range(0, n_steps, stride):
        wanted[int(round(k * frames_per_step))] = k
    rows_by_step = {}
    # STREAM the decode: whole-video imread would hold ~GBs for long episodes.
    saved = {}
    reader = imageio.get_reader(str(ep_dir / "video.mp4"))   # ffmpeg backend
    for fi, frame in enumerate(reader):
        if fi in wanted:
            k = wanted[fi]
            img_path = ep_out / f"step_{k:05d}.jpg"
            if not img_path.exists():
                iio.imwrite(img_path, frame, quality=quality)
            saved[k] = img_path
        if fi >= max(wanted):
            break
    reader.close()
    rows = []
    for k in sorted(saved):
        img_path = saved[k]
        dk = d[k] if k < len(d) else None
        rows.append({
            "image": str(img_path),
            "episode_tag": tag,
            "step": k,
            "distance": None if (dk is None or not np.isfinite(dk)) else float(dk),
            f"return": float(returns[k]) if k < len(returns) else 0.0,
        })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dirs", nargs="+", required=True,
                   help="RL run dirs (each containing rollout/)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--return-gamma", type=float, required=True)
    p.add_argument("--episodes-file", default=None,
                   help="only mine episode dirs whose basename (tag) is listed in this "
                        "file (one per line); pins a fixed offline-eval set")
    p.add_argument("--frame-stride", type=int, default=2,
                   help="keep every k-th policy step (2 halves the store at little "
                        "information cost -- adjacent steps are 0.4 s apart)")
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--gap", type=int, default=10)
    p.add_argument("--dt", type=float, default=0.04)
    p.add_argument("--jpeg-quality", type=int, default=90)
    p.add_argument("--limit-episodes", type=int, default=None)
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "images").mkdir(exist_ok=True)
    labels_path = out / "labels.jsonl"
    done = set()
    if labels_path.exists():
        with open(labels_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["episode_tag"])
                except json.JSONDecodeError:
                    pass
    frames_per_step = args.gap * args.dt * args.fps
    meta = {"return_gamma": args.return_gamma, "frame_stride": args.frame_stride,
            "frames_per_step": frames_per_step, "runs": args.run_dirs}
    (out / "mining_manifest.json").write_text(json.dumps(meta, indent=2))

    n_ep = n_rows = n_fail = 0
    with open(labels_path, "a") as lf, open(out / "failures.log", "a") as failf:
        for run in args.run_dirs:
            only = None
            if args.episodes_file:
                only = {l.strip() for l in open(args.episodes_file) if l.strip()}
            for ep_dir in episode_dirs(Path(run)):
                if only is not None and ep_dir.name not in only:
                    continue
                tag = ep_dir.name.replace("@", "_").replace(":", "_").replace("#", "_")
                if tag in done:
                    continue
                if args.limit_episodes is not None and (n_ep + n_fail) >= args.limit_episodes:
                    break
                try:
                    rows = mine_episode(ep_dir, out / "images", args.return_gamma,
                                        frames_per_step, args.frame_stride,
                                        args.jpeg_quality)
                except Exception as exc:   # decode failures skip, never abort
                    failf.write(f"{type(exc).__name__}: {exc}\t{ep_dir}\n")
                    n_fail += 1
                    continue
                for r in rows:
                    r["run"] = os.path.basename(run.rstrip("/"))
                    lf.write(json.dumps(r) + "\n")
                n_ep += 1
                n_rows += len(rows)
                if n_ep % 200 == 0:
                    print(f"  {n_ep} episodes, {n_rows} rows", flush=True)
    print(f"mined {n_ep} episodes -> {n_rows} rows ({n_fail} decode failures)")


if __name__ == "__main__":
    main()
