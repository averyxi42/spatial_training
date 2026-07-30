"""
Run a trained turn-vector policy on `continuous_demos` episodes -- open loop, or closed
loop inside Habitat.

Two modes, one script:

  open-loop   Feed the *recorded* observation images to the policy and score each
              predicted action chunk against the dataset's ground-truth chunk. No
              simulator, so this runs anywhere the model runs. It answers "did the head
              learn the mapping", and it is scored against the trivial baseline of always
              predicting the dataset's mean chunk -- without that comparison a small RMSE
              on near-stationary data means nothing.

  closed-loop The real thing: Habitat renders what the robot actually sees, the policy
              emits a chunk relative to the robot's *current* pose, the first `gap`
              setpoints are PID-tracked, and the loop repeats from wherever the robot
              ended up. Errors compound -- that is the point.

Almost everything here is `continuous_demos` code: `build_robot_sim`, `resolve_scene_path`
and `extract_demo` from `episode_io`, `PIDPoseController` / `track_trajectory`,
`relative_to_pose` / `reconstruct_reference_trajectory` from `action_chunking`, and
`compute_metrics` / `plot_bev` from `tracking_eval`. The new part is only the policy in
the loop.

------------------------------------------------------------------------------
The environment boundary
------------------------------------------------------------------------------
`habitat_sim` and `transformers` live in different conda envs here (the sim env has no
transformers; the VLM env has no habitat_sim, and neither has Ray installed). So in
closed-loop mode this script runs *itself* in the model env as a subprocess and talks to
it over a unix socket: the sim side sends an RGB frame, the model side returns a chunk.
That is the entire protocol -- no Ray, no shared CUDA context, ~40 lines. Frames cross as
raw bytes plus shape/dtype rather than pickled arrays, because the two envs also disagree
about numpy (1.26 vs 2.2) and unpickling arrays across that boundary is asking for
trouble. `PolicyClient`/`serve_policy` are the only two pieces that know about the
boundary, so swapping them for Ray actors later touches nothing else.

    # open loop (model env)
    python data_scripts/run_vector_policy_habitat.py --mode open-loop \
        --ckpt dump/vector_sft_3090/final \
        --dataset-dir dump/datasets/action_chunks_conversational --split validation

    # closed loop (run from the SIM env; it launches the model env itself)
    python data_scripts/run_vector_policy_habitat.py --mode closed-loop \
        --ckpt dump/vector_sft_3090/final \
        --dataset-dir dump/datasets/action_chunks_conversational --split validation \
        --demo-dataset ~/codes/habitat/continuous_demos/examples/17DRP5sb8fy.json.gz \
        --scene-root ~/codes/habitat/continuous_demos/data/scene_datasets \
        --policy-python ~/anaconda3/envs/longnav/bin/python
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
for p in (str(_ROOT), str(_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

AUTHKEY = b"longnav-vector-policy"
# The sim must render at the resolution the policy was trained on; the images in the
# reference dataset are 640x480 and the token count (hence the visual sparsification)
# depends on it.
DEFAULT_WIDTH, DEFAULT_HEIGHT = 640, 480


# ======================================================================================
# Policy access
# ======================================================================================
class LocalPolicy:
    """Runs `VectorRolloutPolicy` in this process (model env, or an env with both)."""

    def __init__(self, ckpt, device="cuda", merge_lora=True):
        from longnav.utils.vector_rollout import RolloutConfig, VectorRolloutPolicy

        self.policy = VectorRolloutPolicy.from_checkpoint(
            ckpt, RolloutConfig(device=device, merge_lora=merge_lora)
        )

    def reset(self, goal_text):
        self.policy.reset(goal_text=goal_text)

    def act(self, rgb: np.ndarray) -> np.ndarray:
        from PIL import Image

        chunk = self.policy.step(Image.fromarray(rgb))
        return np.asarray(chunk.numpy(), dtype=np.float64)

    @property
    def last_stats(self):
        return self.policy.last_stats

    def close(self):
        pass


class ConstantPolicy:
    """CONTROL: ignores the image and always emits the same chunk.

    Indispensable on this data, not optional. Two thirds of the reference chunks translate
    less than a centimetre, so "stand still" already tracks the recorded trajectory
    closely for a few steps -- a closed-loop ADE means nothing until it is compared
    against this. `kind="zero"` stands still; `kind="mean"` replays the dataset's mean
    chunk (the best possible image-blind constant).
    """

    def __init__(self, target_shape, kind="zero", mean_chunk=None):
        if kind == "mean" and mean_chunk is not None:
            self.chunk = np.broadcast_to(
                np.asarray(mean_chunk, dtype=float), target_shape
            ).copy()
        else:
            self.chunk = np.zeros(target_shape, dtype=float)
        self.kind = kind

    def reset(self, goal_text):
        pass

    def act(self, rgb):
        return self.chunk.copy()

    @property
    def last_stats(self):
        return {"latency_s": 0.0}

    def close(self):
        pass


class PolicyClient:
    """Runs the policy in a subprocess under a different interpreter."""

    def __init__(self, ckpt, policy_python, device="cuda", log_path=None):
        from multiprocessing.connection import Listener

        self.socket_path = Path(tempfile.mkdtemp(prefix="vecpolicy_")) / "sock"
        self.log_path = Path(log_path) if log_path else self.socket_path.with_suffix(".log")
        listener = Listener(str(self.socket_path), family="AF_UNIX", authkey=AUTHKEY)
        cmd = [
            str(policy_python), str(Path(__file__).resolve()),
            "--serve", "--socket", str(self.socket_path),
            "--ckpt", str(ckpt), "--device", device,
        ]
        print(f"[bridge] launching policy server: {' '.join(cmd)}")
        print(f"[bridge] server log -> {self.log_path}")
        self._log = open(self.log_path, "w")
        self.proc = subprocess.Popen(cmd, stdout=self._log, stderr=subprocess.STDOUT)
        self.conn = listener.accept()  # blocks until the model has loaded and connected
        listener.close()
        print("[bridge] policy server connected")

    def _rpc(self, payload):
        try:
            self.conn.send(payload)
            reply = self.conn.recv()
        except (EOFError, BrokenPipeError) as exc:
            raise RuntimeError(
                f"policy server died; see {self.log_path}\n"
                + Path(self.log_path).read_text()[-2000:]
            ) from exc
        if reply.get("error"):
            raise RuntimeError(f"policy server error: {reply['error']}")
        return reply

    def reset(self, goal_text):
        self._rpc({"cmd": "reset", "goal_text": goal_text})

    def act(self, rgb: np.ndarray) -> np.ndarray:
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        reply = self._rpc(
            {"cmd": "act", "shape": rgb.shape, "dtype": "uint8", "data": rgb.tobytes()}
        )
        self._last_stats = reply.get("stats", {})
        return np.frombuffer(reply["chunk"], dtype=np.float32).reshape(reply["chunk_shape"]).astype(np.float64)

    @property
    def last_stats(self):
        return getattr(self, "_last_stats", {})

    def close(self):
        try:
            self.conn.send({"cmd": "stop"})
        except Exception:
            pass
        self.proc.wait(timeout=30)
        self._log.close()


def serve_policy(args):
    """The `--serve` role: load the model, answer reset/act over the socket."""
    from multiprocessing.connection import Client
    from PIL import Image

    from longnav.utils.vector_rollout import RolloutConfig, VectorRolloutPolicy

    print(f"loading {args.ckpt} on {args.device}...", flush=True)
    policy = VectorRolloutPolicy.from_checkpoint(
        args.ckpt, RolloutConfig(device=args.device, merge_lora=True)
    )
    conn = Client(args.socket, family="AF_UNIX", authkey=AUTHKEY)
    print("connected to sim process", flush=True)
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            break
        try:
            if msg["cmd"] == "stop":
                break
            if msg["cmd"] == "reset":
                policy.reset(goal_text=msg.get("goal_text"))
                conn.send({"ok": True})
                continue
            rgb = np.frombuffer(msg["data"], dtype=msg["dtype"]).reshape(msg["shape"])
            chunk = policy.step(Image.fromarray(rgb)).numpy().astype(np.float32)
            conn.send({
                "chunk": chunk.tobytes(),
                "chunk_shape": list(chunk.shape),
                "stats": {k: v for k, v in policy.last_stats.items()},
            })
        except Exception as exc:  # report, don't die silently on the sim side
            import traceback

            traceback.print_exc()
            conn.send({"error": f"{type(exc).__name__}: {exc}"})
    print("policy server exiting", flush=True)


# ======================================================================================
# Dataset helpers
# ======================================================================================
def episode_rows(example):
    """Episode-per-row table -> the per-observation row dicts continuous_demos expects."""
    return [
        {
            "obs_index": int(example["obs_indices"][i]),
            "frame_index": int(example["frame_indices"][i]),
            "obs_pose": np.asarray(example["obs_poses"][i], dtype=float),
            "action_chunk": np.asarray(example["action_chunks"][i], dtype=float),
            "image": example["images"][i],
        }
        for i in range(len(example["images"]))
    ]


def chunk_errors(pred: np.ndarray, gt: np.ndarray):
    """Per-dim RMSE plus displacement errors over the chunk horizon, in target units."""
    err = pred - gt
    return {
        "rmse_dx": float(np.sqrt((err[..., 0] ** 2).mean())),
        "rmse_dy": float(np.sqrt((err[..., 1] ** 2).mean())),
        "rmse_dtheta": float(np.sqrt((err[..., 2] ** 2).mean())),
        # Positional displacement error, averaged over the chunk and at its final step.
        "ade_xy": float(np.linalg.norm(err[..., :2], axis=-1).mean()),
        "fde_xy": float(np.linalg.norm(err[..., -1, :2], axis=-1).mean()),
    }


# ======================================================================================
# Open loop
# ======================================================================================
def moving_mask(gt: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    """Which chunks actually translate. 66% of chunks in the reference data move less
    than a centimetre (the demos turn in place a lot), so an aggregate xy error is mostly
    a measure of how well the model predicts standing still. Reporting the moving subset
    separately is the only way the translation number means anything."""
    return np.linalg.norm(gt[:, -1, :2], axis=-1) > threshold


def run_open_loop(policy, dataset, n_episodes, max_steps, mean_chunk=None, start_obs=0):
    from longnav.utils.vector_sft import load_image

    per_episode = []
    all_pred, all_gt = [], []
    for e in range(min(n_episodes, len(dataset))):
        ex = dataset[e]
        rows = episode_rows(ex)[start_obs:]
        rows = rows[:max_steps] if max_steps else rows
        policy.reset(ex.get("goal_text") or "the goal object")
        preds, t0 = [], time.perf_counter()
        for r in rows:
            preds.append(policy.act(np.asarray(load_image(r["image"]), dtype=np.uint8)))
        pred = np.stack(preds)
        gt = np.stack([r["action_chunk"] for r in rows])
        all_pred.append(pred)
        all_gt.append(gt)
        m = chunk_errors(pred, gt)
        m.update(episode=ex["episode_id"], steps=len(rows),
                 seconds=time.perf_counter() - t0)
        per_episode.append(m)
        print(f"  ep {e} ({len(rows)} obs): ade_xy {m['ade_xy']:.4f} m  "
              f"fde_xy {m['fde_xy']:.4f} m  rmse_dtheta {m['rmse_dtheta']:.4f} rad  "
              f"({m['seconds'] / max(1, len(rows)):.2f} s/step)")

    pred, gt = np.concatenate(all_pred), np.concatenate(all_gt)
    move = moving_mask(gt)
    summary = {
        "model": chunk_errors(pred, gt),
        "n_turns": int(len(pred)),
        "n_moving": int(move.sum()),
        "model_moving_only": chunk_errors(pred[move], gt[move]) if move.any() else None,
    }
    # The baseline that matters: always predict the dataset's mean chunk. On
    # near-stationary trajectories that is already a decent predictor, so a model that
    # does not beat it has learned nothing useful.
    if mean_chunk is not None:
        summary["mean_chunk_baseline"] = chunk_errors(
            np.broadcast_to(mean_chunk, gt.shape), gt
        )
    summary["zero_baseline"] = chunk_errors(np.zeros_like(gt), gt)
    if move.any():
        summary["zero_baseline_moving_only"] = chunk_errors(
            np.zeros_like(gt[move]), gt[move]
        )
    return summary, per_episode


# ======================================================================================
# Closed loop
# ======================================================================================
def run_closed_loop(policy, dataset, args):
    """Render -> policy -> track the chunk's first `gap` setpoints -> repeat."""
    from continuous_demos.action_chunking import (
        reconstruct_reference_trajectory,
        relative_to_pose,
    )
    from continuous_demos.episode_io import (
        DEFAULT_URDF_PATH,
        build_robot_sim,
        extract_demo,
        find_scene_dataset_config,
        load_demo_objectnav,
        resolve_scene_path,
    )
    from continuous_demos.pid_pose_controller import (
        PIDPoseController,
        PIDPoseControllerConfig,
        track_trajectory,
    )
    from continuous_demos.tracking_eval import compute_metrics, plot_bev

    demo = load_demo_objectnav(args.demo_dataset)
    episodes_by_id = {ep["episode_id"]: ep for ep in demo["episodes"]}

    scene_path = resolve_scene_path(dataset[0]["scene_id"], args.scene_root)
    scene_cfg = find_scene_dataset_config(scene_path, args.scene_root)
    robot_sim = build_robot_sim(
        scene_path, scene_cfg, args.width, args.height, str(DEFAULT_URDF_PATH)
    )
    controller = PIDPoseController(robot_sim, PIDPoseControllerConfig())
    out_dir = Path(args.output_dir)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    results = []
    for e in range(min(args.num_episodes, len(dataset))):
        ex = dataset[e]
        if ex["episode_id"] not in episodes_by_id:
            print(f"  skipping {ex['episode_id']}: not in {args.demo_dataset}")
            continue
        episode = episodes_by_id[ex["episode_id"]]
        rows = episode_rows(ex)[args.start_obs:]
        n_steps = min(args.max_steps, len(rows)) if args.max_steps else len(rows)
        dt_native = float(ex["dt_native"])

        # Ground truth for reference: the trajectory the recorded chunks decode to.
        ref_traj = reconstruct_reference_trajectory(rows[:n_steps])

        # Spawn exactly where the recorded episode's first observation was, matching
        # replay_chunked_actions.run_chunked_episode (avoids navmesh-snap drift).
        path_3d, _h, _a = extract_demo(episode)
        x0, y0, th0 = rows[0]["obs_pose"]
        spawn = dict(
            robot_translation=np.array([x0, float(path_3d[0][1]), -y0], dtype=float),
            snap_to_navmesh=False,
        )
        if args.start_obs == 0:
            # Observation 0 is the episode start, so its recorded rotation applies --
            # identical to replay_chunked_actions.run_chunked_episode.
            spawn["robot_rotation"] = episode["start_rotation"].copy()
        else:
            # Starting mid-episode, the episode's start rotation is the wrong heading.
            # `reset` seeds the base yaw through home_joint_positions["joint_th"], so give
            # it this observation's own theta instead.
            spawn["home_joint_positions"] = {"joint_th": float(th0)}
        robot_sim.reset(**spawn)
        controller.reset()
        policy.reset(ex.get("goal_text") or "the goal object")

        actual, pred_chunks, latencies = [robot_sim.get_2d_pose()], [], []
        for i in range(n_steps):
            rgb = np.asarray(robot_sim.get_obs()[args.sensor_uuid])[..., :3]
            pose_now = robot_sim.get_2d_pose()
            chunk = policy.act(np.ascontiguousarray(rgb, dtype=np.uint8))
            pred_chunks.append(chunk)
            latencies.append(policy.last_stats.get("latency_s", float("nan")))

            # Only the actions up to the next observation are ever used; the rest would
            # be superseded (see action_chunking's docstring). The anchor is the robot's
            # CURRENT pose, not the recorded one -- that is what closes the loop.
            gap = (
                rows[i + 1]["frame_index"] - rows[i]["frame_index"]
                if i + 1 < len(rows)
                else len(chunk)
            )
            gap = int(np.clip(gap, 1, len(chunk)))
            setpoints = np.stack([relative_to_pose(pose_now, chunk[j]) for j in range(gap)])
            seg = track_trajectory(
                robot_sim, controller, setpoints, dt=dt_native, initial_pose=pose_now
            )
            actual.extend(seg["actual_poses"])

        actual = np.asarray(actual, dtype=float)
        n = min(len(actual), len(ref_traj))
        metrics = compute_metrics(actual[:n], ref_traj[:n])
        metrics.update(
            episode=ex["episode_id"], model_steps=n_steps, ticks=int(n),
            mean_policy_latency_s=float(np.nanmean(latencies)),
        )
        results.append(metrics)
        plot_bev(
            f"{ex['episode_id']} closed-loop policy vs recorded",
            ref_traj[:n], actual[:n],
            out_dir / "plots" / f"closed_loop_{e}_{ex['episode_id'].replace(':', '_')}.png",
        )
        print(f"  ep {e} {ex['episode_id']}: {n_steps} policy steps, "
              + ", ".join(f"{k} {v:.3f}" for k, v in metrics.items()
                          if isinstance(v, float) and k != "mean_policy_latency_s")
              + f", {metrics['mean_policy_latency_s'] * 1e3:.0f} ms/step")
    return results


# ======================================================================================
def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--mode", default="open-loop", choices=["open-loop", "closed-loop"])
    p.add_argument("--ckpt", required=True, help="trained checkpoint dir (from train_vector_sft)")
    p.add_argument("--dataset-dir", default="dump/datasets/action_chunks_conversational")
    p.add_argument("--split", default="validation")
    p.add_argument("--num-episodes", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=24,
                   help="policy steps per episode (0 = the whole episode). Each step adds "
                        "~330 tokens to the KV cache, so long episodes are a memory cost")
    p.add_argument("--start-obs", type=int, default=0,
                   help="skip this many observations at the episode start. The opening of "
                        "these demos is mostly rotation-in-place ('looking'), so the first "
                        "few observations are not representative of the whole trajectory")
    p.add_argument("--output-dir", default="dump/vector_policy_eval")
    p.add_argument("--device", default="cuda")
    p.add_argument("--policy", default="model", choices=["model", "zero", "mean"],
                   help="CONTROL: 'zero' stands still, 'mean' replays the dataset mean "
                        "chunk. Both ignore the image, so they bound what an image-blind "
                        "policy achieves on this data")

    s = p.add_argument_group("closed loop / simulator")
    s.add_argument("--demo-dataset", default=None,
                   help="the .json.gz Habitat-web ObjectNav file the episodes came from")
    s.add_argument("--scene-root", default=None)
    s.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    s.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    s.add_argument("--sensor-uuid", default="color_sensor")
    s.add_argument("--policy-python", default=os.environ.get("VECTOR_POLICY_PYTHON"),
                   help="interpreter with torch/transformers, if this env lacks them. "
                        "Defaults to $VECTOR_POLICY_PYTHON")

    p.add_argument("--serve", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--socket", default=None, help=argparse.SUPPRESS)
    args = p.parse_args()

    if args.serve:
        return serve_policy(args)

    if args.policy != "model":
        policy = None  # built after the dataset loads (the mean comes from the data)
    elif _importable("transformers") and _importable("torch"):
        policy = LocalPolicy(args.ckpt, device=args.device)
    else:
        if not args.policy_python:
            raise SystemExit(
                "this interpreter has no transformers/torch, so the policy must run in "
                "another env: pass --policy-python <interpreter> (or set "
                "$VECTOR_POLICY_PYTHON)"
            )
        policy = PolicyClient(args.ckpt, args.policy_python, device=args.device)

    from datasets import load_from_disk

    ds = load_from_disk(os.path.expanduser(args.dataset_dir))
    ds = ds[args.split] if hasattr(ds, "keys") else ds
    mean_chunk = dataset_mean_chunk(ds)
    if policy is None:
        shape = target_shape_from_checkpoint(args.ckpt)
        policy = ConstantPolicy(shape, args.policy, mean_chunk)
        print(f"CONTROL policy: {args.policy} (image-blind, chunk shape {shape})")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{args.mode}: {min(args.num_episodes, len(ds))} episode(s) from "
          f"{args.dataset_dir}:{args.split}, checkpoint {args.ckpt}")

    try:
        if args.mode == "open-loop":
            summary, per_episode = run_open_loop(
                policy, ds, args.num_episodes, args.max_steps or None, mean_chunk,
                start_obs=args.start_obs,
            )
            print(f"\n=== chunk prediction, open loop ({summary['n_turns']} turns, "
                  f"{summary['n_moving']} of them translating >1cm) ===")
            for name, m in summary.items():
                if isinstance(m, dict):
                    print(f"  {name:>20}: ade_xy {m['ade_xy']:.4f} m  fde_xy "
                          f"{m['fde_xy']:.4f} m  rmse_dtheta {m['rmse_dtheta']:.4f} rad")
            result = {"summary": summary, "per_episode": per_episode}
        else:
            if not args.demo_dataset:
                raise SystemExit("--demo-dataset is required for closed-loop mode")
            result = {"closed_loop": run_closed_loop(policy, ds, args)}
    finally:
        policy.close()

    tag = "" if args.policy == "model" else f"_{args.policy}"
    out_path = out_dir / f"{args.mode.replace('-', '_')}{tag}_results.json"
    out_path.write_text(json.dumps(result, indent=2, default=float))
    print(f"\nResults -> {out_path}")


def _importable(name):
    import importlib.util

    return importlib.util.find_spec(name) is not None


def target_shape_from_checkpoint(ckpt):
    """Per-turn target shape, read straight from the checkpoint's json.

    Deliberately not `from longnav.utils.vector_sft import HEAD_CONFIG_FILE`: that module
    imports transformers, and the control policies have to run in the sim env, which has
    neither transformers nor torch. The filename mirrors `vector_sft.HEAD_CONFIG_FILE`.
    """
    meta = json.loads((Path(ckpt) / "turn_vector_head_config.json").read_text())
    return tuple(meta["target_shape"])


def dataset_mean_chunk(dataset, max_rows=64):
    """Mean action chunk over the split -- the best image-blind constant predictor.

    Computed from the data with numpy rather than read out of the head's normalizer
    buffers, so the same definition is available in both envs (the sim env has no torch)
    and the baseline is anchored to the data being scored, not to training-time stats.
    """
    acc, n = None, 0
    for i in range(min(max_rows, len(dataset))):
        ch = np.asarray(dataset[i]["action_chunks"], dtype=float)
        acc = ch.sum(axis=0) if acc is None else acc + ch.sum(axis=0)
        n += ch.shape[0]
    return acc / max(1, n)


if __name__ == "__main__":
    main()
