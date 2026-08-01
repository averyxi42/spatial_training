"""
######################################################################################
#  DEPRECATED -- THIS HARNESS SILENTLY PRODUCES CONTAMINATED NUMBERS. DO NOT USE IT.
#
#  Use instead:
#      /Projects/habitat_physical_nav/scripts/eval_objectnav_policy.py
#                --fresh-sim-per-episode
#
#  Why: `HabitatRobotSim.reset()` does not fully restore the physics state between
#  episodes, so an episode's result depends on which episodes ran before it in the
#  same simulator process. This script predates that discovery and has no
#  `--fresh-sim-per-episode` equivalent, so every number it has ever produced is
#  order-dependent, and re-running the same split with different sharding can give
#  a different answer.
#
#  It is not a rounding error. Same checkpoint (vector_sft/checkpoint-2500), same
#  14 val_mini episodes, identical in every respect except the clean-slate flag:
#
#      metric           shared simulator      fresh simulator
#      success                    0.214                0.357
#      oracle_success             0.214                0.357
#      spl                        0.095                0.166
#      oracle_spl                 0.168                0.299
#
#  Two of fourteen episodes flip from failure to success -- roughly 40% relative
#  understatement of the checkpoint. Measured with the *stateless* image-blind
#  `forward` control (so the policy cannot be the cause), the same episode runs a
#  1.4% different trajectory alone vs after another episode, and
#  `--fresh-sim-per-episode` reproduces the run-alone path to 0.000000 m.
#
#  Full write-up: /Projects/spatial_training/dump/eval_system/FINDINGS.md section 5,
#  and /Projects/habitat_physical_nav/docs/OBJECTNAV_EVAL.md.
#
#  The replacement is a superset: it reproduces this script's screening verdicts
#  bit-identically (14/30 runnable, same uids) and its start geodesics to 0.000000 m
#  on all 14 episodes, and it adds the clean-slate flag, a PolicyBackend extension
#  point for new head types, behavioural diagnostics (frozen time, collisions,
#  motion coherence) and paired run comparison.
#
#  This file is kept only so old runs remain readable (`--report` still works) and
#  as the historical record. Running a rollout requires
#  `--i-know-this-is-deprecated`, which exists to make the choice deliberate, not to
#  make the numbers correct -- they will still be contaminated.
######################################################################################

Evaluate a trained turn-vector policy on *standard* Habitat ObjectNav episodes.

This is the companion to `run_vector_policy_habitat.py`, which can only replay recorded
demonstrations: it needs an HF table of observation poses and ground-truth action chunks,
and it scores the policy against that recording. Standard ObjectNav episode files
(`objectnav/hm3d/v1/val_mini/...`) carry no demonstration at all -- just a start pose, a
goal category, and the goal objects' view points. Everything needed to *evaluate* a
navigator is in there, so this script drives the policy from the episode start and scores
it with the usual ObjectNav metrics instead of against a reference trajectory.

The stack underneath is `habitat_sim` + `continuous_demos`, not `habitat-lab`, so the
episode file is parsed directly here rather than through a habitat-lab Dataset/Env.

------------------------------------------------------------------------------
The two navmeshes (this matters, and it is why episodes get skipped)
------------------------------------------------------------------------------
`build_robot_sim` recomputes the navmesh for the *robot's* footprint (radius 0.28 m,
height 1.0 m). The ObjectNav episodes were generated against the navmesh that ships with
the scene, cut for a much slimmer 0.1 m agent. The robot's mesh is therefore strictly
smaller and more fragmented, and on some episodes the goal ends up on a different navmesh
island from the start -- the robot physically cannot fit through whatever connects them.

Those episodes are unwinnable for this embodiment, so they are screened out *before* any
compute is spent on them and reported explicitly: the summary prints how many were
skipped and why, and every skipped episode is listed in the results JSON under
`"skipped"`. They are never silently dropped and never counted as failures. Both
distances are recorded per episode (`geodesic_start_m` on the robot mesh,
`geodesic_start_dataset_m` on the shipped one) so the gap stays visible.

Metrics use the robot's own navmesh: it is the mesh the episode was screened on, so the
shortest path in the SPL denominator is one the robot could actually have taken.

------------------------------------------------------------------------------
Metrics
------------------------------------------------------------------------------
The policy emits pose chunks and has no STOP action, so "did it stop at the goal" has to
be read off the trajectory:

  success        distance to goal <= --success-distance at the FINAL step -- the policy
                 has to still be at the goal when the episode ends, not merely have
                 brushed past it.
  oracle_success distance to goal <= --success-distance at ANY step, i.e. what the policy
                 would have scored with an oracle stop signal that ends the episode on
                 arrival. This is the metric to read for "did it find the object".
  spl            success * d0 / max(d0, path_length)
  oracle_spl     oracle_success * d0 / max(d0, path_length up to the first arrival)

`--auto-stop` ends the episode on arrival, which makes success == oracle_success by
construction; it is off by default so both numbers stay informative.

------------------------------------------------------------------------------
Parallelism
------------------------------------------------------------------------------
Episodes are sharded over GPUs, one worker process per GPU. A worker re-runs this script
with `--worker-shard`, pinned to its GPU via CUDA_VISIBLE_DEVICES, and each worker
launches its own policy subprocess through `PolicyClient` (the sim env has habitat_sim but
a transformers too old for the checkpoint, so the model always runs in a second
interpreter). Shards are grouped by scene so a worker loads each scene once.

    python data_scripts/eval_objectnav_policy.py \
        --ckpt dump/vector_sft/checkpoint-6500 \
        --episodes data/datasets/objectnav/hm3d/v1/val_mini \
        --scene-root data/scene_datasets \
        --output-dir dump/objectnav_eval --max-steps 200 --record-video \
        --gpus 4,5,6,7 --policy-python /workspace/conda/envs/longnav_vlm/bin/python

    # what would run, and what would be skipped, without loading the model
    python data_scripts/eval_objectnav_policy.py --episodes ... --scene-root ... --dry-run
"""

import argparse
import gzip
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "src"), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# The policy bridge lives in the sibling script; there is exactly one implementation of
# the sim-env <-> model-env socket protocol and this is not the place for a second.
from run_vector_policy_habitat import LocalPolicy, PolicyClient, _importable  # noqa: E402

DEFAULT_WIDTH, DEFAULT_HEIGHT = 640, 480
# Execution cadence, from the chunk config the checkpoint was trained on
# (data/continuous_sft/chunk_config.json): 20 Hz control, an observation every 5 ticks.
DEFAULT_DT = 0.05
DEFAULT_GAP = 5


# ======================================================================================
# Episode parsing -- standard Habitat ObjectNav json.gz
# ======================================================================================
def _shard_paths(episodes_path: Path):
    """Every episode file under `episodes_path`.

    Accepts a single `.json.gz`, or a split directory in the standard layout where the
    top-level `<split>.json.gz` holds only the category tables and the episodes live in
    `content/*.json.gz` (which is how val_mini is laid out: its `val_mini.json.gz` has
    zero episodes).
    """
    if episodes_path.is_file():
        return [episodes_path]
    content = episodes_path / "content"
    shards = sorted(content.glob("*.json.gz")) if content.is_dir() else []
    if not shards:
        shards = [p for p in sorted(episodes_path.glob("*.json.gz"))]
    if not shards:
        raise SystemExit(f"no episode .json.gz found under {episodes_path}")
    return shards


def load_episodes(episodes_path: Path):
    """Flatten the episode files into plain dicts with the goal view points resolved.

    ObjectNav episodes carry `goals: []` and a scene-level `goals_by_category` table; the
    per-episode goals are the entry keyed by `<scene_basename>_<category>`. Success is
    defined against the goals' *view points* (navigable poses from which the object is
    visible), not the object centres, which can sit inside furniture.
    """
    out, seen = [], defaultdict(int)
    for shard in _shard_paths(episodes_path):
        with gzip.open(shard, "rt", encoding="utf-8") as fh:
            data = json.load(fh)
        by_cat = data.get("goals_by_category", {})
        for ep in data.get("episodes", []):
            key = f"{os.path.basename(ep['scene_id'])}_{ep['object_category']}"
            goals = by_cat.get(key, [])
            vps = np.asarray(
                [vp["agent_state"]["position"] for g in goals for vp in g.get("view_points", [])],
                dtype=np.float32,
            ).reshape(-1, 3)
            # MP3D val reuses episode_id within a scene (x8F5xyUWy9e has two episodes
            # numbered 3, for different goal categories), so scene:id is not a key. A
            # stable occurrence suffix makes it one -- without it the two share a video
            # filename and a shard-membership test, so one silently overwrites the other.
            scene_name = Path(ep["scene_id"]).stem.replace(".basis", "")
            base = f"{scene_name}:{ep['episode_id']}"
            seen[base] += 1
            out.append({
                "uid": base if seen[base] == 1 else f"{base}#{seen[base]}",
                "episode_id": str(ep["episode_id"]),
                "scene_id": ep["scene_id"],
                "object_category": ep["object_category"],
                "start_position": np.asarray(ep["start_position"], dtype=np.float64),
                "start_rotation": np.asarray(ep["start_rotation"], dtype=np.float64),
                "scene_dataset_config": ep.get("scene_dataset_config"),
                "view_points": vps,
                "shard": shard.name,
            })
    return out


def uid(ep) -> str:
    """The episode's unique key, assigned in `load_episodes` (see the note there)."""
    return ep["uid"]


def filter_episodes(eps, args):
    if args.scenes:
        want = {s.strip() for s in args.scenes.split(",") if s.strip()}
        eps = [e for e in eps if Path(e["scene_id"]).stem.replace(".basis", "") in want]
    if args.categories:
        want = {c.strip() for c in args.categories.split(",") if c.strip()}
        eps = [e for e in eps if e["object_category"] in want]
    if args.episode_ids:
        want = {i.strip() for i in args.episode_ids.split(",") if i.strip()}
        eps = [e for e in eps if e["episode_id"] in want or uid(e) in want]
    eps.sort(key=lambda e: (e["scene_id"], e["episode_id"]))

    # Subsetting, for splits too big to run whole: MP3D val is 2195 episodes over 11
    # scenes. Sample *per scene* rather than taking a head slice -- the episode files are
    # grouped by scene, so a plain head slice would evaluate one scene and call it a split.
    # Seeded, so the same flags always name the same episodes.
    if args.max_scenes or args.sample_per_scene:
        rng = np.random.default_rng(args.seed)
        by_scene = defaultdict(list)
        for e in eps:
            by_scene[e["scene_id"]].append(e)
        scenes = sorted(by_scene)
        if args.max_scenes and len(scenes) > args.max_scenes:
            scenes = sorted(rng.choice(scenes, size=args.max_scenes, replace=False).tolist())
        picked = []
        for scene in scenes:
            group = by_scene[scene]
            if args.sample_per_scene and len(group) > args.sample_per_scene:
                idx = rng.choice(len(group), size=args.sample_per_scene, replace=False)
                group = [group[i] for i in sorted(idx)]
            picked.extend(group)
        eps = picked
        print(f"subset: {len(eps)} episode(s) across {len(scenes)} scene(s) "
              f"(seed {args.seed})")
    return eps[: args.num_episodes] if args.num_episodes else eps


# ======================================================================================
# Scene / navmesh helpers
# ======================================================================================
def resolve_scene(ep, scene_root):
    """Absolute .glb path and the scene-dataset config to load it with.

    `continuous_demos.find_scene_dataset_config` only knows the MP3D config filenames, so
    the episode's own `scene_dataset_config` field is preferred -- for HM3D that is
    `hm3d_annotated_basis.scene_dataset_config.json`, without which the semantic
    annotations and the basis-compressed meshes do not resolve.
    """
    from continuous_demos.episode_io import find_scene_dataset_config, resolve_scene_path

    scene_path = resolve_scene_path(ep["scene_id"], scene_root)
    cfg = ep.get("scene_dataset_config")
    if cfg:
        cand = Path(str(cfg).lstrip("./"))
        for base in (Path(scene_root).parent.parent, Path(scene_root), _ROOT, Path.cwd()):
            for rel in (cand, Path(*cand.parts[2:]) if len(cand.parts) > 2 else cand):
                p = base / rel
                if p.exists():
                    return scene_path, str(p)
    return scene_path, find_scene_dataset_config(scene_path, scene_root)


def dataset_pathfinder(scene_path: Path):
    """The navmesh that ships with the scene -- the one the episodes were generated on.

    Kept only as a reference distance; it is cut for a 0.1 m agent and so overstates what
    this robot can reach.
    """
    import habitat_sim

    # HM3D is `<scene>.basis.glb` -> `<scene>.basis.navmesh`; MP3D is `<scene>.glb` ->
    # `<scene>.navmesh`. Testing the HM3D form first only works if a non-match is rejected:
    # `str.replace` returns the path unchanged for MP3D, and that path exists -- it is the
    # .glb -- so an existence check alone would hand the mesh loader a scene file.
    candidates = []
    hm3d = Path(str(scene_path).replace(".basis.glb", ".basis.navmesh"))
    if hm3d != scene_path:
        candidates.append(hm3d)
    candidates.append(scene_path.with_suffix(".navmesh"))
    for nav in candidates:
        if nav.exists() and nav.suffix == ".navmesh":
            pf = habitat_sim.nav.PathFinder()
            pf.load_nav_mesh(str(nav))
            if pf.is_loaded:
                return pf
    return None


def geodesic(pf, start, ends) -> float:
    """Shortest navmesh distance from `start` to the nearest of `ends` (inf if no path)."""
    import habitat_sim

    ends = np.asarray(ends, dtype=np.float32).reshape(-1, 3)
    if pf is None or len(ends) == 0:
        return float("nan")
    path = habitat_sim.MultiGoalShortestPath()
    path.requested_start = np.asarray(start, dtype=np.float32)
    path.requested_ends = ends
    pf.find_path(path)
    return float(path.geodesic_distance)


def snap_points(pf, pts):
    """Project points onto the navmesh, dropping the ones that fall off it entirely."""
    if pf is None or len(pts) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    snapped = np.asarray([pf.snap_point(p) for p in pts], dtype=np.float32)
    return snapped[~np.isnan(snapped[:, 0])]


def robot_position_3d(robot_sim) -> np.ndarray:
    """The robot's habitat-frame [x, y, z] position.

    NOT `robot.translation`: the base link is fixed at the spawn point and the robot moves
    through its world prismatic joints, so `translation` never changes during an episode.
    The live position lives in the planar control frame instead -- `get_2d_pose()` returns
    [X, Y, theta] with X = habitat_x and Y = -habitat_z -- and the vertical is the base
    translation plus the passive joint_z. Reading `translation` directly pins every goal
    distance to the spawn.
    """
    x, y_planar, _theta = robot_sim.get_2d_pose()
    height = float(robot_sim.robot.translation[1])
    try:
        height += robot_sim.get_joint_state("joint_z")
    except ValueError:
        pass  # passive joint without a motor; the base height is close enough to snap
    return np.array([x, height, -y_planar], dtype=np.float32)


class GoalTracker:
    """Distance from the robot's live position to the nearest goal view point.

    Geodesic on the robot's navmesh (so it respects walls), with the robot's 3D position
    snapped onto the mesh first. The euclidean distance is carried alongside because the
    robot is a physics body that can end up slightly off the navmesh, in which case the
    snap -- and therefore the geodesic -- can jump; a large gap between the two is the
    signal that this happened.
    """

    def __init__(self, pathfinder, view_points):
        self.pf = pathfinder
        self.goals = snap_points(pathfinder, view_points)
        self.raw_goals = np.asarray(view_points, dtype=np.float32).reshape(-1, 3)

    def from_position(self, pos3d):
        pos3d = np.asarray(pos3d, dtype=np.float32)
        if len(self.goals) == 0:
            return float("nan"), float("nan")
        euc = float(np.linalg.norm(self.raw_goals - pos3d[None, :], axis=1).min())
        snapped = self.pf.snap_point(pos3d)
        if np.isnan(snapped[0]):
            return float("inf"), euc
        return geodesic(self.pf, snapped, self.goals), euc


def screen_episode(ep, robot_pf, dataset_pf, success_distance=0.0):
    """Is this episode winnable by this robot? Returns (ok, reason, d_robot, d_dataset).

    An episode is runnable when a path exists on the robot's own navmesh from its start to
    some view point of the goal. Everything else -- start off the mesh, goal off the mesh,
    goal on a different island -- makes the episode unwinnable for this embodiment, and is
    reported rather than scored as a failure.
    """
    start = np.asarray(ep["start_position"], dtype=np.float32)
    d_dataset = geodesic(dataset_pf, start, ep["view_points"]) if dataset_pf else float("nan")
    if len(ep["view_points"]) == 0:
        return False, "no goal view points in the episode file", float("nan"), d_dataset
    snapped_start = robot_pf.snap_point(start)
    if np.isnan(snapped_start[0]):
        return False, "start position is off the robot navmesh", float("inf"), d_dataset
    goals = snap_points(robot_pf, ep["view_points"])
    if len(goals) == 0:
        return False, "no goal view point lands on the robot navmesh", float("inf"), d_dataset
    d_robot = geodesic(robot_pf, snapped_start, goals)
    if not np.isfinite(d_robot):
        return (False, "goal unreachable on the robot navmesh (start and goal are on "
                "different islands -- the robot does not fit through)", d_robot, d_dataset)
    if d_robot <= success_distance:
        # The dataset guarantees a minimum start distance on ITS navmesh, not on the
        # robot's, so a handful of episodes start already inside the success radius.
        # Scoring them would hand the policy a success it did not navigate to.
        return (False, f"starts {d_robot:.2f} m from the goal, already within the "
                f"{success_distance:.1f} m success radius (trivially successful)",
                d_robot, d_dataset)
    return True, "", d_robot, d_dataset


# ======================================================================================
# Video
# ======================================================================================
def annotate(rgb, *, step, t, goal, dist, dist_start, reached, is_obs):
    """Burn the step, elapsed sim time, goal and distance-to-goal into a copy of the frame.

    A copy: the policy is handed the raw render, never these pixels.
    """
    import cv2

    img = np.ascontiguousarray(rgb).copy()
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, 34), (0, 0, 0), -1)
    cv2.putText(img, f"step {step:>3}  t={t:6.2f}s  goal: {goal}", (8, 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    colour = (120, 255, 160) if dist <= dist_start else (140, 160, 255)
    cv2.putText(img, f"dist {dist:5.2f} m  (start {dist_start:5.2f})", (w - 285, 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 1, cv2.LINE_AA)
    if is_obs:
        # Marks the frames the policy actually acts on; the rest are intermediate control
        # ticks, present so the video plays back in real time. Its own row, clear of the
        # right-aligned distance readout.
        cv2.circle(img, (14, 50), 5, (80, 200, 255), -1)
        cv2.putText(img, "obs", (26, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (80, 200, 255), 1, cv2.LINE_AA)
    if reached:
        cv2.putText(img, "REACHED", (w - 285, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (120, 255, 160), 2, cv2.LINE_AA)
    return img


class Recorder:
    """Streams frames straight to the encoder.

    A 200-step episode is 1000 frames; at 640x480 that is ~0.9 GiB if held in a list, per
    episode, per worker. Appending as we go keeps a worker's footprint flat.
    """

    def __init__(self, path, fps, scale=1.0, enabled=True):
        self.enabled, self.n, self.scale, self.path = enabled, 0, scale, path
        if not enabled:
            return
        import imageio.v2 as imageio

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # macro_block_size=1: 640x480 is already divisible, but a --video-scale can make it
        # odd, and silent resizing would desync the overlay from the frame.
        self.writer = imageio.get_writer(str(path), fps=fps, macro_block_size=1)

    def add(self, frame):
        if not self.enabled:
            return
        if self.scale != 1.0:
            import cv2

            frame = cv2.resize(frame, None, fx=self.scale, fy=self.scale,
                               interpolation=cv2.INTER_AREA)
        self.writer.append_data(frame)
        self.n += 1

    def close(self):
        if self.enabled:
            self.writer.close()


# ======================================================================================
# One episode
# ======================================================================================
def run_episode(robot_sim, controller, policy, ep, robot_pf, d0, d0_dataset, args):
    """Spawn at the episode start, run the policy, score it.

    One policy step = one render -> one chunk -> `--gap` control ticks of PID tracking
    against the first `gap` setpoints of that chunk, anchored on the robot's *current*
    pose. The rest of the chunk is discarded: it would be superseded by the next
    observation, exactly as in training.
    """
    from continuous_demos.action_chunking import relative_to_pose
    from continuous_demos.pid_pose_controller import track_trajectory

    tracker = GoalTracker(robot_pf, ep["view_points"])

    # Spawn verbatim at the episode's start pose. No navmesh snap by default: the start is
    # guaranteed navigable on the dataset's mesh, and snapping it onto the robot's smaller
    # mesh would silently move the agent away from the pose the episode specifies.
    robot_sim.reset(
        robot_translation=np.asarray(ep["start_position"], dtype=float).copy(),
        robot_rotation=np.asarray(ep["start_rotation"], dtype=float),
        snap_to_navmesh=args.snap_start,
    )
    controller.reset()
    policy.reset(ep["object_category"])

    dist, euc = tracker.from_position(robot_position_3d(robot_sim))
    dist_start = dist
    tag = "" if args.policy == "model" else f"{args.policy}_"
    rec = Recorder(
        Path(args.output_dir) / "videos" / f"{tag}{uid(ep).replace(':', '_')}.mp4",
        fps=round(1.0 / args.dt), scale=args.video_scale, enabled=args.record_video,
    )

    poses = [robot_sim.get_2d_pose()]
    dist_trace, latencies = [dist], []
    path_length = 0.0
    # Path length at the moment the goal was first reached -- the oracle-stop denominator.
    path_at_first_reach, first_reach_step = None, None
    t_sim, steps_run, terminated = 0.0, 0, None
    t0 = time.perf_counter()

    try:
        for i in range(args.max_steps):
            rgb = np.asarray(robot_sim.get_obs()[args.sensor_uuid])[..., :3]
            rec.add(annotate(rgb, step=i, t=t_sim, goal=ep["object_category"], dist=dist,
                             dist_start=dist_start, reached=first_reach_step is not None,
                             is_obs=True))

            chunk = policy.act(np.ascontiguousarray(rgb, dtype=np.uint8))
            latencies.append(policy.last_stats.get("latency_s", float("nan")))
            steps_run = i + 1

            pose_now = robot_sim.get_2d_pose()
            gap = min(args.gap, len(chunk))
            setpoints = np.stack([relative_to_pose(pose_now, chunk[j]) for j in range(gap)])

            # One tick at a time so the video gets a frame per dt and plays back in real
            # time. Feeding setpoints singly with initial_pose=<previous setpoint>
            # reproduces track_trajectory's own finite-difference feedforward exactly, and
            # the controller carries its integral state across calls, so the control the
            # robot sees is identical to handing it the whole block.
            prev = pose_now
            for j, setpoint in enumerate(setpoints):
                seg = track_trajectory(robot_sim, controller, setpoint[None, :], dt=args.dt,
                                       initial_pose=prev)
                prev = setpoint
                new_pose = seg["actual_poses"][-1]
                path_length += float(np.linalg.norm(new_pose[:2] - poses[-1][:2]))
                poses.append(new_pose)
                t_sim += args.dt

                dist, euc = tracker.from_position(robot_position_3d(robot_sim))
                if dist <= args.success_distance and first_reach_step is None:
                    first_reach_step, path_at_first_reach = i, path_length
                # The next step's [obs] frame covers the final tick of this chunk.
                if j < gap - 1:
                    rec.add(annotate(
                        np.asarray(robot_sim.get_obs()[args.sensor_uuid])[..., :3],
                        step=i, t=t_sim, goal=ep["object_category"], dist=dist,
                        dist_start=dist_start, reached=first_reach_step is not None,
                        is_obs=False))

            dist_trace.append(dist)
            if robot_sim.is_tipped_over():
                terminated = "robot tipped over"
                break
            if args.auto_stop and first_reach_step is not None:
                terminated = "auto-stop on arrival"
                break
    finally:
        rec.close()

    reached = first_reach_step is not None
    success = bool(np.isfinite(dist) and dist <= args.success_distance)
    denom = max(d0, path_length) if path_length > 0 else d0
    oracle_len = path_at_first_reach if path_at_first_reach is not None else path_length
    return {
        "uid": uid(ep),
        "episode_id": ep["episode_id"],
        "scene": Path(ep["scene_id"]).stem.replace(".basis", ""),
        "object_category": ep["object_category"],
        "success": float(success),
        "oracle_success": float(reached),
        "spl": float(success) * d0 / denom if denom > 0 else 0.0,
        "oracle_spl": (float(reached) * d0 / max(d0, oracle_len)) if oracle_len > 0 else 0.0,
        "distance_to_goal_m": float(dist),
        "distance_to_goal_euclidean_m": float(euc),
        "min_distance_to_goal_m": float(np.nanmin(dist_trace)),
        "geodesic_start_m": float(d0),
        "geodesic_start_dataset_m": float(d0_dataset),
        "path_length_m": float(path_length),
        "steps": int(steps_run),
        "ticks": int(len(poses) - 1),
        "sim_seconds": float(t_sim),
        "wall_seconds": float(time.perf_counter() - t0),
        "mean_policy_latency_s": float(np.nanmean(latencies)) if latencies else float("nan"),
        "first_reach_step": first_reach_step,
        "terminated": terminated,
        "video": str(rec.path) if args.record_video else None,
        "video_frames": int(rec.n),
        "dist_trace": [round(float(v), 3) for v in dist_trace],
    }


# ======================================================================================
# Worker: one GPU, a list of episodes, grouped by scene
# ======================================================================================
class ScriptedPolicy:
    """CONTROL: ignores the image and emits a fixed chunk.

    Not a baseline worth reporting -- a check that the harness itself is sound. `forward`
    drives the base straight ahead at a constant speed, so if it does not walk into a wall
    and shrink the distance to a goal in front of it, chunk decoding, anchoring or PID
    tracking is broken and no model number measured through this loop means anything.
    `zero` stands still, which bounds what a policy that has learned nothing scores.
    """

    def __init__(self, kind, chunk_len=10, dt=DEFAULT_DT, speed=1.0):
        step = speed * dt
        # Each row is an offset from the SAME anchor (the pose at chunk emission), so a
        # constant-velocity chunk ramps linearly rather than repeating one offset.
        self.chunk = np.zeros((chunk_len, 3), dtype=np.float64)
        if kind == "forward":
            self.chunk[:, 0] = step * np.arange(1, chunk_len + 1)
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


def build_policy(args):
    if args.policy != "model":
        print(f"CONTROL policy: {args.policy} (image-blind)")
        return ScriptedPolicy(args.policy, dt=args.dt)
    if _importable("transformers") and _importable("torch") and not args.policy_python:
        return LocalPolicy(args.ckpt, device=args.device,
                           max_context_tokens=args.max_context_tokens,
                           placeholder=args.placeholder)
    if not args.policy_python:
        raise SystemExit(
            "this interpreter cannot load the checkpoint; pass --policy-python <interpreter> "
            "(an env with a transformers new enough for the model), or set "
            "$VECTOR_POLICY_PYTHON"
        )
    log = Path(args.output_dir) / "logs" / f"policy_shard{args.shard_index}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    return PolicyClient(args.ckpt, args.policy_python, device=args.device, log_path=log,
                        max_context_tokens=args.max_context_tokens,
                        placeholder=args.placeholder)


def run_shard(episodes, args):
    """Run every episode in this shard, reusing one simulator per scene."""
    from continuous_demos.episode_io import DEFAULT_URDF_PATH, build_robot_sim
    from continuous_demos.pid_pose_controller import PIDPoseController, PIDPoseControllerConfig

    by_scene = defaultdict(list)
    for ep in episodes:
        by_scene[ep["scene_id"]].append(ep)

    policy = build_policy(args)
    results, skipped = [], []
    try:
        for scene_id, eps in by_scene.items():
            scene_path, scene_cfg = resolve_scene(eps[0], args.scene_root)
            print(f"[shard {args.shard_index}] scene {Path(scene_path).stem} "
                  f"({len(eps)} episodes)", flush=True)
            robot_sim = build_robot_sim(scene_path, scene_cfg, args.width, args.height,
                                        str(DEFAULT_URDF_PATH))
            controller = PIDPoseController(robot_sim, PIDPoseControllerConfig())
            robot_pf = robot_sim.sim.pathfinder
            data_pf = dataset_pathfinder(scene_path)
            try:
                for ep in eps:
                    ok, reason, d0, d0_ds = screen_episode(ep, robot_pf, data_pf,
                                                   args.success_distance)
                    if not ok:
                        print(f"[shard {args.shard_index}] SKIP {uid(ep)} "
                              f"({ep['object_category']}): {reason}", flush=True)
                        skipped.append({
                            "uid": uid(ep), "episode_id": ep["episode_id"],
                            "scene": Path(ep["scene_id"]).stem.replace(".basis", ""),
                            "object_category": ep["object_category"], "reason": reason,
                            "geodesic_start_m": float(d0),
                            "geodesic_start_dataset_m": float(d0_ds),
                        })
                        continue
                    r = run_episode(robot_sim, controller, policy, ep, robot_pf, d0, d0_ds,
                                    args)
                    results.append(r)
                    print(f"[shard {args.shard_index}] {r['uid']} ({r['object_category']}): "
                          f"dist {r['geodesic_start_m']:.2f} -> {r['distance_to_goal_m']:.2f} m "
                          f"(min {r['min_distance_to_goal_m']:.2f})  "
                          f"success {int(r['success'])} oracle {int(r['oracle_success'])}  "
                          f"spl {r['spl']:.3f} oracle_spl {r['oracle_spl']:.3f}  "
                          f"path {r['path_length_m']:.1f} m in {r['steps']} steps "
                          f"({r['wall_seconds']:.0f}s)", flush=True)
            finally:
                robot_sim.sim.close()
    finally:
        policy.close()
    return results, skipped


# ======================================================================================
# Aggregation
# ======================================================================================
def summarize(results, skipped, total):
    """Aggregate, averaging distances over the episodes where they are defined.

    A run can end with the robot off its navmesh -- driven into a nook the mesh does not
    cover, or onto a disconnected island -- and then the geodesic to the goal is `inf`.
    That is a legitimate (failed) outcome and stays a failure in `success`/`spl`, but
    averaging it into a mean distance makes the whole mean `inf` and destroys the one
    number that says how close the policy got. Distances are therefore averaged over the
    finite episodes and the excluded count is reported as `episodes_off_navmesh`.
    """
    rate_keys = ["success", "oracle_success", "spl", "oracle_spl"]
    dist_keys = ["distance_to_goal_m", "min_distance_to_goal_m", "path_length_m",
                 "geodesic_start_m", "distance_to_goal_euclidean_m", "steps"]
    s = {}
    if results:
        s = {k: float(np.mean([r[k] for r in results])) for k in rate_keys}
        for k in dist_keys:
            vals = [r[k] for r in results if np.isfinite(r[k])]
            s[k] = float(np.mean(vals)) if vals else float("nan")
    s.update(
        episodes_evaluated=len(results), episodes_skipped=len(skipped),
        episodes_total=total,
        episodes_off_navmesh=sum(
            1 for r in results if not np.isfinite(r["distance_to_goal_m"])),
    )
    return s


def _jsonable(v, ndigits=3):
    """Round floats and turn inf/nan into null, so the line is valid strict JSON.

    `json.dumps` emits bare `Infinity` for a non-finite float, which most parsers reject;
    an off-navmesh episode would otherwise produce a line that will not load.
    """
    if isinstance(v, float):
        return round(v, ndigits) if np.isfinite(v) else None
    return v


def write_episode_lines(path, results, skipped):
    """One compact JSON object per line: the at-a-glance view of a run.

    Skipped episodes are included, with `status` and `reason`, so the file is a complete
    record of every episode the run considered -- reading only the evaluated ones would
    silently overstate coverage. Sorted by scene then episode so runs diff cleanly.
    """
    rows = []
    for r in results:
        rows.append({
            "uid": r["uid"], "scene": r["scene"], "episode_id": r["episode_id"],
            "category": r["object_category"], "status": "evaluated",
            "success": int(r["success"]), "oracle_success": int(r["oracle_success"]),
            "spl": _jsonable(r["spl"]), "oracle_spl": _jsonable(r["oracle_spl"]),
            "start_m": _jsonable(r["geodesic_start_m"], 2),
            "final_m": _jsonable(r["distance_to_goal_m"], 2),
            "min_m": _jsonable(r["min_distance_to_goal_m"], 2),
            "path_m": _jsonable(r["path_length_m"], 2),
            "steps": r["steps"],
            "video": Path(r["video"]).name if r.get("video") else None,
        })
    for s in skipped:
        rows.append({
            "uid": s["uid"], "scene": s["scene"], "episode_id": s["episode_id"],
            "category": s["object_category"], "status": "skipped",
            "reason": s["reason"],
            "start_m": _jsonable(s["geodesic_start_m"], 2),
            "start_dataset_m": _jsonable(s["geodesic_start_dataset_m"], 2),
        })
    rows.sort(key=lambda r: (r["scene"], r["episode_id"]))
    Path(path).write_text(
        "\n".join(json.dumps(r, separators=(",", ":")) for r in rows) + "\n")
    return path


def print_summary(summary, results, skipped):
    print(f"\n=== ObjectNav evaluation: {summary['episodes_evaluated']}/"
          f"{summary['episodes_total']} episodes evaluated, "
          f"{summary['episodes_skipped']} skipped ===")
    if skipped:
        print("  skipped (unwinnable for this robot, NOT counted as failures):")
        by_reason = defaultdict(list)
        for sk in skipped:
            by_reason[sk["reason"]].append(sk)
        for reason, items in by_reason.items():
            print(f"    {len(items)}x  {reason}")
            for sk in items:
                print(f"        {sk['uid']:<28} {sk['object_category']:<14} "
                      f"dataset-navmesh geodesic {sk['geodesic_start_dataset_m']:.2f} m")
    if not results:
        print("  no episodes were evaluated")
        return
    print(f"  success          {summary['success']:.3f}")
    print(f"  oracle_success   {summary['oracle_success']:.3f}")
    print(f"  spl              {summary['spl']:.3f}")
    print(f"  oracle_spl       {summary['oracle_spl']:.3f}")
    print(f"  final distance   {summary['distance_to_goal_m']:.2f} m  "
          f"(min reached {summary['min_distance_to_goal_m']:.2f} m, "
          f"start {summary['geodesic_start_m']:.2f} m)")
    if summary.get("episodes_off_navmesh"):
        print(f"                   {summary['episodes_off_navmesh']} episode(s) ended off "
              f"the navmesh (geodesic inf); scored as failures, excluded from the distance "
              f"means above")
    print(f"  path length      {summary['path_length_m']:.2f} m over "
          f"{summary['steps']:.0f} policy steps")
    by_cat = defaultdict(list)
    for r in results:
        by_cat[r["object_category"]].append(r)
    print("  by category:")
    for cat, rs in sorted(by_cat.items()):
        print(f"    {cat:<16} n={len(rs):<3} success {np.mean([r['success'] for r in rs]):.2f}  "
              f"oracle {np.mean([r['oracle_success'] for r in rs]):.2f}  "
              f"spl {np.mean([r['spl'] for r in rs]):.3f}")


# ======================================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--i-know-this-is-deprecated", dest="deprecated_ok",
                   action="store_true",
                   help="DEPRECATED HARNESS. Required to run a rollout. This script "
                        "predates the HabitatRobotSim.reset() state-leakage fix and has "
                        "no --fresh-sim-per-episode, so its numbers are order-dependent "
                        "and understated (checkpoint-2500 on val_mini: success 0.214 "
                        "here vs 0.357 with a clean slate). Use "
                        "/Projects/habitat_physical_nav/scripts/eval_objectnav_policy.py "
                        "--fresh-sim-per-episode instead. --report and --dry-run do not "
                        "need this flag")
    p.add_argument("--ckpt", help="trained checkpoint dir (from train_vector_sft)")
    p.add_argument("--episodes", required=True,
                   help="ObjectNav split dir (with content/*.json.gz) or a single .json.gz")
    p.add_argument("--scene-root", default="data/scene_datasets")
    p.add_argument("--output-dir", default="dump/objectnav_eval")

    sel = p.add_argument_group("episode selection")
    sel.add_argument("--num-episodes", type=int, default=0, help="0 = all")
    sel.add_argument("--scenes", default=None, help="comma-separated scene names")
    sel.add_argument("--categories", default=None, help="comma-separated goal categories")
    sel.add_argument("--episode-ids", default=None,
                     help="comma-separated episode ids, or scene:id for uniqueness")
    sel.add_argument("--max-scenes", type=int, default=0,
                     help="keep at most this many scenes, chosen at random under --seed. "
                          "Scene load is the fixed cost per worker, so this is the knob "
                          "that actually shrinks a run")
    sel.add_argument("--sample-per-scene", type=int, default=0,
                     help="sample at most this many episodes from each kept scene. With "
                          "--max-scenes this gives a stratified small subset of a big "
                          "split (MP3D val is 2195 episodes over 11 scenes)")
    sel.add_argument("--seed", type=int, default=0,
                     help="seed for --max-scenes / --sample-per-scene, so a subset is "
                          "reproducible from the flags alone")

    run = p.add_argument_group("rollout")
    run.add_argument("--max-steps", type=int, default=200, help="policy steps per episode")
    run.add_argument("--gap", type=int, default=DEFAULT_GAP,
                     help="control ticks executed per chunk, = the training observation "
                          "stride (5 ticks at 20 Hz = 4 Hz observations)")
    run.add_argument("--dt", type=float, default=DEFAULT_DT, help="control tick, seconds")
    run.add_argument("--success-distance", type=float, default=1.0,
                     help="geodesic distance to a goal view point that counts as reaching it")
    run.add_argument("--auto-stop", action="store_true",
                     help="end the episode on arrival; makes success == oracle_success")
    run.add_argument("--snap-start", action="store_true",
                     help="snap the spawn onto the robot navmesh instead of using the "
                          "episode's start position verbatim")
    run.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    run.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    run.add_argument("--sensor-uuid", default="color_sensor")

    vid = p.add_argument_group("video")
    vid.add_argument("--record-video", action="store_true",
                     help="one MP4 per episode, one frame per control tick, so playback at "
                          "1/--dt fps is real time")
    vid.add_argument("--video-scale", type=float, default=1.0)

    mdl = p.add_argument_group("policy")
    mdl.add_argument("--policy", default="model", choices=["model", "forward", "zero"],
                     help="CONTROL: 'forward' drives straight ahead and 'zero' stands "
                          "still, both ignoring the image. They verify the harness (and "
                          "bound a policy that learned nothing) rather than the checkpoint")
    mdl.add_argument("--device", default="cuda")
    mdl.add_argument("--max-context-tokens", type=int, default=110000)
    mdl.add_argument("--placeholder", default=None)
    mdl.add_argument("--policy-python", default=os.environ.get("VECTOR_POLICY_PYTHON"),
                     help="interpreter that can load the checkpoint, when this env cannot")

    par = p.add_argument_group("parallelism")
    par.add_argument("--gpus", default=None,
                     help="comma-separated GPU ids to shard over, e.g. '4,5,6,7'. "
                          "Default: a single worker on the current CUDA_VISIBLE_DEVICES")
    par.add_argument("--workers-per-gpu", type=int, default=1)
    p.add_argument("--report", default=None,
                   help="re-aggregate and re-print an existing results JSON (or a shards/ "
                        "directory of per-shard results) and exit. Recovers the summary "
                        "after a partial run without repeating any rollout")
    par.add_argument("--dry-run", action="store_true",
                     help="screen the episodes and print what would run, no model, no sim "
                          "rollout")

    p.add_argument("--worker-shard", default=None, help=argparse.SUPPRESS)
    p.add_argument("--shard-index", type=int, default=0, help=argparse.SUPPRESS)
    return p.parse_args(argv)


def worker_main(args):
    """The `--worker-shard` role: run the episodes listed in a shard file."""
    shard = json.loads(Path(args.worker_shard).read_text())
    wanted = set(shard["episodes"])
    episodes = [e for e in load_episodes(Path(args.episodes)) if uid(e) in wanted]
    if len(episodes) != len(wanted):
        raise SystemExit(f"shard {args.shard_index}: resolved {len(episodes)} of "
                         f"{len(wanted)} episode uids -- the episode list is not stable")
    results, skipped = run_shard(episodes, args)
    Path(shard["out"]).write_text(
        json.dumps({"results": results, "skipped": skipped}, indent=2, default=float))
    print(f"[shard {args.shard_index}] done: {len(results)} evaluated, {len(skipped)} skipped",
          flush=True)


def dry_run(episodes, args):
    """Screen every episode without running the policy, and print the verdict table."""
    from continuous_demos.episode_io import DEFAULT_URDF_PATH, build_robot_sim

    by_scene = defaultdict(list)
    for ep in episodes:
        by_scene[ep["scene_id"]].append(ep)
    runnable, skipped = [], []
    for scene_id, eps in by_scene.items():
        scene_path, scene_cfg = resolve_scene(eps[0], args.scene_root)
        sim = build_robot_sim(scene_path, scene_cfg, 160, 120, str(DEFAULT_URDF_PATH))
        robot_pf, data_pf = sim.sim.pathfinder, dataset_pathfinder(scene_path)
        print(f"\n=== {Path(scene_path).stem}: {len(eps)} episodes")
        print(f"    robot navmesh   {robot_pf.navigable_area:7.1f} m2 / "
              f"{robot_pf.num_islands} islands")
        if data_pf:
            print(f"    dataset navmesh {data_pf.navigable_area:7.1f} m2 / "
                  f"{data_pf.num_islands} islands")
        for ep in eps:
            ok, reason, d0, d0_ds = screen_episode(ep, robot_pf, data_pf,
                                                   args.success_distance)
            print(f"    {uid(ep):<26} {ep['object_category']:<14} "
                  f"dataset {d0_ds:7.2f} m | robot {d0:7.2f} m"
                  f"{'' if ok else '   SKIP: ' + reason}")
            (runnable if ok else skipped).append(ep)
        sim.sim.close()
    print(f"\n{len(runnable)}/{len(episodes)} episodes runnable, {len(skipped)} skipped")


def report_only(path: Path):
    """Rebuild the summary from results already on disk, and rewrite it in place."""
    if path.is_dir():
        payloads = [json.loads(p.read_text()) for p in sorted(path.glob("*_results.json"))]
        results = [r for p in payloads for r in p.get("results", [])]
        skipped = [s for p in payloads for s in p.get("skipped", [])]
        out_path = path / "objectnav_eval_results.json"
        payload = {"results": results, "skipped": skipped}
    else:
        payload = json.loads(path.read_text())
        results, skipped, out_path = payload["results"], payload["skipped"], path
    summary = summarize(results, skipped,
                        payload.get("summary", {}).get("episodes_total",
                                                       len(results) + len(skipped)))
    print_summary(summary, results, skipped)
    payload["summary"] = summary
    out_path.write_text(json.dumps(payload, indent=2, default=float))
    lines_path = write_episode_lines(out_path.with_suffix(".jsonl"), results, skipped)
    print(f"\nResults -> {out_path}")
    print(f"Per-episode -> {lines_path} (one line per episode)")


DEPRECATION_BANNER = """
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  DEPRECATED HARNESS -- the numbers it produces are contaminated.

  This script predates the discovery that HabitatRobotSim.reset() does not fully
  restore the physics state between episodes, and it has no --fresh-sim-per-episode
  equivalent. Episodes therefore affect each other's results depending on execution
  order. Measured on checkpoint-2500 / val_mini, correcting for it moved success
  from 0.214 to 0.357 -- roughly 40% relative understatement.

  Use instead:
      /Projects/habitat_physical_nav/scripts/eval_objectnav_policy.py \\
          --fresh-sim-per-episode

  See /Projects/spatial_training/dump/eval_system/FINDINGS.md section 5.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""


def main():
    args = parse_args()
    # Printed on every invocation, including --report and --dry-run: the point of
    # the notice is that two similarly-named harnesses exist and one is quietly
    # wrong, and someone reading a --report of an old run needs to know its numbers
    # carry the defect too.
    print(DEPRECATION_BANNER, file=sys.stderr, flush=True)
    if args.report:
        return report_only(Path(args.report))
    if args.worker_shard:
        # A worker is only ever spawned by a parent that already cleared the gate
        # below, so re-checking here would just break the parallel path.
        return worker_main(args)

    episodes = filter_episodes(load_episodes(Path(args.episodes)), args)
    if not episodes:
        raise SystemExit("no episodes selected")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(episodes)} episode(s) from {args.episodes}")

    if args.dry_run:
        # Screening only -- no rollout, so no contaminated numbers to produce.
        return dry_run(episodes, args)
    if not args.deprecated_ok:
        raise SystemExit(
            "refusing to run: this harness is deprecated and its closed-loop numbers "
            "are order-dependent (see the banner above and the module docstring).\n"
            "  Use  /Projects/habitat_physical_nav/scripts/eval_objectnav_policy.py "
            "--fresh-sim-per-episode\n"
            "  If you genuinely need this one -- to reproduce a historical run, say -- "
            "pass --i-know-this-is-deprecated.\n"
            "  That flag does not make the numbers correct. It only records that you "
            "chose them knowingly."
        )
    if not args.ckpt and args.policy == "model":
        raise SystemExit("--ckpt is required unless --dry-run or --policy is a control")

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()] if args.gpus else [None]
    slots = [g for g in gpus for _ in range(max(1, args.workers_per_gpu))]

    if len(slots) == 1 and slots[0] is None:
        results, skipped = run_shard(episodes, args)
    else:
        results, skipped = run_parallel(episodes, slots, args)

    summary = summarize(results, skipped, len(episodes))
    print_summary(summary, results, skipped)
    results.sort(key=lambda r: r["uid"])
    out_path = out_dir / (f"objectnav_eval_results"
                          f"{'' if args.policy == 'model' else '_' + args.policy}.json")
    out_path.write_text(json.dumps(
        {"summary": summary, "results": results, "skipped": skipped,
         "config": {k: v for k, v in vars(args).items() if k != "worker_shard"}},
        indent=2, default=float))
    lines_path = write_episode_lines(out_path.with_suffix(".jsonl"), results, skipped)
    print(f"\nResults -> {out_path}")
    print(f"Per-episode -> {lines_path} (one line per episode)")
    if args.record_video:
        print(f"Videos  -> {out_dir / 'videos'} "
              f"(real time at {round(1.0 / args.dt)} fps)")


def shard_episodes(episodes, n_slots):
    """Split episodes across workers, keeping scene locality but filling every worker.

    Scene load dominates a short episode, so episodes of one scene are kept together --
    but only up to a point: sharding purely by scene would idle every worker beyond the
    scene count (val_mini has two scenes, so four GPUs would leave two doing nothing).
    Scenes are therefore cut into blocks no larger than an even share and the blocks are
    packed onto the least-loaded worker, so a scene is reloaded only when that is the
    price of using an otherwise idle GPU.
    """
    by_scene = defaultdict(list)
    for ep in episodes:
        by_scene[ep["scene_id"]].append(ep)
    block = max(1, -(-len(episodes) // n_slots))
    groups = [eps[i:i + block]
              for _scene, eps in sorted(by_scene.items(), key=lambda kv: -len(kv[1]))
              for i in range(0, len(eps), block)]
    shards = [[] for _ in range(n_slots)]
    for g in sorted(groups, key=len, reverse=True):
        min(shards, key=len).extend(g)
    return shards


def run_parallel(episodes, slots, args):
    """Shard the episodes over GPUs and re-run this script once per shard.

    Sharding is round-robin over *scenes* rather than episodes so that a worker loads each
    scene once; scene load dominates a short episode. Workers are separate processes
    because CUDA_VISIBLE_DEVICES has to be set before habitat_sim and torch initialise.
    """
    shards = shard_episodes(episodes, len(slots))

    shard_dir = Path(args.output_dir) / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    procs = []
    for i, (gpu, eps) in enumerate(zip(slots, shards)):
        if not eps:
            continue
        spec = shard_dir / f"shard_{i}.json"
        out = shard_dir / f"shard_{i}_results.json"
        spec.write_text(json.dumps({"out": str(out), "episodes": [uid(e) for e in eps]},
                                   indent=2))
        env = dict(os.environ)
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env.setdefault("MAGNUM_LOG", "quiet")
        env.setdefault("HABITAT_SIM_LOG", "quiet")
        cmd = [sys.executable, str(Path(__file__).resolve()),
               "--worker-shard", str(spec), "--shard-index", str(i)] + _worker_argv(args)
        log = shard_dir / f"shard_{i}.log"
        print(f"  shard {i}: {len(eps)} episode(s) on GPU {gpu} -> {log}")
        procs.append((i, out, log, subprocess.Popen(
            cmd, env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT)))

    results, skipped, failed = [], [], []
    for i, out, log, proc in procs:
        rc = proc.wait()
        if rc != 0 or not out.exists():
            failed.append(i)
            print(f"  shard {i} FAILED (exit {rc}); see {log}")
            print("  " + "\n  ".join(Path(log).read_text().splitlines()[-15:]))
            continue
        payload = json.loads(out.read_text())
        results.extend(payload["results"])
        skipped.extend(payload["skipped"])
        print(f"  shard {i} ok: {len(payload['results'])} evaluated, "
              f"{len(payload['skipped'])} skipped")
    if failed:
        print(f"\nWARNING: {len(failed)} shard(s) failed: {failed}. "
              f"The summary below covers only the shards that completed.")
    return results, skipped


def _worker_argv(args):
    """Rebuild this invocation's flags for a worker, minus the parallelism controls."""
    argv = ["--episodes", str(args.episodes), "--policy", args.policy,
            "--scene-root", str(args.scene_root), "--output-dir", str(args.output_dir),
            "--max-steps", str(args.max_steps), "--gap", str(args.gap),
            "--dt", str(args.dt), "--success-distance", str(args.success_distance),
            "--width", str(args.width), "--height", str(args.height),
            "--sensor-uuid", args.sensor_uuid, "--device", args.device,
            "--max-context-tokens", str(args.max_context_tokens),
            "--video-scale", str(args.video_scale)]
    if args.ckpt:
        argv += ["--ckpt", str(args.ckpt)]
    if args.record_video:
        argv.append("--record-video")
    if args.auto_stop:
        argv.append("--auto-stop")
    if args.snap_start:
        argv.append("--snap-start")
    if args.placeholder:
        argv += ["--placeholder", args.placeholder]
    if args.policy_python:
        argv += ["--policy-python", args.policy_python]
    return argv


if __name__ == "__main__":
    main()
