"""Kinematic drop-in for the HabitatRobotSim stack: cylinder agent, no physics.

Motivation (2026-08-30): isolate the EMBODIMENT term of the sim-to-sim gap. The
physical stack simulates the robot in Bullet and PID-tracks each chunk, so executed
motion carries tracking lag. This body executes the chunk FAITHFULLY: each tick's
setpoint pose is applied directly, constrained only by the navmesh
(``pathfinder.try_step`` -- the same mechanism habitat-lab's ``VelocityAction`` and
``BaseVelAction`` use). No controller in the loop; what the policy commands is what
happens, up to navmesh sliding.

``KinematicRobotSim`` duck-types the subset of ``continuous_demos.sim.HabitatRobotSim``
that the eval stack actually touches (task layer, executor, screening, the actor);
``KinematicChunkExecutor`` replaces the PID ``track_trajectory`` inner loop of
``objectnav_eval.executor.ChunkExecutor`` with exact per-tick placement. Camera
geometry (mount offset + hfov) is parsed from the SAME URDF the physical rig loads,
so the rendered viewpoint matches the corpus by construction.

Planar control frame throughout, identical to ``continuous_demos.sim``:
X = habitat_x, Y = -habitat_z, theta CCW from +X.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def _camera_mount_from_urdf(urdf_path: str) -> np.ndarray:
    """Agent-local habitat-frame camera position from the robot URDF.

    URDF frame: x forward, y left, z up. Habitat agent frame: -z forward, y up,
    x right. base_camera_joint origin + joint_x's own z offset give the mount.
    """
    root = ET.parse(urdf_path).getroot()
    base = np.zeros(3)
    cam = None
    for j in root.findall("joint"):
        o = j.find("origin")
        xyz = np.array([float(v) for v in (o.get("xyz") or "0 0 0").split()]) if o is not None else np.zeros(3)
        if j.get("name") == "joint_x":
            base = xyz
        if j.get("name") == "base_camera_joint":
            cam = xyz
    if cam is None:
        raise ValueError(f"no base_camera_joint in {urdf_path}")
    x_f, y_l, z_u = cam + np.array([0.0, 0.0, base[2]])
    return np.array([-y_l, z_u, -x_f])   # habitat [right, up, back]


class KinematicRobotSim:
    """Cylinder-agent habitat sim; motion = twist/pose integration on the navmesh."""

    def __init__(self, scene_path, scene_dataset_config: Optional[str],
                 width: int, height: int, urdf_path: str, hfov_deg: float = 79.0):
        import habitat_sim

        self._hs = habitat_sim
        cam_pos = _camera_mount_from_urdf(str(urdf_path))
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = str(scene_path)
        if scene_dataset_config:
            sim_cfg.scene_dataset_config_file = str(scene_dataset_config)
        sim_cfg.enable_physics = False
        spec = habitat_sim.sensor.CameraSensorSpec()
        spec.uuid = "color_sensor"
        spec.sensor_type = habitat_sim.sensor.SensorType.COLOR
        spec.resolution = [int(height), int(width)]
        spec.hfov = float(hfov_deg)
        spec.position = [float(v) for v in cam_pos]
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = [spec]
        self.agent_id = 0
        self.sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
        self._agent = self.sim.get_agent(self.agent_id)
        self._scene_path = str(scene_path)
        self._scene_dataset_config = scene_dataset_config
        self._spec = spec
        self._pose2d = np.zeros(3)
        self._nav_y = 0.0
        self._twist_world = np.zeros(3)     # [vX, vY, w] planar world frame
        self._navmesh_recomputed = False    # read by apply_navmesh_choice
        print(f"[kinematic_sim] camera mount (habitat frame) {np.round(cam_pos, 3)}, hfov {hfov_deg}")

    # -- pose plumbing (exact inverses of each other) -----------------------------------
    def _write_state(self) -> None:
        st = self._hs.agent.AgentState()
        X, Y, th = self._pose2d
        st.position = np.array([X, self._nav_y, -Y], dtype=np.float32)
        phi = th - math.pi / 2.0            # forward (-sin phi, 0, -cos phi) -> theta
        import quaternion  # numpy-quaternion, a habitat_sim dependency

        st.rotation = quaternion.from_rotation_vector([0.0, phi, 0.0])
        self._agent.set_state(st, reset_sensors=False)

    def _read_pose_from(self, position, rotation_xyzw) -> None:
        import quaternion

        q = np.quaternion(rotation_xyzw[3], *rotation_xyzw[:3])
        f = quaternion.rotate_vectors(q, np.array([0.0, 0.0, -1.0]))
        self._pose2d = np.array([position[0], -position[2],
                                 math.atan2(-f[2], f[0])])
        self._nav_y = float(position[1])

    @property
    def robot(self):
        """Tiny proxy for the task layer's height derivation (NavSimAdapter._height):
        cumulative_bb.min.y == 0 and translation[1] == navmesh y makes it return the
        navmesh-level height exactly -- which is what a cylinder agent's height is."""
        import types

        import magnum as mn

        node = types.SimpleNamespace(cumulative_bb=mn.Range3D(
            mn.Vector3(0.0, 0.0, 0.0), mn.Vector3(0.0, 0.0, 0.0)))
        return types.SimpleNamespace(
            root_scene_node=node,
            translation=[self._pose2d[0], self._nav_y, -self._pose2d[1]])

    # -- HabitatRobotSim surface --------------------------------------------------------
    def reset(self, scene_id: Optional[str] = None, robot_translation=None,
              robot_rotation=None, snap_to_navmesh: bool = True,
              vertical_offset: Optional[float] = None,
              home_joint_positions=None, scene_dataset_config: Optional[str] = None) -> dict:
        if scene_id is not None and str(scene_id) != self._scene_path:
            raise NotImplementedError(
                "KinematicRobotSim is built per scene (the actor builds one sim per "
                f"actor); asked to switch {self._scene_path} -> {scene_id}")
        if robot_translation is not None:
            pos = np.asarray(robot_translation, dtype=np.float64).copy()
            if snap_to_navmesh:
                snapped = np.array(self.sim.pathfinder.snap_point(pos))
                if np.isfinite(snapped).all():
                    pos = snapped
            rot = list(robot_rotation) if robot_rotation is not None else [0, 0, 0, 1]
            self._read_pose_from(pos, rot)
            self._write_state()
        self._twist_world[:] = 0.0
        return {}

    def step(self, chassis_twist, joint_targets=None, twist_frame: str = "robot",
             dt: float = 0.04) -> dict:
        """Integrate a [v_x, v_y, w] twist for dt on the navmesh (PID-compatible path)."""
        v = np.asarray(chassis_twist, dtype=np.float64)
        X, Y, th = self._pose2d
        if twist_frame == "robot":
            c, s = math.cos(th), math.sin(th)
            vw = np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]])
        else:
            vw = v.copy()
        self._teleport_planar(X + vw[0] * dt, Y + vw[1] * dt, th + vw[2] * dt)
        self._twist_world = vw
        return {}

    def execute_pose(self, pose2d) -> np.ndarray:
        """Exact-mode primitive: place the base at pose2d, navmesh-clamped."""
        self._teleport_planar(*np.asarray(pose2d, dtype=np.float64))
        return self.get_2d_pose()

    def _teleport_planar(self, X: float, Y: float, th: float) -> None:
        cur = np.array([self._pose2d[0], self._nav_y, -self._pose2d[1]])
        target = np.array([X, self._nav_y, -Y])
        end = np.array(self.sim.pathfinder.try_step(cur, target))
        self._pose2d = np.array([end[0], -end[2],
                                 (th + math.pi) % (2 * math.pi) - math.pi])
        self._nav_y = float(end[1])
        self._write_state()

    def get_2d_pose(self) -> np.ndarray:
        return self._pose2d.copy()

    def get_2d_pose_and_twist(self, virtual_heading=None):
        th = self._pose2d[2] if virtual_heading is None else virtual_heading
        v_fwd = self._twist_world[0] * math.cos(th) + self._twist_world[1] * math.sin(th)
        return self._pose2d.copy(), np.array([v_fwd, self._twist_world[2]])

    def get_joint_state(self, joint_name: str, is_vel: bool = False) -> float:
        if is_vel:
            return {"joint_x": self._twist_world[0], "joint_y": self._twist_world[1],
                    "joint_th": self._twist_world[2]}.get(joint_name, 0.0)
        if joint_name == "joint_x":
            return float(self._pose2d[0])
        if joint_name == "joint_y":
            return float(self._pose2d[1])
        if joint_name == "joint_th":
            return float(self._pose2d[2])
        return 0.0                                   # joint_z and friends: rigid body

    def is_stationary(self, lin_vel_threshold, ang_vel_threshold) -> bool:
        return (np.linalg.norm(self._twist_world[:2]) < lin_vel_threshold
                and abs(self._twist_world[2]) < ang_vel_threshold)

    def get_obs(self) -> dict:
        return self.sim.get_sensor_observations(self.agent_id)

    def close(self) -> None:
        self.sim.close()


class KinematicChunkExecutor:
    """ChunkExecutor-compatible exact executor: one setpoint applied per tick."""

    def __init__(self, robot_sim: KinematicRobotSim, controller=None, dt: float = 0.04,
                 gap: int = 10, collect_contacts: bool = False, log_camera: bool = False,
                 sensor_uuid: str = "color_sensor"):
        if collect_contacts:
            raise ValueError("kinematic body has no contact manifold; "
                             "set collision_penalty to 0")
        self.robot_sim = robot_sim
        self.dt = float(dt)
        self.gap = int(gap)
        self.log_camera = bool(log_camera)
        self.sensor_uuid = sensor_uuid
        self.tick = 0
        self.sim_time = 0.0

    def reset(self) -> None:
        self.tick = 0
        self.sim_time = 0.0

    def execute(self, chunk, chunk_index: int, on_tick=None, stop_when=None):
        from continuous_demos.action_chunking import relative_to_pose
        from objectnav_eval.executor import ChunkExecution, TickRecord

        poses = np.asarray(getattr(chunk, "poses", chunk), dtype=np.float64).reshape(-1, 3)
        gap = min(self.gap, len(poses))
        if gap == 0:
            raise ValueError("policy returned an empty action chunk")
        anchor = np.asarray(self.robot_sim.get_2d_pose(), dtype=np.float64)
        setpoints = np.stack([relative_to_pose(anchor, poses[j]) for j in range(gap)])
        reached = np.zeros((gap, 3), dtype=np.float64)
        ticks: List[Any] = []
        stopped_early = False
        previous_setpoint, previous_pose = anchor, anchor
        for j, setpoint in enumerate(setpoints):
            pose = self.robot_sim.execute_pose(setpoint)
            reached[j] = pose
            self.tick += 1
            self.sim_time += self.dt
            record = TickRecord(
                tick=self.tick, chunk_index=chunk_index, tick_in_chunk=j,
                setpoint=setpoint, previous_setpoint=previous_setpoint,
                pose=pose, previous_pose=previous_pose, sim_time=self.sim_time,
                saturation={}, contacts=None, height=0.0, camera=None)
            ticks.append(record)
            if on_tick is not None:
                on_tick(record)
            previous_setpoint, previous_pose = setpoint, pose
            if stop_when is not None and stop_when(record):
                stopped_early = True
                break
        return ChunkExecution(chunk_index=chunk_index, anchor=anchor,
                              setpoints=setpoints[: len(ticks)],
                              poses=reached[: len(ticks)], ticks=ticks,
                              stopped_early=stopped_early, chunk=poses)


def build_kinematic_sim(episode, config: Any) -> tuple:
    """Mirror of ``objectnav_eval.harness.build_scene_simulator`` for the kinematic body."""
    from continuous_demos.episode_io import DEFAULT_URDF_PATH

    from objectnav_eval.navmesh import apply_navmesh_choice, resolve_scene

    scene_path, scene_dataset_config = resolve_scene(episode, config.scene_root)
    robot_sim = KinematicRobotSim(scene_path, scene_dataset_config,
                                  config.width, config.height, str(DEFAULT_URDF_PATH))
    print(f"    navmesh         "
          f"{apply_navmesh_choice(robot_sim, scene_path, config.navmesh)}", flush=True)
    return robot_sim, scene_path
