# based on official dwpp code:
# https://github.com/decwest/dwpp
import numpy as np
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

@dataclass
class ControllerConfig:
    """Configuration parameters for the Pure Pursuit controller variants."""
    min_look_ahead_distance: float
    max_look_ahead_distance: float
    look_ahead_time: float
    v_max: float
    v_min: float
    w_max: float
    w_min: float
    a_max: float
    aw_max: float
    dt: float
    approach_velocity_scaling_dist: float
    min_approach_linear_velocity: float
    goal_tolerance_dist: float  # Note: preserved original spelling (torelance)
    regulated_linear_scaling_min_radius: float
    regulated_linear_scaling_min_speed: float


class PurePursuitController:
    """
    Encapsulates Pure Pursuit, Adaptive Pure Pursuit, Regulated Pure Pursuit, 
    and Dynamic Window Pure Pursuit logic.
    """
    def __init__(self, config: ControllerConfig, method_name: str = "dwpp"):
        self.config = config
        self.method_name = method_name
        self.path: Optional[np.ndarray] = None
        self._path_distances: Optional[np.ndarray] = None

    def set_path(self, path: np.ndarray) -> None:
        """
        Sets the reference path and pre-calculates cumulative path distances 
        to optimize control loop performance.
        
        Args:
            path: np.ndarray of shape (N, 2) or (N, 3) representing the path coordinates.
        """
        self.path = self.densify_path(path, resolution=self.config.min_look_ahead_distance/4.0)
        self._path_distances = self._calc_path_distances(self.path)

    def compute_velocity(self, current_pose: np.ndarray, current_velocity: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Calculates the next optimal velocity command based on the selected pursuit method.
        
        Args:
            current_pose: The robot's current pose array [x, y, theta].
            current_velocity: The robot's current velocity array [v, w].
            
        Returns:
            next_velocity_ref: The calculated command velocity [v_ref, w_ref].
            debug_info: Dictionary containing look_ahead_pos, break_constraints_flag, 
                        curvature, and regulated_v.
        """
        if self.path is None or self._path_distances is None:
            raise ValueError("Path not set. Call set_path() first.")

        # calc index of current position
        current_idx = self._calc_index(current_pose)
        
        # calc look ahead distance (Adaptive Pure Pursuit)
        look_ahead_distance = self._calc_look_ahead_distance(current_velocity)
        
        # calc curvature to the look ahead position
        curvature, look_ahead_pos = self._calc_curvature_to_look_ahead_position(
            current_pose, current_idx, look_ahead_distance
        )
        
        if self.method_name in ["rpp", "dwpp"]:
            # calc regulated translational velocity (Regulated Pure Pursuit)
            regulated_v = self._calc_regulated_translational_velocity(curvature)
        else:
            regulated_v = self.config.v_max
        
        if self.method_name in ["pp", "app", "rpp"]:
            # calc translational velocity
            v_ref = self._calc_reference_translational_velocity(current_pose, self.path[-1])
            
            # regulate translational velocity
            if self.method_name == "rpp":
                v_ref = min(v_ref, regulated_v)
            
            # calc angular velocity
            w_ref = curvature * v_ref
            next_velocity_ref = np.array([v_ref, w_ref])
            
        else:
            # decide accel or decel
            is_accel = self._decide_accel_or_decel(current_idx)
            # calc dynamic window and optimal next velocity
            next_velocity_ref = self._calc_optimal_velocity_considering_dynamic_window(
                current_velocity, regulated_v, curvature, is_accel
            )

        break_constraints_flag = self._evaluate_accelaration_constraints(current_velocity, next_velocity_ref)
        
        debug_info = {
            "look_ahead_pos": look_ahead_pos,
            "break_constraints_flag": break_constraints_flag,
            "curvature": curvature,
            "regulated_v": regulated_v
        }
        
        return next_velocity_ref, debug_info

    # ---------------------------------------------------------
    # Private helper methods
    # ---------------------------------------------------------

    def _evaluate_accelaration_constraints(self, current_velocity: np.ndarray, next_velocity_ref: np.ndarray) -> list[bool]:
        """Checks if the proposed velocity violates maximum acceleration/deceleration limits."""
        break_constraints_flag = [False, False]
        if current_velocity[0] - self.config.a_max * self.config.dt > next_velocity_ref[0] or \
           current_velocity[0] + self.config.a_max * self.config.dt < next_velocity_ref[0]:
            break_constraints_flag[0] = True
        if current_velocity[1] - self.config.aw_max * self.config.dt > next_velocity_ref[1] or \
           current_velocity[1] + self.config.aw_max * self.config.dt < next_velocity_ref[1]:
            break_constraints_flag[1] = True
        
        return break_constraints_flag

    def _calc_reference_translational_velocity(self, current_pose: np.ndarray, goal_pose: np.ndarray) -> float:
        """Scales down the reference velocity as the robot approaches the goal."""
        v_ref = self.config.v_max
        
        distance_to_goal = float(np.linalg.norm(goal_pose[:2] - current_pose[:2]))
        if distance_to_goal < self.config.approach_velocity_scaling_dist:
            v_ref = max(v_ref * distance_to_goal / self.config.approach_velocity_scaling_dist, 
                        self.config.min_approach_linear_velocity)
        if distance_to_goal < self.config.goal_tolerance_dist:
            v_ref = 0.0
        
        return v_ref

    def _calc_index(self, current_pose: np.ndarray) -> np.intp:
        """Finds the index of the closest point on the reference path to the current pose."""
        distances = np.linalg.norm(self.path[:, :2] - current_pose[:2], axis=1)
        idx = np.argmin(distances)
        return idx

    def _calc_path_distances(self, path: np.ndarray) -> np.ndarray:
        """Computes the cumulative distance along the path array."""
        differences = np.diff(path, axis=0)
        distances = np.linalg.norm(differences, axis=1)
        path_distances = np.concatenate(([0.0], np.cumsum(distances)))
        return path_distances

    def _calc_look_ahead_distance(self, current_velocity: np.ndarray) -> float:
        """Dynamically calculates the look-ahead distance based on current speed."""
        if self.method_name in ["app", "rpp", "dwpp", "dwpp_wo_rpp"]:
            look_ahead_distance = self.config.look_ahead_time * current_velocity[0]
            look_ahead_distance = min(max(look_ahead_distance, self.config.min_look_ahead_distance), 
                                      self.config.max_look_ahead_distance)
        else:
            look_ahead_distance = self.config.min_look_ahead_distance
            
        return look_ahead_distance

    def _calc_curvature_to_look_ahead_position(self, current_pose: np.ndarray, current_idx: np.intp, look_ahead_distance: float) -> Tuple[float, np.ndarray]:
        """Calculates the target look-ahead coordinate and the required path curvature to reach it."""
        current_distance = self._path_distances[current_idx]
        look_ahead_pos_distance = current_distance + look_ahead_distance
        look_ahead_idx = min(np.searchsorted(self._path_distances, look_ahead_pos_distance), len(self.path) - 1)
        look_ahead_pos = self.path[look_ahead_idx]
        
        look_ahead_angle = (math.atan2(look_ahead_pos[1] - current_pose[1], look_ahead_pos[0] - current_pose[0]) - current_pose[2])
        L = float(np.linalg.norm(look_ahead_pos - current_pose[:2]))
        curvature = 2.0 * math.sin(look_ahead_angle) / L
        
        return curvature, look_ahead_pos

    def _calc_regulated_translational_velocity(self, curvature: float) -> float:
        """Reduces the maximum linear velocity during sharp turns to maintain stability."""
        if curvature == 0.0:
            return self.config.v_max
        
        curvature_radius = 1.0 / abs(curvature)
        if curvature_radius <= self.config.regulated_linear_scaling_min_radius:
            regulated_v = self.config.v_max * curvature_radius / self.config.regulated_linear_scaling_min_radius
        else:
            regulated_v = self.config.v_max
        
        regulated_v = max(regulated_v, self.config.regulated_linear_scaling_min_speed)
        return regulated_v

    def _decide_accel_or_decel(self, current_idx: np.intp) -> bool:
        """Determines if the robot should accelerate or decelerate based on distance to goal vs braking distance."""
        goal_distance = self._path_distances[-1] - self._path_distances[current_idx]
        decel_distance = (self.config.v_max ** 2) / (2 * self.config.a_max)
        
        if goal_distance > decel_distance:
            return True
        else:
            return False

    def _calc_optimal_velocity_considering_dynamic_window(self, current_velocity: np.ndarray, regulated_v: float, curvature: float, is_accel: bool) -> np.ndarray:
        """Intersects the curvature constraint with the dynamic window limits to find optimal reachable velocities."""
        dw_vmax = min(current_velocity[0] + self.config.a_max * self.config.dt, self.config.v_max)
        dw_vmin = max(current_velocity[0] - self.config.a_max * self.config.dt, self.config.v_min)
        dw_wmax = min(current_velocity[1] + self.config.aw_max * self.config.dt, self.config.w_max)
        dw_wmin = max(current_velocity[1] - self.config.aw_max * self.config.dt, self.config.w_min)
        
        if dw_vmax > regulated_v:
            dw_vmax = max(dw_vmin, regulated_v)
        
        velocity_candidates = []
        p1 = (dw_vmin, curvature * dw_vmin)
        p2 = (dw_vmax, curvature * dw_vmax)
        velocity_candidates.append(p1)
        velocity_candidates.append(p2)
        if curvature != 0.0:
            p3 = (dw_wmin / curvature, dw_wmin)
            p4 = (dw_wmax / curvature, dw_wmax)
            velocity_candidates.append(p3)
            velocity_candidates.append(p4)
            
        valid_velocity_candidates = []
        for v in velocity_candidates:
            if dw_vmin <= v[0] <= dw_vmax and dw_wmin <= v[1] <= dw_wmax:
                valid_velocity_candidates.append(v)
        
        if len(valid_velocity_candidates) > 0:
            valid_velocity_candidates.sort(key=lambda x: x[0])
            if is_accel:
                next_velocity = valid_velocity_candidates[-1]
            else:
                next_velocity = valid_velocity_candidates[0]
        else:
            distance_from_coords = []
            dw_coords = [
                (dw_vmin, dw_wmin),
                (dw_vmin, dw_wmax),
                (dw_vmax, dw_wmin),
                (dw_vmax, dw_wmax),
            ]
            for p in dw_coords:
                dist = abs(curvature * p[0] - p[1]) / math.sqrt(curvature**2 + 1)
                distance_from_coords.append(dist)
            
            min_dist = min(distance_from_coords)
            min_dist_dw_coords = []
            for p, dist in zip(dw_coords, distance_from_coords):
                if dist == min_dist:
                    min_dist_dw_coords.append(p)
            
            min_dist_dw_coords.sort(key=lambda x: x[0])
            if is_accel:
                next_velocity = min_dist_dw_coords[-1]
            else:
                next_velocity = min_dist_dw_coords[0]
        
        return np.array(next_velocity)

    def densify_path(self,path: np.ndarray, resolution: float = 0.1) -> np.ndarray:
        """Linearly interpolates between waypoints to guarantee a maximum spatial resolution."""
        dense_path = [path[0]]
        for i in range(len(path) - 1):
            p1, p2 = path[i], path[i+1]
            dist = np.linalg.norm(p2[:2] - p1[:2])
            num_points = int(np.ceil(dist / resolution))
            
            if num_points > 1:
                # Generate intermediate points (skipping the start to avoid duplicates)
                interp = np.linspace(p1, p2, num_points + 1)[1:]
                dense_path.extend(interp)
            else:
                dense_path.append(p2)
                
        return np.array(dense_path)